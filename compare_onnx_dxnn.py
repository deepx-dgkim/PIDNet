#!/usr/bin/env python3
"""Compare PIDNet ONNX and DXNN output consistency (cosine similarity).

Dataset-level mIoU (see eval_cityscapes_onnx.py / eval_cityscapes_dxnn.py) can hide
a quantization-induced shift that happens to still pick the same argmax class most
of the time. This script instead compares the raw "masks" (Softmax) tensor the ONNX
model and the compiled DXNN model produce for the *same* image, at the network's
native 1/8-resolution stride, before any upsampling:

  - Per-pixel cosine similarity between the 19-class score vectors at each spatial
    location, pooled across every image in the dataset.
  - Per-pixel argmax agreement rate, as a cross-check against the mIoU delta between
    the two eval scripts: a large cosine-similarity drop with little argmax-agreement
    drop usually means quantization is shifting confidence without yet flipping the
    predicted class.

Usage:
    python compare_onnx_dxnn.py \\
        --onnx-model pretrained_dynamic.onnx \\
        --dxnn-model pidnet_argmax_softmax.dxnn \\
        --dataset-root cityscapes_small \\
        --save-json metrics/onnx_vs_dxnn.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from dx_engine import InferenceEngine
from tqdm import tqdm


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ONNX vs. DXNN PIDNet 'masks' output via cosine similarity."
    )
    parser.add_argument("--onnx-model", required=True, help="ONNX model path (e.g. pretrained_dynamic.onnx).")
    parser.add_argument("--dxnn-model", required=True, help="DXNN model path (e.g. pidnet_argmax_softmax.dxnn).")
    parser.add_argument(
        "--dataset-root",
        "--cityscapes-root",
        dest="dataset_root",
        default="cityscapes_small",
        help="Dataset root. Images are read from <dataset-root>/images unless --images-dir is given.",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Image directory. Defaults to <dataset-root>/images.",
    )
    parser.add_argument(
        "--limit",
        "--max-images",
        dest="limit",
        type=int,
        default=None,
        help="Compare only the first N images.",
    )
    parser.add_argument(
        "--dxnn-input-color",
        choices=("bgr", "rgb"),
        default="rgb",
        help=(
            "Color order sent to DXRT. Defaults to rgb, matching export_argmax_softmax_onnx.py's "
            "preprocessing -- see the README note on this exact pitfall."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.98,
        help="Per-pixel cosine-similarity threshold used to report a 'low similarity' ratio.",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=None,
        help="ONNX Runtime providers, e.g. CUDAExecutionProvider CPUExecutionProvider",
    )
    parser.add_argument("--save-json", default=None, help="Optional metrics JSON output path.")
    return parser.parse_args()


def find_images(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.glob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def choose_providers(requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    available = ort.get_available_providers()
    preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return [p for p in preferred if p in available] or available


def infer_static_input_size(shape: list[object]) -> tuple[int, int] | None:
    if len(shape) != 4:
        return None
    height, width = shape[2], shape[3]
    if isinstance(height, int) and isinstance(width, int) and height > 0 and width > 0:
        return height, width
    return None


def preprocess_onnx(image_bgr: np.ndarray, input_size: tuple[int, int]) -> np.ndarray:
    """Matches pidnet_demo_onnx.py::preprocess() -- always-normalized float32 NCHW."""
    height, width = input_size
    resized = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image_rgb -= IMAGENET_MEAN
    image_rgb /= IMAGENET_STD
    return np.expand_dims(image_rgb.transpose(2, 0, 1), axis=0).astype(np.float32)


def dxnn_input_layout(shape: list[int]) -> str:
    if len(shape) != 4:
        raise ValueError(f"Unsupported DXNN input shape: {shape}")
    if shape[-1] in (1, 3, 4):
        return "nhwc"
    if shape[1] in (1, 3, 4):
        return "nchw"
    raise ValueError(f"Could not infer DXNN input layout from shape: {shape}")


def dxnn_static_input_size(shape: list[int], layout: str) -> tuple[int, int]:
    height, width = (shape[1], shape[2]) if layout == "nhwc" else (shape[2], shape[3])
    if height <= 0 or width <= 0:
        raise ValueError(f"DXNN input shape is not static: {shape}")
    return height, width


def preprocess_dxnn(
    image_bgr: np.ndarray,
    input_size: tuple[int, int],
    layout: str,
    dtype: np.dtype,
    input_color: str,
) -> np.ndarray:
    """Matches pidnet_demo_dxnn.py::PIDNet.preprocess() -- dtype-adaptive normalization."""
    height, width = input_size
    resized = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    if input_color == "rgb":
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    if np.issubdtype(dtype, np.floating):
        tensor = resized.astype(np.float32) / 255.0
        tensor -= IMAGENET_MEAN
        tensor /= IMAGENET_STD
    else:
        tensor = resized.astype(dtype, copy=False)

    if layout == "nchw":
        tensor = tensor.transpose(2, 0, 1)

    return np.ascontiguousarray(np.expand_dims(tensor, axis=0), dtype=dtype)


def reshape_output(output: np.ndarray, shape: list[int] | None) -> np.ndarray:
    if output.ndim == 1 and shape and output.size == int(np.prod(shape)):
        return output.reshape(tuple(shape))
    return output


class OnnxMasksRunner:
    def __init__(self, model_path: Path, providers: list[str] | None) -> None:
        self.session = ort.InferenceSession(str(model_path), providers=choose_providers(providers))
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.input_size = infer_static_input_size(input_meta.shape) or (512, 512)
        output_names = {out.name for out in self.session.get_outputs()}
        if "masks" not in output_names:
            raise ValueError(f"ONNX model has no 'masks' output. Available outputs: {sorted(output_names)}")
        print(f"[onnx] {model_path}: providers={self.session.get_providers()}, input_size={self.input_size}")

    def masks(self, image_bgr: np.ndarray) -> np.ndarray:
        tensor = preprocess_onnx(image_bgr, self.input_size)
        (output,) = self.session.run(["masks"], {self.input_name: tensor})
        return np.squeeze(output, axis=0)  # [19, h, w]


class DxnnMasksRunner:
    def __init__(self, model_path: Path, input_color: str) -> None:
        self.engine = InferenceEngine(str(model_path))
        self.input_color = input_color

        input_info = self.engine.get_input_tensors_info()[0]
        output_info = self.engine.get_output_tensors_info()
        self.input_dtype = np.dtype(input_info["dtype"])
        self.input_layout = dxnn_input_layout(input_info["shape"])
        self.input_size = dxnn_static_input_size(input_info["shape"], self.input_layout)
        self.output_info = output_info
        self.output_names = [info["name"] for info in output_info]
        if "masks" not in self.output_names:
            raise ValueError(f"DXNN model has no 'masks' output. Available outputs: {self.output_names}")
        self.masks_index = self.output_names.index("masks")
        print(
            f"[dxnn] {model_path}: input_size={self.input_size}, "
            f"dtype={self.input_dtype.name}, layout={self.input_layout}, color={input_color}"
        )

    def close(self) -> None:
        self.engine.dispose()

    def masks(self, image_bgr: np.ndarray) -> np.ndarray:
        tensor = preprocess_dxnn(image_bgr, self.input_size, self.input_layout, self.input_dtype, self.input_color)
        outputs = self.engine.run([tensor])
        output = reshape_output(outputs[self.masks_index], self.output_info[self.masks_index].get("shape"))
        return np.squeeze(output, axis=0).astype(np.float32)  # [19, h, w]


def align_shapes(onnx_masks: np.ndarray, dxnn_masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if onnx_masks.shape == dxnn_masks.shape:
        return onnx_masks, dxnn_masks

    target_hw = onnx_masks.shape[-2:]
    resized = np.empty((dxnn_masks.shape[0], target_hw[0], target_hw[1]), dtype=np.float32)
    for class_idx in range(dxnn_masks.shape[0]):
        resized[class_idx] = cv2.resize(
            dxnn_masks[class_idx], (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR
        )
    return onnx_masks, resized


def per_pixel_cosine_similarity(onnx_masks: np.ndarray, dxnn_masks: np.ndarray) -> np.ndarray:
    """[19, H, W] x2 -> [H*W] cosine similarity of the 19-class vector at each pixel."""
    a = onnx_masks.reshape(onnx_masks.shape[0], -1).astype(np.float64)
    b = dxnn_masks.reshape(dxnn_masks.shape[0], -1).astype(np.float64)
    dot = (a * b).sum(axis=0)
    norm = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0) + 1e-12
    return dot / norm


def argmax_agreement(onnx_masks: np.ndarray, dxnn_masks: np.ndarray) -> float:
    onnx_pred = np.argmax(onnx_masks, axis=0)
    dxnn_pred = np.argmax(dxnn_masks, axis=0)
    return float((onnx_pred == dxnn_pred).mean())


def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    images_dir = Path(args.images_dir) if args.images_dir else dataset_root / "images"
    image_paths = find_images(images_dir)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {images_dir}")

    onnx_runner = OnnxMasksRunner(Path(args.onnx_model), args.providers)
    dxnn_runner = DxnnMasksRunner(Path(args.dxnn_model), args.dxnn_input_color)

    per_image: list[dict[str, Any]] = []
    all_pixel_sims: list[np.ndarray] = []

    try:
        for image_path in tqdm(image_paths, desc="Comparing"):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue

            onnx_masks = onnx_runner.masks(image)
            dxnn_masks = dxnn_runner.masks(image)
            onnx_masks, dxnn_masks = align_shapes(onnx_masks, dxnn_masks)

            pixel_sims = per_pixel_cosine_similarity(onnx_masks, dxnn_masks)
            all_pixel_sims.append(pixel_sims)

            per_image.append(
                {
                    "image": image_path.name,
                    "mean_cosine_similarity": float(pixel_sims.mean()),
                    "min_cosine_similarity": float(pixel_sims.min()),
                    "argmax_agreement": argmax_agreement(onnx_masks, dxnn_masks),
                }
            )
    finally:
        dxnn_runner.close()

    pooled = np.concatenate(all_pixel_sims) if all_pixel_sims else np.array([])
    summary = {
        "num_images": len(per_image),
        "mean_cosine_similarity": float(pooled.mean()) if pooled.size else None,
        "min_cosine_similarity": float(pooled.min()) if pooled.size else None,
        "low_similarity_ratio": float((pooled < args.threshold).mean()) if pooled.size else None,
        "threshold": args.threshold,
        "mean_argmax_agreement": float(np.mean([r["argmax_agreement"] for r in per_image])) if per_image else None,
    }

    print(
        "Summary: "
        f"images={summary['num_images']}, "
        f"mean_cos_sim={summary['mean_cosine_similarity']:.4f}, "
        f"min_cos_sim={summary['min_cosine_similarity']:.4f}, "
        f"low_sim_ratio(<{args.threshold})={summary['low_similarity_ratio']:.4f}, "
        f"mean_argmax_agreement={summary['mean_argmax_agreement']:.4f}"
    )

    if args.save_json:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w") as f:
            json.dump({"summary": summary, "per_image": per_image}, f, indent=2)
        print(f"Saved metrics to: {save_path}")


if __name__ == "__main__":
    main()
