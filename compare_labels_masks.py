#!/usr/bin/env python3
"""Check PIDNet labels/Softmax consistency for ONNX and DXNN outputs.

For logits ``z``, Softmax preserves class ordering, so a correctly converted model
must satisfy ``labels == argmax(masks)`` when both outputs come from the same logits.
This script measures that invariant inside each model and also compares the ONNX and
DXNN predictions with one another at the models' native output resolution.
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

from compare_onnx_dxnn import (
    choose_providers,
    dxnn_input_layout,
    dxnn_static_input_size,
    find_images,
    infer_static_input_size,
    preprocess_dxnn,
    preprocess_onnx,
    reshape_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check labels == argmax(masks) inside PIDNet ONNX/DXNN models and "
            "compare the two models' class predictions."
        )
    )
    parser.add_argument("--onnx-model", required=True, help="ONNX model path.")
    parser.add_argument("--dxnn-model", required=True, help="DXNN model path.")
    parser.add_argument(
        "--dataset-root",
        default="cityscapes_small",
        help="Dataset root. Images default to <dataset-root>/images.",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Optional image directory instead of <dataset-root>/images.",
    )
    parser.add_argument(
        "--limit",
        "--max-images",
        dest="limit",
        type=int,
        default=None,
        help="Check only the first N images.",
    )
    parser.add_argument(
        "--dxnn-input-color",
        choices=("bgr", "rgb"),
        default="rgb",
        help="Color order sent to DXRT (default: rgb).",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=None,
        help="ONNX Runtime providers, e.g. CUDAExecutionProvider CPUExecutionProvider.",
    )
    parser.add_argument("--save-json", default=None, help="Optional result JSON path.")
    return parser.parse_args()


def labels_to_hw(output: np.ndarray, model_name: str) -> np.ndarray:
    labels = np.squeeze(output)
    if labels.ndim != 2:
        raise ValueError(f"{model_name} labels must be 2-D after squeeze, got {output.shape}")
    return labels.astype(np.int64, copy=False)


def masks_to_chw(output: np.ndarray, label_hw: tuple[int, int], model_name: str) -> np.ndarray:
    masks = np.asarray(output)
    if masks.ndim == 4 and masks.shape[0] == 1:
        masks = masks[0]
    if masks.ndim != 3:
        raise ValueError(f"{model_name} masks must be 3-D after batch removal, got {output.shape}")

    if masks.shape[1:] == label_hw:
        return masks.astype(np.float32, copy=False)
    if masks.shape[:2] == label_hw:
        return masks.transpose(2, 0, 1).astype(np.float32, copy=False)
    raise ValueError(
        f"{model_name} masks shape {masks.shape} is incompatible with labels shape {label_hw}"
    )


def resize_labels(labels: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    if labels.shape == target_hw:
        return labels
    return cv2.resize(
        labels.astype(np.int32),
        (target_hw[1], target_hw[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int64)


def resize_masks(masks: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    if masks.shape[1:] == target_hw:
        return masks
    resized = np.empty((masks.shape[0], *target_hw), dtype=np.float32)
    for class_idx in range(masks.shape[0]):
        resized[class_idx] = cv2.resize(
            masks[class_idx],
            (target_hw[1], target_hw[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    return resized


def agreement(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    if a.shape != b.shape:
        raise ValueError(f"Cannot compare different shapes: {a.shape} vs. {b.shape}")
    equal = int(np.count_nonzero(a == b))
    total = int(a.size)
    return equal, total, equal / total if total else 0.0


def main() -> None:
    args = parse_args()
    onnx_path = Path(args.onnx_model)
    dxnn_path = Path(args.dxnn_model)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if not dxnn_path.exists():
        raise FileNotFoundError(f"DXNN model not found: {dxnn_path}")

    images_dir = Path(args.images_dir) if args.images_dir else Path(args.dataset_root) / "images"
    image_paths = find_images(images_dir)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {images_dir}")

    session = ort.InferenceSession(str(onnx_path), providers=choose_providers(args.providers))
    onnx_input = session.get_inputs()[0]
    onnx_input_size = infer_static_input_size(onnx_input.shape) or (512, 512)
    onnx_output_names = {output.name for output in session.get_outputs()}
    missing_onnx = {"labels", "masks"} - onnx_output_names
    if missing_onnx:
        raise ValueError(
            f"ONNX model is missing outputs {sorted(missing_onnx)}; "
            f"available={sorted(onnx_output_names)}"
        )

    totals = {
        "onnx_labels_vs_masks_argmax": [0, 0],
        "dxnn_labels_vs_masks_argmax": [0, 0],
        "onnx_vs_dxnn_labels": [0, 0],
        "onnx_vs_dxnn_masks_argmax": [0, 0],
    }
    per_image: list[dict[str, Any]] = []

    with InferenceEngine(str(dxnn_path)) as engine:
        dxnn_input_info = engine.get_input_tensors_info()[0]
        dxnn_output_info = engine.get_output_tensors_info()
        dxnn_output_names = [info["name"] for info in dxnn_output_info]
        missing_dxnn = {"labels", "masks"} - set(dxnn_output_names)
        if missing_dxnn:
            raise ValueError(
                f"DXNN model is missing outputs {sorted(missing_dxnn)}; "
                f"available={dxnn_output_names}"
            )

        labels_index = dxnn_output_names.index("labels")
        masks_index = dxnn_output_names.index("masks")
        dxnn_input_dtype = np.dtype(dxnn_input_info["dtype"])
        dxnn_layout = dxnn_input_layout(dxnn_input_info["shape"])
        dxnn_input_size = dxnn_static_input_size(dxnn_input_info["shape"], dxnn_layout)
        expected_input_bytes = engine.get_input_size()

        print(
            f"[onnx] {onnx_path}: providers={session.get_providers()}, "
            f"input_size={onnx_input_size}"
        )
        print(
            f"[dxnn] {dxnn_path}: input_size={dxnn_input_size}, "
            f"dtype={dxnn_input_dtype.name}, layout={dxnn_layout}, "
            f"color={args.dxnn_input_color}"
        )

        for image_path in tqdm(image_paths, desc="Checking", unit="image"):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {image_path}")

            onnx_tensor = preprocess_onnx(image, onnx_input_size)
            onnx_labels_raw, onnx_masks_raw = session.run(
                ["labels", "masks"], {onnx_input.name: onnx_tensor}
            )
            onnx_labels = labels_to_hw(onnx_labels_raw, "ONNX")
            onnx_masks = masks_to_chw(onnx_masks_raw, onnx_labels.shape, "ONNX")
            onnx_masks_argmax = np.argmax(onnx_masks, axis=0)

            dxnn_tensor = preprocess_dxnn(
                image,
                dxnn_input_size,
                dxnn_layout,
                dxnn_input_dtype,
                args.dxnn_input_color,
            )
            if dxnn_tensor.nbytes != expected_input_bytes:
                raise ValueError(
                    f"DXNN input byte size mismatch: got {dxnn_tensor.nbytes}, "
                    f"expected {expected_input_bytes}"
                )
            dxnn_outputs = engine.run([dxnn_tensor])
            dxnn_labels_raw = reshape_output(
                np.asarray(dxnn_outputs[labels_index]),
                dxnn_output_info[labels_index].get("shape"),
            )
            dxnn_masks_raw = reshape_output(
                np.asarray(dxnn_outputs[masks_index]),
                dxnn_output_info[masks_index].get("shape"),
            )
            dxnn_labels = labels_to_hw(dxnn_labels_raw, "DXNN")
            dxnn_masks = masks_to_chw(dxnn_masks_raw, dxnn_labels.shape, "DXNN")
            dxnn_masks_argmax = np.argmax(dxnn_masks, axis=0)

            aligned_dxnn_labels = resize_labels(dxnn_labels, onnx_labels.shape)
            aligned_dxnn_masks = resize_masks(dxnn_masks, onnx_labels.shape)
            aligned_dxnn_masks_argmax = np.argmax(aligned_dxnn_masks, axis=0)

            comparisons = {
                "onnx_labels_vs_masks_argmax": (onnx_labels, onnx_masks_argmax),
                "dxnn_labels_vs_masks_argmax": (dxnn_labels, dxnn_masks_argmax),
                "onnx_vs_dxnn_labels": (onnx_labels, aligned_dxnn_labels),
                "onnx_vs_dxnn_masks_argmax": (
                    onnx_masks_argmax,
                    aligned_dxnn_masks_argmax,
                ),
            }
            image_result: dict[str, Any] = {"image": image_path.name}
            for name, (left, right) in comparisons.items():
                equal, total, ratio = agreement(left, right)
                totals[name][0] += equal
                totals[name][1] += total
                image_result[name] = ratio
            per_image.append(image_result)

    summary: dict[str, Any] = {
        "num_images": len(per_image),
        "onnx_model": str(onnx_path),
        "dxnn_model": str(dxnn_path),
        "images_dir": str(images_dir),
        "dxnn_input_color": args.dxnn_input_color,
    }
    for name, (equal, total) in totals.items():
        summary[name] = equal / total if total else None
        summary[f"{name}_equal_pixels"] = equal
        summary[f"{name}_total_pixels"] = total

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.save_json:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(
            json.dumps({"summary": summary, "per_image": per_image}, indent=2, ensure_ascii=False)
            + "\n"
        )
        print(f"Saved metrics to: {save_path}")


if __name__ == "__main__":
    main()
