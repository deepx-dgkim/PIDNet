#!/usr/bin/env python3
"""Evaluate a PIDNet DXNN model on the local cityscapes_small dataset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dx_engine import InferenceEngine
from tqdm import tqdm


CLASS_NAMES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

CITYSCAPES_ID_TO_TRAIN_ID = {
    -1: 255,
    0: 255,
    1: 255,
    2: 255,
    3: 255,
    4: 255,
    5: 255,
    6: 255,
    7: 0,
    8: 1,
    9: 255,
    10: 255,
    11: 2,
    12: 3,
    13: 4,
    14: 255,
    15: 255,
    16: 255,
    17: 5,
    18: 255,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    29: 255,
    30: 255,
    31: 16,
    32: 17,
    33: 18,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a PIDNet DXNN model on cityscapes_small/images + masks."
    )
    parser.add_argument(
        "--model",
        default="dxnn/pidnet_s_cityscapes_val.dxnn",
        help="DXNN model path.",
    )
    parser.add_argument(
        "--dataset-root",
        "--cityscapes-root",
        dest="dataset_root",
        default="cityscapes_small",
        help="Dataset root. Defaults to cityscapes_small.",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Image directory. Defaults to <dataset-root>/images.",
    )
    parser.add_argument(
        "--masks-dir",
        default=None,
        help="Mask directory. Defaults to <dataset-root>/masks.",
    )
    parser.add_argument(
        "--mask-suffix",
        default="_mask.png",
        help="Mask filename suffix for simple datasets. 000000.png -> 000000_mask.png",
    )
    parser.add_argument(
        "--limit",
        "--max-images",
        dest="limit",
        type=int,
        default=None,
        help="Evaluate only the first N images.",
    )
    parser.add_argument(
        "--label-format",
        choices=("auto", "id", "trainid"),
        default="auto",
        help="Mask value format. cityscapes_small uses Cityscapes label IDs.",
    )
    parser.add_argument(
        "--input-size",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="Resize input before inference. Defaults to the DXNN static input size.",
    )
    parser.add_argument(
        "--input-color",
        choices=("bgr", "rgb"),
        default="bgr",
        help=(
            "Color order sent to DXRT. Defaults to bgr for DXNNs compiled with "
            "BGR2RGB preprocessing."
        ),
    )
    parser.add_argument(
        "--input-layout",
        choices=("auto", "nhwc", "nchw"),
        default="auto",
        help="Input tensor layout. Defaults to auto from DXNN input shape.",
    )
    parser.add_argument(
        "--output-index",
        type=int,
        default=-1,
        help="Output tensor index when the DXNN graph has multiple outputs.",
    )
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--save-json", default=None, help="Optional metrics JSON output path")
    parser.add_argument(
        "--save-preds",
        default=None,
        help="Optional directory for predicted trainId PNG masks.",
    )
    return parser.parse_args()


def find_simple_pairs(root: Path, images_dir: str | None, masks_dir: str | None, mask_suffix: str) -> list[tuple[Path, Path]]:
    image_root = Path(images_dir) if images_dir else root / "images"
    mask_root = Path(masks_dir) if masks_dir else root / "masks"
    if not image_root.exists() or not mask_root.exists():
        return []

    image_paths = sorted(
        path
        for pattern in ("*.png", "*.jpg", "*.jpeg")
        for path in image_root.glob(pattern)
    )
    pairs: list[tuple[Path, Path]] = []
    for image_path in image_paths:
        mask_path = mask_root / f"{image_path.stem}{mask_suffix}"
        if mask_path.exists():
            pairs.append((image_path, mask_path))
    return pairs


def find_official_cityscapes_pairs(root: Path, split: str = "val") -> list[tuple[Path, Path]]:
    image_paths = sorted((root / "leftImg8bit" / split).glob("*/*_leftImg8bit.png"))
    pairs: list[tuple[Path, Path]] = []
    for image_path in image_paths:
        city = image_path.parent.name
        stem = image_path.name.replace("_leftImg8bit.png", "")
        label_path = root / "gtFine" / split / city / f"{stem}_gtFine_labelTrainIds.png"
        if not label_path.exists():
            label_path = root / "gtFine" / split / city / f"{stem}_gtFine_labelIds.png"
        if label_path.exists():
            pairs.append((image_path, label_path))
    return pairs


def resolve_label_format(label_path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if label_path.name.endswith("_gtFine_labelTrainIds.png"):
        return "trainid"
    return "id"


def label_to_train_id(label: np.ndarray, ignore_label: int, label_format: str) -> np.ndarray:
    if label.size == 0 or label_format == "trainid":
        return label.astype(np.uint8, copy=False)

    mapped = np.full(label.shape, ignore_label, dtype=np.uint8)
    for label_id, train_id in CITYSCAPES_ID_TO_TRAIN_ID.items():
        if label_id >= 0:
            mapped[label == label_id] = train_id
    return mapped


def dtype_name(dtype: Any) -> str:
    try:
        return np.dtype(dtype).name
    except TypeError:
        return str(dtype)


def json_safe_tensor_info(info: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe_info = []
    for tensor in info:
        safe_tensor = dict(tensor)
        if "dtype" in safe_tensor:
            safe_tensor["dtype"] = dtype_name(safe_tensor["dtype"])
        safe_info.append(safe_tensor)
    return safe_info


def infer_input_layout(input_shape: list[int], requested: str) -> str:
    if requested != "auto":
        return requested
    if len(input_shape) != 4:
        raise ValueError(f"Unsupported DXNN input shape: {input_shape}")

    if input_shape[-1] in (1, 3, 4) and input_shape[1] > 4 and input_shape[2] > 4:
        return "nhwc"
    if input_shape[1] in (1, 3, 4) and input_shape[2] > 4 and input_shape[3] > 4:
        return "nchw"
    raise ValueError(f"Could not infer DXNN input layout from shape: {input_shape}")


def infer_static_input_size(input_shape: list[int], layout: str) -> tuple[int, int]:
    if len(input_shape) != 4:
        raise ValueError(f"Unsupported DXNN input shape: {input_shape}")
    if layout == "nhwc":
        height, width = input_shape[1], input_shape[2]
    else:
        height, width = input_shape[2], input_shape[3]
    if height <= 0 or width <= 0:
        raise ValueError(f"DXNN input shape is not static: {input_shape}")
    return height, width


def preprocess_dxnn(
    image_bgr: np.ndarray,
    input_size: tuple[int, int],
    layout: str,
    dtype: np.dtype,
    input_color: str,
) -> np.ndarray:
    height, width = input_size
    resized = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    if input_color == "rgb":
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    if np.issubdtype(dtype, np.floating):
        tensor = resized.astype(np.float32) / 255.0
        tensor -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
        tensor /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
    else:
        tensor = resized.astype(dtype, copy=False)

    if layout == "nchw":
        tensor = tensor.transpose(2, 0, 1)

    tensor = np.expand_dims(tensor, axis=0)
    return np.ascontiguousarray(tensor, dtype=dtype)


def reshape_output_if_needed(output: np.ndarray, output_info: dict[str, Any] | None) -> np.ndarray:
    if output_info is None or output.ndim != 1:
        return output

    shape = output_info.get("shape")
    if not shape:
        return output

    expected_elements = int(np.prod(shape))
    if output.size == expected_elements:
        return output.reshape(tuple(shape))
    return output


def resize_scores(scores: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    if scores.shape[-2:] == target_hw:
        return scores

    resized = np.empty((scores.shape[0], target_h, target_w), dtype=np.float32)
    for class_idx in range(scores.shape[0]):
        resized[class_idx] = cv2.resize(
            scores[class_idx], (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )
    return resized


def output_to_prediction(
    output: np.ndarray,
    target_hw: tuple[int, int],
    num_classes: int,
) -> np.ndarray:
    if output.ndim == 4 and output.shape[1] == num_classes:
        scores = resize_scores(output[0], target_hw)
        return np.argmax(scores, axis=0).astype(np.uint8)

    if output.ndim == 4 and output.shape[-1] == num_classes:
        scores = output[0].transpose(2, 0, 1)
        scores = resize_scores(scores, target_hw)
        return np.argmax(scores, axis=0).astype(np.uint8)

    if output.ndim == 3 and output.shape[0] == num_classes:
        scores = resize_scores(output, target_hw)
        return np.argmax(scores, axis=0).astype(np.uint8)

    if output.ndim == 3 and output.shape[-1] == num_classes:
        scores = output.transpose(2, 0, 1)
        scores = resize_scores(scores, target_hw)
        return np.argmax(scores, axis=0).astype(np.uint8)

    prediction = np.squeeze(output)
    if prediction.ndim != 2:
        raise ValueError(f"Unsupported DXNN output shape: {output.shape}")
    if prediction.shape != target_hw:
        prediction = cv2.resize(
            prediction.astype(np.uint8),
            (target_hw[1], target_hw[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return prediction.astype(np.uint8)


def update_confusion_matrix(
    hist: np.ndarray,
    pred: np.ndarray,
    label: np.ndarray,
    num_classes: int,
    ignore_label: int,
) -> None:
    mask = label != ignore_label
    gt = label[mask].astype(np.int64)
    pd = pred[mask].astype(np.int64)
    valid = (gt >= 0) & (gt < num_classes) & (pd >= 0) & (pd < num_classes)
    bins = np.bincount(
        num_classes * gt[valid] + pd[valid],
        minlength=num_classes * num_classes,
    )
    hist += bins.reshape(num_classes, num_classes)


def compute_metrics(hist: np.ndarray) -> dict[str, object]:
    true_positive = np.diag(hist)
    gt_pixels = hist.sum(axis=1)
    pred_pixels = hist.sum(axis=0)
    union = gt_pixels + pred_pixels - true_positive

    iou = true_positive / np.maximum(union, 1.0)
    class_acc = true_positive / np.maximum(gt_pixels, 1.0)
    pixel_acc = float(true_positive.sum() / np.maximum(hist.sum(), 1.0))

    return {
        "mIoU": float(np.mean(iou)),
        "pixel_accuracy": pixel_acc,
        "mean_accuracy": float(np.mean(class_acc)),
        "class_iou": {name: float(value) for name, value in zip(CLASS_NAMES, iou)},
    }


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    root = Path(args.dataset_root)

    if not model_path.exists():
        raise FileNotFoundError(f"DXNN model not found: {model_path}")

    pairs = find_simple_pairs(root, args.images_dir, args.masks_dir, args.mask_suffix)
    if not pairs:
        pairs = find_official_cityscapes_pairs(root)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if not pairs:
        raise FileNotFoundError(
            f"No image/mask pairs found under {root}. Expected images/ and masks/ directories."
        )

    hist = np.zeros((args.num_classes, args.num_classes), dtype=np.float64)
    save_pred_dir = Path(args.save_preds) if args.save_preds else None
    if save_pred_dir:
        save_pred_dir.mkdir(parents=True, exist_ok=True)

    with InferenceEngine(str(model_path)) as engine:
        input_info = engine.get_input_tensors_info()
        output_info = engine.get_output_tensors_info()
        if not input_info:
            raise RuntimeError("DXNN model has no input tensor metadata.")

        input_shape = input_info[0]["shape"]
        input_dtype = np.dtype(input_info[0]["dtype"])
        input_layout = infer_input_layout(input_shape, args.input_layout)
        input_size = tuple(args.input_size) if args.input_size else infer_static_input_size(input_shape, input_layout)
        expected_input_bytes = engine.get_input_size()

        start = time.perf_counter()
        for image_path, label_path in tqdm(pairs, desc="Evaluating", unit="image"):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(image_path)
            if label is None:
                raise FileNotFoundError(label_path)

            label = label_to_train_id(
                label,
                args.ignore_label,
                label_format=resolve_label_format(label_path, args.label_format),
            )
            dxnn_input = preprocess_dxnn(
                image,
                input_size,
                input_layout,
                input_dtype,
                args.input_color,
            )
            if dxnn_input.nbytes != expected_input_bytes:
                raise ValueError(
                    "DXNN input byte size mismatch: "
                    f"got {dxnn_input.nbytes}, expected {expected_input_bytes}. "
                    f"input_shape={input_shape}, input_size={input_size}, layout={input_layout}"
                )

            outputs = engine.run([dxnn_input])
            selected = outputs[args.output_index]
            selected_info = output_info[args.output_index] if output_info else None
            selected = reshape_output_if_needed(selected, selected_info)
            pred = output_to_prediction(selected, label.shape, args.num_classes)

            update_confusion_matrix(hist, pred, label, args.num_classes, args.ignore_label)

            if save_pred_dir:
                out_name = image_path.name.replace("_leftImg8bit.png", "_predTrainIds.png")
                cv2.imwrite(str(save_pred_dir / out_name), pred)

        elapsed = time.perf_counter() - start
        metrics = compute_metrics(hist)
        metrics.update(
            {
                "images": len(pairs),
                "model": str(model_path),
                "dataset_root": str(root),
                "label_format": args.label_format,
                "runtime": "dxrt",
                "seconds": elapsed,
                "fps": len(pairs) / elapsed if elapsed > 0 else None,
                "input_layout": input_layout,
                "input_color": args.input_color,
                "input_size": list(input_size),
                "input_info": json_safe_tensor_info(input_info),
                "output_info": json_safe_tensor_info(output_info),
            }
        )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.save_json:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
