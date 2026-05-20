#!/usr/bin/env python3
"""Export the official PIDNet-S Cityscapes PyTorch checkpoint to ONNX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PIDNet-S Cityscapes checkpoint to ONNX.")
    parser.add_argument(
        "--pidnet-repo",
        default="external/PIDNet",
        help="Path to a checkout of https://github.com/XuJiacong/PIDNet.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Cityscapes PIDNet-S checkpoint, e.g. PIDNet_S_Cityscapes_val.pt.",
    )
    parser.add_argument("--output", default="pidnet_s_cityscapes.onnx")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def load_pidnet_model(pidnet_repo: Path, checkpoint_path: Path, num_classes: int) -> torch.nn.Module:
    if not pidnet_repo.exists():
        raise FileNotFoundError(f"PIDNet repo not found: {pidnet_repo}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    sys.path.insert(0, str(pidnet_repo.resolve()))
    from models.pidnet import get_pred_model  # noqa: PLC0415

    model = get_pred_model(name="pidnet-s", num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state = model.state_dict()

    loaded = {}
    for key, value in state_dict.items():
        clean_key = key
        for prefix in ("model.module.", "module.", "model."):
            clean_key = clean_key.removeprefix(prefix)
        if clean_key in model_state and value.shape == model_state[clean_key].shape:
            loaded[clean_key] = value

    if not loaded:
        raise RuntimeError(f"No compatible parameters loaded from {checkpoint_path}")

    model_state.update(loaded)
    model.load_state_dict(model_state, strict=True)
    model.eval()
    print(f"Loaded {len(loaded)} tensors from {checkpoint_path}")
    return model


def add_metadata(onnx_path: Path, args: argparse.Namespace) -> None:
    try:
        import onnx
    except ImportError:
        print("onnx package is not installed; skipped metadata injection.")
        return

    model = onnx.load(str(onnx_path))
    metadata = {
        "architecture": "PIDNet-S",
        "dataset": "Cityscapes",
        "num_classes": str(args.num_classes),
        "checkpoint": str(Path(args.checkpoint)),
        "input_batch_size": "1",
        "input_height": str(args.height),
        "input_width": str(args.width),
    }
    del model.metadata_props[:]
    for key, value in metadata.items():
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = value
    onnx.save(model, str(onnx_path))


def main() -> None:
    args = parse_args()
    pidnet_repo = Path(args.pidnet_repo)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_pidnet_model(pidnet_repo, checkpoint_path, args.num_classes)
    dummy = torch.randn(1, 3, args.height, args.width, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=args.opset,
        do_constant_folding=True,
    )
    add_metadata(output_path, args)
    print(f"Exported {output_path}")


if __name__ == "__main__":
    main()
