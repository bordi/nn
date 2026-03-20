from __future__ import annotations

from pathlib import Path

import torch

from .config import DEFAULT_IMAGE_SIZE, ONNX_DIR, ensure_directories
from .model import load_checkpoint_model


def run_export_onnx(args) -> str:
    ensure_directories()
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output) if args.output else ONNX_DIR / f"{checkpoint_path.stem}.onnx"

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_names = checkpoint["class_names"]
    model, _ = load_checkpoint_model(
        checkpoint_path=checkpoint_path,
        num_classes=len(class_names),
        map_location="cpu",
    )
    model.eval()

    dummy_input = torch.randn(1, 3, args.image_size, args.image_size)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"ONNX model exported to: {output_path}")
    return str(output_path)


def build_export_defaults() -> dict:
    return {"image_size": DEFAULT_IMAGE_SIZE}
