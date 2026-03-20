from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

from src.benchmark import build_benchmark_defaults, run_benchmark
from src.config import CHECKPOINTS_DIR, DEFAULT_IMAGE_SIZE, GRADCAM_DIR, default_num_workers
from src.evaluate import build_eval_arg_defaults, run_evaluation
from src.export_onnx import build_export_defaults, run_export_onnx
from src.gradcam import build_gradcam_defaults, run_gradcam
from src.train import run_training


def add_shared_data_args(parser: argparse.ArgumentParser, defaults: dict) -> None:
    parser.add_argument("--batch-size", type=int, default=defaults.get("batch_size", 32))
    parser.add_argument("--image-size", type=int, default=defaults.get("image_size", DEFAULT_IMAGE_SIZE))
    parser.add_argument("--num-workers", type=int, default=defaults.get("num_workers", default_num_workers()))
    parser.add_argument("--debug-samples", type=int, default=defaults.get("debug_samples"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Edge CV lab on Food-101.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train baseline and improved models.")
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    train_parser.add_argument("--num-workers", type=int, default=default_num_workers())
    train_parser.add_argument("--baseline-epochs", type=int, default=3)
    train_parser.add_argument("--improved-epochs", type=int, default=5)
    train_parser.add_argument("--baseline-learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--improved-learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--patience", type=int, default=3)
    train_parser.add_argument("--unfreeze-blocks", type=int, default=3)
    train_parser.add_argument("--debug-samples", type=int, default=None)
    train_parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    train_parser.set_defaults(func=run_training)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a saved checkpoint.")
    add_shared_data_args(eval_parser, build_eval_arg_defaults())
    eval_parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINTS_DIR / "improved_best.pt"))
    eval_parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    eval_parser.set_defaults(func=run_evaluation)

    gradcam_defaults = build_gradcam_defaults()
    gradcam_parser = subparsers.add_parser("gradcam", help="Generate Grad-CAM visualizations.")
    add_shared_data_args(gradcam_parser, gradcam_defaults)
    gradcam_parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINTS_DIR / "improved_best.pt"))
    gradcam_parser.add_argument("--output-dir", type=str, default=gradcam_defaults["output_dir"])
    gradcam_parser.add_argument("--correct-examples", type=int, default=gradcam_defaults["correct_examples"])
    gradcam_parser.add_argument("--incorrect-examples", type=int, default=gradcam_defaults["incorrect_examples"])
    gradcam_parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    gradcam_parser.set_defaults(func=run_gradcam)

    export_parser = subparsers.add_parser("export-onnx", help="Export a checkpoint to ONNX.")
    export_parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINTS_DIR / "improved_best.pt"))
    export_parser.add_argument("--output", type=str, default=None)
    export_parser.add_argument("--image-size", type=int, default=build_export_defaults()["image_size"])
    export_parser.set_defaults(func=run_export_onnx)

    benchmark_parser = subparsers.add_parser("benchmark", help="Compare PyTorch and ONNX on CPU.")
    add_shared_data_args(benchmark_parser, build_benchmark_defaults())
    benchmark_parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINTS_DIR / "improved_best.pt"))
    benchmark_parser.add_argument("--onnx", type=str, default=str(CHECKPOINTS_DIR.parent / "onnx" / "improved_best.onnx"))
    benchmark_parser.add_argument("--latency-samples", type=int, default=100)
    benchmark_parser.add_argument("--warmup", type=int, default=10)
    benchmark_parser.add_argument("--eval-limit", type=int, default=None)
    benchmark_parser.set_defaults(func=run_benchmark)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
