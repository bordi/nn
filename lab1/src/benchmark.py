from __future__ import annotations

import json
import time
from math import ceil
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from .config import ARTIFACTS_DIR, DEFAULT_IMAGE_SIZE, default_num_workers
from .data import create_eval_loader
from .model import load_checkpoint_model
from .progress import TerminalProgressBar


def _estimate_total_batches(dataloader, max_samples: int | None = None) -> int:
    if max_samples is None:
        return len(dataloader)
    batch_size = getattr(dataloader, "batch_size", None) or 1
    return max(1, min(len(dataloader), ceil(max_samples / batch_size)))


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def evaluate_pytorch(model: torch.nn.Module, dataloader, max_samples: int | None = None) -> tuple[float, float]:
    model.eval()
    predictions: list[int] = []
    targets: list[int] = []
    seen_samples = 0
    total_batches = _estimate_total_batches(dataloader, max_samples=max_samples)
    progress_bar = TerminalProgressBar(total=total_batches, description="bench pt eval")

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"]
            labels = batch["label"]
            logits = model(images)
            batch_predictions = logits.argmax(dim=1)
            predictions.extend(batch_predictions.cpu().tolist())
            targets.extend(labels.cpu().tolist())
            seen_samples += images.size(0)
            current_accuracy = float((np.array(predictions) == np.array(targets)).mean())
            progress_bar.update(postfix=f"acc={current_accuracy:.4f}")
            if max_samples is not None and seen_samples >= max_samples:
                predictions = predictions[:max_samples]
                targets = targets[:max_samples]
                break

    final_accuracy = float((np.array(predictions) == np.array(targets)).mean())
    progress_bar.close(postfix=f"acc={final_accuracy:.4f}")

    accuracy = float((np.array(predictions) == np.array(targets)).mean())
    macro_f1 = float(f1_score(targets, predictions, average="macro"))
    return accuracy, macro_f1


def benchmark_pytorch_latency(
    model: torch.nn.Module,
    dataloader,
    warmup: int,
    num_samples: int,
) -> float:
    model.eval()
    latencies_ms: list[float] = []
    progress_bar = TerminalProgressBar(total=warmup + num_samples, description="bench pt lat")

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            images = batch["image"]
            if batch_index < warmup:
                _ = model(images)
                progress_bar.update(postfix=f"warmup={batch_index + 1}/{warmup}")
                continue
            started_at = time.perf_counter()
            _ = model(images)
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            latencies_ms.append(elapsed_ms / images.size(0))
            progress_bar.update(postfix=f"avg_ms={float(np.mean(latencies_ms)):.2f}")
            if len(latencies_ms) >= num_samples:
                break

    progress_bar.close(postfix=f"avg_ms={float(np.mean(latencies_ms)):.2f}")
    return float(np.mean(latencies_ms))


def evaluate_onnx(session, dataloader, max_samples: int | None = None) -> tuple[float, float]:
    predictions: list[int] = []
    targets: list[int] = []
    seen_samples = 0
    input_name = session.get_inputs()[0].name
    total_batches = _estimate_total_batches(dataloader, max_samples=max_samples)
    progress_bar = TerminalProgressBar(total=total_batches, description="bench onnx eval")

    for batch in dataloader:
        images = batch["image"].numpy()
        labels = batch["label"].numpy()
        logits = session.run(None, {input_name: images})[0]
        batch_predictions = logits.argmax(axis=1)
        predictions.extend(batch_predictions.tolist())
        targets.extend(labels.tolist())
        seen_samples += len(images)
        current_accuracy = float((np.array(predictions) == np.array(targets)).mean())
        progress_bar.update(postfix=f"acc={current_accuracy:.4f}")
        if max_samples is not None and seen_samples >= max_samples:
            predictions = predictions[:max_samples]
            targets = targets[:max_samples]
            break

    final_accuracy = float((np.array(predictions) == np.array(targets)).mean())
    progress_bar.close(postfix=f"acc={final_accuracy:.4f}")
    accuracy = float((np.array(predictions) == np.array(targets)).mean())
    macro_f1 = float(f1_score(targets, predictions, average="macro"))
    return accuracy, macro_f1


def benchmark_onnx_latency(session, dataloader, warmup: int, num_samples: int) -> float:
    latencies_ms: list[float] = []
    input_name = session.get_inputs()[0].name
    progress_bar = TerminalProgressBar(total=warmup + num_samples, description="bench onnx lat")

    for batch_index, batch in enumerate(dataloader):
        images = batch["image"].numpy()
        if batch_index < warmup:
            _ = session.run(None, {input_name: images})
            progress_bar.update(postfix=f"warmup={batch_index + 1}/{warmup}")
            continue
        started_at = time.perf_counter()
        _ = session.run(None, {input_name: images})
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        latencies_ms.append(elapsed_ms / len(images))
        progress_bar.update(postfix=f"avg_ms={float(np.mean(latencies_ms)):.2f}")
        if len(latencies_ms) >= num_samples:
            break

    progress_bar.close(postfix=f"avg_ms={float(np.mean(latencies_ms)):.2f}")
    return float(np.mean(latencies_ms))


def print_comparison_table(rows: list[dict]) -> None:
    print("| version | accuracy | macro_f1 | size_mb | latency_ms |")
    print("|---|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['version']} | "
            f"{row['accuracy']:.4f} | "
            f"{row['macro_f1']:.4f} | "
            f"{row['size_mb']:.2f} | "
            f"{row['latency_ms']:.2f} |"
        )


def run_benchmark(args) -> dict:
    import onnxruntime as ort

    checkpoint_path = Path(args.checkpoint)
    onnx_path = Path(args.onnx)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_names = checkpoint["class_names"]
    model, _ = load_checkpoint_model(
        checkpoint_path=checkpoint_path,
        num_classes=len(class_names),
        map_location="cpu",
    )
    model.eval()

    metric_loader, _ = create_eval_loader(
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        debug_samples=args.debug_samples,
    )
    latency_loader, _ = create_eval_loader(
        batch_size=1,
        image_size=args.image_size,
        num_workers=args.num_workers,
        debug_samples=args.debug_samples,
    )

    pytorch_accuracy, pytorch_macro_f1 = evaluate_pytorch(
        model=model,
        dataloader=metric_loader,
        max_samples=args.eval_limit,
    )
    pytorch_latency = benchmark_pytorch_latency(
        model=model,
        dataloader=latency_loader,
        warmup=args.warmup,
        num_samples=args.latency_samples,
    )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_accuracy, onnx_macro_f1 = evaluate_onnx(
        session=session,
        dataloader=metric_loader,
        max_samples=args.eval_limit,
    )
    onnx_latency = benchmark_onnx_latency(
        session=session,
        dataloader=latency_loader,
        warmup=args.warmup,
        num_samples=args.latency_samples,
    )

    rows = [
        {
            "version": "pytorch",
            "accuracy": pytorch_accuracy,
            "macro_f1": pytorch_macro_f1,
            "size_mb": file_size_mb(checkpoint_path),
            "latency_ms": pytorch_latency,
        },
        {
            "version": "onnx",
            "accuracy": onnx_accuracy,
            "macro_f1": onnx_macro_f1,
            "size_mb": file_size_mb(onnx_path),
            "latency_ms": onnx_latency,
        },
    ]

    print_comparison_table(rows)
    summary = {
        "checkpoint": str(checkpoint_path),
        "onnx": str(onnx_path),
        "rows": rows,
    }
    output_path = ARTIFACTS_DIR / "benchmark_summary.json"
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Benchmark summary saved to: {output_path}")
    return summary


def build_benchmark_defaults() -> dict:
    return {
        "batch_size": 32,
        "image_size": DEFAULT_IMAGE_SIZE,
        "num_workers": default_num_workers(),
        "latency_samples": 100,
        "warmup": 10,
        "eval_limit": None,
        "debug_samples": None,
    }
