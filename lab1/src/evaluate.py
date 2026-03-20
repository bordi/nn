from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from .config import ARTIFACTS_DIR, TrainingConfig, default_num_workers, get_device
from .data import create_eval_loader
from .model import load_checkpoint_model
from .progress import TerminalProgressBar


def evaluate_model(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict:
    model.eval()
    all_predictions: list[int] = []
    all_targets: list[int] = []
    total_batches = min(len(dataloader), max_batches) if max_batches is not None else len(dataloader)
    progress_bar = TerminalProgressBar(total=total_batches, description="evaluate")

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = batch["image"].to(device)
            targets = batch["label"].to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1)
            all_predictions.extend(predictions.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
            current_accuracy = float(
                (np.array(all_predictions) == np.array(all_targets)).mean()
            )
            progress_bar.update(postfix=f"acc={current_accuracy:.4f}")

    final_accuracy = float((np.array(all_predictions) == np.array(all_targets)).mean())
    progress_bar.close(postfix=f"acc={final_accuracy:.4f}")

    predictions_np = np.array(all_predictions)
    targets_np = np.array(all_targets)
    accuracy = float((predictions_np == targets_np).mean())
    macro_f1 = float(f1_score(targets_np, predictions_np, average="macro"))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "predictions": all_predictions,
        "targets": all_targets,
    }


def build_confusion_summary(
    targets: list[int],
    predictions: list[int],
    class_names: list[str],
    top_k: int = 10,
) -> list[dict]:
    matrix = confusion_matrix(targets, predictions, labels=list(range(len(class_names))))
    confused_pairs: list[dict] = []
    for true_index in range(matrix.shape[0]):
        for predicted_index in range(matrix.shape[1]):
            if true_index == predicted_index:
                continue
            count = int(matrix[true_index, predicted_index])
            if count == 0:
                continue
            confused_pairs.append(
                {
                    "true_label": class_names[true_index],
                    "predicted_label": class_names[predicted_index],
                    "count": count,
                }
            )
    confused_pairs.sort(key=lambda item: item["count"], reverse=True)
    return confused_pairs[:top_k]


def format_metrics(metrics: dict) -> str:
    return (
        f"accuracy={metrics['accuracy']:.4f}, "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )


def run_evaluation(args) -> dict:
    checkpoint_path = Path(args.checkpoint)
    config = TrainingConfig(
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        debug_samples=args.debug_samples,
    )
    test_loader, class_names = create_eval_loader(
        batch_size=config.batch_size,
        image_size=config.image_size,
        num_workers=config.num_workers,
        debug_samples=config.debug_samples,
    )
    model, checkpoint = load_checkpoint_model(
        checkpoint_path=checkpoint_path,
        num_classes=len(class_names),
        map_location="cpu",
    )
    device = get_device(prefer_cpu=args.device == "cpu")
    model.to(device)
    metrics = evaluate_model(model, test_loader, device=device)
    confusion_summary = build_confusion_summary(
        targets=metrics["targets"],
        predictions=metrics["predictions"],
        class_names=class_names,
    )

    summary = {
        "checkpoint": str(checkpoint_path),
        "stage": checkpoint.get("stage", "unknown"),
        "epoch": checkpoint.get("epoch", "unknown"),
        "metrics": {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
        },
        "top_confusions": confusion_summary,
    }
    output_path = ARTIFACTS_DIR / "evaluation_summary.json"
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"Evaluation summary saved to: {output_path}")
    print(format_metrics(summary["metrics"]))
    if confusion_summary:
        print("Top confusion pairs:")
        for item in confusion_summary:
            print(
                f"  {item['true_label']} -> {item['predicted_label']}: {item['count']}"
            )
    return summary


def build_eval_arg_defaults() -> dict:
    return {
        "batch_size": 32,
        "image_size": 224,
        "num_workers": default_num_workers(),
        "debug_samples": None,
    }
