from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .config import CHECKPOINTS_DIR, TrainingConfig, ensure_directories, get_device, set_seed
from .data import create_dataloaders
from .model import count_trainable_parameters, create_model, freeze_backbone, unfreeze_last_blocks
from .progress import TerminalProgressBar


def run_epoch(
    model: torch.nn.Module,
    dataloader,
    criterion,
    optimizer,
    device: torch.device,
    training: bool,
    progress_label: str,
) -> dict:
    if training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_examples = 0

    progress_bar = TerminalProgressBar(total=len(dataloader), description=progress_label)

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_examples += images.size(0)
        progress_bar.update(
            postfix=(
                f"loss={running_loss / max(1, total_examples):.4f} "
                f"acc={correct_predictions / max(1, total_examples):.4f}"
            )
        )

    progress_bar.close(
        postfix=(
            f"loss={running_loss / max(1, total_examples):.4f} "
            f"acc={correct_predictions / max(1, total_examples):.4f}"
        )
    )

    return {
        "loss": running_loss / max(1, total_examples),
        "accuracy": correct_predictions / max(1, total_examples),
    }


def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    class_names: list[str],
    stage: str,
    epoch: int,
    metrics: dict,
    config: TrainingConfig,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "stage": stage,
        "epoch": epoch,
        "metrics": metrics,
        "config": asdict(config),
    }
    torch.save(checkpoint, checkpoint_path)


def train_stage(
    stage_name: str,
    model: torch.nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    checkpoint_path: Path,
    class_names: list[str],
    config: TrainingConfig,
) -> tuple[list[dict], Path]:
    criterion = torch.nn.CrossEntropyLoss()
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict] = []

    print(f"\nStarting stage: {stage_name}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    for epoch in range(1, epochs + 1):
        started_at = time.perf_counter()
        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            training=True,
            progress_label=f"{stage_name} train {epoch}/{epochs}",
        )
        val_metrics = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            training=False,
            progress_label=f"{stage_name} val {epoch}/{epochs}",
        )
        elapsed = time.perf_counter() - started_at

        epoch_summary = {
            "stage": stage_name,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "elapsed_seconds": elapsed,
        }
        history.append(epoch_summary)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"time={elapsed:.1f}s"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                class_names=class_names,
                stage=stage_name,
                epoch=epoch,
                metrics=epoch_summary,
                config=config,
            )
            print(f"Saved new best checkpoint to: {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered for stage '{stage_name}'.")
                break

    return history, checkpoint_path


def run_training(args) -> dict:
    ensure_directories()
    config = TrainingConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        baseline_epochs=args.baseline_epochs,
        improved_epochs=args.improved_epochs,
        baseline_learning_rate=args.baseline_learning_rate,
        improved_learning_rate=args.improved_learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        debug_samples=args.debug_samples,
        unfreeze_blocks=args.unfreeze_blocks,
    )
    set_seed(config.random_seed)
    device = get_device(prefer_cpu=args.device == "cpu")

    baseline_train_loader, baseline_val_loader, _, class_names = create_dataloaders(
        config=config,
        use_augmentations=False,
    )
    improved_train_loader, improved_val_loader, _, _ = create_dataloaders(
        config=config,
        use_augmentations=True,
    )

    model = create_model(num_classes=len(class_names), pretrained=True)
    model.to(device)

    baseline_checkpoint = CHECKPOINTS_DIR / "baseline_best.pt"
    improved_checkpoint = CHECKPOINTS_DIR / "improved_best.pt"

    freeze_backbone(model)
    baseline_history, _ = train_stage(
        stage_name="baseline",
        model=model,
        train_loader=baseline_train_loader,
        val_loader=baseline_val_loader,
        device=device,
        epochs=config.baseline_epochs,
        learning_rate=config.baseline_learning_rate,
        weight_decay=config.weight_decay,
        patience=config.patience,
        checkpoint_path=baseline_checkpoint,
        class_names=class_names,
        config=config,
    )

    if improved_checkpoint.exists():
        improved_checkpoint.unlink()

    baseline_state = torch.load(baseline_checkpoint, map_location=device)
    model.load_state_dict(baseline_state["model_state_dict"])
    unfreeze_last_blocks(model, num_blocks=config.unfreeze_blocks)
    improved_history, _ = train_stage(
        stage_name="improved",
        model=model,
        train_loader=improved_train_loader,
        val_loader=improved_val_loader,
        device=device,
        epochs=config.improved_epochs,
        learning_rate=config.improved_learning_rate,
        weight_decay=config.weight_decay,
        patience=config.patience,
        checkpoint_path=improved_checkpoint,
        class_names=class_names,
        config=config,
    )

    summary = {
        "device": str(device),
        "class_count": len(class_names),
        "baseline_checkpoint": str(baseline_checkpoint),
        "improved_checkpoint": str(improved_checkpoint),
        "baseline_history": baseline_history,
        "improved_history": improved_history,
    }
    summary_path = CHECKPOINTS_DIR / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nTraining summary saved to: {summary_path}")
    return summary
