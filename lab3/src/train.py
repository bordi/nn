from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .baselines import load_prepared_series
from .config import Config
from .models import build_model
from .utils import read_json, set_seed, write_json
from .windows import build_eval_windows, build_train_windows, windows_to_arrays


BEST_CHECKPOINT_FILENAME = "gru_best.pt"
TRAINING_SUMMARY_FILENAME = "training_summary.json"


def fit_standardizer(records: list[dict[str, Any]]) -> dict[str, float | int]:
    train_values = [float(record["value"]) for record in records if record.get("split") == "train"]
    if not train_values:
        raise ValueError("Prepared records must include at least one train row for normalization")

    mean = sum(train_values) / len(train_values)
    variance = sum((value - mean) ** 2 for value in train_values) / len(train_values)
    std = math.sqrt(variance)
    if std <= 1e-8:
        std = 1.0

    return {
        "count": len(train_values),
        "mean": mean,
        "std": std,
    }


def compute_normalization_stats(records: list[dict[str, Any]]) -> dict[str, float | int]:
    return fit_standardizer(records)


def apply_standardizer(
    records: list[dict[str, Any]],
    normalization_stats: dict[str, float | int],
) -> list[dict[str, Any]]:
    mean = float(normalization_stats["mean"])
    std = float(normalization_stats["std"])
    return [{**record, "value": (float(record["value"]) - mean) / std} for record in records]


def prepare_batch_tensors(window_records: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    input_values, targets = windows_to_arrays(window_records)
    inputs = torch.tensor(input_values, dtype=torch.float32).unsqueeze(-1)
    target_tensor = torch.tensor(targets, dtype=torch.float32)
    return inputs, target_tensor


def create_data_loader(
    window_records: list[dict[str, Any]],
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    inputs, targets = prepare_batch_tensors(window_records)
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_dataloaders(
    train_windows: list[dict[str, Any]],
    val_windows: list[dict[str, Any]],
    *,
    batch_size: int,
) -> tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], DataLoader[tuple[torch.Tensor, torch.Tensor]]]:
    train_loader = create_data_loader(train_windows, batch_size=batch_size, shuffle=True)
    val_loader = create_data_loader(val_windows, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def save_checkpoint(
    *,
    best_state: dict[str, Any] | None,
    epoch: int,
    validation_mae_normalized: float,
    model: nn.Module,
    checkpoint_path: Path,
    model_hyperparameters: dict[str, Any],
    normalization_stats: dict[str, float | int],
) -> dict[str, Any]:
    if best_state is not None and validation_mae_normalized >= float(best_state["best_validation_mae_normalized"]):
        return best_state

    checkpoint = {
        "best_epoch": epoch,
        "best_validation_mae_normalized": validation_mae_normalized,
        "model_hyperparameters": model_hyperparameters,
        "normalization_stats": normalization_stats,
        "model_state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    return checkpoint


def select_best_checkpoint(
    *,
    best_state: dict[str, Any] | None,
    epoch: int,
    validation_mae_normalized: float,
    model: nn.Module,
    checkpoint_path: Path,
    model_hyperparameters: dict[str, Any],
    normalization_stats: dict[str, float | int],
) -> dict[str, Any]:
    return save_checkpoint(
        best_state=best_state,
        epoch=epoch,
        validation_mae_normalized=validation_mae_normalized,
        model=model,
        checkpoint_path=checkpoint_path,
        model_hyperparameters=model_hyperparameters,
        normalization_stats=normalization_stats,
    )


def update_early_stopping_counter(
    best_validation_mae: float,
    current_validation_mae: float,
    patience_counter: int,
) -> tuple[float, int]:
    if current_validation_mae < best_validation_mae:
        return current_validation_mae, 0
    return best_validation_mae, patience_counter + 1


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for inputs, targets in loader:
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training loader produced no samples")
    return total_loss / total_samples


def evaluate_model(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: nn.Module,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_absolute_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in loader:
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            absolute_error = torch.abs(predictions - targets).sum()

            batch_size = targets.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_absolute_error += float(absolute_error.item())
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Validation loader produced no samples")
    return total_loss / total_samples, total_absolute_error / total_samples


def save_training_summary(path: Path, summary: dict[str, Any]) -> None:
    write_json(path, summary)


def train_model(config: Config) -> dict[str, Any]:
    set_seed(config.seed)
    config.ensure_directories()

    prepared_series_path = config.prepared_dir / "series.csv"
    prepared_metadata_path = config.prepared_dir / "prepared_metadata.json"
    records = load_prepared_series(prepared_series_path)
    metadata = read_json(prepared_metadata_path)

    window_size = int(metadata["window_size"])
    horizon = int(metadata["horizon"])
    normalization_stats = fit_standardizer(records)
    normalized_records = apply_standardizer(records, normalization_stats)
    normalization_std = float(normalization_stats["std"])
    model_hyperparameters = {
        "input_size": config.input_size,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
    }

    train_windows = build_train_windows(normalized_records, window_size=window_size, horizon=horizon)
    val_windows = build_eval_windows(normalized_records, "val", window_size=window_size, horizon=horizon)

    if not train_windows:
        raise ValueError("No training windows could be built from prepared artifacts")
    if not val_windows:
        raise ValueError("No validation windows could be built from prepared artifacts")

    train_loader, val_loader = build_dataloaders(train_windows, val_windows, batch_size=config.batch_size)

    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    history: list[dict[str, float | int]] = []
    best_state: dict[str, Any] | None = None
    best_validation_mae = float("inf")
    patience_counter = 0
    checkpoint_path = config.models_dir / BEST_CHECKPOINT_FILENAME

    for epoch in range(1, config.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        validation_loss, validation_mae_normalized = evaluate_model(model, val_loader, loss_fn)
        validation_mae_original = validation_mae_normalized * normalization_std
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": validation_loss,
                "val_mae_normalized": validation_mae_normalized,
                "val_mae_original": validation_mae_original,
            }
        )

        best_state = save_checkpoint(
            best_state=best_state,
            epoch=epoch,
            validation_mae_normalized=validation_mae_normalized,
            model=model,
            checkpoint_path=checkpoint_path,
            model_hyperparameters=model_hyperparameters,
            normalization_stats=normalization_stats,
        )
        best_validation_mae, patience_counter = update_early_stopping_counter(
            best_validation_mae,
            validation_mae_normalized,
            patience_counter,
        )
        if patience_counter >= config.early_stopping_patience:
            break

    if best_state is None:
        raise ValueError("Training did not produce a best checkpoint")

    summary = {
        "model_hyperparameters": model_hyperparameters,
        "training_hyperparameters": {
            "optimizer": "Adam",
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "max_epochs": config.max_epochs,
            "early_stopping_patience": config.early_stopping_patience,
            "loss": "MSELoss",
        },
        "train_val_history": history,
        "best_epoch": int(best_state["best_epoch"]),
        "best_validation_mae_normalized": float(best_state["best_validation_mae_normalized"]),
        "best_validation_mae_original": float(best_state["best_validation_mae_normalized"]) * normalization_std,
        "normalization_stats": normalization_stats,
    }
    summary_path = config.models_dir / TRAINING_SUMMARY_FILENAME
    save_training_summary(summary_path, summary)

    return {
        "checkpoint_path": checkpoint_path,
        "summary_path": summary_path,
        "summary": summary,
    }


def run_training_pipeline(config: Config) -> dict[str, Any]:
    return train_model(config)
