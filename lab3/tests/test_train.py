from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab3.src.config import load_config
from lab3.src.models import build_model
from lab3.src.train import (
    apply_standardizer,
    build_dataloaders,
    evaluate_model,
    fit_standardizer,
    save_checkpoint,
    save_training_summary,
    select_best_checkpoint,
    train_one_epoch,
    update_early_stopping_counter,
)
from lab3.src.utils import read_json


def test_fit_standardizer_uses_train_split_only() -> None:
    records = [
        {"value": 1.0, "split": "train"},
        {"value": 3.0, "split": "train"},
        {"value": 100.0, "split": "val"},
        {"value": 200.0, "split": "test"},
    ]

    stats = fit_standardizer(records)

    assert stats["count"] == 2
    assert stats["mean"] == 2.0
    assert stats["std"] == 1.0


def test_apply_standardizer_preserves_shape_and_uses_given_stats() -> None:
    window_records = [
        {"value": 1.0, "split": "train"},
        {"value": 3.0, "split": "val"},
    ]
    standardized = apply_standardizer(window_records, {"mean": 2.0, "std": 1.0, "count": 1})

    assert [row["value"] for row in standardized] == [-1.0, 1.0]
    assert [row["split"] for row in standardized] == ["train", "val"]


def test_build_dataloaders_returns_gru_ready_batch_tensors() -> None:
    train_windows = [
        {"input_values": [1.0, 2.0, 3.0], "target": 4.0},
        {"input_values": [5.0, 6.0, 7.0], "target": 8.0},
    ]
    val_windows = [
        {"input_values": [2.0, 3.0, 4.0], "target": 5.0},
    ]

    train_loader, val_loader = build_dataloaders(train_windows, val_windows, batch_size=2)
    train_inputs, train_targets = next(iter(train_loader))
    val_inputs, val_targets = next(iter(val_loader))

    assert train_inputs.shape == torch.Size([2, 3, 1])
    assert train_targets.shape == torch.Size([2])
    assert val_inputs.shape == torch.Size([1, 3, 1])
    assert val_targets.shape == torch.Size([1])
    assert train_inputs.dtype == torch.float32
    assert train_targets.dtype == torch.float32


def test_save_checkpoint_tracks_lowest_validation_mae(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    checkpoint_path = tmp_path / "gru_best.pt"
    model_hyperparameters = {"input_size": 1, "hidden_size": 64, "num_layers": 1, "dropout": 0.0}
    normalization_stats = {"count": 2, "mean": 2.0, "std": 1.0}

    best = save_checkpoint(
        best_state=None,
        epoch=1,
        validation_mae_normalized=0.8,
        model=model,
        checkpoint_path=checkpoint_path,
        model_hyperparameters=model_hyperparameters,
        normalization_stats=normalization_stats,
    )
    first_saved = torch.load(checkpoint_path)

    unchanged = save_checkpoint(
        best_state=best,
        epoch=2,
        validation_mae_normalized=1.2,
        model=model,
        checkpoint_path=checkpoint_path,
        model_hyperparameters=model_hyperparameters,
        normalization_stats=normalization_stats,
    )
    after_worse = torch.load(checkpoint_path)

    improved = save_checkpoint(
        best_state=unchanged,
        epoch=3,
        validation_mae_normalized=0.4,
        model=model,
        checkpoint_path=checkpoint_path,
        model_hyperparameters=model_hyperparameters,
        normalization_stats=normalization_stats,
    )
    after_better = torch.load(checkpoint_path)

    assert best["best_epoch"] == 1
    assert best["best_validation_mae_normalized"] == 0.8
    assert unchanged == best
    assert first_saved["best_validation_mae_normalized"] == 0.8
    assert first_saved["model_hyperparameters"] == model_hyperparameters
    assert first_saved["normalization_stats"] == normalization_stats
    assert after_worse == first_saved
    assert improved["best_epoch"] == 3
    assert improved["best_validation_mae_normalized"] == 0.4
    assert after_better["best_validation_mae_normalized"] == 0.4


def test_train_one_epoch_and_evaluate_model_return_scalar_metrics(tmp_path: Path) -> None:
    model = build_model(load_config(project_root=tmp_path))
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor([[[1.0]], [[2.0]]], dtype=torch.float32),
        torch.tensor([1.0, 2.0], dtype=torch.float32),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

    train_loss = train_one_epoch(model, loader, optimizer, loss_fn)
    val_loss, val_mae = evaluate_model(model, loader, loss_fn)

    assert isinstance(train_loss, float)
    assert isinstance(val_loss, float)
    assert isinstance(val_mae, float)
    assert train_loss >= 0.0
    assert val_loss >= 0.0
    assert val_mae >= 0.0


def test_update_early_stopping_counter_resets_only_after_improvement() -> None:
    best_mae = float("inf")
    patience_counter = 0

    best_mae, patience_counter = update_early_stopping_counter(best_mae, 1.0, patience_counter)
    assert best_mae == 1.0
    assert patience_counter == 0

    best_mae, patience_counter = update_early_stopping_counter(best_mae, 1.2, patience_counter)
    assert best_mae == 1.0
    assert patience_counter == 1

    best_mae, patience_counter = update_early_stopping_counter(best_mae, 1.3, patience_counter)
    assert best_mae == 1.0
    assert patience_counter == 2

    best_mae, patience_counter = update_early_stopping_counter(best_mae, 0.9, patience_counter)
    assert best_mae == 0.9
    assert patience_counter == 0


def test_save_training_summary_writes_expected_payload(tmp_path: Path) -> None:
    summary_path = tmp_path / "training_summary.json"
    summary = {
        "model_hyperparameters": {"input_size": 1, "hidden_size": 64, "num_layers": 1, "dropout": 0.0},
        "training_hyperparameters": {
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "batch_size": 64,
            "max_epochs": 1,
            "early_stopping_patience": 5,
            "loss": "MSELoss",
        },
        "train_val_history": [
            {
                "epoch": 1,
                "train_loss": 0.1,
                "val_loss": 0.2,
                "val_mae_normalized": 0.3,
                "val_mae_original": 1.7,
            }
        ],
        "best_epoch": 1,
        "best_validation_mae_normalized": 0.3,
        "best_validation_mae_original": 1.7,
        "normalization_stats": {"count": 2, "mean": 2.0, "std": 1.0},
    }

    save_training_summary(summary_path, summary)

    assert read_json(summary_path) == summary


def test_training_artifacts_use_self_describing_schema(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    checkpoint_path = tmp_path / "gru_best.pt"
    summary_path = tmp_path / "training_summary.json"
    model_hyperparameters = {"input_size": 1, "hidden_size": 64, "num_layers": 1, "dropout": 0.0}
    normalization_stats = {"count": 2, "mean": 2.0, "std": 1.0}

    checkpoint = save_checkpoint(
        best_state=None,
        epoch=1,
        validation_mae_normalized=0.25,
        model=model,
        checkpoint_path=checkpoint_path,
        model_hyperparameters=model_hyperparameters,
        normalization_stats=normalization_stats,
    )
    summary = {
        "model_hyperparameters": model_hyperparameters,
        "training_hyperparameters": {
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "batch_size": 64,
            "max_epochs": 1,
            "early_stopping_patience": 5,
            "loss": "MSELoss",
        },
        "train_val_history": [
            {
                "epoch": 1,
                "train_loss": 0.1,
                "val_loss": 0.2,
                "val_mae_normalized": 0.25,
                "val_mae_original": 1.5,
            }
        ],
        "best_epoch": 1,
        "best_validation_mae_normalized": 0.25,
        "best_validation_mae_original": 1.5,
        "normalization_stats": normalization_stats,
    }

    save_training_summary(summary_path, summary)

    saved_checkpoint = torch.load(checkpoint_path)
    saved_summary = read_json(summary_path)

    assert checkpoint["model_hyperparameters"] == model_hyperparameters
    assert checkpoint["normalization_stats"] == normalization_stats
    assert saved_checkpoint["model_hyperparameters"] == saved_summary["model_hyperparameters"]
    assert saved_checkpoint["normalization_stats"] == saved_summary["normalization_stats"]
    assert "best_validation_mae_normalized" in saved_summary
    assert "best_validation_mae_original" in saved_summary


def test_select_best_checkpoint_preserves_self_describing_schema(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    checkpoint_path = tmp_path / "gru_best.pt"
    model_hyperparameters = {"input_size": 1, "hidden_size": 64, "num_layers": 1, "dropout": 0.0}
    normalization_stats = {"count": 2, "mean": 2.0, "std": 1.0}

    best = select_best_checkpoint(
        best_state=None,
        epoch=1,
        validation_mae_normalized=0.6,
        model=model,
        checkpoint_path=checkpoint_path,
        model_hyperparameters=model_hyperparameters,
        normalization_stats=normalization_stats,
    )
    saved_checkpoint = torch.load(checkpoint_path)

    assert best["best_validation_mae_normalized"] == 0.6
    assert best["model_hyperparameters"] == model_hyperparameters
    assert best["normalization_stats"] == normalization_stats
    assert saved_checkpoint["model_hyperparameters"] == model_hyperparameters
    assert saved_checkpoint["normalization_stats"] == normalization_stats
