from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab3.src.anomaly import (
    compute_threshold,
    flag_anomalies,
    load_checkpoint_and_predict,
    save_anomaly_rows,
    save_forecast_rows,
    save_threshold_summary,
)
from lab3.src.config import load_config
from lab3.src.data import save_prepared_metadata, save_prepared_series
from lab3.src.models import GRUForecastModel
from lab3.src.train import save_checkpoint


def test_threshold_uses_only_normal_validation_rows() -> None:
    rows = [
        {"timestamp": datetime(2020, 1, 1, 0, 0), "split": "val", "residual_abs": 1.0, "is_anomaly": 0},
        {"timestamp": datetime(2020, 1, 1, 0, 30), "split": "val", "residual_abs": 3.0, "is_anomaly": 0},
        {"timestamp": datetime(2020, 1, 1, 1, 0), "split": "val", "residual_abs": 100.0, "is_anomaly": 1},
        {"timestamp": datetime(2020, 1, 1, 1, 30), "split": "test", "residual_abs": 200.0, "is_anomaly": 0},
    ]

    threshold = compute_threshold(rows, percentile=95)

    assert threshold == 2.9


def test_threshold_uses_95th_percentile() -> None:
    rows = [
        {"timestamp": datetime(2020, 1, 1, 0, 0), "split": "val", "residual_abs": 1.0, "is_anomaly": 0},
        {"timestamp": datetime(2020, 1, 1, 0, 30), "split": "val", "residual_abs": 2.0, "is_anomaly": 0},
        {"timestamp": datetime(2020, 1, 1, 1, 0), "split": "val", "residual_abs": 3.0, "is_anomaly": 0},
        {"timestamp": datetime(2020, 1, 1, 1, 30), "split": "val", "residual_abs": 4.0, "is_anomaly": 0},
    ]

    threshold = compute_threshold(rows, percentile=95)

    assert round(threshold, 6) == 3.85


def test_threshold_percentile_must_be_within_closed_0_to_100_interval() -> None:
    rows = [
        {"timestamp": datetime(2020, 1, 1, 0, 0), "split": "val", "residual_abs": 1.0, "is_anomaly": 0},
        {"timestamp": datetime(2020, 1, 1, 0, 30), "split": "val", "residual_abs": 2.0, "is_anomaly": 0},
    ]

    try:
        compute_threshold(rows, percentile=101)
    except ValueError as exc:
        assert "between 0 and 100" in str(exc)
    else:
        raise AssertionError("Expected compute_threshold to reject percentiles outside [0, 100]")


def test_predicted_anomaly_is_one_when_residual_abs_meets_or_exceeds_threshold() -> None:
    rows = [
        {"timestamp": datetime(2020, 1, 1, 0, 0), "split": "val", "residual_abs": 2.9, "is_anomaly": 0},
        {"timestamp": datetime(2020, 1, 1, 0, 30), "split": "test", "residual_abs": 2.89, "is_anomaly": 1},
    ]

    flagged_rows = flag_anomalies(rows, threshold=2.9)

    assert [row["predicted_anomaly"] for row in flagged_rows] == [1, 0]


def test_save_helpers_write_spec_facing_forecast_and_anomaly_csvs(tmp_path: Path) -> None:
    forecast_rows = [
        {
            "timestamp": datetime(2020, 1, 1, 0, 0),
            "split": "val",
            "target": 10.0,
            "prediction": 9.0,
            "residual": 1.0,
            "residual_abs": 1.0,
            "is_anomaly": 0,
        }
    ]
    anomaly_rows = [
        {
            "timestamp": datetime(2020, 1, 1, 0, 0),
            "split": "val",
            "target": 10.0,
            "prediction": 9.0,
            "residual_abs": 1.0,
            "is_anomaly": 0,
            "predicted_anomaly": 1,
        }
    ]

    forecast_path = tmp_path / "gru_predictions.csv"
    anomaly_path = tmp_path / "anomaly_predictions.csv"
    save_forecast_rows(forecast_path, forecast_rows)
    save_anomaly_rows(anomaly_path, anomaly_rows)

    with forecast_path.open("r", encoding="utf-8", newline="") as handle:
        forecast_reader = csv.DictReader(handle)
        assert forecast_reader.fieldnames == [
            "timestamp",
            "split",
            "target",
            "prediction",
            "residual",
            "residual_abs",
            "is_anomaly",
        ]
        assert next(forecast_reader)["residual"] == "1.0"

    with anomaly_path.open("r", encoding="utf-8", newline="") as handle:
        anomaly_reader = csv.DictReader(handle)
        assert anomaly_reader.fieldnames == [
            "timestamp",
            "split",
            "target",
            "prediction",
            "residual_abs",
            "is_anomaly",
            "predicted_anomaly",
        ]
        assert next(anomaly_reader)["predicted_anomaly"] == "1"


def test_save_threshold_summary_writes_spec_schema(tmp_path: Path) -> None:
    threshold_summary = {
        "threshold_strategy": "validation_normal_residual_abs_percentile",
        "threshold_percentile": 95,
        "threshold_value": 2.9,
        "validation_points_used": 10,
    }

    path = tmp_path / "threshold_summary.json"
    save_threshold_summary(path, threshold_summary)

    saved_payload = json.loads(path.read_text(encoding="utf-8"))
    assert saved_payload == threshold_summary
    assert "threshold_percentile" in saved_payload
    assert "threshold_value" in saved_payload
    assert "percentile" not in saved_payload
    assert "threshold" not in saved_payload


def test_load_checkpoint_and_predict_reuses_saved_normalization_stats(tmp_path: Path) -> None:
    config = load_config(project_root=tmp_path)
    config.ensure_directories()

    records = [
        {
            "timestamp": datetime(2020, 1, 1, 0, 0),
            "value": 10.0,
            "is_anomaly": 0,
            "split": "train",
        },
        {
            "timestamp": datetime(2020, 1, 1, 0, 30),
            "value": 20.0,
            "is_anomaly": 0,
            "split": "train",
        },
        {
            "timestamp": datetime(2020, 1, 1, 1, 0),
            "value": 30.0,
            "is_anomaly": 0,
            "split": "val",
        },
        {
            "timestamp": datetime(2020, 1, 1, 1, 30),
            "value": 40.0,
            "is_anomaly": 1,
            "split": "test",
        },
    ]
    save_prepared_series(config.prepared_dir / "series.csv", records)
    save_prepared_metadata(
        config.prepared_dir / "prepared_metadata.json",
        {"window_size": 2, "horizon": 1},
    )

    import torch

    model = GRUForecastModel(input_size=1, hidden_size=1, num_layers=1, dropout=0.0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    checkpoint_path = config.models_dir / "gru_best.pt"
    save_checkpoint(
        best_state=None,
        epoch=1,
        validation_mae_normalized=0.0,
        model=model,
        checkpoint_path=checkpoint_path,
        model_hyperparameters={"input_size": 1, "hidden_size": 1, "num_layers": 1, "dropout": 0.0},
        normalization_stats={"count": 2, "mean": 100.0, "std": 10.0},
    )
    (config.models_dir / "training_summary.json").write_text(
        '{"normalization_stats": {"count": 2, "mean": 100.0, "std": 10.0}}',
        encoding="utf-8",
    )

    rows = load_checkpoint_and_predict(
        prepared_series_path=config.prepared_dir / "series.csv",
        prepared_metadata_path=config.prepared_dir / "prepared_metadata.json",
        training_summary_path=config.models_dir / "training_summary.json",
        checkpoint_path=checkpoint_path,
    )

    assert len(rows) == 2
    assert [row["split"] for row in rows] == ["val", "test"]
    assert rows[0]["target"] == 30.0
    assert rows[0]["prediction"] == 100.0


def test_load_checkpoint_and_predict_fails_when_summary_and_checkpoint_normalization_diverge(
    tmp_path: Path,
) -> None:
    config = load_config(project_root=tmp_path)
    config.ensure_directories()

    records = [
        {"timestamp": datetime(2020, 1, 1, 0, 0), "value": 10.0, "is_anomaly": 0, "split": "train"},
        {"timestamp": datetime(2020, 1, 1, 0, 30), "value": 20.0, "is_anomaly": 0, "split": "train"},
        {"timestamp": datetime(2020, 1, 1, 1, 0), "value": 30.0, "is_anomaly": 0, "split": "val"},
        {"timestamp": datetime(2020, 1, 1, 1, 30), "value": 40.0, "is_anomaly": 1, "split": "test"},
    ]
    save_prepared_series(config.prepared_dir / "series.csv", records)
    save_prepared_metadata(config.prepared_dir / "prepared_metadata.json", {"window_size": 2, "horizon": 1})

    import torch

    model = GRUForecastModel(input_size=1, hidden_size=1, num_layers=1, dropout=0.0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    checkpoint_path = config.models_dir / "gru_best.pt"
    save_checkpoint(
        best_state=None,
        epoch=1,
        validation_mae_normalized=0.0,
        model=model,
        checkpoint_path=checkpoint_path,
        model_hyperparameters={"input_size": 1, "hidden_size": 1, "num_layers": 1, "dropout": 0.0},
        normalization_stats={"count": 2, "mean": 100.0, "std": 10.0},
    )
    (config.models_dir / "training_summary.json").write_text(
        '{"normalization_stats": {"count": 2, "mean": 101.0, "std": 10.0}}',
        encoding="utf-8",
    )

    try:
        load_checkpoint_and_predict(
            prepared_series_path=config.prepared_dir / "series.csv",
            prepared_metadata_path=config.prepared_dir / "prepared_metadata.json",
            training_summary_path=config.models_dir / "training_summary.json",
            checkpoint_path=checkpoint_path,
        )
    except ValueError as exc:
        assert "normalization" in str(exc).lower()
        assert "checkpoint" in str(exc).lower()
        assert "training_summary" in str(exc).lower()
    else:
        raise AssertionError("Expected mismatched normalization stats to fail loudly")
