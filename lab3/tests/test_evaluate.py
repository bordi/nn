from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab3.src.evaluate import (
    build_error_analysis,
    compute_anomaly_metrics,
    compute_forecast_metrics,
    compute_mape,
    compute_precision_recall_f1,
    load_anomaly_predictions,
    load_forecast_predictions,
    save_anomaly_metrics,
    save_error_analysis,
    save_forecast_metrics,
)


def _make_rows() -> list[dict[str, object]]:
    return [
        {
            "timestamp": datetime(2020, 1, 1, 0, 0),
            "split": "val",
            "target": 10.0,
            "prediction": 8.0,
            "is_anomaly": 0,
            "predicted_anomaly": 1,
        },
        {
            "timestamp": datetime(2020, 1, 1, 0, 30),
            "split": "val",
            "target": 0.0,
            "prediction": 1.0,
            "is_anomaly": 0,
            "predicted_anomaly": 0,
        },
        {
            "timestamp": datetime(2020, 1, 1, 1, 0),
            "split": "test",
            "target": 4.0,
            "prediction": 1.0,
            "is_anomaly": 1,
            "predicted_anomaly": 1,
        },
        {
            "timestamp": datetime(2020, 1, 1, 1, 30),
            "split": "test",
            "target": 5.0,
            "prediction": 7.0,
            "is_anomaly": 0,
            "predicted_anomaly": 1,
        },
        {
            "timestamp": datetime(2020, 1, 1, 2, 0),
            "split": "test",
            "target": 6.0,
            "prediction": 6.0,
            "is_anomaly": 0,
            "predicted_anomaly": 0,
        },
        {
            "timestamp": datetime(2020, 1, 1, 2, 30),
            "split": "test",
            "target": 3.0,
            "prediction": 2.0,
            "is_anomaly": 1,
            "predicted_anomaly": 0,
        },
    ]


def test_compute_forecast_metrics_reports_mae_and_safe_mape_by_split() -> None:
    metrics = compute_forecast_metrics(_make_rows())

    assert metrics["val"]["count"] == 2
    assert metrics["val"]["mae"] == 1.5
    assert metrics["val"]["mape"] == 20.0
    assert metrics["test"]["count"] == 4
    assert metrics["test"]["mae"] == 1.5
    assert round(metrics["test"]["mape"], 6) == round(37.08333333333333, 6)


def test_compute_mape_skips_zero_targets_using_epsilon_guard() -> None:
    rows = [
        {"target": 0.0, "prediction": 2.0},
        {"target": 10.0, "prediction": 8.0},
    ]

    assert compute_mape(rows, epsilon=1e-8) == 20.0


def test_compute_anomaly_metrics_is_pointwise_and_restricted_to_test_split() -> None:
    metrics = compute_anomaly_metrics(_make_rows())

    assert metrics["count"] == 4
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fn"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_compute_precision_recall_f1_returns_pointwise_scores() -> None:
    metrics = compute_precision_recall_f1(_make_rows())

    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_error_analysis_includes_confusion_counts_and_fp_fn_examples() -> None:
    analysis = build_error_analysis(_make_rows(), max_examples=2)

    assert analysis["counts"] == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    assert len(analysis["false_positives"]) == 1
    assert len(analysis["false_negatives"]) == 1
    assert analysis["false_positives"][0]["split"] == "test"
    assert analysis["false_positives"][0]["timestamp"] == "2020-01-01 01:30:00"
    assert analysis["false_negatives"][0]["timestamp"] == "2020-01-01 02:30:00"


def test_save_metric_helpers_and_loaders_round_trip_forecast_and_anomaly_artifacts(tmp_path: Path) -> None:
    forecast_metrics = {"val": {"count": 1, "mae": 1.0, "mape": 10.0}}
    anomaly_metrics = {"count": 2, "precision": 0.5, "recall": 0.5, "f1": 0.5, "tp": 1, "fp": 1, "tn": 0, "fn": 0}
    error_analysis = {"counts": {"tp": 1, "fp": 1, "tn": 0, "fn": 0}, "false_positives": [], "false_negatives": []}

    save_forecast_metrics(tmp_path / "forecast_metrics.json", forecast_metrics)
    save_anomaly_metrics(tmp_path / "anomaly_metrics.json", anomaly_metrics)
    save_error_analysis(tmp_path / "error_analysis.json", error_analysis)

    assert json.loads((tmp_path / "forecast_metrics.json").read_text(encoding="utf-8")) == forecast_metrics
    assert json.loads((tmp_path / "anomaly_metrics.json").read_text(encoding="utf-8")) == anomaly_metrics
    assert json.loads((tmp_path / "error_analysis.json").read_text(encoding="utf-8")) == error_analysis

    forecast_csv = tmp_path / "gru_predictions.csv"
    forecast_csv.write_text(
        "timestamp,split,target,prediction,residual,residual_abs,is_anomaly\n"
        "2020-01-01 00:00:00,val,10,9,1,1,0\n",
        encoding="utf-8",
    )
    anomaly_csv = tmp_path / "anomaly_predictions.csv"
    anomaly_csv.write_text(
        "timestamp,split,target,prediction,residual_abs,is_anomaly,predicted_anomaly\n"
        "2020-01-01 00:00:00,test,10,9,1,0,1\n",
        encoding="utf-8",
    )

    forecast_rows = load_forecast_predictions(forecast_csv)
    anomaly_rows = load_anomaly_predictions(anomaly_csv)

    assert forecast_rows[0]["residual"] == 1.0
    assert anomaly_rows[0]["predicted_anomaly"] == 1
    assert forecast_rows[0]["split"] == "val"
    assert anomaly_rows[0]["split"] == "test"
