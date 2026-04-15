from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab3.src.baselines import (
    compute_forecast_metrics,
    run_persistence_baseline,
    save_baseline_metrics,
    save_baseline_predictions,
)


def _make_window_records() -> list[dict[str, object]]:
    start = datetime(2020, 1, 1, 0, 0)
    return [
        {
            "input_values": [1.0, 2.0],
            "target": 3.0,
            "target_timestamp": start + timedelta(minutes=60),
            "split": "val",
            "target_is_anomaly": 0,
        },
        {
            "input_values": [4.0, 5.0],
            "target": 8.0,
            "target_timestamp": start + timedelta(minutes=90),
            "split": "test",
            "target_is_anomaly": 1,
        },
    ]


def test_persistence_prediction_equals_last_value_in_window() -> None:
    rows = run_persistence_baseline(_make_window_records())

    assert [row["prediction"] for row in rows] == [2.0, 5.0]
    assert [row["target"] for row in rows] == [3.0, 8.0]


def test_residual_columns_are_computed_correctly(tmp_path: Path) -> None:
    rows = run_persistence_baseline(_make_window_records())

    assert rows[0]["residual"] == 1.0
    assert rows[0]["residual_abs"] == 1.0
    assert rows[1]["residual"] == 3.0
    assert rows[1]["residual_abs"] == 3.0
    assert rows[0]["is_anomaly"] == 0
    assert rows[1]["is_anomaly"] == 1

    predictions_path = tmp_path / "baseline_predictions.csv"
    metrics_path = tmp_path / "baseline_metrics.json"
    save_baseline_predictions(predictions_path, rows)
    save_baseline_metrics(metrics_path, compute_forecast_metrics(rows))

    saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert saved_metrics["val"]["mae"] == 1.0
    assert saved_metrics["test"]["mae"] == 3.0


def test_baseline_produces_split_specific_metrics() -> None:
    rows = run_persistence_baseline(_make_window_records())
    metrics = compute_forecast_metrics(rows)

    assert set(metrics) == {"val", "test"}
    assert metrics["val"]["count"] == 1
    assert metrics["test"]["count"] == 1
    assert metrics["val"]["mae"] == 1.0
    assert metrics["test"]["mape"] == 37.5
