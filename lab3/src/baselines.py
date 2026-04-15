from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .utils import format_timestamp, parse_timestamp, read_json, write_csv, write_json
from .windows import build_eval_windows


PREDICTIONS_FILENAME = "baseline_predictions.csv"
METRICS_FILENAME = "baseline_metrics.json"


def load_prepared_series(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        records: list[dict[str, Any]] = []
        for row in reader:
            records.append(
                {
                    "timestamp": parse_timestamp(row["timestamp"]),
                    "value": float(row["value"]),
                    "is_anomaly": int(row["is_anomaly"]),
                    "split": row["split"],
                }
            )
    records.sort(key=lambda record: record["timestamp"])
    return records


def run_persistence_baseline(window_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prediction_rows: list[dict[str, Any]] = []
    for window in window_records:
        prediction = float(window["input_values"][-1])
        target = float(window["target"])
        residual = target - prediction
        prediction_rows.append(
            {
                "timestamp": window["target_timestamp"],
                "split": window["split"],
                "target": target,
                "prediction": prediction,
                "residual": residual,
                "residual_abs": abs(residual),
                "is_anomaly": int(window["target_is_anomaly"]),
            }
        )
    return prediction_rows


def compute_forecast_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["split"]), []).append(row)

    metrics: dict[str, dict[str, float | int]] = {}
    for split in sorted(grouped):
        split_rows = grouped[split]
        if not split_rows:
            continue
        mae = sum(abs(float(row["target"]) - float(row["prediction"])) for row in split_rows) / len(split_rows)
        mape = (
            sum(
                abs(float(row["target"]) - float(row["prediction"]))
                / max(abs(float(row["target"])), 1e-8)
                * 100
                for row in split_rows
            )
            / len(split_rows)
        )
        metrics[split] = {
            "count": len(split_rows),
            "mae": mae,
            "mape": mape,
        }

    return metrics


def save_baseline_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    csv_rows = [
        {
            "timestamp": format_timestamp(row["timestamp"]),
            "split": row["split"],
            "target": row["target"],
            "prediction": row["prediction"],
            "residual": row["residual"],
            "residual_abs": row["residual_abs"],
            "is_anomaly": row["is_anomaly"],
        }
        for row in rows
    ]
    write_csv(
        path,
        csv_rows,
        ["timestamp", "split", "target", "prediction", "residual", "residual_abs", "is_anomaly"],
    )


def save_baseline_metrics(path: Path, metrics: dict[str, dict[str, float | int]]) -> None:
    write_json(path, metrics)


def run_baseline_pipeline(
    *,
    prepared_series_path: Path,
    prepared_metadata_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    records = load_prepared_series(prepared_series_path)
    metadata = read_json(prepared_metadata_path)
    window_size = int(metadata["window_size"])
    horizon = int(metadata["horizon"])

    val_windows = build_eval_windows(records, "val", window_size=window_size, horizon=horizon)
    test_windows = build_eval_windows(records, "test", window_size=window_size, horizon=horizon)
    window_records = val_windows + test_windows
    if not window_records:
        raise ValueError("No validation or test windows could be built from prepared artifacts")

    prediction_rows = run_persistence_baseline(window_records)
    metrics = compute_forecast_metrics(prediction_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / PREDICTIONS_FILENAME
    metrics_path = output_dir / METRICS_FILENAME
    save_baseline_predictions(predictions_path, prediction_rows)
    save_baseline_metrics(metrics_path, metrics)

    return {
        "predictions_path": predictions_path,
        "metrics_path": metrics_path,
        "metrics": metrics,
        "rows": prediction_rows,
    }
