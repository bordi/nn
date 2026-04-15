from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .config import Config
from .utils import format_timestamp, parse_timestamp, write_json


FORECAST_METRICS_FILENAME = "forecast_metrics.json"
ANOMALY_METRICS_FILENAME = "anomaly_metrics.json"
ERROR_ANALYSIS_FILENAME = "error_analysis.json"


def compute_mae(rows: list[dict[str, Any]]) -> float:
    if not rows:
        raise ValueError("Need at least one row to compute MAE")
    return sum(abs(float(row["target"]) - float(row["prediction"])) for row in rows) / len(rows)


def compute_mape(rows: list[dict[str, Any]], epsilon: float = 1e-8) -> float:
    if not rows:
        raise ValueError("Need at least one row to compute MAPE")

    denominators = []
    for row in rows:
        target = abs(float(row["target"]))
        if target <= epsilon:
            continue
        error = abs(float(row["target"]) - float(row["prediction"]))
        denominators.append(error / target * 100.0)

    if not denominators:
        return 0.0
    return sum(denominators) / len(denominators)


def compute_safe_mape(rows: list[dict[str, Any]]) -> float:
    return compute_mape(rows)


def compute_forecast_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    for split_name in ("val", "test"):
        split_rows = [row for row in rows if row.get("split") == split_name]
        if not split_rows:
            continue
        metrics[split_name] = {
            "count": len(split_rows),
            "mae": compute_mae(split_rows),
            "mape": compute_mape(split_rows),
        }
    return metrics


def compute_precision_recall_f1(rows: list[dict[str, Any]]) -> dict[str, float]:
    metrics = compute_anomaly_metrics(rows)
    return {
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
    }


def compute_anomaly_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    test_rows = [row for row in rows if row.get("split") == "test"]
    if not test_rows:
        raise ValueError("Need at least one test row to compute anomaly metrics")

    tp = fp = tn = fn = 0
    for row in test_rows:
        actual = int(row["is_anomaly"])
        predicted = int(row["predicted_anomaly"])
        if predicted == 1 and actual == 1:
            tp += 1
        elif predicted == 1 and actual == 0:
            fp += 1
        elif predicted == 0 and actual == 0:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    return {
        "count": len(test_rows),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _example_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": format_timestamp(parse_timestamp(row["timestamp"])),
        "split": row["split"],
        "target": float(row["target"]),
        "prediction": float(row["prediction"]),
        "residual_abs": abs(float(row["target"]) - float(row["prediction"]))
        if "residual_abs" not in row
        else float(row["residual_abs"]),
        "is_anomaly": int(row["is_anomaly"]),
        "predicted_anomaly": int(row["predicted_anomaly"]),
    }


def build_error_analysis(rows: list[dict[str, Any]], max_examples: int = 3) -> dict[str, Any]:
    test_rows = [row for row in rows if row.get("split") == "test"]
    metrics = compute_anomaly_metrics(test_rows)
    false_positives = [
        _example_row(row)
        for row in test_rows
        if int(row["predicted_anomaly"]) == 1 and int(row["is_anomaly"]) == 0
    ][:max_examples]
    false_negatives = [
        _example_row(row)
        for row in test_rows
        if int(row["predicted_anomaly"]) == 0 and int(row["is_anomaly"]) == 1
    ][:max_examples]

    return {
        "counts": {
            "tp": int(metrics["tp"]),
            "fp": int(metrics["fp"]),
            "tn": int(metrics["tn"]),
            "fn": int(metrics["fn"]),
        },
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def load_anomaly_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, Any]] = []
        for row in reader:
            rows.append(
                {
                    "timestamp": parse_timestamp(row["timestamp"]),
                    "split": row["split"],
                    "target": float(row["target"]),
                    "prediction": float(row["prediction"]),
                    "residual_abs": float(row["residual_abs"]),
                    "is_anomaly": int(row["is_anomaly"]),
                    "predicted_anomaly": int(row["predicted_anomaly"]),
                }
            )
    return rows


def load_forecast_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, Any]] = []
        for row in reader:
            rows.append(
                {
                    "timestamp": parse_timestamp(row["timestamp"]),
                    "split": row["split"],
                    "target": float(row["target"]),
                    "prediction": float(row["prediction"]),
                    "residual": float(row["residual"]),
                    "residual_abs": float(row["residual_abs"]),
                    "is_anomaly": int(row["is_anomaly"]),
                }
            )
    return rows


def save_forecast_metrics(path: Path, metrics: dict[str, Any]) -> None:
    write_json(path, metrics)


def save_anomaly_metrics(path: Path, metrics: dict[str, Any]) -> None:
    write_json(path, metrics)


def save_error_analysis(path: Path, analysis: dict[str, Any]) -> None:
    write_json(path, analysis)


def run_evaluation_pipeline(config: Config) -> dict[str, Any]:
    config.ensure_directories()

    forecast_predictions_path = config.forecasts_dir / "gru_predictions.csv"
    anomaly_predictions_path = config.anomalies_dir / "anomaly_predictions.csv"
    forecast_rows = load_forecast_predictions(forecast_predictions_path)
    anomaly_rows = load_anomaly_predictions(anomaly_predictions_path)
    forecast_metrics = compute_forecast_metrics(forecast_rows)
    anomaly_metrics = compute_anomaly_metrics(anomaly_rows)
    error_analysis = build_error_analysis(anomaly_rows)

    forecast_metrics_path = config.eval_dir / FORECAST_METRICS_FILENAME
    anomaly_metrics_path = config.eval_dir / ANOMALY_METRICS_FILENAME
    error_analysis_path = config.eval_dir / ERROR_ANALYSIS_FILENAME

    save_forecast_metrics(forecast_metrics_path, forecast_metrics)
    save_anomaly_metrics(anomaly_metrics_path, anomaly_metrics)
    save_error_analysis(error_analysis_path, error_analysis)

    return {
        "forecast_metrics_path": forecast_metrics_path,
        "anomaly_metrics_path": anomaly_metrics_path,
        "error_analysis_path": error_analysis_path,
        "forecast_metrics": forecast_metrics,
        "anomaly_metrics": anomaly_metrics,
        "error_analysis": error_analysis,
    }
