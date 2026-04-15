from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from .baselines import load_prepared_series
from .config import Config
from .models import GRUForecastModel
from .train import TRAINING_SUMMARY_FILENAME, apply_standardizer
from .utils import format_timestamp, read_json, write_csv, write_json
from .windows import build_eval_windows


PREDICTIONS_FILENAME = "gru_predictions.csv"
ANOMALY_PREDICTIONS_FILENAME = "anomaly_predictions.csv"
THRESHOLD_SUMMARY_FILENAME = "threshold_summary.json"
CHECKPOINT_FILENAME = "gru_best.pt"


def denormalize_value(value: float, normalization_stats: dict[str, float | int]) -> float:
    return float(value) * float(normalization_stats["std"]) + float(normalization_stats["mean"])


def percentile(values: list[float], percentile_rank: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile for an empty sequence")
    if not 0 <= float(percentile_rank) <= 100:
        raise ValueError("threshold_percentile must be between 0 and 100 inclusive")

    sorted_values = sorted(float(value) for value in values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (len(sorted_values) - 1) * (float(percentile_rank) / 100.0)
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    if lower_index == upper_index:
        return sorted_values[lower_index]

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    weight = rank - lower_index
    return lower_value + (upper_value - lower_value) * weight


def compute_threshold(rows: list[dict[str, Any]], percentile: int = 95) -> float:
    validation_normals = [
        float(row["residual_abs"])
        for row in rows
        if row.get("split") == "val" and int(row.get("is_anomaly", 0)) == 0
    ]
    if not validation_normals:
        raise ValueError("Need at least one normal validation residual to compute threshold")

    return percentile_fn(validation_normals, percentile)


def percentile_fn(values: list[float], percentile_rank: int) -> float:
    return percentile(values, percentile_rank)


def resolve_normalization_stats(
    training_summary_stats: dict[str, float | int],
    checkpoint_stats: dict[str, float | int],
) -> dict[str, float | int]:
    if training_summary_stats != checkpoint_stats:
        raise ValueError(
            "Normalization stats mismatch between checkpoint and training_summary artifacts"
        )
    return training_summary_stats


def apply_threshold(rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    flagged_rows: list[dict[str, Any]] = []
    for row in rows:
        residual_abs = float(row["residual_abs"])
        flagged_rows.append({**row, "predicted_anomaly": int(residual_abs >= threshold)})
    return flagged_rows


def flag_anomalies(rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    return apply_threshold(rows, threshold)


def _load_model(checkpoint_path: Path) -> tuple[GRUForecastModel, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    hyperparameters = checkpoint["model_hyperparameters"]
    model = GRUForecastModel(
        input_size=int(hyperparameters["input_size"]),
        hidden_size=int(hyperparameters["hidden_size"]),
        num_layers=int(hyperparameters["num_layers"]),
        dropout=float(hyperparameters["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def build_prediction_rows(
    records: list[dict[str, Any]],
    *,
    model: GRUForecastModel,
    normalization_stats: dict[str, float | int],
    window_size: int,
    horizon: int,
) -> list[dict[str, Any]]:
    normalized_records = apply_standardizer(records, normalization_stats)
    rows: list[dict[str, Any]] = []

    for split_name in ("val", "test"):
        for window in build_eval_windows(
            normalized_records,
            split_name,
            window_size=window_size,
            horizon=horizon,
        ):
            inputs = torch.tensor([window["input_values"]], dtype=torch.float32).unsqueeze(-1)
            with torch.no_grad():
                predicted_normalized = float(model(inputs).item())

            target = denormalize_value(float(window["target"]), normalization_stats)
            prediction = denormalize_value(predicted_normalized, normalization_stats)
            residual = target - prediction
            rows.append(
                {
                    "timestamp": window["target_timestamp"],
                    "split": split_name,
                    "target": target,
                    "prediction": prediction,
                    "residual": residual,
                    "residual_abs": abs(residual),
                    "is_anomaly": int(window["target_is_anomaly"]),
                }
            )

    return rows


def load_checkpoint_and_predict(
    *,
    prepared_series_path: Path,
    prepared_metadata_path: Path,
    training_summary_path: Path,
    checkpoint_path: Path,
) -> list[dict[str, Any]]:
    records = load_prepared_series(prepared_series_path)
    metadata = read_json(prepared_metadata_path)
    training_summary = read_json(training_summary_path)
    model, checkpoint = _load_model(checkpoint_path)
    normalization_stats = resolve_normalization_stats(
        training_summary["normalization_stats"],
        checkpoint["normalization_stats"],
    )
    return build_prediction_rows(
        records,
        model=model,
        normalization_stats=normalization_stats,
        window_size=int(metadata["window_size"]),
        horizon=int(metadata["horizon"]),
    )


def save_forecast_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
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


def save_forecast_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    save_forecast_predictions(path, rows)


def save_anomaly_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    csv_rows = [
        {
            "timestamp": format_timestamp(row["timestamp"]),
            "split": row["split"],
            "target": row["target"],
            "prediction": row["prediction"],
            "residual_abs": row["residual_abs"],
            "is_anomaly": row["is_anomaly"],
            "predicted_anomaly": row["predicted_anomaly"],
        }
        for row in rows
    ]
    write_csv(
        path,
        csv_rows,
        ["timestamp", "split", "target", "prediction", "residual_abs", "is_anomaly", "predicted_anomaly"],
    )


def save_anomaly_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    save_anomaly_predictions(path, rows)


def save_threshold_summary(path: Path, summary: dict[str, Any]) -> None:
    write_json(path, summary)


def run_detection_pipeline(config: Config) -> dict[str, Any]:
    config.ensure_directories()

    prepared_series_path = config.prepared_dir / "series.csv"
    prepared_metadata_path = config.prepared_dir / "prepared_metadata.json"
    training_summary_path = config.models_dir / TRAINING_SUMMARY_FILENAME
    checkpoint_path = config.models_dir / CHECKPOINT_FILENAME

    prediction_rows = load_checkpoint_and_predict(
        prepared_series_path=prepared_series_path,
        prepared_metadata_path=prepared_metadata_path,
        training_summary_path=training_summary_path,
        checkpoint_path=checkpoint_path,
    )
    threshold = compute_threshold(prediction_rows, percentile=config.threshold_percentile)
    anomaly_rows = flag_anomalies(prediction_rows, threshold)
    validation_points_used = sum(
        1 for row in prediction_rows if row["split"] == "val" and int(row["is_anomaly"]) == 0
    )

    forecasts_path = config.forecasts_dir / PREDICTIONS_FILENAME
    anomalies_path = config.anomalies_dir / ANOMALY_PREDICTIONS_FILENAME
    threshold_path = config.anomalies_dir / THRESHOLD_SUMMARY_FILENAME

    save_forecast_rows(forecasts_path, prediction_rows)
    save_anomaly_rows(anomalies_path, anomaly_rows)
    save_threshold_summary(
        threshold_path,
        {
            "threshold_strategy": "validation_normal_residual_abs_percentile",
            "threshold_percentile": config.threshold_percentile,
            "threshold_value": threshold,
            "validation_points_used": validation_points_used,
        },
    )

    return {
        "forecast_predictions_path": forecasts_path,
        "anomaly_predictions_path": anomalies_path,
        "threshold_summary_path": threshold_path,
        "threshold": threshold,
        "prediction_rows": prediction_rows,
        "anomaly_rows": anomaly_rows,
    }
