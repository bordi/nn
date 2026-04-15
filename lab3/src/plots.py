from __future__ import annotations

import os
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable
from tempfile import gettempdir

os.environ.setdefault("MPLCONFIGDIR", str(Path(gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .baselines import load_prepared_series
from .config import Config
from .evaluate import load_anomaly_predictions, load_forecast_predictions
from .utils import read_json


PLOT_FILENAMES = OrderedDict(
    (
        ("series_anomalies", "test_series_anomalies.png"),
        ("forecast_zoom", "test_forecast_zoom.png"),
        ("residuals_threshold", "test_residuals_threshold.png"),
    )
)

FALLBACK_ANOMALY_SPAN = timedelta(minutes=30)


def build_plot_paths(config: Config) -> dict[str, Path]:
    return {name: config.plots_dir / filename for name, filename in PLOT_FILENAMES.items()}


def _sorted_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: row["timestamp"])


def _infer_cadence(rows: list[dict[str, Any]]) -> timedelta:
    timestamps = [row["timestamp"] for row in _sorted_rows(rows)]
    for left, right in zip(timestamps, timestamps[1:]):
        delta = right - left
        if delta > timedelta(0):
            return delta
    return timedelta(0)


def _true_anomaly_spans(rows: list[dict[str, Any]]) -> list[tuple[Any, Any]]:
    sorted_rows = _sorted_rows(rows)
    cadence = _infer_cadence(sorted_rows)
    span_width = cadence if cadence > timedelta(0) else FALLBACK_ANOMALY_SPAN
    spans: list[tuple[Any, Any]] = []
    current_start = None
    current_end = None

    for row in sorted_rows:
        if int(row.get("is_anomaly", 0)) != 1:
            continue
        timestamp = row["timestamp"]
        if current_start is None:
            current_start = timestamp
            current_end = timestamp
            continue
        if span_width and timestamp <= current_end + span_width:
            current_end = timestamp
            continue
        spans.append((current_start, current_end + span_width))
        current_start = timestamp
        current_end = timestamp

    if current_start is not None and current_end is not None:
        spans.append((current_start, current_end + span_width))

    return spans


def _shade_true_anomalies(ax: plt.Axes, rows: list[dict[str, Any]]) -> None:
    for index, (start, end) in enumerate(_true_anomaly_spans(rows)):
        ax.axvspan(
            start,
            end,
            color="tab:orange",
            alpha=0.18,
            label="true anomaly" if index == 0 else None,
        )


def _scatter_predicted_anomalies(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    y_key: str,
    label: str = "predicted anomaly",
) -> None:
    predicted_rows = [row for row in _sorted_rows(rows) if int(row.get("predicted_anomaly", 0)) == 1]
    if not predicted_rows:
        return

    ax.scatter(
        [row["timestamp"] for row in predicted_rows],
        [float(row[y_key]) for row in predicted_rows],
        color="tab:red",
        marker="x",
        s=35,
        linewidths=1.4,
        label=label,
        zorder=4,
    )


def _finalize_plot(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("timestamp")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")


def plot_test_series_with_anomalies(
    prepared_rows: list[dict[str, Any]],
    anomaly_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    test_rows = [row for row in _sorted_rows(prepared_rows) if row.get("split") == "test"]
    test_anomaly_rows = [row for row in _sorted_rows(anomaly_rows) if row.get("split") == "test"]

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(
        [row["timestamp"] for row in test_rows],
        [float(row["value"]) for row in test_rows],
        color="tab:blue",
        linewidth=1.6,
        label="series",
        zorder=2,
    )
    _shade_true_anomalies(ax, test_rows)
    _scatter_predicted_anomalies(ax, test_anomaly_rows, y_key="target")
    _finalize_plot(ax, "Test series with anomalies", "value")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_test_forecast_zoom(
    forecast_rows: list[dict[str, Any]],
    anomaly_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    test_forecast_rows = [row for row in _sorted_rows(forecast_rows) if row.get("split") == "test"]
    test_anomaly_rows = [row for row in _sorted_rows(anomaly_rows) if row.get("split") == "test"]

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(
        [row["timestamp"] for row in test_forecast_rows],
        [float(row["target"]) for row in test_forecast_rows],
        color="tab:blue",
        linewidth=1.6,
        label="target",
        zorder=2,
    )
    ax.plot(
        [row["timestamp"] for row in test_forecast_rows],
        [float(row["prediction"]) for row in test_forecast_rows],
        color="tab:green",
        linewidth=1.6,
        label="prediction",
        zorder=3,
    )
    _shade_true_anomalies(ax, test_forecast_rows)
    _scatter_predicted_anomalies(ax, test_anomaly_rows, y_key="target")
    _finalize_plot(ax, "Test forecast zoom", "value")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_residuals_with_threshold(
    forecast_rows: list[dict[str, Any]],
    anomaly_rows: list[dict[str, Any]],
    threshold: float,
    output_path: Path,
) -> Path:
    test_forecast_rows = [row for row in _sorted_rows(forecast_rows) if row.get("split") == "test"]
    test_anomaly_rows = [row for row in _sorted_rows(anomaly_rows) if row.get("split") == "test"]

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(
        [row["timestamp"] for row in test_forecast_rows],
        [float(row["residual_abs"]) for row in test_forecast_rows],
        color="tab:purple",
        linewidth=1.6,
        label="residual_abs",
        zorder=2,
    )
    ax.axhline(threshold, color="tab:red", linestyle="--", linewidth=1.4, label="threshold")
    _shade_true_anomalies(ax, test_forecast_rows)
    _scatter_predicted_anomalies(ax, test_anomaly_rows, y_key="residual_abs")
    _finalize_plot(ax, "Residuals with threshold", "residual_abs")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_all_plots(config: Config) -> dict[str, Path]:
    config.ensure_directories()
    plot_paths = build_plot_paths(config)

    prepared_rows = load_prepared_series(config.prepared_dir / "series.csv")
    forecast_rows = load_forecast_predictions(config.forecasts_dir / "gru_predictions.csv")
    anomaly_rows = load_anomaly_predictions(config.anomalies_dir / "anomaly_predictions.csv")
    threshold_summary = read_json(config.anomalies_dir / "threshold_summary.json")
    threshold_value = float(threshold_summary["threshold_value"])

    plot_test_series_with_anomalies(
        prepared_rows=prepared_rows,
        anomaly_rows=anomaly_rows,
        output_path=plot_paths["series_anomalies"],
    )
    plot_test_forecast_zoom(
        forecast_rows=forecast_rows,
        anomaly_rows=anomaly_rows,
        output_path=plot_paths["forecast_zoom"],
    )
    plot_residuals_with_threshold(
        forecast_rows=forecast_rows,
        anomaly_rows=anomaly_rows,
        threshold=threshold_value,
        output_path=plot_paths["residuals_threshold"],
    )

    return plot_paths
