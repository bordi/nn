from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .config import Config
from .utils import format_timestamp, parse_timestamp, write_csv, write_json

SERIES_FILENAME = "series.csv"
METADATA_FILENAME = "prepared_metadata.json"
EXPECTED_CADENCE = timedelta(minutes=30)
LABELING_RULE = "start <= timestamp <= end"


def _parse_value(value: Any) -> float:
    if value is None:
        raise ValueError("Missing value")
    if isinstance(value, str) and not value.strip():
        raise ValueError("Missing value")
    return float(value)


def _validate_split_ratios(train_ratio: float, val_ratio: float) -> tuple[float, float, float]:
    if train_ratio <= 0:
        raise ValueError("train_ratio must be greater than 0")
    if val_ratio < 0:
        raise ValueError("val_ratio must be greater than or equal to 0")

    test_ratio = 1 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    return train_ratio, val_ratio, test_ratio


def load_series_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, Any]] = []
        for row in reader:
            timestamp = parse_timestamp(row.get("timestamp"))
            value = _parse_value(row.get("value"))
            rows.append({"timestamp": timestamp, "value": value})
    return validate_series(rows)


def load_label_windows(path: Path, series_key: str) -> list[dict[str, datetime]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    raw_windows = payload.get(series_key)
    if raw_windows is None:
        raise KeyError(f"Missing labels for series_key={series_key!r}")

    windows: list[dict[str, datetime]] = []
    for index, window in enumerate(raw_windows):
        if not isinstance(window, (list, tuple)) or len(window) != 2:
            raise ValueError(
                f"Malformed label window at index {index}: expected a 2-item pair of [start, end]"
            )
        start_raw, end_raw = window
        start = parse_timestamp(start_raw)
        end = parse_timestamp(end_raw)
        if start > end:
            raise ValueError(
                f"Malformed label window at index {index}: start must be <= end"
            )
        windows.append({"start": start, "end": end})
    windows.sort(key=lambda window: window["start"])
    return windows


def validate_series(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        raise ValueError("Series is empty")

    cleaned: list[dict[str, Any]] = []
    for record in records:
        timestamp = parse_timestamp(record.get("timestamp"))
        value = _parse_value(record.get("value"))
        cleaned.append({"timestamp": timestamp, "value": value})

    cleaned.sort(key=lambda record: record["timestamp"])

    deduped: list[dict[str, Any]] = []
    latest_by_timestamp: dict[datetime, dict[str, Any]] = {}
    for record in cleaned:
        latest_by_timestamp[record["timestamp"]] = record
    for timestamp in sorted(latest_by_timestamp):
        deduped.append(latest_by_timestamp[timestamp])

    if len(deduped) > 1:
        for previous, current in zip(deduped, deduped[1:], strict=False):
            if current["timestamp"] - previous["timestamp"] != EXPECTED_CADENCE:
                raise ValueError("Series cadence must be exactly 30 minutes")

    return deduped


def label_series_points(
    records: list[dict[str, Any]],
    windows: list[dict[str, datetime]],
) -> list[dict[str, Any]]:
    labeled: list[dict[str, Any]] = []
    for record in records:
        timestamp = parse_timestamp(record["timestamp"])
        is_anomaly = int(any(window["start"] <= timestamp <= window["end"] for window in windows))
        labeled.append({**record, "timestamp": timestamp, "is_anomaly": is_anomaly})
    return labeled


def split_series_by_time(
    records: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[dict[str, Any]], dict[str, datetime]]:
    if not records:
        raise ValueError("Series is empty")
    train_ratio, val_ratio, _ = _validate_split_ratios(train_ratio, val_ratio)

    total = len(records)
    train_rows = int(total * train_ratio)
    val_rows = int(total * val_ratio)
    if train_rows < 1:
        train_rows = 1
    if train_rows + val_rows >= total:
        val_rows = max(0, total - train_rows - 1)
    test_rows = total - train_rows - val_rows
    if test_rows < 1:
        test_rows = 1
        if val_rows > 0:
            val_rows -= 1
        else:
            train_rows -= 1

    boundaries: dict[str, datetime] = {}
    if train_rows:
        boundaries["train_end"] = records[train_rows - 1]["timestamp"]
    if val_rows:
        boundaries["val_end"] = records[train_rows + val_rows - 1]["timestamp"]

    split_records: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if index < train_rows:
            split = "train"
        elif index < train_rows + val_rows:
            split = "val"
        else:
            split = "test"
        split_records.append({**record, "split": split})

    return split_records, boundaries


def _ensure_split_usability(
    split_counts: dict[str, int],
    *,
    window_size: int,
    horizon: int,
) -> None:
    required_train_rows = window_size + horizon
    if split_counts["train"] < required_train_rows:
        raise ValueError(
            "Prepared train split needs at least window_size + horizon rows "
            f"({required_train_rows})"
        )
    if split_counts["val"] < horizon:
        raise ValueError(f"Prepared val split needs at least horizon rows ({horizon})")
    if split_counts["test"] < horizon:
        raise ValueError(f"Prepared test split needs at least horizon rows ({horizon})")


def save_prepared_series(path: Path, records: list[dict[str, Any]]) -> None:
    rows = [
        {
            "timestamp": format_timestamp(record["timestamp"]),
            "value": record["value"],
            "is_anomaly": record["is_anomaly"],
            "split": record["split"],
        }
        for record in records
    ]
    write_csv(path, rows, ["timestamp", "value", "is_anomaly", "split"])


def save_prepared_metadata(path: Path, metadata: dict[str, Any]) -> None:
    write_json(path, metadata)


def prepare_data(config: Config) -> dict[str, Any]:
    series_path = config.series_csv_path
    labels_path = config.labels_path

    records = load_series_csv(series_path)
    windows = load_label_windows(labels_path, config.series_key)
    labeled_records = label_series_points(records, windows)
    split_records, split_boundaries = split_series_by_time(
        labeled_records,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
    )

    split_counts = {name: 0 for name in ("train", "val", "test")}
    anomaly_counts = {name: 0 for name in ("train", "val", "test")}
    for record in split_records:
        split = record["split"]
        split_counts[split] += 1
        anomaly_counts[split] += int(record["is_anomaly"])

    _ensure_split_usability(
        split_counts,
        window_size=config.window_size,
        horizon=config.horizon,
    )

    train_ratio, val_ratio, test_ratio = _validate_split_ratios(
        config.train_ratio,
        config.val_ratio,
    )

    metadata = {
        "series_key": config.series_key,
        "split_boundaries": {key: format_timestamp(value) for key, value in split_boundaries.items()},
        "num_rows": len(split_records),
        "train_rows": split_counts["train"],
        "val_rows": split_counts["val"],
        "test_rows": split_counts["test"],
        "window_size": config.window_size,
        "horizon": config.horizon,
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": round(test_ratio, 10),
        },
        "labeling_rule": LABELING_RULE,
    }

    config.ensure_directories()
    save_prepared_series(config.prepared_dir / SERIES_FILENAME, split_records)
    save_prepared_metadata(config.prepared_dir / METADATA_FILENAME, metadata)

    summary = {
        "counts": split_counts,
        "anomalies": anomaly_counts,
        "metadata_path": config.prepared_dir / METADATA_FILENAME,
        "series_path": config.prepared_dir / SERIES_FILENAME,
    }
    return summary
