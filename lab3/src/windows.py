from __future__ import annotations

from typing import Any

from .utils import parse_timestamp


def _normalize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for record in records:
        normalized.append({**record, "timestamp": parse_timestamp(record["timestamp"])})
    normalized.sort(key=lambda record: record["timestamp"])
    return normalized


def _build_windows(
    records: list[dict[str, Any]],
    *,
    target_split: str,
    window_size: int,
    horizon: int,
    exclude_anomalous_train_targets: bool = False,
) -> list[dict[str, Any]]:
    if window_size <= 0:
        raise ValueError("window_size must be greater than 0")
    if horizon <= 0:
        raise ValueError("horizon must be greater than 0")

    normalized = _normalize_records(records)
    windows: list[dict[str, Any]] = []
    target_offset = window_size + horizon - 1

    for target_index in range(target_offset, len(normalized)):
        target_record = normalized[target_index]
        if target_record.get("split") != target_split:
            continue
        if exclude_anomalous_train_targets and int(target_record.get("is_anomaly", 0)) == 1:
            continue

        input_start = target_index - target_offset
        input_end = target_index - horizon + 1
        input_records = normalized[input_start:input_end]
        if len(input_records) != window_size:
            continue

        windows.append(
            {
                "input_values": [float(record["value"]) for record in input_records],
                "target": float(target_record["value"]),
                "target_timestamp": target_record["timestamp"],
                "split": target_split,
                "target_is_anomaly": int(target_record.get("is_anomaly", 0)),
            }
        )

    return windows


def build_train_windows(
    records: list[dict[str, Any]],
    window_size: int,
    horizon: int,
) -> list[dict[str, Any]]:
    return _build_windows(
        records,
        target_split="train",
        window_size=window_size,
        horizon=horizon,
        exclude_anomalous_train_targets=True,
    )


def build_eval_windows(
    records: list[dict[str, Any]],
    split_name: str,
    window_size: int,
    horizon: int,
) -> list[dict[str, Any]]:
    if split_name not in {"val", "test"}:
        raise ValueError("split_name must be 'val' or 'test'")

    return _build_windows(
        records,
        target_split=split_name,
        window_size=window_size,
        horizon=horizon,
    )


def windows_to_arrays(window_records: list[dict[str, Any]]) -> tuple[list[list[float]], list[float]]:
    input_values = [list(window["input_values"]) for window in window_records]
    targets = [float(window["target"]) for window in window_records]
    return input_values, targets
