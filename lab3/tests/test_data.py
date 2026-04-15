from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab3.src.config import load_config
from lab3.src.data import (
    label_series_points,
    load_label_windows,
    load_series_csv,
    prepare_data,
    split_series_by_time,
    validate_series,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "value"])
        writer.writeheader()
        writer.writerows(rows)


def _make_series_rows() -> list[dict[str, object]]:
    return [
        {"timestamp": "2020-01-01 01:00:00", "value": "3"},
        {"timestamp": "2020-01-01 00:00:00", "value": "1"},
        {"timestamp": "2020-01-01 00:30:00", "value": "2"},
        {"timestamp": "2020-01-01 00:30:00", "value": "9"},
        {"timestamp": "2020-01-01 01:30:00", "value": "4"},
    ]


def test_load_series_csv_reads_nyc_taxi_format(tmp_path: Path) -> None:
    csv_path = tmp_path / "nyc_taxi.csv"
    _write_csv(
        csv_path,
        [
            {"timestamp": "2014-07-01 00:00:00", "value": "10844"},
            {"timestamp": "2014-07-01 00:30:00", "value": "8127"},
        ],
    )

    records = load_series_csv(csv_path)

    assert records == [
        {"timestamp": datetime(2014, 7, 1, 0, 0), "value": 10844.0},
        {"timestamp": datetime(2014, 7, 1, 0, 30), "value": 8127.0},
    ]


def test_validate_series_sorts_and_keeps_last_duplicate_then_checks_cadence() -> None:
    records = [
        {"timestamp": datetime(2020, 1, 1, 1, 0), "value": 3.0},
        {"timestamp": datetime(2020, 1, 1, 0, 0), "value": 1.0},
        {"timestamp": datetime(2020, 1, 1, 0, 30), "value": 2.0},
        {"timestamp": datetime(2020, 1, 1, 0, 30), "value": 9.0},
        {"timestamp": datetime(2020, 1, 1, 1, 30), "value": 4.0},
    ]

    validated = validate_series(records)

    assert [row["timestamp"] for row in validated] == [
        datetime(2020, 1, 1, 0, 0),
        datetime(2020, 1, 1, 0, 30),
        datetime(2020, 1, 1, 1, 0),
        datetime(2020, 1, 1, 1, 30),
    ]
    assert [row["value"] for row in validated] == [1.0, 9.0, 3.0, 4.0]


def test_validate_series_rejects_non_regular_30_minute_cadence() -> None:
    records = [
        {"timestamp": datetime(2020, 1, 1, 0, 0), "value": 1.0},
        {"timestamp": datetime(2020, 1, 1, 0, 45), "value": 2.0},
    ]

    try:
        validate_series(records)
    except ValueError as exc:
        assert "30 minutes" in str(exc)
    else:
        raise AssertionError("Expected validate_series to reject irregular cadence")


def test_label_series_points_uses_inclusive_window_boundaries() -> None:
    records = [
        {"timestamp": datetime(2020, 1, 1, 0, 0), "value": 1.0},
        {"timestamp": datetime(2020, 1, 1, 0, 30), "value": 2.0},
        {"timestamp": datetime(2020, 1, 1, 1, 0), "value": 3.0},
    ]
    windows = [
        {
            "start": datetime(2020, 1, 1, 0, 30),
            "end": datetime(2020, 1, 1, 1, 0),
        }
    ]

    labeled = label_series_points(records, windows)

    assert [row["is_anomaly"] for row in labeled] == [0, 1, 1]


def test_split_series_by_time_assigns_60_20_20_split() -> None:
    records = [
        {"timestamp": datetime(2020, 1, 1, 0, 0), "value": 1.0},
        {"timestamp": datetime(2020, 1, 1, 0, 30), "value": 2.0},
        {"timestamp": datetime(2020, 1, 1, 1, 0), "value": 3.0},
        {"timestamp": datetime(2020, 1, 1, 1, 30), "value": 4.0},
        {"timestamp": datetime(2020, 1, 1, 2, 0), "value": 5.0},
    ]

    split_records, boundaries = split_series_by_time(records, train_ratio=0.6, val_ratio=0.2)

    assert [row["split"] for row in split_records] == ["train", "train", "train", "val", "test"]
    assert boundaries == {
        "train_end": datetime(2020, 1, 1, 1, 0),
        "val_end": datetime(2020, 1, 1, 1, 30),
    }


def test_split_series_by_time_rejects_invalid_ratio_sum() -> None:
    records = [
        {"timestamp": datetime(2020, 1, 1, 0, 0), "value": 1.0},
        {"timestamp": datetime(2020, 1, 1, 0, 30), "value": 2.0},
        {"timestamp": datetime(2020, 1, 1, 1, 0), "value": 3.0},
    ]

    try:
        split_series_by_time(records, train_ratio=0.8, val_ratio=0.25)
    except ValueError as exc:
        assert "train_ratio + val_ratio" in str(exc)
    else:
        raise AssertionError("Expected split_series_by_time to reject invalid ratio sums")


def test_load_label_windows_rejects_malformed_window_structure(tmp_path: Path) -> None:
    labels_path = tmp_path / "combined_windows.json"
    labels_path.write_text(
        json.dumps({"realKnownCause/nyc_taxi.csv": [["2020-01-01 00:00:00.000000"]]}),
        encoding="utf-8",
    )

    try:
        load_label_windows(labels_path, "realKnownCause/nyc_taxi.csv")
    except ValueError as exc:
        assert "2-item pair" in str(exc)
    else:
        raise AssertionError("Expected load_label_windows to reject malformed windows")


def test_load_label_windows_rejects_inverted_window_bounds(tmp_path: Path) -> None:
    labels_path = tmp_path / "combined_windows.json"
    labels_path.write_text(
        json.dumps(
            {
                "realKnownCause/nyc_taxi.csv": [
                    ["2020-01-01 01:00:00.000000", "2020-01-01 00:00:00.000000"]
                ]
            }
        ),
        encoding="utf-8",
    )

    try:
        load_label_windows(labels_path, "realKnownCause/nyc_taxi.csv")
    except ValueError as exc:
        assert "start must be <= end" in str(exc)
    else:
        raise AssertionError("Expected load_label_windows to reject inverted windows")


def test_prepare_data_writes_metadata_and_series_artifacts(tmp_path: Path) -> None:
    config = load_config(project_root=tmp_path)
    config.ensure_directories()

    dataset_dir = config.dataset_root / "data" / "realKnownCause"
    labels_dir = config.dataset_root / "labels"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    csv_path = dataset_dir / "nyc_taxi.csv"
    rows = [
        {
            "timestamp": (datetime(2020, 1, 1, 0, 0) + index * timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S"),
            "value": str(index + 1),
        }
        for index in range(82)
    ]
    _write_csv(csv_path, rows)
    (labels_dir / "combined_windows.json").write_text(
        json.dumps({"realKnownCause/nyc_taxi.csv": [["2020-01-01 00:30:00.000000", "2020-01-01 00:30:00.000000"]]}),
        encoding="utf-8",
    )

    summary = prepare_data(config)

    series_path = config.prepared_dir / "series.csv"
    metadata_path = config.prepared_dir / "prepared_metadata.json"
    assert series_path.exists()
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["series_key"] == "realKnownCause/nyc_taxi.csv"
    assert metadata["window_size"] == 48
    assert metadata["horizon"] == 1
    assert metadata["split_ratios"] == {"train": 0.6, "val": 0.2, "test": 0.2}
    assert metadata["labeling_rule"] == "start <= timestamp <= end"
    assert metadata["num_rows"] == 82
    assert metadata["train_rows"] == 49
    assert metadata["val_rows"] == 16
    assert metadata["test_rows"] == 17
    assert "split_boundaries" in metadata

    assert summary["counts"] == {"train": 49, "val": 16, "test": 17}
    assert summary["anomalies"] == {"train": 1, "val": 0, "test": 0}


def test_prepare_data_rejects_series_too_short_for_windows(tmp_path: Path) -> None:
    config = load_config(project_root=tmp_path)
    config.ensure_directories()

    dataset_dir = config.dataset_root / "data" / "realKnownCause"
    labels_dir = config.dataset_root / "labels"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(
        dataset_dir / "nyc_taxi.csv",
        [
            {"timestamp": "2020-01-01 00:00:00", "value": "1"},
            {"timestamp": "2020-01-01 00:30:00", "value": "2"},
            {"timestamp": "2020-01-01 01:00:00", "value": "3"},
            {"timestamp": "2020-01-01 01:30:00", "value": "4"},
            {"timestamp": "2020-01-01 02:00:00", "value": "5"},
        ],
    )
    (labels_dir / "combined_windows.json").write_text(
        json.dumps({"realKnownCause/nyc_taxi.csv": [["2020-01-01 00:30:00.000000", "2020-01-01 00:30:00.000000"]]}),
        encoding="utf-8",
    )

    try:
        prepare_data(config)
    except ValueError as exc:
        assert "train split" in str(exc)
    else:
        raise AssertionError("Expected prepare_data to reject too-short series")
