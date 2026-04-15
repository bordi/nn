from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab3.src.windows import build_eval_windows, build_train_windows


def _make_records() -> list[dict[str, object]]:
    start = datetime(2020, 1, 1, 0, 0)
    return [
        {
            "timestamp": start + index * timedelta(minutes=30),
            "value": float(index + 1),
            "is_anomaly": int(index == 2 or index == 6),
            "split": "train" if index < 4 else "val" if index < 6 else "test",
        }
        for index in range(8)
    ]


def test_build_train_windows_uses_only_train_targets() -> None:
    records = _make_records()

    windows = build_train_windows(records, window_size=2, horizon=1)

    assert [window["target_timestamp"] for window in windows] == [
        datetime(2020, 1, 1, 1, 30),
    ]
    assert [window["split"] for window in windows] == ["train"]
    assert [window["target_is_anomaly"] for window in windows] == [0]
    assert windows[0]["input_values"] == [2.0, 3.0]
    assert windows[0]["target"] == 4.0


def test_build_eval_windows_allows_context_from_previous_split() -> None:
    records = _make_records()

    val_windows = build_eval_windows(records, "val", window_size=2, horizon=1)
    test_windows = build_eval_windows(records, "test", window_size=2, horizon=1)

    assert [window["target_timestamp"] for window in val_windows] == [
        datetime(2020, 1, 1, 2, 0),
        datetime(2020, 1, 1, 2, 30),
    ]
    assert val_windows[0]["input_values"] == [3.0, 4.0]
    assert val_windows[1]["input_values"] == [4.0, 5.0]
    assert [window["split"] for window in test_windows] == ["test", "test"]
    assert test_windows[0]["input_values"] == [5.0, 6.0]
    assert test_windows[1]["input_values"] == [6.0, 7.0]
    assert test_windows[0]["target_timestamp"] == datetime(2020, 1, 1, 3, 0)


def test_build_train_windows_excludes_anomalous_targets() -> None:
    records = _make_records()

    windows = build_train_windows(records, window_size=1, horizon=1)

    assert [window["target_timestamp"] for window in windows] == [
        datetime(2020, 1, 1, 0, 30),
        datetime(2020, 1, 1, 1, 30),
    ]


def test_window_target_timestamps_are_preserved() -> None:
    records = _make_records()

    windows = build_eval_windows(records, "val", window_size=2, horizon=1)

    assert windows[0]["target_timestamp"] == datetime(2020, 1, 1, 2, 0)
    assert windows[1]["target_timestamp"] == datetime(2020, 1, 1, 2, 30)
