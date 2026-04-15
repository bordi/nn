from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab3.main import main
from lab3.src.config import load_config
from lab3.src.plots import build_plot_paths, save_all_plots, _true_anomaly_spans


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_build_plot_paths_returns_three_stable_plot_files(tmp_path: Path) -> None:
    config = load_config(project_root=tmp_path)

    plot_paths = build_plot_paths(config)

    assert list(plot_paths) == [
        "series_anomalies",
        "forecast_zoom",
        "residuals_threshold",
    ]
    assert [path.name for path in plot_paths.values()] == [
        "test_series_anomalies.png",
        "test_forecast_zoom.png",
        "test_residuals_threshold.png",
    ]
    assert [path.parent for path in plot_paths.values()] == [config.plots_dir] * 3


def test_save_all_plots_creates_three_pngs(tmp_path: Path) -> None:
    config = load_config(project_root=tmp_path)
    config.ensure_directories()

    prepared_rows = [
        {"timestamp": datetime(2020, 1, 1, 0, 0), "value": 10.0, "is_anomaly": 0, "split": "train"},
        {"timestamp": datetime(2020, 1, 1, 0, 30), "value": 11.0, "is_anomaly": 0, "split": "train"},
        {"timestamp": datetime(2020, 1, 1, 1, 0), "value": 12.0, "is_anomaly": 1, "split": "val"},
        {"timestamp": datetime(2020, 1, 1, 1, 30), "value": 13.0, "is_anomaly": 0, "split": "test"},
        {"timestamp": datetime(2020, 1, 1, 2, 0), "value": 14.0, "is_anomaly": 1, "split": "test"},
        {"timestamp": datetime(2020, 1, 1, 2, 30), "value": 15.0, "is_anomaly": 0, "split": "test"},
    ]
    forecast_rows = [
        {
            "timestamp": datetime(2020, 1, 1, 1, 30),
            "split": "test",
            "target": 13.0,
            "prediction": 12.5,
            "residual": 0.5,
            "residual_abs": 0.5,
            "is_anomaly": 0,
        },
        {
            "timestamp": datetime(2020, 1, 1, 2, 0),
            "split": "test",
            "target": 14.0,
            "prediction": 12.0,
            "residual": 2.0,
            "residual_abs": 2.0,
            "is_anomaly": 1,
        },
    ]
    anomaly_rows = [
        {
            "timestamp": datetime(2020, 1, 1, 1, 30),
            "split": "test",
            "target": 13.0,
            "prediction": 12.5,
            "residual_abs": 0.5,
            "is_anomaly": 0,
            "predicted_anomaly": 0,
        },
        {
            "timestamp": datetime(2020, 1, 1, 2, 0),
            "split": "test",
            "target": 14.0,
            "prediction": 12.0,
            "residual_abs": 2.0,
            "is_anomaly": 1,
            "predicted_anomaly": 1,
        },
    ]

    _write_csv(
        config.prepared_dir / "series.csv",
        prepared_rows,
        ["timestamp", "value", "is_anomaly", "split"],
    )
    _write_csv(
        config.forecasts_dir / "gru_predictions.csv",
        forecast_rows,
        ["timestamp", "split", "target", "prediction", "residual", "residual_abs", "is_anomaly"],
    )
    _write_csv(
        config.anomalies_dir / "anomaly_predictions.csv",
        anomaly_rows,
        ["timestamp", "split", "target", "prediction", "residual_abs", "is_anomaly", "predicted_anomaly"],
    )
    (config.anomalies_dir / "threshold_summary.json").write_text(
        json.dumps(
            {
                "threshold_strategy": "validation_normal_residual_abs_percentile",
                "threshold_percentile": 95,
                "threshold_value": 1.5,
                "validation_points_used": 1,
            }
        ),
        encoding="utf-8",
    )

    plot_paths = save_all_plots(config)

    assert list(plot_paths) == [
        "series_anomalies",
        "forecast_zoom",
        "residuals_threshold",
    ]
    assert [path.name for path in plot_paths.values()] == [
        "test_series_anomalies.png",
        "test_forecast_zoom.png",
        "test_residuals_threshold.png",
    ]
    for path in plot_paths.values():
        assert path.exists()
        assert path.suffix == ".png"


def test_true_anomaly_spans_use_a_non_degenerate_fallback_for_single_point() -> None:
    rows = [
        {
            "timestamp": datetime(2020, 1, 1, 0, 0),
            "value": 10.0,
            "is_anomaly": 1,
            "split": "test",
        }
    ]

    spans = _true_anomaly_spans(rows)

    assert len(spans) == 1
    start, end = spans[0]
    assert end > start


def test_run_all_executes_pipeline_steps_in_order_and_forwards_train_overrides(monkeypatch) -> None:
    calls: list[tuple[str, int | None]] = []

    def record_step(name: str):
        def _handler(args):
            calls.append((name, getattr(args, "max_epochs", None)))
            return 0

        return _handler

    from lab3 import main as lab3_main

    monkeypatch.setitem(lab3_main.HANDLERS, "prepare-data", record_step("prepare-data"))
    monkeypatch.setitem(lab3_main.HANDLERS, "baseline", record_step("baseline"))
    monkeypatch.setitem(lab3_main.HANDLERS, "train", record_step("train"))
    monkeypatch.setitem(lab3_main.HANDLERS, "detect", record_step("detect"))
    monkeypatch.setitem(lab3_main.HANDLERS, "evaluate", record_step("evaluate"))
    monkeypatch.setitem(lab3_main.HANDLERS, "plot", record_step("plot"))

    exit_code = main(["run-all", "--max-epochs", "1"])

    assert exit_code == 0
    assert calls == [
        ("prepare-data", None),
        ("baseline", None),
        ("train", 1),
        ("detect", None),
        ("evaluate", None),
        ("plot", None),
    ]


def test_run_all_stops_after_first_non_zero_handler_result(monkeypatch) -> None:
    calls: list[str] = []

    def record_step(name: str, exit_code: int = 0):
        def _handler(_: object) -> int:
            calls.append(name)
            return exit_code

        return _handler

    from lab3 import main as lab3_main

    monkeypatch.setitem(lab3_main.HANDLERS, "prepare-data", record_step("prepare-data"))
    monkeypatch.setitem(lab3_main.HANDLERS, "baseline", record_step("baseline", exit_code=3))
    monkeypatch.setitem(lab3_main.HANDLERS, "train", record_step("train"))
    monkeypatch.setitem(lab3_main.HANDLERS, "detect", record_step("detect"))
    monkeypatch.setitem(lab3_main.HANDLERS, "evaluate", record_step("evaluate"))
    monkeypatch.setitem(lab3_main.HANDLERS, "plot", record_step("plot"))

    exit_code = main(["run-all"])

    assert exit_code == 3
    assert calls == ["prepare-data", "baseline"]
