from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lab3.src.config import load_config
from lab3.src.console import render_key_values, render_metrics, render_paths
from lab3.src.anomaly import run_detection_pipeline
from lab3.src.baselines import run_baseline_pipeline
from lab3.src.data import prepare_data
from lab3.src.evaluate import run_evaluation_pipeline
from lab3.src.plots import save_all_plots
from lab3.src.train import run_training_pipeline


COMMANDS = (
    "prepare-data",
    "baseline",
    "train",
    "detect",
    "evaluate",
    "plot",
    "run-all",
)

PIPELINE_COMMANDS = COMMANDS[:-1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lab3", description="Lab 3 time-series pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in COMMANDS:
        command_parser = subparsers.add_parser(command, help=f"Run the {command} step")
        if command in {"train", "run-all"}:
            _add_train_arguments(command_parser)

    return parser


def _add_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)


def _stub_handler(command: str, _: argparse.Namespace) -> int:
    config = load_config()
    config.ensure_directories()
    print(
        render_key_values(
            f"{command} scaffold",
            {
                "series_key": config.series_key,
                "artifacts_root": config.artifacts_root,
                "status": "not implemented yet",
            },
        )
    )
    return 0


def _prepare_data_handler(_: argparse.Namespace) -> int:
    config = load_config()
    summary = prepare_data(config)
    print(
        render_key_values(
            "prepare-data summary",
            {
                "train": f"{summary['counts']['train']} rows, {summary['anomalies']['train']} anomalies",
                "val": f"{summary['counts']['val']} rows, {summary['anomalies']['val']} anomalies",
                "test": f"{summary['counts']['test']} rows, {summary['anomalies']['test']} anomalies",
            },
        )
    )
    return 0


def _baseline_handler(_: argparse.Namespace) -> int:
    config = load_config()
    config.ensure_directories()
    summary = run_baseline_pipeline(
        prepared_series_path=config.prepared_dir / "series.csv",
        prepared_metadata_path=config.prepared_dir / "prepared_metadata.json",
        output_dir=config.baselines_dir,
    )
    metrics = summary["metrics"]
    print(
        render_key_values(
            "baseline summary",
            {
                "val": f"MAE={metrics['val']['mae']:.4f}, MAPE={metrics['val']['mape']:.4f}",
                "test": f"MAE={metrics['test']['mae']:.4f}, MAPE={metrics['test']['mape']:.4f}",
            },
        )
    )
    return 0


def _train_handler(args: argparse.Namespace) -> int:
    base_config = load_config()
    config = replace(
        base_config,
        max_epochs=base_config.max_epochs if args.max_epochs is None else args.max_epochs,
        batch_size=base_config.batch_size if args.batch_size is None else args.batch_size,
        hidden_size=base_config.hidden_size if args.hidden_size is None else args.hidden_size,
    )
    result = run_training_pipeline(config)
    summary = result["summary"]
    print(
        render_key_values(
            "train summary",
            {
                "best_epoch": summary["best_epoch"],
                "best_val_mae_normalized": f"{summary['best_validation_mae_normalized']:.6f}",
                "best_val_mae_original": f"{summary['best_validation_mae_original']:.6f}",
                "epochs_ran": len(summary["train_val_history"]),
            },
        )
    )
    return 0


def _detect_handler(_: argparse.Namespace) -> int:
    config = load_config()
    result = run_detection_pipeline(config)
    print(
        render_key_values(
            "detect summary",
            {
                "threshold": f"{result['threshold']:.6f}",
                "forecast_rows": len(result["prediction_rows"]),
                "anomaly_rows": len(result["anomaly_rows"]),
            },
        )
    )
    return 0


def _evaluate_handler(_: argparse.Namespace) -> int:
    config = load_config()
    result = run_evaluation_pipeline(config)
    forecast_metrics = result["forecast_metrics"]
    anomaly_metrics = result["anomaly_metrics"]
    print(
        render_metrics(
            "evaluate summary",
            {
                "val_forecast": (
                    f"MAE={forecast_metrics['val']['mae']:.4f}, "
                    f"MAPE={forecast_metrics['val']['mape']:.4f}"
                ),
                "test_forecast": (
                    f"MAE={forecast_metrics['test']['mae']:.4f}, "
                    f"MAPE={forecast_metrics['test']['mape']:.4f}"
                ),
                "test_anomaly": (
                    f"P={anomaly_metrics['precision']:.4f}, "
                    f"R={anomaly_metrics['recall']:.4f}, "
                    f"F1={anomaly_metrics['f1']:.4f}"
                ),
            },
        )
    )
    return 0


def _plot_handler(_: argparse.Namespace) -> int:
    config = load_config()
    plot_paths = save_all_plots(config)
    print(render_paths("plot summary", plot_paths))
    return 0


def _run_all_handler(args: argparse.Namespace) -> int:
    train_args = argparse.Namespace(
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
    )
    return _run_pipeline(train_args)


def _run_pipeline(train_args: argparse.Namespace) -> int:
    for command in PIPELINE_COMMANDS:
        if command == "train":
            command_args = train_args
        else:
            command_args = argparse.Namespace()

        result = HANDLERS[command](command_args)
        if result != 0:
            return result
    return 0


HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "prepare-data": _prepare_data_handler,
    "baseline": _baseline_handler,
    "train": _train_handler,
    "detect": _detect_handler,
    "evaluate": _evaluate_handler,
    "plot": _plot_handler,
    "run-all": _run_all_handler,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return HANDLERS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
