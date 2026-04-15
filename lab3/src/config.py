from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .utils import ensure_directories as ensure_directories_for_paths


ARTIFACT_DIR_NAMES = (
    "prepared",
    "baselines",
    "models",
    "forecasts",
    "anomalies",
    "eval",
    "plots",
)


@dataclass(frozen=True, slots=True)
class Config:
    project_root: Path
    dataset_root: Path
    artifacts_root: Path
    prepared_dir: Path
    baselines_dir: Path
    models_dir: Path
    forecasts_dir: Path
    anomalies_dir: Path
    eval_dir: Path
    plots_dir: Path
    series_key: str = "realKnownCause/nyc_taxi.csv"
    window_size: int = 48
    horizon: int = 1
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    input_size: int = 1
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 20
    early_stopping_patience: int = 5
    threshold_percentile: int = 95
    seed: int = 42

    @property
    def series_csv_path(self) -> Path:
        return self.dataset_root / "data" / self.series_key

    @property
    def labels_path(self) -> Path:
        return self.dataset_root / "labels" / "combined_windows.json"

    def ensure_directories(self) -> list[Path]:
        return ensure_directories_for_paths(self.artifact_directories())

    def artifact_directories(self) -> list[Path]:
        return [
            self.prepared_dir,
            self.baselines_dir,
            self.models_dir,
            self.forecasts_dir,
            self.anomalies_dir,
            self.eval_dir,
            self.plots_dir,
        ]


def load_config(project_root: Path | None = None) -> Config:
    root = project_root or Path(__file__).resolve().parents[1]
    artifacts_root = root / "artifacts"
    return Config(
        project_root=root,
        dataset_root=root / "dataset" / "nab",
        artifacts_root=artifacts_root,
        prepared_dir=artifacts_root / "prepared",
        baselines_dir=artifacts_root / "baselines",
        models_dir=artifacts_root / "models",
        forecasts_dir=artifacts_root / "forecasts",
        anomalies_dir=artifacts_root / "anomalies",
        eval_dir=artifacts_root / "eval",
        plots_dir=artifacts_root / "plots",
    )
