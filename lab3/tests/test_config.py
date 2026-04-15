import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab3.main import COMMANDS, build_parser
from lab3.src.config import load_config


def test_load_config_paths_and_defaults(tmp_path: Path) -> None:
    config = load_config(project_root=tmp_path)

    assert config.series_key == "realKnownCause/nyc_taxi.csv"
    assert config.horizon == 1
    assert config.threshold_percentile == 95
    assert config.dataset_root == tmp_path / "dataset" / "nab"
    assert config.artifacts_root == tmp_path / "artifacts"
    assert config.prepared_dir == tmp_path / "artifacts" / "prepared"
    assert config.baselines_dir == tmp_path / "artifacts" / "baselines"
    assert config.models_dir == tmp_path / "artifacts" / "models"
    assert config.forecasts_dir == tmp_path / "artifacts" / "forecasts"
    assert config.anomalies_dir == tmp_path / "artifacts" / "anomalies"
    assert config.eval_dir == tmp_path / "artifacts" / "eval"
    assert config.plots_dir == tmp_path / "artifacts" / "plots"


def test_ensure_directories_creates_artifact_tree(tmp_path: Path) -> None:
    config = load_config(project_root=tmp_path)

    created = config.ensure_directories()

    expected_dirs = [
        config.prepared_dir,
        config.baselines_dir,
        config.models_dir,
        config.forecasts_dir,
        config.anomalies_dir,
        config.eval_dir,
        config.plots_dir,
    ]

    assert created == expected_dirs
    for path in expected_dirs:
        assert path.exists()
        assert path.is_dir()


def test_cli_parser_exposes_expected_commands() -> None:
    parser = build_parser()
    parsed_commands = [parser.parse_args([command]).command for command in COMMANDS]

    assert parsed_commands == list(COMMANDS)
