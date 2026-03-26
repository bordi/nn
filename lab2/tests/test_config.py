from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main as main_module
from main import build_parser, main
from src import config as config_module
from src.config import (
    ARTIFACTS_DIR,
    CORPUS_DIR,
    DEMOS_DIR,
    DEV_DATASET_PATH,
    EMBEDDINGS_DIR,
    EVAL_DIR,
    INDEX_DIR,
    RUNS_DIR,
    SQUAD_DIR,
    TRAIN_DATASET_PATH,
    ensure_directories,
    load_config,
)


def test_paths_and_defaults_match_lab2_layout(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        config_module,
        "ENV_LOCAL_PATH",
        Path(__file__).resolve().parents[1] / ".env.test.missing",
    )

    config = load_config()

    assert SQUAD_DIR.name == "squad"
    assert TRAIN_DATASET_PATH == SQUAD_DIR / "train-v2.0.json"
    assert DEV_DATASET_PATH == SQUAD_DIR / "dev-v2.0.json"
    assert config.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.default_top_k == 5
    assert config.openai_api_key is None


def test_load_config_reads_openai_api_key_from_env_local(tmp_path, monkeypatch) -> None:
    env_local_path = tmp_path / ".env.local"
    env_local_path.write_text("OPENAI_API_KEY=test-key-from-env-local\n", encoding="utf-8")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(config_module, "ENV_LOCAL_PATH", env_local_path, raising=False)

    config = load_config()

    assert config.openai_api_key == "test-key-from-env-local"


def test_load_config_reloads_env_local_without_sticky_process_state(tmp_path, monkeypatch) -> None:
    first_env_local_path = tmp_path / "first.env.local"
    second_env_local_path = tmp_path / "second.env.local"
    first_env_local_path.write_text("OPENAI_API_KEY=first-key\n", encoding="utf-8")
    second_env_local_path.write_text("OPENAI_API_KEY=second-key\n", encoding="utf-8")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(config_module, "ENV_LOCAL_PATH", first_env_local_path, raising=False)
    first_config = load_config()

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(config_module, "ENV_LOCAL_PATH", second_env_local_path, raising=False)
    second_config = load_config()

    assert first_config.openai_api_key == "first-key"
    assert second_config.openai_api_key == "second-key"


def test_load_config_prefers_env_local_over_inherited_shell_value(tmp_path, monkeypatch) -> None:
    env_local_path = tmp_path / ".env.local"
    env_local_path.write_text("OPENAI_API_KEY=local-key\n", encoding="utf-8")

    monkeypatch.setenv("OPENAI_API_KEY", "shell-key")
    monkeypatch.setattr(config_module, "ENV_LOCAL_PATH", env_local_path, raising=False)

    config = load_config()

    assert config.openai_api_key == "local-key"


def test_env_example_exists_for_local_setup() -> None:
    assert Path(__file__).resolve().parents[1].joinpath(".env.example").exists()


def test_missing_api_key_does_not_break_non_generation_command(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        config_module,
        "ENV_LOCAL_PATH",
        Path(__file__).resolve().parents[1] / ".env.test.missing",
    )
    captured_call: dict[str, object] = {}

    def fake_run_ask(query: str, *, top_k: int, no_llm: bool) -> None:
        captured_call["query"] = query
        captured_call["top_k"] = top_k
        captured_call["no_llm"] = no_llm

    monkeypatch.setattr(main_module, "run_ask", fake_run_ask)
    monkeypatch.setattr(sys, "argv", ["main.py", "ask", "--query", "What is SQuAD?", "--no-llm"])

    main()

    assert captured_call == {
        "query": "What is SQuAD?",
        "top_k": 5,
        "no_llm": True,
    }


def test_ensure_directories_makes_required_artifact_directories() -> None:
    ensure_directories()

    for path in (
        ARTIFACTS_DIR,
        CORPUS_DIR,
        EMBEDDINGS_DIR,
        INDEX_DIR,
        EVAL_DIR,
        RUNS_DIR,
        DEMOS_DIR,
    ):
        assert path.exists()
        assert path.is_dir()


def test_cli_parser_exposes_planned_commands() -> None:
    parser = build_parser()
    subparsers_action = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )

    assert set(subparsers_action.choices) == {
        "prepare-corpus",
        "build-index",
        "ask",
        "chat",
        "eval-retrieval",
        "eval-answers",
    }


def test_ask_command_requires_query_argument() -> None:
    parser = build_parser()

    try:
        parser.parse_args(["ask"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("ask should require --query")


def test_retrieval_commands_require_positive_top_k() -> None:
    parser = build_parser()

    try:
        parser.parse_args(["ask", "--query", "What is SQuAD?", "--top-k", "0"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("ask should reject non-positive --top-k values")

    try:
        parser.parse_args(["eval-retrieval", "--top-k", "0"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("eval-retrieval should reject non-positive --top-k values")

    try:
        parser.parse_args(["eval-answers", "--sample-size", "0"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("eval-answers should reject non-positive --sample-size values")
