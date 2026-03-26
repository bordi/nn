from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chat import ChatRuntime, answer_question, run_ask, run_chat
from src.config import load_config
from src.console import render_answer_report


class DummyEncoder:
    pass


class DummyIndex:
    pass


def _build_runtime() -> ChatRuntime:
    config = load_config()
    return ChatRuntime(
        config=config,
        chunk_records=[],
        index_metadata={"chunk_ids": ["chunk-001", "chunk-002"]},
        encoder=DummyEncoder(),
        index=DummyIndex(),
    )


def test_answer_question_returns_generated_answer_and_retrieval_metadata() -> None:
    runtime = _build_runtime()

    def fake_retrieval(**_: object) -> list[dict[str, object]]:
        return [
            {
                "chunk_id": "chunk-001",
                "document_id": "doc-001",
                "paragraph_id": "para-001",
                "title": "Alpha",
                "source_split": "dev",
                "score": 0.9,
                "context_preview": "alpha beta",
            }
        ]

    def fake_prompt_builder(question: str, chunks: list[dict[str, object]]) -> str:
        return f"prompt::{question}::{chunks[0]['chunk_id']}"

    def fake_generator(prompt: str, **_: object) -> dict[str, object]:
        return {"answer": f"generated from {prompt}", "used_chunk_ids": ["chunk-001"]}

    result = answer_question(
        runtime,
        "What is alpha?",
        top_k=3,
        retrieval_fn=fake_retrieval,
        prompt_builder=fake_prompt_builder,
        generator=fake_generator,
    )

    assert result["answer"] == "generated from prompt::What is alpha?::chunk-001"
    assert result["used_chunk_ids"] == ["chunk-001"]
    assert result["retrieved_chunks"][0]["chunk_id"] == "chunk-001"


def test_answer_question_supports_no_llm_mode() -> None:
    runtime = _build_runtime()

    def fake_retrieval(**_: object) -> list[dict[str, object]]:
        return [
            {
                "chunk_id": "chunk-001",
                "document_id": "doc-001",
                "paragraph_id": "para-001",
                "score": 0.9,
                "context_preview": "alpha beta",
            }
        ]

    result = answer_question(
        runtime,
        "What is alpha?",
        top_k=3,
        no_llm=True,
        retrieval_fn=fake_retrieval,
    )

    assert result["mode"] == "retrieval-only"
    assert result["used_chunk_ids"] == []
    assert result["retrieved_chunks"][0]["chunk_id"] == "chunk-001"


def test_run_ask_prints_answer_then_used_chunk_ids_then_retrieval_metadata() -> None:
    runtime = _build_runtime()
    output = io.StringIO()

    def fake_retrieval(**_: object) -> list[dict[str, object]]:
        return [
            {
                "chunk_id": "chunk-001",
                "document_id": "doc-001",
                "paragraph_id": "para-001",
                "title": "Alpha",
                "source_split": "dev",
                "score": 0.9,
                "context_preview": "alpha beta",
            }
        ]

    def fake_generator(prompt: str, **_: object) -> dict[str, object]:
        return {"answer": "Alpha answer", "used_chunk_ids": ["chunk-001"]}

    run_ask(
        "What is alpha?",
        top_k=3,
        output=output,
        runtime=runtime,
        retrieval_fn=fake_retrieval,
        generator=fake_generator,
        prompt_builder=lambda question, _: question,
    )

    rendered = output.getvalue()
    assert rendered.index("Answer") < rendered.index("used_chunk_ids")
    assert rendered.index("used_chunk_ids") < rendered.index("Retrieval metadata")
    assert "Alpha answer" in rendered
    assert "chunk-001" in rendered


def test_render_answer_report_uses_compact_sections_without_ansi_when_not_tty() -> None:
    output = io.StringIO()
    result = {
        "answer": "Alpha answer",
        "used_chunk_ids": ["chunk-001", "chunk-002"],
        "retrieved_chunks": [
            {
                "chunk_id": "chunk-001",
                "document_id": "doc-001",
                "paragraph_id": "para-001",
                "title": "Alpha",
                "source_split": "dev",
                "score": 0.9,
                "context_preview": "alpha beta",
            }
        ],
    }

    render_answer_report(result, output=output, is_tty=False)

    rendered = output.getvalue()
    assert "\x1b[" not in rendered
    assert rendered.index("Answer") < rendered.index("used_chunk_ids") < rendered.index("Retrieval metadata")
    assert "Alpha answer" in rendered
    assert "chunk-001, chunk-002" in rendered
    assert "score=0.9000" in rendered
    assert "document_id=doc-001" in rendered
    assert "paragraph_id=para-001" in rendered


def test_render_answer_report_emits_ansi_when_tty() -> None:
    class TtyStringIO(io.StringIO):
        def isatty(self) -> bool:  # pragma: no cover - exercised by assertion
            return True

    output = TtyStringIO()
    result = {
        "answer": "Alpha answer",
        "used_chunk_ids": [],
        "retrieved_chunks": [],
    }

    render_answer_report(result, output=output)

    rendered = output.getvalue()
    assert "\x1b[" in rendered
    assert "Answer" in rendered


def test_run_chat_stops_on_exit() -> None:
    runtime = _build_runtime()
    output = io.StringIO()
    prompts: list[str] = []

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return "exit"

    run_chat(
        top_k=3,
        output=output,
        input_func=fake_input,
        runtime=runtime,
    )

    rendered = output.getvalue()
    assert "chat: type a question" in rendered
    assert prompts == ["You> "]


def test_run_chat_stops_on_quit() -> None:
    runtime = _build_runtime()
    output = io.StringIO()
    prompts: list[str] = []

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return "quit"

    run_chat(
        top_k=3,
        output=output,
        input_func=fake_input,
        runtime=runtime,
    )

    rendered = output.getvalue()
    assert "chat: type a question" in rendered
    assert prompts == ["You> "]
