from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prompt_builder import build_grounded_prompt
from src.generation import generate_structured_answer


def test_build_grounded_prompt_includes_question_context_and_chunk_ids() -> None:
    retrieved_chunks = [
        {
            "chunk_id": "chunk-001",
            "document_id": "doc-001",
            "paragraph_id": "para-001",
            "title": "Alpha Article",
            "context": "alpha beta",
            "score": 0.91,
        },
        {
            "chunk_id": "chunk-002",
            "document_id": "doc-002",
            "paragraph_id": "para-002",
            "title": "Beta Article",
            "context": "beta gamma",
            "score": 0.42,
        },
    ]

    prompt = build_grounded_prompt("What is alpha?", retrieved_chunks)

    assert "What is alpha?" in prompt
    assert "chunk-001" in prompt
    assert "chunk-002" in prompt
    assert "alpha beta" in prompt
    assert "beta gamma" in prompt


def test_build_grounded_prompt_requires_grounded_answer_and_used_chunk_ids() -> None:
    prompt = build_grounded_prompt(
        "Where is the answer?",
        [
            {
                "chunk_id": "chunk-001",
                "document_id": "doc-001",
                "paragraph_id": "para-001",
                "title": "Grounded Source",
                "context": "The answer is here.",
                "score": 1.0,
            }
        ],
    )

    assert "only the provided context" in prompt.lower()
    assert "do not invent" in prompt.lower() or "do not make up" in prompt.lower()
    assert "information was not found" in prompt.lower() or "not found" in prompt.lower()
    assert "used_chunk_ids" in prompt
    assert "chunk_id" in prompt


def test_generate_structured_answer_raises_clear_error_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        generate_structured_answer("prompt", client=object())


def test_generate_structured_answer_parses_structured_json_response(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeResponse:
        def __init__(self, content: str) -> None:
            self.choices = [type("Choice", (), {"message": type("Message", (), {"content": content})()})()]

    class FakeCompletions:
        def __init__(self) -> None:
            self.called_kwargs: dict[str, object] = {}

        def create(self, **kwargs: object) -> FakeResponse:
            self.called_kwargs = dict(kwargs)
            return FakeResponse('{"answer": "Alpha", "used_chunk_ids": ["chunk-001", "chunk-002"]}')

    class FakeClient:
        def __init__(self) -> None:
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    client = FakeClient()

    result = generate_structured_answer("prompt", client=client)

    assert result == {"answer": "Alpha", "used_chunk_ids": ["chunk-001", "chunk-002"]}
    assert client.chat.completions.called_kwargs["messages"][0]["content"] == "prompt"
    assert client.chat.completions.called_kwargs["model"]
