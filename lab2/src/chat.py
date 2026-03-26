from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TextIO
import sys

from src.config import Config, load_config
from src.embeddings import SentenceTransformerEncoder
from src.generation import generate_structured_answer
from src.index import load_faiss_index, load_index_metadata
from src.console import render_answer_report
from src.prompt_builder import build_grounded_prompt
from src.retrieval import load_chunk_records, retrieve_top_k


@dataclass(frozen=True)
class ChatRuntime:
    config: Config
    chunk_records: list[dict[str, Any]]
    index_metadata: dict[str, Any]
    encoder: Any
    index: Any


def load_chat_runtime(config: Config | None = None, *, command_name: str = "chat") -> ChatRuntime:
    resolved_config = config or load_config()
    _ensure_artifacts_exist(
        resolved_config.chunks_path,
        resolved_config.faiss_index_path,
        resolved_config.index_metadata_path,
        command_name=command_name,
    )

    chunk_records = load_chunk_records(resolved_config.chunks_path)
    index_metadata = load_index_metadata(resolved_config.index_metadata_path)
    if index_metadata.get("model_name") != resolved_config.embedding_model_name:
        raise SystemExit(
            f"{command_name}: index model mismatch. "
            f"Index was built with {index_metadata.get('model_name')}, "
            f"but config expects {resolved_config.embedding_model_name}."
        )

    encoder = SentenceTransformerEncoder(resolved_config.embedding_model_name)
    index = load_faiss_index(resolved_config.faiss_index_path)
    return ChatRuntime(
        config=resolved_config,
        chunk_records=chunk_records,
        index_metadata=index_metadata,
        encoder=encoder,
        index=index,
    )


def answer_question(
    runtime: ChatRuntime,
    query: str,
    *,
    top_k: int,
    no_llm: bool = False,
    generator: Callable[..., dict[str, Any]] = generate_structured_answer,
    retrieval_fn: Callable[..., list[dict[str, Any]]] = retrieve_top_k,
    prompt_builder: Callable[[str, list[dict[str, Any]]], str] = build_grounded_prompt,
) -> dict[str, Any]:
    retrieved_chunks = retrieval_fn(
        query=query,
        encoder=runtime.encoder,
        index=runtime.index,
        chunk_records=runtime.chunk_records,
        top_k=top_k,
        index_chunk_ids=runtime.index_metadata.get("chunk_ids"),
    )

    if no_llm:
        return {
            "query": query,
            "answer": "[generation skipped with --no-llm]",
            "used_chunk_ids": [],
            "retrieved_chunks": retrieved_chunks,
            "mode": "retrieval-only",
        }

    prompt = prompt_builder(query, retrieved_chunks)
    generated = generator(prompt, api_key=runtime.config.openai_api_key)
    used_chunk_ids = _normalize_used_chunk_ids(generated.get("used_chunk_ids", []))

    return {
        "query": query,
        "answer": generated["answer"],
        "used_chunk_ids": used_chunk_ids,
        "retrieved_chunks": retrieved_chunks,
        "mode": "llm",
        "prompt": prompt,
    }


def run_ask(
    query: str,
    *,
    top_k: int,
    output: TextIO | None = None,
    no_llm: bool = False,
    runtime: ChatRuntime | None = None,
    question_runner: Callable[..., dict[str, Any]] = answer_question,
    generator: Callable[..., dict[str, Any]] = generate_structured_answer,
    retrieval_fn: Callable[..., list[dict[str, Any]]] = retrieve_top_k,
    prompt_builder: Callable[[str, list[dict[str, Any]]], str] = build_grounded_prompt,
) -> None:
    resolved_output = output or sys.stdout
    resolved_runtime = runtime or load_chat_runtime(command_name="ask")

    try:
        result = question_runner(
            resolved_runtime,
            query,
            top_k=top_k,
            no_llm=no_llm,
            generator=generator,
            retrieval_fn=retrieval_fn,
            prompt_builder=prompt_builder,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"ask: {exc}") from exc

    render_answer_report(result, output=resolved_output)


def run_chat(
    *,
    top_k: int,
    output: TextIO | None = None,
    input_func: Callable[[str], str] = input,
    runtime: ChatRuntime | None = None,
    question_runner: Callable[..., dict[str, Any]] = answer_question,
    generator: Callable[..., dict[str, Any]] = generate_structured_answer,
    retrieval_fn: Callable[..., list[dict[str, Any]]] = retrieve_top_k,
    prompt_builder: Callable[[str, list[dict[str, Any]]], str] = build_grounded_prompt,
) -> None:
    resolved_output = output or sys.stdout
    resolved_runtime = runtime or load_chat_runtime(command_name="chat")

    print("chat: type a question, or 'exit'/'quit' to stop.", file=resolved_output)
    while True:
        try:
            raw_query = input_func("You> ")
        except EOFError:
            print(file=resolved_output)
            break

        query = raw_query.strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        try:
            result = question_runner(
                resolved_runtime,
                query,
                top_k=top_k,
                generator=generator,
                retrieval_fn=retrieval_fn,
                prompt_builder=prompt_builder,
            )
        except (RuntimeError, ValueError) as exc:
            print(f"chat: {exc}", file=resolved_output)
            continue

        render_answer_report(result, output=resolved_output)


def _ensure_artifacts_exist(*paths: Path, command_name: str) -> None:
    missing_paths = [path for path in paths if not path.exists()]
    if missing_paths:
        formatted_paths = ", ".join(str(path) for path in missing_paths)
        raise SystemExit(
            f"{command_name}: missing required artifacts. "
            f"Run `python main.py prepare-corpus` and `python main.py build-index` first. "
            f"Missing: {formatted_paths}"
        )


def _normalize_used_chunk_ids(used_chunk_ids: list[Any]) -> list[str]:
    normalized_ids: list[str] = []
    seen: set[str] = set()
    for chunk_id in used_chunk_ids:
        resolved_chunk_id = str(chunk_id)
        if not resolved_chunk_id or resolved_chunk_id in seen:
            continue
        seen.add(resolved_chunk_id)
        normalized_ids.append(resolved_chunk_id)
    return normalized_ids
