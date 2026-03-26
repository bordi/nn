from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence


EVAL_ANSWER_FIELDS = [
    "query",
    "gold_reference",
    "model_answer",
    "used_chunk_ids",
    "faithfulness_score",
    "helpfulness_score",
    "notes",
]


def select_answerable_eval_queries(
    eval_queries: Sequence[dict[str, Any]],
    *,
    sample_size: int,
) -> list[dict[str, Any]]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    answerable_queries = [dict(query) for query in eval_queries if query.get("is_answerable", True)]
    return answerable_queries[:sample_size]


def build_answer_eval_row(
    query: dict[str, Any],
    answer_result: dict[str, Any],
) -> dict[str, str]:
    return {
        "query": str(query.get("question", "")),
        "gold_reference": _build_gold_reference(query),
        "model_answer": str(answer_result.get("answer", "")),
        "used_chunk_ids": ", ".join(str(chunk_id) for chunk_id in answer_result.get("used_chunk_ids", [])),
        "faithfulness_score": "",
        "helpfulness_score": "",
        "notes": "",
    }


def generate_answer_eval_rows(
    eval_queries: Sequence[dict[str, Any]],
    *,
    sample_size: int,
    top_k: int,
    answer_for_query: Callable[[dict[str, Any], int], dict[str, Any]],
) -> list[dict[str, str]]:
    selected_queries = select_answerable_eval_queries(eval_queries, sample_size=sample_size)
    rows: list[dict[str, str]] = []

    for query in selected_queries:
        answer_result = answer_for_query(query, top_k)
        rows.append(build_answer_eval_row(query, answer_result))

    return rows


def export_answer_eval_run(
    *,
    runs_dir: Path,
    rows: Sequence[dict[str, str]],
    sample_size: int,
    top_k: int,
    run_label: str = "answer-eval",
    created_at: datetime | None = None,
) -> dict[str, Path]:
    timestamp = (created_at or datetime.now()).strftime("%Y%m%d-%H%M%S")
    run_dir = runs_dir / f"{run_label}-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / "answers.csv"
    jsonl_path = run_dir / "answers.jsonl"
    metadata_path = run_dir / "metadata.json"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVAL_ANSWER_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in EVAL_ANSWER_FIELDS})

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = {
        "run_label": run_label,
        "sample_size": sample_size,
        "top_k": top_k,
        "num_rows": len(rows),
        "created_at": (created_at or datetime.now()).isoformat(timespec="seconds"),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "run_dir": run_dir,
        "csv_path": csv_path,
        "jsonl_path": jsonl_path,
        "metadata_path": metadata_path,
    }


def _build_gold_reference(query: dict[str, Any]) -> str:
    answer_texts: list[str] = []
    seen: set[str] = set()
    for answer in query.get("answers", []):
        text = str(answer.get("text", "")).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        answer_texts.append(text)
    return " | ".join(answer_texts)
