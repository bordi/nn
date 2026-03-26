from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate_answers import (
    build_answer_eval_row,
    export_answer_eval_run,
    select_answerable_eval_queries,
)


def test_select_answerable_eval_queries_respects_order_and_sample_size() -> None:
    eval_queries = [
        {"query_id": "q-001", "question": "Q1", "is_answerable": True},
        {"query_id": "q-002", "question": "Q2", "is_answerable": False},
        {"query_id": "q-003", "question": "Q3", "is_answerable": True},
        {"query_id": "q-004", "question": "Q4", "is_answerable": True},
    ]

    selected = select_answerable_eval_queries(eval_queries, sample_size=2)

    assert [query["query_id"] for query in selected] == ["q-001", "q-003"]


def test_build_answer_eval_row_includes_required_manual_scoring_columns() -> None:
    query = {
        "query_id": "q-001",
        "question": "What is alpha?",
        "answers": [{"text": "Alpha is the first letter."}],
    }
    answer_result = {
        "answer": "Alpha is the first letter of the Greek alphabet.",
        "used_chunk_ids": ["chunk-001", "chunk-002"],
    }

    row = build_answer_eval_row(query, answer_result)

    assert row["query"] == "What is alpha?"
    assert row["gold_reference"] == "Alpha is the first letter."
    assert row["model_answer"] == "Alpha is the first letter of the Greek alphabet."
    assert row["used_chunk_ids"] == "chunk-001, chunk-002"
    assert row["faithfulness_score"] == ""
    assert row["helpfulness_score"] == ""
    assert row["notes"] == ""


def test_export_answer_eval_run_writes_run_folder_with_csv_jsonl_and_metadata(tmp_path) -> None:
    rows = [
        {
            "query": "What is alpha?",
            "gold_reference": "Alpha is the first letter.",
            "model_answer": "Alpha is the first letter of the Greek alphabet.",
            "used_chunk_ids": "chunk-001",
            "faithfulness_score": "",
            "helpfulness_score": "",
            "notes": "",
        }
    ]

    run_paths = export_answer_eval_run(
        runs_dir=tmp_path,
        rows=rows,
        sample_size=1,
        top_k=5,
        run_label="test-run",
    )

    csv_path = run_paths["csv_path"]
    jsonl_path = run_paths["jsonl_path"]
    metadata_path = run_paths["metadata_path"]

    assert csv_path.exists()
    assert jsonl_path.exists()
    assert metadata_path.exists()

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["query"] == "What is alpha?"

    with jsonl_path.open("r", encoding="utf-8") as handle:
        jsonl_rows = [json.loads(line) for line in handle if line.strip()]
    assert jsonl_rows[0]["used_chunk_ids"] == "chunk-001"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["sample_size"] == 1
    assert metadata["top_k"] == 5
    assert metadata["num_rows"] == 1
