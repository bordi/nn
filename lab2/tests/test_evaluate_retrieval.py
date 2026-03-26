from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate_retrieval import (
    aggregate_retrieval_metrics,
    compute_mrr,
    compute_recall_at_k,
    evaluate_query_retrieval,
    evaluate_retrieval,
    filter_answerable_eval_queries,
    run_retrieval_evaluation,
)


def test_compute_recall_at_k_returns_one_when_relevant_chunk_is_within_cutoff() -> None:
    retrieved_chunks = [
        {"paragraph_id": "para-999"},
        {"paragraph_id": "para-001"},
        {"paragraph_id": "para-123"},
    ]

    assert compute_recall_at_k(retrieved_chunks, gold_paragraph_id="para-001", k=2) == 1.0
    assert compute_recall_at_k(retrieved_chunks, gold_paragraph_id="para-001", k=1) == 0.0


def test_compute_mrr_uses_rank_of_first_relevant_chunk() -> None:
    retrieved_chunks = [
        {"paragraph_id": "para-999"},
        {"paragraph_id": "para-001"},
        {"paragraph_id": "para-001"},
    ]

    assert compute_mrr(retrieved_chunks, gold_paragraph_id="para-001") == 0.5


def test_evaluate_retrieval_aggregates_across_multiple_answerable_queries() -> None:
    eval_queries = [
        {"query_id": "q-001", "question": "Q1", "gold_paragraph_id": "para-001", "is_answerable": True},
        {"query_id": "q-002", "question": "Q2", "gold_paragraph_id": "para-002", "is_answerable": True},
        {"query_id": "q-003", "question": "Q3", "gold_paragraph_id": "para-003", "is_answerable": False},
    ]

    retrieved_by_query = {
        "Q1": [
            {"paragraph_id": "para-001"},
            {"paragraph_id": "para-999"},
        ],
        "Q2": [
            {"paragraph_id": "para-888"},
            {"paragraph_id": "para-002"},
        ],
        "Q3": [
            {"paragraph_id": "para-003"},
        ],
    }

    def retrieve(query: str, top_k: int) -> list[dict[str, str]]:
        assert top_k == 2
        return retrieved_by_query[query]

    metrics = evaluate_retrieval(eval_queries, retrieve, top_k=2)

    assert metrics["n_queries"] == 2
    assert metrics["recall@1"] == 0.5
    assert metrics["recall@2"] == 1.0
    assert metrics["mrr"] == 0.75


def test_filter_answerable_eval_queries_excludes_impossible_questions() -> None:
    eval_queries = [
        {"query_id": "q-001", "is_answerable": True},
        {"query_id": "q-002", "is_answerable": False},
        {"query_id": "q-003", "is_answerable": True},
    ]

    filtered_queries = filter_answerable_eval_queries(eval_queries)

    assert [query["query_id"] for query in filtered_queries] == ["q-001", "q-003"]


def test_evaluate_query_retrieval_reports_hit_and_first_rank() -> None:
    retrieved_chunks = [
        {"chunk_id": "chunk-001", "paragraph_id": "para-999", "score": 0.9},
        {"chunk_id": "chunk-002", "paragraph_id": "para-002", "score": 0.8},
        {"chunk_id": "chunk-003", "paragraph_id": "para-777", "score": 0.7},
    ]

    result = evaluate_query_retrieval(
        query_id="q-002",
        question="Q2",
        gold_paragraph_id="para-002",
        retrieved_chunks=retrieved_chunks,
        top_k=3,
    )

    assert result["hit"] is True
    assert result["first_relevant_rank"] == 2
    assert result["reciprocal_rank"] == 0.5
    assert result["retrieved_chunk_ids"] == ["chunk-001", "chunk-002", "chunk-003"]


def test_run_retrieval_evaluation_scores_only_answerable_queries() -> None:
    eval_queries = [
        {"query_id": "q-001", "question": "Q1", "gold_paragraph_id": "para-001", "is_answerable": True},
        {"query_id": "q-002", "question": "Q2", "gold_paragraph_id": "para-002", "is_answerable": False},
        {"query_id": "q-003", "question": "Q3", "gold_paragraph_id": "para-003", "is_answerable": True},
    ]
    retrieved_by_query = {
        "Q1": [{"chunk_id": "chunk-001", "paragraph_id": "para-001", "score": 0.9}],
        "Q3": [
            {"chunk_id": "chunk-009", "paragraph_id": "para-999", "score": 0.7},
            {"chunk_id": "chunk-010", "paragraph_id": "para-003", "score": 0.6},
        ],
    }

    def retrieve(query: str, top_k: int) -> list[dict[str, str | float]]:
        assert top_k == 5
        return retrieved_by_query[query]

    metrics, examples = run_retrieval_evaluation(
        eval_queries,
        top_k=5,
        retrieve_for_query=retrieve,
    )

    assert metrics["num_queries_total"] == 3
    assert metrics["num_queries_scored"] == 2
    assert metrics["num_queries_skipped"] == 1
    assert metrics["recall_at_k"] == 1.0
    assert metrics["mrr_at_k"] == 0.75
    assert [example["query_id"] for example in examples] == ["q-001", "q-003"]


def test_aggregate_retrieval_metrics_uses_hit_and_rr_fields() -> None:
    query_results = [
        {"query_id": "q-001", "hit": True, "reciprocal_rank": 1.0},
        {"query_id": "q-002", "hit": True, "reciprocal_rank": 0.5},
        {"query_id": "q-003", "hit": False, "reciprocal_rank": 0.0},
    ]

    metrics = aggregate_retrieval_metrics(query_results, top_k=5)

    assert metrics["recall_at_k"] == 2 / 3
    assert metrics["mrr_at_k"] == 0.5
