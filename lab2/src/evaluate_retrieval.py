from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


def compute_recall_at_k(
    retrieved_chunks: Sequence[dict[str, Any]],
    gold_paragraph_id: str,
    k: int,
) -> float:
    """Return 1.0 when a relevant paragraph appears in the top-k results."""

    if k <= 0:
        raise ValueError("k must be positive")

    top_k_results = retrieved_chunks[:k]
    return 1.0 if any(_paragraph_id(chunk) == gold_paragraph_id for chunk in top_k_results) else 0.0


def compute_mrr(
    retrieved_chunks: Sequence[dict[str, Any]],
    gold_paragraph_id: str,
) -> float:
    """Return reciprocal rank of the first relevant retrieved chunk."""

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if _paragraph_id(chunk) == gold_paragraph_id:
            return 1.0 / rank
    return 0.0


def filter_answerable_eval_queries(eval_queries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(query) for query in eval_queries if query.get("is_answerable", True)]


def evaluate_query_retrieval(
    *,
    query_id: str,
    question: str,
    gold_paragraph_id: str,
    retrieved_chunks: Sequence[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    top_results = [dict(chunk) for chunk in retrieved_chunks[:top_k]]
    recall_at_k = compute_recall_at_k(top_results, gold_paragraph_id, top_k)
    reciprocal_rank = compute_mrr(top_results, gold_paragraph_id)
    first_relevant_rank = None if reciprocal_rank == 0.0 else int(round(1.0 / reciprocal_rank))

    return {
        "query_id": query_id,
        "question": question,
        "gold_paragraph_id": gold_paragraph_id,
        "top_k": top_k,
        "hit": bool(recall_at_k),
        "first_relevant_rank": first_relevant_rank,
        "reciprocal_rank": reciprocal_rank,
        "retrieved_chunk_ids": [str(chunk.get("chunk_id", "")) for chunk in top_results],
        "retrieved_paragraph_ids": [str(chunk.get("paragraph_id", "")) for chunk in top_results],
        "top_score": float(top_results[0].get("score", 0.0)) if top_results else 0.0,
    }


def aggregate_retrieval_metrics(
    query_results: Sequence[dict[str, Any]],
    *,
    top_k: int,
    num_queries_total: int | None = None,
    num_queries_skipped: int | None = None,
) -> dict[str, float | int]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    scored_queries = list(query_results)
    num_scored = len(scored_queries)
    total_queries = num_scored if num_queries_total is None else num_queries_total
    skipped_queries = max(total_queries - num_scored, 0) if num_queries_skipped is None else num_queries_skipped

    if not scored_queries:
        return {
            "top_k": top_k,
            "num_queries_total": total_queries,
            "num_queries_scored": 0,
            "num_queries_skipped": skipped_queries,
            "recall_at_k": 0.0,
            "mrr_at_k": 0.0,
        }

    recall_total = sum(1.0 if result.get("hit") else 0.0 for result in scored_queries)
    reciprocal_rank_total = sum(float(result.get("reciprocal_rank", 0.0)) for result in scored_queries)

    return {
        "top_k": top_k,
        "num_queries_total": total_queries,
        "num_queries_scored": num_scored,
        "num_queries_skipped": skipped_queries,
        "recall_at_k": recall_total / num_scored,
        "mrr_at_k": reciprocal_rank_total / num_scored,
    }


def run_retrieval_evaluation(
    eval_queries: Sequence[dict[str, Any]],
    *,
    top_k: int,
    retrieve_for_query: Callable[[str, int], Sequence[dict[str, Any]]],
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    answerable_queries = filter_answerable_eval_queries(eval_queries)
    query_results: list[dict[str, Any]] = []

    for query in answerable_queries:
        retrieved_chunks = retrieve_for_query(str(query.get("question", "")), top_k)
        query_results.append(
            evaluate_query_retrieval(
                query_id=str(query.get("query_id", "")),
                question=str(query.get("question", "")),
                gold_paragraph_id=str(query.get("gold_paragraph_id", "")),
                retrieved_chunks=retrieved_chunks,
                top_k=top_k,
            )
        )

    metrics = aggregate_retrieval_metrics(
        query_results,
        top_k=top_k,
        num_queries_total=len(eval_queries),
        num_queries_skipped=len(eval_queries) - len(answerable_queries),
    )
    return metrics, query_results


def evaluate_retrieval(
    eval_queries: Sequence[dict[str, Any]],
    retrieve_top_k: Callable[[str, int], Sequence[dict[str, Any]]],
    *,
    top_k: int,
) -> dict[str, float | int]:
    """Backward-compatible aggregate metrics used by current tests."""

    answerable_queries = filter_answerable_eval_queries(eval_queries)
    if not answerable_queries:
        metrics: dict[str, float | int] = {"n_queries": 0, "mrr": 0.0}
        for k in range(1, top_k + 1):
            metrics[f"recall@{k}"] = 0.0
        return metrics

    recall_totals = {k: 0.0 for k in range(1, top_k + 1)}
    reciprocal_rank_total = 0.0

    for query in answerable_queries:
        retrieved_chunks = retrieve_top_k(str(query.get("question", "")), top_k)
        gold_paragraph_id = str(query.get("gold_paragraph_id", ""))

        for k in range(1, top_k + 1):
            recall_totals[k] += compute_recall_at_k(retrieved_chunks, gold_paragraph_id, k)
        reciprocal_rank_total += compute_mrr(retrieved_chunks, gold_paragraph_id)

    query_count = len(answerable_queries)
    backward_compatible_metrics: dict[str, float | int] = {
        "n_queries": query_count,
        "mrr": reciprocal_rank_total / query_count,
    }
    for k in range(1, top_k + 1):
        backward_compatible_metrics[f"recall@{k}"] = recall_totals[k] / query_count

    return backward_compatible_metrics


def save_retrieval_metrics(metrics_path: Path, metrics: dict[str, Any]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def save_retrieval_examples(examples_path: Path, examples: Sequence[dict[str, Any]]) -> None:
    examples_path.parent.mkdir(parents=True, exist_ok=True)
    with examples_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")


def _paragraph_id(chunk: dict[str, Any]) -> str:
    paragraph_id = chunk.get("paragraph_id")
    if paragraph_id is not None:
        return str(paragraph_id)

    chunk_id = chunk.get("chunk_id")
    if chunk_id is not None:
        return str(chunk_id)

    return ""
