from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.index import search_faiss_index


def load_chunk_records(path: Path) -> list[dict[str, Any]]:
    """Load chunk records from a JSONL artifact file."""

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def retrieve_top_k(
    query: str,
    encoder: Any,
    index: Any,
    chunk_records: list[dict[str, Any]],
    top_k: int,
    *,
    index_chunk_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve the best matching chunks for a query."""

    query_embedding = encoder.encode_query(query)
    search_results = search_faiss_index(index, query_embedding, top_k=top_k)
    chunk_lookup = {str(record.get("chunk_id", "")): record for record in chunk_records}

    retrieved_chunks: list[dict[str, Any]] = []
    for result in search_results:
        row_index = int(result["row_index"])
        if index_chunk_ids is None:
            if row_index >= len(chunk_records):
                raise ValueError(
                    f"index row {row_index} exceeds available chunk records ({len(chunk_records)})"
                )
            chunk = dict(chunk_records[row_index])
        else:
            if row_index >= len(index_chunk_ids):
                raise ValueError(
                    f"index row {row_index} exceeds available index chunk ids ({len(index_chunk_ids)})"
                )
            chunk_id = str(index_chunk_ids[row_index])
            if chunk_id not in chunk_lookup:
                raise ValueError(f"index references missing chunk_id '{chunk_id}'")
            chunk = dict(chunk_lookup[chunk_id])

        context = str(chunk.get("context", ""))
        chunk["score"] = result["score"]
        chunk["context_preview"] = context[:200]
        retrieved_chunks.append(chunk)

    return retrieved_chunks
