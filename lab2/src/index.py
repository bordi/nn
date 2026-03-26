from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import faiss
import numpy as np


class _QueryEncoder(Protocol):
    def encode_query(self, text: str) -> list[float]:  # pragma: no cover - typing helper
        ...


def build_faiss_index(matrix: np.ndarray) -> faiss.Index:
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")

    dimension = int(matrix.shape[1])
    index = faiss.IndexFlatIP(dimension)

    if matrix.size:
        index.add(np.ascontiguousarray(matrix, dtype=np.float32))

    return index


def search_faiss_index(
    index: faiss.Index,
    query_vector: list[float] | np.ndarray,
    top_k: int,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    query = np.asarray(query_vector, dtype=np.float32)
    if query.ndim == 1:
        query = np.expand_dims(query, axis=0)
    if query.ndim != 2:
        raise ValueError("query_vector must be 1-dimensional or 2-dimensional")
    if query.shape[1] != index.d:
        raise ValueError(
            f"query vector dimension {query.shape[1]} does not match index dimension {index.d}"
        )

    scores, row_indices = index.search(np.ascontiguousarray(query, dtype=np.float32), top_k)

    results: list[dict[str, Any]] = []
    for row_index, score in zip(row_indices[0], scores[0]):
        if row_index < 0:
            continue
        results.append({"row_index": int(row_index), "score": float(score)})

    results.sort(key=lambda result: result["score"], reverse=True)
    return results


def save_faiss_index(index: faiss.Index, index_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))


def load_faiss_index(index_path: Path) -> faiss.Index:
    return faiss.read_index(str(index_path))


def load_index_metadata(metadata_path: Path) -> dict[str, Any]:
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def save_index_metadata(
    metadata_path: Path,
    *,
    model_name: str,
    chunk_ids: list[str],
    embedding_dimension: int,
) -> None:
    metadata = {
        "model_name": model_name,
        "num_vectors": len(chunk_ids),
        "embedding_dimension": embedding_dimension,
        "chunk_ids": chunk_ids,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
