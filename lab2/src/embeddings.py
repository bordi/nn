from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer


class _DocumentEncoder(Protocol):
    def encode_documents(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - typing helper
        ...


class SentenceTransformerEncoder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        cached_model_path = _resolve_cached_model_path(model_name)
        if cached_model_path is not None:
            self._model = SentenceTransformer(cached_model_path, local_files_only=True)
        else:
            self._model = SentenceTransformer(model_name)

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def encode_query(self, text: str) -> list[float]:
        embedding = self._model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embedding[0].tolist()


def _resolve_cached_model_path(model_name: str) -> str | None:
    local_model_path = Path(model_name).expanduser()
    if local_model_path.exists():
        return str(local_model_path)

    try:
        return snapshot_download(repo_id=model_name, local_files_only=True)
    except Exception:
        return None


def build_embedding_matrix(
    chunk_records: list[dict[str, Any]],
    encoder: _DocumentEncoder,
) -> tuple[np.ndarray, list[str]]:
    texts = [str(record.get("context", "")) for record in chunk_records]
    chunk_ids = [str(record.get("chunk_id", "")) for record in chunk_records]

    if not texts:
        return np.zeros((0, 0), dtype=np.float32), chunk_ids

    embeddings = encoder.encode_documents(texts)
    matrix = np.asarray(embeddings, dtype=np.float32)

    if matrix.ndim == 1:
        matrix = np.expand_dims(matrix, axis=0)

    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    return matrix, chunk_ids


def save_embedding_artifacts(
    matrix: np.ndarray,
    chunk_ids: list[str],
    embeddings_path: Path,
    metadata_path: Path,
    *,
    model_name: str,
) -> None:
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, np.ascontiguousarray(matrix, dtype=np.float32))
    metadata = {
        "model_name": model_name,
        "num_embeddings": int(matrix.shape[0]) if matrix.ndim == 2 else 0,
        "embedding_dimension": int(matrix.shape[1]) if matrix.ndim == 2 and matrix.size else 0,
        "chunk_ids": chunk_ids,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
