from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import embeddings as embeddings_module
from src.embeddings import SentenceTransformerEncoder, build_embedding_matrix
from src.index import build_faiss_index, search_faiss_index
from src.retrieval import load_chunk_records, retrieve_top_k


class FakeEncoder:
    def __init__(self) -> None:
        self._vectors = {
            "alpha beta": [1.0, 0.0],
            "beta gamma": [0.6, 0.4],
            "delta epsilon": [0.0, 1.0],
            "alpha": [1.0, 0.0],
        }

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors[text] for text in texts]

    def encode_query(self, text: str) -> list[float]:
        return self._vectors[text]


def test_load_chunk_records_reads_artifact_jsonl(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    records = [
        {
            "chunk_id": "chunk-001",
            "document_id": "doc-001",
            "paragraph_id": "para-001",
            "context": "alpha beta",
            "title": "Doc 1",
            "source_split": "train",
        },
        {
            "chunk_id": "chunk-002",
            "document_id": "doc-002",
            "paragraph_id": "para-002",
            "context": "beta gamma",
            "title": "Doc 2",
            "source_split": "dev",
        },
    ]
    chunks_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    loaded_records = load_chunk_records(chunks_path)

    assert len(loaded_records) == 2
    assert loaded_records[0]["chunk_id"] == "chunk-001"
    assert loaded_records[1]["paragraph_id"] == "para-002"


def test_build_embedding_matrix_preserves_row_to_chunk_mapping() -> None:
    chunk_records = [
        {"chunk_id": "chunk-001", "context": "alpha beta"},
        {"chunk_id": "chunk-002", "context": "beta gamma"},
        {"chunk_id": "chunk-003", "context": "delta epsilon"},
    ]

    matrix, chunk_ids = build_embedding_matrix(chunk_records, encoder=FakeEncoder())

    assert matrix.shape == (3, 2)
    assert chunk_ids == ["chunk-001", "chunk-002", "chunk-003"]


def test_search_faiss_index_returns_sorted_top_k_results() -> None:
    chunk_records = [
        {"chunk_id": "chunk-001", "context": "alpha beta"},
        {"chunk_id": "chunk-002", "context": "beta gamma"},
        {"chunk_id": "chunk-003", "context": "delta epsilon"},
    ]
    encoder = FakeEncoder()
    matrix, _ = build_embedding_matrix(chunk_records, encoder=encoder)
    index = build_faiss_index(matrix)

    results = search_faiss_index(index, encoder.encode_query("alpha"), top_k=2)

    assert len(results) == 2
    assert results[0]["row_index"] == 0
    assert results[1]["row_index"] == 1
    assert results[0]["score"] >= results[1]["score"]


def test_search_faiss_index_rejects_non_positive_top_k() -> None:
    chunk_records = [
        {"chunk_id": "chunk-001", "context": "alpha beta"},
        {"chunk_id": "chunk-002", "context": "beta gamma"},
    ]
    encoder = FakeEncoder()
    matrix, _ = build_embedding_matrix(chunk_records, encoder=encoder)
    index = build_faiss_index(matrix)

    try:
        search_faiss_index(index, encoder.encode_query("alpha"), top_k=0)
    except ValueError as exc:
        assert "top_k" in str(exc)
    else:
        raise AssertionError("search_faiss_index should reject top_k <= 0")


def test_search_faiss_index_rejects_query_dimension_mismatch() -> None:
    chunk_records = [
        {"chunk_id": "chunk-001", "context": "alpha beta"},
        {"chunk_id": "chunk-002", "context": "beta gamma"},
    ]
    encoder = FakeEncoder()
    matrix, _ = build_embedding_matrix(chunk_records, encoder=encoder)
    index = build_faiss_index(matrix)

    try:
        search_faiss_index(index, [1.0, 0.0, 0.5], top_k=1)
    except ValueError as exc:
        assert "dimension" in str(exc)
    else:
        raise AssertionError("search_faiss_index should reject mismatched query dimensions")


def test_retrieve_top_k_maps_search_results_back_to_chunks() -> None:
    chunk_records = [
        {
            "chunk_id": "chunk-001",
            "document_id": "doc-001",
            "paragraph_id": "para-001",
            "context": "alpha beta",
            "title": "Doc 1",
            "source_split": "train",
        },
        {
            "chunk_id": "chunk-002",
            "document_id": "doc-002",
            "paragraph_id": "para-002",
            "context": "beta gamma",
            "title": "Doc 2",
            "source_split": "dev",
        },
        {
            "chunk_id": "chunk-003",
            "document_id": "doc-003",
            "paragraph_id": "para-003",
            "context": "delta epsilon",
            "title": "Doc 3",
            "source_split": "train",
        },
    ]
    encoder = FakeEncoder()
    matrix, _ = build_embedding_matrix(chunk_records, encoder=encoder)
    index = build_faiss_index(matrix)

    results = retrieve_top_k(
        query="alpha",
        encoder=encoder,
        index=index,
        chunk_records=chunk_records,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["chunk_id"] == "chunk-001"
    assert results[1]["chunk_id"] == "chunk-002"
    assert results[0]["score"] >= results[1]["score"]
    assert results[0]["context_preview"]


def test_retrieve_top_k_uses_index_chunk_ids_instead_of_chunk_record_order() -> None:
    indexed_chunk_records = [
        {
            "chunk_id": "chunk-001",
            "document_id": "doc-001",
            "paragraph_id": "para-001",
            "context": "alpha beta",
            "title": "Doc 1",
            "source_split": "train",
        },
        {
            "chunk_id": "chunk-002",
            "document_id": "doc-002",
            "paragraph_id": "para-002",
            "context": "beta gamma",
            "title": "Doc 2",
            "source_split": "dev",
        },
        {
            "chunk_id": "chunk-003",
            "document_id": "doc-003",
            "paragraph_id": "para-003",
            "context": "delta epsilon",
            "title": "Doc 3",
            "source_split": "train",
        },
    ]
    shuffled_chunk_records = [
        indexed_chunk_records[2],
        indexed_chunk_records[1],
        indexed_chunk_records[0],
    ]
    encoder = FakeEncoder()
    matrix, chunk_ids = build_embedding_matrix(indexed_chunk_records, encoder=encoder)
    index = build_faiss_index(matrix)

    results = retrieve_top_k(
        query="alpha",
        encoder=encoder,
        index=index,
        chunk_records=shuffled_chunk_records,
        top_k=2,
        index_chunk_ids=chunk_ids,
    )

    assert [result["chunk_id"] for result in results] == ["chunk-001", "chunk-002"]


def test_retrieve_top_k_fails_if_index_references_missing_chunk_id() -> None:
    chunk_records = [
        {
            "chunk_id": "chunk-001",
            "document_id": "doc-001",
            "paragraph_id": "para-001",
            "context": "alpha beta",
            "title": "Doc 1",
            "source_split": "train",
        }
    ]
    encoder = FakeEncoder()
    matrix, _ = build_embedding_matrix(chunk_records, encoder=encoder)
    index = build_faiss_index(matrix)

    try:
        retrieve_top_k(
            query="alpha",
            encoder=encoder,
            index=index,
            chunk_records=chunk_records,
            top_k=1,
            index_chunk_ids=["chunk-999"],
        )
    except ValueError as exc:
        assert "chunk-999" in str(exc)
    else:
        raise AssertionError("retrieve_top_k should fail for missing chunk ids")


def test_sentence_transformer_encoder_uses_local_files_only(monkeypatch) -> None:
    captured_kwargs: dict[str, object] = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, **kwargs: object) -> None:
            captured_kwargs["model_name"] = model_name
            captured_kwargs.update(kwargs)

        def encode(self, texts: list[str], **_: object) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(embeddings_module, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(
        embeddings_module,
        "_resolve_cached_model_path",
        lambda model_name: "/tmp/fake-cached-model" if model_name == "sentence-transformers/all-MiniLM-L6-v2" else None,
    )

    encoder = SentenceTransformerEncoder("sentence-transformers/all-MiniLM-L6-v2")

    assert encoder.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert captured_kwargs["model_name"] == "/tmp/fake-cached-model"
    assert captured_kwargs["local_files_only"] is True
