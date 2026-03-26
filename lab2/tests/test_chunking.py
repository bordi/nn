from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import chunk_documents


def test_chunk_documents_assigns_stable_chunk_ids() -> None:
    documents = [
        {
            "document_id": "train-0000-0000",
            "paragraph_id": "train-0000-0000",
            "title": "Sample Title",
            "context": "one two three four five six seven eight nine ten",
            "source_split": "train",
            "paragraph_index": 0,
        }
    ]

    chunks = chunk_documents(documents, max_words=4, overlap_words=1)

    assert [chunk["chunk_id"] for chunk in chunks] == [
        "train-0000-0000-chunk-000",
        "train-0000-0000-chunk-001",
        "train-0000-0000-chunk-002",
    ]


def test_chunk_documents_respects_chunk_size_limit() -> None:
    documents = [
        {
            "document_id": "train-0000-0000",
            "paragraph_id": "train-0000-0000",
            "title": "Sample Title",
            "context": "alpha beta gamma delta epsilon zeta eta theta",
            "source_split": "train",
            "paragraph_index": 0,
        }
    ]

    chunks = chunk_documents(documents, max_words=3, overlap_words=1)

    assert chunks
    assert all(len(chunk["context"].split()) <= 3 for chunk in chunks)


def test_chunk_documents_preserves_source_metadata() -> None:
    documents = [
        {
            "document_id": "train-0000-0000",
            "paragraph_id": "train-0000-0000",
            "title": "Sample Title",
            "context": "one two three four five",
            "source_split": "train",
            "paragraph_index": 7,
        }
    ]

    chunks = chunk_documents(documents, max_words=2, overlap_words=0)

    first_chunk = chunks[0]

    assert first_chunk["document_id"] == "train-0000-0000"
    assert first_chunk["paragraph_id"] == "train-0000-0000"
    assert first_chunk["title"] == "Sample Title"
    assert first_chunk["source_split"] == "train"
    assert first_chunk["paragraph_index"] == 7
    assert first_chunk["chunk_index"] == 0
