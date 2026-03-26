from __future__ import annotations

from typing import Any


def chunk_documents(
    documents: list[dict[str, Any]],
    max_words: int,
    overlap_words: int,
) -> list[dict[str, Any]]:
    """Split documents into overlapping word chunks.

    Each output chunk preserves the source metadata and adds:
    - chunk_id: stable identifier based on document_id and chunk_index
    - chunk_index: zero-based position within the document
    - context: the chunk text
    """

    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if overlap_words < 0:
        raise ValueError("overlap_words must be non-negative")
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")

    step = max_words - overlap_words
    chunks: list[dict[str, Any]] = []

    for document in documents:
        words = str(document.get("context", "")).split()
        document_id = str(document.get("document_id", "document"))

        if not words:
            continue

        chunk_index = 0
        start = 0

        while start < len(words):
            end = min(start + max_words, len(words))
            chunk_words = words[start:end]
            if not chunk_words:
                break

            chunk = {
                "chunk_id": f"{document_id}-chunk-{chunk_index:03d}",
                "chunk_index": chunk_index,
                "document_id": document_id,
                "paragraph_id": document.get("paragraph_id", document_id),
                "title": document.get("title", ""),
                "context": " ".join(chunk_words),
                "source_split": document.get("source_split", ""),
                "article_index": document.get("article_index"),
                "paragraph_index": document.get("paragraph_index"),
            }
            chunks.append(chunk)

            if end >= len(words):
                break

            chunk_index += 1
            start += step

    return chunks
