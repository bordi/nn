from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def build_grounded_prompt(question: str, retrieved_chunks: Iterable[Mapping[str, Any]]) -> str:
    """Build a grounded prompt for answer generation.

    The prompt explicitly tells the model to answer only from the supplied
    context, avoid hallucinations, and return a structured JSON response.
    """

    chunk_blocks = [_format_chunk(chunk) for chunk in retrieved_chunks]
    chunks_section = "\n\n".join(chunk_blocks) if chunk_blocks else "[no retrieved chunks]"

    return "\n".join(
        [
            "You are a grounded question-answering assistant.",
            "Use only the provided context.",
            "Do not invent or make up facts.",
            "If the answer is not found in the provided context, say that the information was not found.",
            'Return valid JSON with the keys "answer" and "used_chunk_ids".',
            "When you answer, include only chunk_id values from the retrieved context in used_chunk_ids.",
            "",
            "Question:",
            question,
            "",
            "Retrieved chunks:",
            chunks_section,
        ]
    )


def _format_chunk(chunk: Mapping[str, Any]) -> str:
    chunk_id = str(chunk.get("chunk_id", ""))
    document_id = str(chunk.get("document_id", ""))
    paragraph_id = str(chunk.get("paragraph_id", ""))
    title = str(chunk.get("title", ""))
    score = chunk.get("score")
    context = str(chunk.get("context") or chunk.get("context_preview") or "")

    metadata_bits = [f"chunk_id={chunk_id}"]
    if document_id:
        metadata_bits.append(f"document_id={document_id}")
    if paragraph_id:
        metadata_bits.append(f"paragraph_id={paragraph_id}")
    if title:
        metadata_bits.append(f"title={title}")
    if score is not None:
        metadata_bits.append(f"score={score}")

    header = "[" + " ".join(metadata_bits) + "]"
    return "\n".join([header, context])
