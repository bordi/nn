from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_squad_dataset(path: Path) -> dict[str, Any]:
    """Load a local SQuAD JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_corpus_documents(dataset: dict[str, Any], source_split: str) -> list[dict[str, Any]]:
    """Flatten SQuAD articles into paragraph-level corpus documents."""
    documents: list[dict[str, Any]] = []

    for article_index, article in enumerate(dataset.get("data", [])):
        title = article.get("title", "")
        for paragraph_index, paragraph in enumerate(article.get("paragraphs", [])):
            paragraph_id = _build_paragraph_id(source_split, article_index, paragraph_index)
            qas = paragraph.get("qas", [])
            questions = [_build_question_record(qa, qas, question_index) for question_index, qa in enumerate(qas)]

            documents.append(
                {
                    "document_id": paragraph_id,
                    "paragraph_id": paragraph_id,
                    "title": title,
                    "context": paragraph.get("context", ""),
                    "source_split": source_split,
                    "article_index": article_index,
                    "paragraph_index": paragraph_index,
                    "questions": questions,
                    "related_questions": [question["question"] for question in questions],
                    "has_answerable": any(question["is_answerable"] for question in questions),
                }
            )

    return documents


def build_combined_corpus_documents(
    datasets_with_splits: list[tuple[dict[str, Any], str]],
) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []

    for dataset, source_split in datasets_with_splits:
        documents.extend(build_corpus_documents(dataset, source_split=source_split))

    return documents


def build_eval_queries(dataset: dict[str, Any], source_split: str) -> list[dict[str, Any]]:
    """Flatten SQuAD questions into paragraph-linked evaluation queries."""
    eval_queries: list[dict[str, Any]] = []

    for article_index, article in enumerate(dataset.get("data", [])):
        title = article.get("title", "")
        for paragraph_index, paragraph in enumerate(article.get("paragraphs", [])):
            paragraph_id = _build_paragraph_id(source_split, article_index, paragraph_index)
            context = paragraph.get("context", "")
            qas = paragraph.get("qas", [])
            related_questions = [qa.get("question", "") for qa in qas]

            for question_index, qa in enumerate(qas):
                is_answerable = not bool(qa.get("is_impossible", False))
                answers = [] if not is_answerable else [dict(answer) for answer in qa.get("answers", [])]

                eval_queries.append(
                    {
                        "query_id": qa.get("id", ""),
                        "document_id": paragraph_id,
                        "gold_paragraph_id": paragraph_id,
                        "title": title,
                        "context": context,
                        "source_split": source_split,
                        "article_index": article_index,
                        "paragraph_index": paragraph_index,
                        "question_index": question_index,
                        "question": qa.get("question", ""),
                        "answers": answers,
                        "is_answerable": is_answerable,
                        "related_questions": [
                            other_qa.get("question", "")
                            for other_index, other_qa in enumerate(qas)
                            if other_index != question_index
                        ],
                    }
                )

    return eval_queries


def _build_document_id(source_split: str, article_index: int, paragraph_index: int) -> str:
    return f"{source_split}-{article_index:04d}-{paragraph_index:04d}"


def _build_paragraph_id(source_split: str, article_index: int, paragraph_index: int) -> str:
    return _build_document_id(source_split, article_index, paragraph_index)


def _build_question_record(qa: dict[str, Any], all_qas: list[dict[str, Any]], question_index: int) -> dict[str, Any]:
    question_text = qa.get("question", "")
    answers = [dict(answer) for answer in qa.get("answers", [])]
    is_answerable = not bool(qa.get("is_impossible", False))

    return {
        "question_id": qa.get("id", ""),
        "question": question_text,
        "answers": answers,
        "is_answerable": is_answerable,
        "related_questions": [
            other_qa.get("question", "")
            for other_index, other_qa in enumerate(all_qas)
            if other_index != question_index
        ],
    }
