from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DEV_DATASET_PATH, TRAIN_DATASET_PATH
from src.data import (
    build_combined_corpus_documents,
    build_corpus_documents,
    build_eval_queries,
    load_squad_dataset,
)


def test_load_squad_dataset_reads_local_train_and_dev_files() -> None:
    train_dataset = load_squad_dataset(TRAIN_DATASET_PATH)
    dev_dataset = load_squad_dataset(DEV_DATASET_PATH)

    assert train_dataset["version"] == "v2.0"
    assert dev_dataset["version"] == "v2.0"
    assert len(train_dataset["data"]) > 0
    assert len(dev_dataset["data"]) > 0


def test_build_corpus_documents_flattens_train_paragraphs_with_metadata() -> None:
    train_dataset = load_squad_dataset(TRAIN_DATASET_PATH)

    documents = build_corpus_documents(train_dataset, source_split="train")

    first_document = documents[0]

    assert len(documents) > 0
    assert first_document["document_id"] == "train-0000-0000"
    assert first_document["paragraph_id"] == "train-0000-0000"
    assert first_document["title"] == "Beyoncé"
    assert first_document["source_split"] == "train"
    assert first_document["paragraph_index"] == 0
    assert first_document["context"]
    assert isinstance(first_document["questions"], list)
    assert first_document["questions"][0]["question"]
    assert isinstance(first_document["questions"][0]["is_answerable"], bool)
    assert first_document["has_answerable"] is True


def test_build_eval_queries_extracts_flat_queries_from_dev_split() -> None:
    dev_dataset = load_squad_dataset(DEV_DATASET_PATH)

    eval_queries = build_eval_queries(dev_dataset, source_split="dev")

    first_query = eval_queries[0]

    assert len(eval_queries) > 0
    assert first_query["query_id"]
    assert first_query["document_id"] == "dev-0000-0000"
    assert first_query["gold_paragraph_id"] == "dev-0000-0000"
    assert first_query["title"] == "Normans"
    assert first_query["source_split"] == "dev"
    assert first_query["question"] == "In what country is Normandy located?"
    assert isinstance(first_query["is_answerable"], bool)
    assert isinstance(first_query["answers"], list)
    assert "answer_start" in first_query["answers"][0]


def test_build_eval_queries_marks_impossible_questions_without_gold_answers() -> None:
    dev_dataset = load_squad_dataset(DEV_DATASET_PATH)

    eval_queries = build_eval_queries(dev_dataset, source_split="dev")
    impossible_query = next(query for query in eval_queries if query["query_id"] == "5ad39d53604f3c001a3fe8d1")

    assert impossible_query["question"] == "Who gave their name to Normandy in the 1000's and 1100's"
    assert impossible_query["is_answerable"] is False
    assert impossible_query["answers"] == []
    assert impossible_query["gold_paragraph_id"] == "dev-0000-0000"


def test_build_combined_corpus_documents_merges_train_and_dev_splits() -> None:
    train_dataset = load_squad_dataset(TRAIN_DATASET_PATH)
    dev_dataset = load_squad_dataset(DEV_DATASET_PATH)

    documents = build_combined_corpus_documents(
        [
            (train_dataset, "train"),
            (dev_dataset, "dev"),
        ]
    )

    assert documents
    assert any(document["source_split"] == "train" for document in documents)
    assert any(document["source_split"] == "dev" for document in documents)
    assert any(document["paragraph_id"] == "dev-0000-0000" for document in documents)
