from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_LOCAL_PATH = PROJECT_ROOT / ".env.local"
DATASET_DIR = PROJECT_ROOT / "dataset"
SQUAD_DIR = DATASET_DIR / "squad"
TRAIN_DATASET_PATH = SQUAD_DIR / "train-v2.0.json"
DEV_DATASET_PATH = SQUAD_DIR / "dev-v2.0.json"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CORPUS_DIR = ARTIFACTS_DIR / "corpus"
DOCUMENTS_PATH = CORPUS_DIR / "documents.jsonl"
CHUNKS_PATH = CORPUS_DIR / "chunks.jsonl"
EVAL_QUERIES_PATH = CORPUS_DIR / "eval_queries.jsonl"
EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"
CHUNK_EMBEDDINGS_PATH = EMBEDDINGS_DIR / "chunk_embeddings.npy"
EMBEDDINGS_METADATA_PATH = EMBEDDINGS_DIR / "embeddings_metadata.json"
INDEX_DIR = ARTIFACTS_DIR / "index"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
INDEX_METADATA_PATH = INDEX_DIR / "index_metadata.json"
EVAL_DIR = ARTIFACTS_DIR / "eval"
RUNS_DIR = ARTIFACTS_DIR / "runs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEMOS_DIR = OUTPUTS_DIR / "demos"

DEFAULT_TOP_K = 5
DEFAULT_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_MAX_WORDS = 120
DEFAULT_CHUNK_OVERLAP_WORDS = 20


@dataclass(frozen=True)
class Config:
    openai_api_key: str | None
    project_root: Path = PROJECT_ROOT
    dataset_dir: Path = DATASET_DIR
    squad_dir: Path = SQUAD_DIR
    train_dataset_path: Path = TRAIN_DATASET_PATH
    dev_dataset_path: Path = DEV_DATASET_PATH
    artifacts_dir: Path = ARTIFACTS_DIR
    corpus_dir: Path = CORPUS_DIR
    documents_path: Path = DOCUMENTS_PATH
    chunks_path: Path = CHUNKS_PATH
    eval_queries_path: Path = EVAL_QUERIES_PATH
    embeddings_dir: Path = EMBEDDINGS_DIR
    chunk_embeddings_path: Path = CHUNK_EMBEDDINGS_PATH
    embeddings_metadata_path: Path = EMBEDDINGS_METADATA_PATH
    index_dir: Path = INDEX_DIR
    faiss_index_path: Path = FAISS_INDEX_PATH
    index_metadata_path: Path = INDEX_METADATA_PATH
    eval_dir: Path = EVAL_DIR
    runs_dir: Path = RUNS_DIR
    outputs_dir: Path = OUTPUTS_DIR
    demos_dir: Path = DEMOS_DIR
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME
    default_top_k: int = DEFAULT_TOP_K
    default_chunk_max_words: int = DEFAULT_CHUNK_MAX_WORDS
    default_chunk_overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS


def load_config() -> Config:
    env_local_values = dotenv_values(ENV_LOCAL_PATH) if ENV_LOCAL_PATH.exists() else {}
    api_key = env_local_values.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    return Config(openai_api_key=api_key or None)


def ensure_directories() -> None:
    for path in (
        ARTIFACTS_DIR,
        CORPUS_DIR,
        EMBEDDINGS_DIR,
        INDEX_DIR,
        EVAL_DIR,
        RUNS_DIR,
        OUTPUTS_DIR,
        DEMOS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
