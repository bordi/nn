from __future__ import annotations

import argparse

from src.chunking import chunk_documents
from src.chat import answer_question, load_chat_runtime, run_ask, run_chat
from src.config import (
    DEFAULT_CHUNK_MAX_WORDS,
    DEFAULT_CHUNK_OVERLAP_WORDS,
    DEFAULT_TOP_K,
    ensure_directories,
    load_config,
)
from src.console import render_export_report, render_metrics_report
from src.data import build_combined_corpus_documents, build_eval_queries, load_squad_dataset
from src.embeddings import SentenceTransformerEncoder, build_embedding_matrix, save_embedding_artifacts
from src.evaluate_answers import export_answer_eval_run, generate_answer_eval_rows
from src.evaluate_retrieval import run_retrieval_evaluation, save_retrieval_examples, save_retrieval_metrics
from src.index import build_faiss_index, save_faiss_index, save_index_metadata
from src.retrieval import load_chunk_records, retrieve_top_k
from src.utils import read_jsonl, write_jsonl


def _print_stub_message(command_name: str, *, requires_openai: bool = False) -> None:
    config = load_config()
    if requires_openai and not config.openai_api_key:
        print(f"{command_name}: OpenAI API key is not set; this command is a stub for now.")
        raise SystemExit(2)
    print(f"{command_name}: not implemented yet.")
    raise SystemExit(2)


def _add_common_retrieval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--top-k",
        type=_positive_int,
        default=DEFAULT_TOP_K,
        help="Number of results to return.",
    )


def _positive_int(value: str) -> int:
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("--top-k must be a positive integer")
    return parsed_value


def _run_prepare_corpus(args: argparse.Namespace) -> None:
    config = load_config()
    ensure_directories()

    train_dataset = load_squad_dataset(config.train_dataset_path)
    dev_dataset = load_squad_dataset(config.dev_dataset_path)

    documents = build_combined_corpus_documents(
        [
            (train_dataset, "train"),
            (dev_dataset, "dev"),
        ]
    )
    chunks = chunk_documents(
        documents,
        max_words=args.max_words,
        overlap_words=args.overlap_words,
    )
    eval_queries = build_eval_queries(dev_dataset, source_split="dev")

    write_jsonl(config.documents_path, documents)
    write_jsonl(config.chunks_path, chunks)
    write_jsonl(config.eval_queries_path, eval_queries)

    print(
        "prepare-corpus: "
        f"saved {len(documents)} documents, {len(chunks)} chunks, "
        f"{len(eval_queries)} eval queries "
        f"(train docs: {sum(document['source_split'] == 'train' for document in documents)}, "
        f"dev docs: {sum(document['source_split'] == 'dev' for document in documents)})."
    )


def _run_build_index(_: argparse.Namespace) -> None:
    config = load_config()
    ensure_directories()
    if not config.chunks_path.exists():
        raise SystemExit("build-index: chunks.jsonl not found. Run `python main.py prepare-corpus` first.")

    chunk_records = load_chunk_records(config.chunks_path)
    encoder = SentenceTransformerEncoder(config.embedding_model_name)
    matrix, chunk_ids = build_embedding_matrix(chunk_records, encoder=encoder)
    index = build_faiss_index(matrix)

    save_embedding_artifacts(
        matrix,
        chunk_ids,
        config.chunk_embeddings_path,
        config.embeddings_metadata_path,
        model_name=config.embedding_model_name,
    )
    save_faiss_index(index, config.faiss_index_path)
    save_index_metadata(
        config.index_metadata_path,
        model_name=config.embedding_model_name,
        chunk_ids=chunk_ids,
        embedding_dimension=matrix.shape[1] if matrix.ndim == 2 and matrix.size else 0,
    )

    print(
        "build-index: "
        f"saved {len(chunk_ids)} embeddings and FAISS index "
        f"for model {config.embedding_model_name}."
    )


def _run_ask(args: argparse.Namespace) -> None:
    run_ask(args.query, top_k=args.top_k, no_llm=args.no_llm)


def _run_chat(args: argparse.Namespace) -> None:
    run_chat(top_k=args.top_k)


def _run_eval_retrieval(args: argparse.Namespace) -> None:
    config = load_config()
    ensure_directories()
    if not config.eval_queries_path.exists():
        raise SystemExit(
            "eval-retrieval: eval_queries.jsonl not found. Run `python main.py prepare-corpus` first."
        )

    runtime = load_chat_runtime(config, command_name="eval-retrieval")
    eval_queries = read_jsonl(config.eval_queries_path)

    def retrieve_for_query(question: str, top_k: int) -> list[dict[str, object]]:
        return retrieve_top_k(
            query=question,
            encoder=runtime.encoder,
            index=runtime.index,
            chunk_records=runtime.chunk_records,
            top_k=top_k,
            index_chunk_ids=runtime.index_metadata.get("chunk_ids"),
        )

    metrics, examples = run_retrieval_evaluation(
        eval_queries,
        top_k=args.top_k,
        retrieve_for_query=retrieve_for_query,
    )

    metrics_path = config.eval_dir / "retrieval_metrics.json"
    examples_path = config.eval_dir / "retrieval_examples.jsonl"
    save_retrieval_metrics(metrics_path, metrics)
    save_retrieval_examples(examples_path, examples)

    render_metrics_report(metrics, examples, output=__import__("sys").stdout)
    print()
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved examples to {examples_path}")


def _run_eval_answers(args: argparse.Namespace) -> None:
    config = load_config()
    ensure_directories()
    if not config.eval_queries_path.exists():
        raise SystemExit(
            "eval-answers: eval_queries.jsonl not found. Run `python main.py prepare-corpus` first."
        )

    runtime = load_chat_runtime(config, command_name="eval-answers")
    eval_queries = read_jsonl(config.eval_queries_path)

    def answer_for_query(query: dict[str, object], top_k: int) -> dict[str, object]:
        try:
            return answer_question(
                runtime,
                str(query.get("question", "")),
                top_k=top_k,
            )
        except (RuntimeError, ValueError) as exc:
            raise SystemExit(f"eval-answers: {exc}") from exc

    rows = generate_answer_eval_rows(
        eval_queries,
        sample_size=args.sample_size,
        top_k=args.top_k,
        answer_for_query=answer_for_query,
    )
    run_paths = export_answer_eval_run(
        runs_dir=config.runs_dir,
        rows=rows,
        sample_size=args.sample_size,
        top_k=args.top_k,
    )

    render_export_report(
        "Answer evaluation export",
        [
            ("sample_size", args.sample_size),
            ("top_k", args.top_k),
            ("num_rows", len(rows)),
            ("run_dir", run_paths["run_dir"]),
            ("csv", run_paths["csv_path"]),
            ("jsonl", run_paths["jsonl_path"]),
            ("metadata", run_paths["metadata_path"]),
        ],
        output=__import__("sys").stdout,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lab2 RAG assistant CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare-corpus", help="Prepare corpus artifacts from SQuAD.")
    prepare_parser.add_argument(
        "--max-words",
        type=int,
        default=DEFAULT_CHUNK_MAX_WORDS,
        help="Maximum number of words per chunk.",
    )
    prepare_parser.add_argument(
        "--overlap-words",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP_WORDS,
        help="Number of overlapping words between adjacent chunks.",
    )
    prepare_parser.set_defaults(func=_run_prepare_corpus)

    build_index_parser = subparsers.add_parser("build-index", help="Build the vector index.")
    build_index_parser.set_defaults(func=_run_build_index)

    ask_parser = subparsers.add_parser("ask", help="Ask a single question and exit.")
    ask_parser.add_argument("--query", type=str, required=True, help="Question to ask.")
    _add_common_retrieval_args(ask_parser)
    ask_parser.add_argument("--no-llm", action="store_true", help="Skip answer generation.")
    ask_parser.set_defaults(func=_run_ask)

    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat session.")
    _add_common_retrieval_args(chat_parser)
    chat_parser.set_defaults(func=_run_chat)

    eval_retrieval_parser = subparsers.add_parser(
        "eval-retrieval",
        help="Evaluate retrieval quality on the dev split.",
    )
    _add_common_retrieval_args(eval_retrieval_parser)
    eval_retrieval_parser.set_defaults(func=_run_eval_retrieval)

    eval_answers_parser = subparsers.add_parser(
        "eval-answers",
        help="Export answer evaluation data.",
    )
    eval_answers_parser.add_argument(
        "--sample-size",
        type=_positive_int,
        default=20,
        help="Number of answerable dev queries to export for manual answer evaluation.",
    )
    _add_common_retrieval_args(eval_answers_parser)
    eval_answers_parser.set_defaults(func=_run_eval_answers)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
