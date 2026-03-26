from __future__ import annotations

from typing import Any, TextIO


def render_metrics_report(
    metrics: dict[str, Any],
    examples: list[dict[str, Any]],
    *,
    output: TextIO,
    is_tty: bool | None = None,
) -> None:
    tty_enabled = output.isatty() if is_tty is None else is_tty
    sections = [
        _render_metrics_summary_section(metrics, tty_enabled),
        _render_hard_misses_section(examples, tty_enabled),
    ]
    output.write("\n\n".join(section for section in sections if section))
    output.write("\n")


def render_export_report(
    title: str,
    items: list[tuple[str, Any]],
    *,
    output: TextIO,
    is_tty: bool | None = None,
) -> None:
    tty_enabled = output.isatty() if is_tty is None else is_tty
    header = _section_header(title, tty_enabled)
    width = max(len(label) for label, _ in items) if items else 0
    lines = [f"{_label(label, tty_enabled):<{width}} : {value}" for label, value in items]
    output.write("\n".join([header, *_indent_block(lines)]))
    output.write("\n")


def render_answer_report(
    result: dict[str, Any],
    *,
    output: TextIO,
    is_tty: bool | None = None,
) -> None:
    tty_enabled = output.isatty() if is_tty is None else is_tty

    sections = [
        _render_answer_section(str(result.get("answer", "")), tty_enabled),
        _render_used_chunk_ids_section(result.get("used_chunk_ids", []), tty_enabled),
        _render_retrieval_metadata_section(result.get("retrieved_chunks", []), tty_enabled),
    ]
    output.write("\n\n".join(section for section in sections if section))
    output.write("\n")


def _render_metrics_summary_section(metrics: dict[str, Any], tty_enabled: bool) -> str:
    header = _section_header("Retrieval metrics", tty_enabled)
    top_k = int(metrics.get("top_k", 0))
    rows = [
        ("queries_total", metrics.get("num_queries_total", 0)),
        ("queries_scored", metrics.get("num_queries_scored", 0)),
        ("queries_skipped", metrics.get("num_queries_skipped", 0)),
        (f"recall@{top_k}", _format_float(metrics.get("recall_at_k", 0.0))),
        (f"mrr@{top_k}", _format_float(metrics.get("mrr_at_k", 0.0))),
    ]
    width = max(len(label) for label, _ in rows)
    lines = [f"{_label(label, tty_enabled):<{width}} : {value}" for label, value in rows]
    return "\n".join([header, *_indent_block(lines)])


def _render_hard_misses_section(examples: list[dict[str, Any]], tty_enabled: bool) -> str:
    misses = [example for example in examples if not example.get("hit")]
    header = _section_header("Hard misses", tty_enabled)
    if not misses:
        return f"{header}\n{_indent_lines('none')}"

    lines: list[str] = []
    for position, example in enumerate(misses[:3], start=1):
        lines.append(
            f"{position}. {_label('query_id', tty_enabled)}={example.get('query_id', '')}  "
            f"{_label('gold_paragraph_id', tty_enabled)}={example.get('gold_paragraph_id', '')}"
        )
        lines.append(f"   {_label('question', tty_enabled)}={example.get('question', '')}")
        lines.append(
            f"   {_label('retrieved_chunk_ids', tty_enabled)}="
            + ", ".join(str(chunk_id) for chunk_id in example.get("retrieved_chunk_ids", []))
        )
    return "\n".join([header, *lines])


def _render_answer_section(answer: str, tty_enabled: bool) -> str:
    header = _section_header("Answer", tty_enabled)
    body = answer.strip() if answer.strip() else "[empty answer]"
    return f"{header}\n{_indent_lines(body)}"


def _render_used_chunk_ids_section(used_chunk_ids: list[Any], tty_enabled: bool) -> str:
    header = _section_header("used_chunk_ids", tty_enabled)
    chunk_ids = [str(chunk_id) for chunk_id in used_chunk_ids if str(chunk_id)]
    body = ", ".join(chunk_ids) if chunk_ids else "[]"
    return f"{header}\n{_indent_lines(body)}"


def _render_retrieval_metadata_section(
    retrieved_chunks: list[dict[str, Any]],
    tty_enabled: bool,
) -> str:
    header = _section_header("Retrieval metadata", tty_enabled)
    if not retrieved_chunks:
        return f"{header}\n{_indent_lines('[]')}"

    lines: list[str] = []
    for position, chunk in enumerate(retrieved_chunks, start=1):
        lines.extend(_render_chunk_block(position, chunk, tty_enabled))
    return "\n".join([header, *lines])


def _render_chunk_block(position: int, chunk: dict[str, Any], tty_enabled: bool) -> list[str]:
    chunk_id = str(chunk.get("chunk_id", ""))
    score = float(chunk.get("score", 0.0))
    document_id = str(chunk.get("document_id", ""))
    paragraph_id = str(chunk.get("paragraph_id", ""))
    title = str(chunk.get("title", ""))
    source_split = str(chunk.get("source_split", ""))
    preview = str(chunk.get("context_preview") or chunk.get("context") or "")

    first_line = f"{position}. {_label('chunk_id', tty_enabled)}={chunk_id}"
    first_line += f"  {_label('score', tty_enabled)}={score:.4f}"

    second_line = f"   {_label('document_id', tty_enabled)}={document_id}"
    second_line += f"  {_label('paragraph_id', tty_enabled)}={paragraph_id}"

    third_line_bits: list[str] = []
    if title:
        third_line_bits.append(f"{_label('title', tty_enabled)}={title}")
    if source_split:
        third_line_bits.append(f"{_label('source_split', tty_enabled)}={source_split}")
    third_line = f"   {'  '.join(third_line_bits)}" if third_line_bits else ""

    lines = [first_line, second_line]
    if third_line:
        lines.append(third_line)
    if preview:
        lines.append(f"   {_label('preview', tty_enabled)}={preview}")
    return lines


def _section_header(title: str, tty_enabled: bool) -> str:
    if not tty_enabled:
        return title
    return f"{_ansi('bold', 'cyan')}{title}{_ansi('reset')}"


def _label(text: str, tty_enabled: bool) -> str:
    if not tty_enabled:
        return text
    return f"{_ansi('bold', 'green')}{text}{_ansi('reset')}"


def _indent_lines(text: str, indent: str = "  ") -> str:
    lines = text.splitlines() or [""]
    return "\n".join(f"{indent}{line}" for line in lines)


def _indent_block(lines: list[str], indent: str = "  ") -> list[str]:
    return [f"{indent}{line}" for line in lines]


def _format_float(value: Any) -> str:
    return f"{float(value):.4f}"


def _ansi(*codes: str) -> str:
    mapping = {
        "reset": "\x1b[0m",
        "bold": "\x1b[1m",
        "cyan": "\x1b[36m",
        "green": "\x1b[32m",
    }
    return "".join(mapping[code] for code in codes)
