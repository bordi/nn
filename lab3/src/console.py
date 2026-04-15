from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence


def render_summary(title: str, rows: Sequence[tuple[str, object]]) -> str:
    width = max((len(label) for label, _ in rows), default=0)
    lines = [title]
    for label, value in rows:
        lines.append(f"  {label.ljust(width)} : {value}")
    return "\n".join(lines)


def render_key_values(title: str, values: Mapping[str, object]) -> str:
    return render_summary(title, list(values.items()))


def render_metrics(title: str, metrics: Mapping[str, object]) -> str:
    return render_key_values(title, metrics)


def render_paths(title: str, paths: Mapping[str, Path]) -> str:
    return render_key_values(title, {name: str(path) for name, path in paths.items()})
