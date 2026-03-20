from __future__ import annotations

import shutil
import sys
import time


class TerminalProgressBar:
    def __init__(self, total: int, description: str, enabled: bool = True) -> None:
        self.total = max(1, total)
        self.description = description
        self.enabled = enabled
        self.current = 0
        self.started_at = time.perf_counter()
        self.last_rendered_line = ""

        if self.enabled:
            self._render()

    def update(self, step: int = 1, postfix: str = "") -> None:
        self.current = min(self.total, self.current + step)
        if self.enabled:
            self._render(postfix=postfix)

    def close(self, postfix: str = "") -> None:
        self.current = self.total
        if self.enabled:
            final_line = self._compose_line(postfix=postfix)
            if final_line != self.last_rendered_line:
                self._render(postfix=postfix, final=True)
            sys.stdout.write("\n")
            sys.stdout.flush()

    def _compose_line(self, postfix: str = "") -> str:
        terminal_width = shutil.get_terminal_size((100, 20)).columns
        bar_width = max(10, min(28, terminal_width // 4))
        fraction = self.current / self.total
        filled = int(bar_width * fraction)
        bar = "#" * filled + "-" * (bar_width - filled)
        elapsed = time.perf_counter() - self.started_at
        rate = self.current / elapsed if elapsed > 0 else 0.0
        remaining = max(0.0, (self.total - self.current) / rate) if rate > 0 else 0.0

        line = (
            f"{self.description:<20} "
            f"[{bar}] "
            f"{self.current:>4}/{self.total:<4} "
            f"{fraction * 100:>6.2f}% "
            f"eta={remaining:>5.1f}s"
        )
        if postfix:
            line += f" | {postfix}"

        if len(line) > terminal_width - 1:
            line = line[: terminal_width - 4] + "..."

        return line

    def _render(self, postfix: str = "", final: bool = False) -> None:
        line = self._compose_line(postfix=postfix)

        prefix = "\r" if not final else "\r"
        padding = ""
        if len(self.last_rendered_line) > len(line):
            padding = " " * (len(self.last_rendered_line) - len(line))

        sys.stdout.write(prefix + line + padding)
        sys.stdout.flush()
        self.last_rendered_line = line
