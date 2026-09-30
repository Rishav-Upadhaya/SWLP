"""Terminal presentation for the SWLP CLI, built on ``rich``.

One console, one palette, a handful of building blocks — every command renders
through these so the CLI looks the same everywhere (header panel, dim
secondary text, one accent colour, a status line after each answer). ``rich``
handles NO_COLOR, non-TTY output and terminal width (incl. editor terminals).
"""
from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

_THEME = Theme({
    "accent": "bold #d97757",       # warm orange accent (headings, prompt caret)
    "muted": "grey58",
    "ok": "green",
    "warn": "yellow",
    "err": "bold red",
    "key": "bold",
    "cmd": "#8ab4f8",
})
console = Console(theme=_THEME, highlight=False)
err_console = Console(theme=_THEME, stderr=True, highlight=False)

# Answers re-render as markdown at most this often while streaming.
_LIVE_REFRESH_PER_SECOND = 12


def header(title: str, rows: list[tuple[str, str]]) -> None:
    """Session header: a rounded panel of ``key  value`` rows."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="muted", no_wrap=True)
    grid.add_column()
    for key, value in rows:
        grid.add_row(key, value)
    console.print(Panel(grid, title=f"[accent]{title}[/]", title_align="left",
                        border_style="muted", expand=False, padding=(0, 2)))


def section(title: str) -> None:
    console.print(f"\n[accent]{title}[/]")


def commands(rows: Iterable[tuple[str, str]]) -> None:
    """Two-column ``command  description`` list (help screens)."""
    grid = Table.grid(padding=(0, 3))
    grid.add_column(style="cmd", no_wrap=True)
    grid.add_column(style="muted")
    for cmd, desc in rows:
        grid.add_row(cmd, desc)
    console.print(Padding(grid, (0, 0, 0, 2)))


def table(columns: list[str], rows: list[list[str]], title: str = "") -> None:
    t = Table(title=title or None, title_style="accent", title_justify="left",
              header_style="muted", border_style="muted", box=None, pad_edge=False,
              padding=(0, 2), show_header=any(columns))
    for col in columns:
        t.add_column(col)
    for row in rows:
        t.add_row(*row)
    console.print(Padding(t, (0, 0, 0, 2)))


def status(parts: list[str]) -> None:
    """Dim one-line summary under an answer: ``14.5 tok/s · 58 ms/tok · …``."""
    console.print(Text("  " + "  ·  ".join(p for p in parts if p), style="muted"))


def ok(message: str) -> None:
    console.print(f"[ok]✓[/] {message}")


def note(message: str) -> None:
    console.print(f"[muted]{message}[/]")


def warn(message: str) -> None:
    err_console.print(f"[warn]![/] {message}")


def error(message: str, hint: str = "") -> None:
    err_console.print(f"[err]✗[/] {message}")
    if hint:
        err_console.print(f"  [muted]{hint}[/]")


@contextmanager
def spinner(message: str) -> Iterator[dict[str, float]]:
    """``with ui.spinner("loading …") as t:`` → ``t["elapsed"]`` afterwards."""
    timing = {"elapsed": 0.0}
    started = time.perf_counter()
    with console.status(f"[muted]{message}[/]", spinner="dots"):
        yield timing
    timing["elapsed"] = time.perf_counter() - started


def progress() -> Progress:
    """Bar for long jobs (download / shard)."""
    return Progress(
        TextColumn("  [muted]{task.description}[/]"), BarColumn(bar_width=32),
        TextColumn("{task.completed}/{task.total}"), TimeRemainingColumn(),
        console=console, transient=False,
    )


def stream_markdown(chunks: Iterable[str]) -> tuple[str, int, float]:
    """Render streamed text as live-updating markdown; returns
    ``(full_text, n_chunks, seconds)``. Ctrl+C stops the stream cleanly."""
    parts: list[str] = []
    started = time.perf_counter()
    with Live(console=console, refresh_per_second=_LIVE_REFRESH_PER_SECOND,
              vertical_overflow="visible") as live:
        try:
            for chunk in chunks:
                parts.append(chunk)
                live.update(Padding(Markdown(_hard_breaks("".join(parts))), (0, 0, 0, 2)))
        except KeyboardInterrupt:  # stops this answer only; the chat continues
            parts.append("\n\n*[stopped]*")
            live.update(Padding(Markdown(_hard_breaks("".join(parts))), (0, 0, 0, 2)))
    return "".join(parts), len(parts), time.perf_counter() - started


def markdown(text: str) -> None:
    console.print(Padding(Markdown(_hard_breaks(text)), (0, 0, 0, 2)))


def _hard_breaks(text: str) -> str:
    """Keep the model's line breaks, as chat UIs do: CommonMark would join
    "Red\nBlue\nYellow" into one line. Code fences are left untouched."""
    pieces = text.split("```")
    for i in range(0, len(pieces), 2):  # even pieces are outside code fences
        pieces[i] = pieces[i].replace("\n", "  \n")
    return "```".join(pieces)
