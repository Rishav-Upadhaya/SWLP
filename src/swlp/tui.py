"""Terminal presentation for SWLP — colours, boxes, spinners, stream wrapping.

Pure presentation: no model, no config, no I/O beyond stdout. Kept separate
from ``chat.py`` so the REPL is conversation logic and this is how it looks.

Design rules, in priority order:

1. **Degrade cleanly.** Colour only when stdout is a TTY and ``NO_COLOR`` is
   unset; box drawing only when the encoding can represent it. Piping SWLP
   into a file must produce readable plain text, not escape-code soup.
2. **Respect the terminal width.** A generated paragraph that runs off the
   right edge is unreadable, and models emit long lines. Streamed text is
   word-wrapped as it arrives.
3. **One accent colour.** Cyan for structure, green for the model, yellow for
   warnings, dim for anything secondary. More colours look like a toy.
"""
from __future__ import annotations

import itertools
import os
import shutil
import sys
import threading
import time

# ── capability detection ─────────────────────────────────────────────────────

def _supports_colour() -> bool:
    if os.environ.get("NO_COLOR"):          # no-color.org
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return sys.stdout.isatty()


def _supports_unicode() -> bool:
    enc = (getattr(sys.stdout, "encoding", "") or "").lower()
    return "utf" in enc


COLOUR = _supports_colour()
UNICODE = _supports_unicode()


def width(default: int = 80, cap: int = 100) -> int:
    """Usable text width: the terminal, capped so long lines stay readable."""
    try:
        return max(40, min(cap, shutil.get_terminal_size((default, 24)).columns - 2))
    except Exception:
        return default


# ── styling ──────────────────────────────────────────────────────────────────

def _c(code: str, text: str) -> str:
    return text if not COLOUR else f"\033[{code}m{text}\033[0m"


def bold(t: str) -> str:    return _c("1", t)
def dim(t: str) -> str:     return _c("2", t)
def italic(t: str) -> str:  return _c("3", t)
def red(t: str) -> str:     return _c("31", t)
def green(t: str) -> str:   return _c("32", t)
def yellow(t: str) -> str:  return _c("33", t)
def blue(t: str) -> str:    return _c("34", t)
def cyan(t: str) -> str:    return _c("36", t)
def accent(t: str) -> str:  return _c("1;36", t)


# ── box drawing ──────────────────────────────────────────────────────────────

_BOX = {
    True:  {"tl": "╭", "tr": "╮", "bl": "╰", "br": "╯", "h": "─", "v": "│"},
    False: {"tl": "+", "tr": "+", "bl": "+", "br": "+", "h": "-", "v": "|"},
}


def rule(label: str = "", char: str = "") -> str:
    """A horizontal rule, optionally with an inline label."""
    w = width()
    char = char or (_BOX[UNICODE]["h"])
    if not label:
        return dim(char * w)
    text = f" {label} "
    pad = max(0, w - len(text) - 3)
    return dim(char * 2) + dim(text) + dim(char * pad)


def box(lines: list[str], title: str = "") -> str:
    """Render lines inside a box, sized to content and clamped to the terminal.

    ``lines`` may contain ANSI codes; visible length is measured without them.
    """
    b = _BOX[UNICODE]
    inner = max([_visible_len(x) for x in lines] + [_visible_len(title)]) + 2
    inner = min(inner, width() - 2)

    out = [dim(b["tl"] + b["h"] * inner + b["tr"])]
    if title:
        pad = inner - _visible_len(title)
        left = pad // 2
        out.append(dim(b["v"]) + " " * left + accent(title) + " " * (pad - left) + dim(b["v"]))
    for line in lines:
        pad = max(0, inner - _visible_len(line) - 1)
        out.append(dim(b["v"]) + " " + line + " " * pad + dim(b["v"]))
    out.append(dim(b["bl"] + b["h"] * inner + b["br"]))
    return "\n".join(out)


def _visible_len(s: str) -> int:
    """Length ignoring ANSI escape sequences."""
    out, i = 0, 0
    while i < len(s):
        if s[i] == "\033":
            while i < len(s) and s[i] != "m":
                i += 1
            i += 1
        else:
            out += 1
            i += 1
    return out


def kv(key: str, value: str, key_width: int = 16) -> str:
    """An aligned ``key   value`` line."""
    return f"  {dim(key.ljust(key_width))} {value}"


# ── spinner ──────────────────────────────────────────────────────────────────

_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏" if UNICODE else "|/-\\"


class Spinner:
    """A background spinner for slow, un-instrumentable waits (model load).

    Silent when stdout is not a TTY — a progress animation in a log file is
    noise. Always use as a context manager so the line is cleared on error.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start = 0.0

    def __enter__(self) -> Spinner:
        self._start = time.perf_counter()
        if COLOUR:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        else:
            print(f"  {self.message}...", flush=True)
        return self

    def _spin(self) -> None:
        for frame in itertools.cycle(_FRAMES):
            if self._stop.is_set():
                return
            elapsed = time.perf_counter() - self._start
            sys.stdout.write(f"\r  {cyan(frame)} {self.message}  {dim(f'{elapsed:.0f}s')}   ")
            sys.stdout.flush()
            time.sleep(0.08)

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.3)
            sys.stdout.write("\r" + " " * (width()) + "\r")
            sys.stdout.flush()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._start


# ── streaming word wrap ──────────────────────────────────────────────────────

class StreamWrapper:
    """Word-wrap a token stream to the terminal width as it arrives.

    Tokens arrive as arbitrary fragments ("Hel", "lo wor", "ld"), so this
    buffers the trailing partial word and only commits it once a boundary
    shows up. Without it, generated paragraphs wrap mid-word at the terminal
    edge and long answers are genuinely hard to read.
    """

    def __init__(self, indent: str = "  ", text_width: int | None = None) -> None:
        self.indent = indent
        self.width = (text_width or width()) - len(indent)
        self._col = 0
        self._pending = ""
        self._started = False

    def feed(self, text: str) -> str:
        """Return the text to print for this fragment (may be empty)."""
        out: list[str] = []
        for ch in text:
            if ch == "\n":
                out.append(self._flush_word())
                out.append("\n" + self.indent)
                self._col = 0
            elif ch.isspace():
                out.append(self._flush_word())
                if self._col > 0:
                    out.append(ch)
                    self._col += 1
            else:
                self._pending += ch
        return "".join(out)

    def _flush_word(self) -> str:
        if not self._pending:
            return ""
        word, self._pending = self._pending, ""
        prefix = ""
        if not self._started:
            prefix, self._started = self.indent, True
        if self._col + len(word) > self.width and self._col > 0:
            prefix = "\n" + self.indent
            self._col = 0
        self._col += len(word)
        return prefix + word

    def finish(self) -> str:
        """Flush any trailing partial word."""
        return self._flush_word()


# ── message helpers ──────────────────────────────────────────────────────────

def warn(message: str) -> None:
    print(f"  {yellow('!')} {message}")


def note(message: str) -> None:
    print(f"  {dim('·')} {dim(message)}")


def error(message: str) -> None:
    print(f"  {red('✗')} {message}", file=sys.stderr)


def ok(message: str) -> None:
    print(f"  {green('✓')} {message}")
