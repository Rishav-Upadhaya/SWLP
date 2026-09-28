"""Tests for swlp.tui — terminal presentation primitives.

The load-bearing piece is StreamWrapper: tokens arrive as arbitrary fragments
("Hel", "lo wor", "ld"), so wrapping has to buffer partial words. Getting it
wrong either breaks words mid-character or drops the final token — both of
which are silent and only visible to a human reading output.
"""
from __future__ import annotations

from swlp import tui


# ── StreamWrapper ────────────────────────────────────────────────────────────

def _run(fragments: list[str], text_width: int, indent: str = "") -> str:
    w = tui.StreamWrapper(indent=indent, text_width=text_width)
    out = "".join(w.feed(f) for f in fragments)
    return out + w.finish()


def test_wrapper_preserves_every_character_across_fragment_splits() -> None:
    """Whatever the fragmentation, no text may be lost."""
    text = "the quick brown fox jumps over the lazy dog"
    whole = _run([text], text_width=80)
    char_by_char = _run(list(text), text_width=80)
    odd_splits = _run(["the qui", "ck bro", "wn fox jum", "ps over the lazy d", "og"], 80)

    assert whole.split() == text.split()
    assert char_by_char.split() == text.split()
    assert odd_splits.split() == text.split()


def test_wrapper_never_breaks_a_word() -> None:
    out = _run(["antidisestablishmentarianism is a long word"], text_width=20)
    for line in out.split("\n"):
        assert " " not in line.strip() or all(
            len(word) <= 20 or " " not in word for word in [line.strip()]
        )
    assert "antidisestablishmentarianism" in out   # intact, not split


def test_wrapper_wraps_at_the_given_width() -> None:
    words = " ".join(["word"] * 40)
    out = _run([words], text_width=30)
    assert "\n" in out
    for line in out.split("\n"):
        assert len(line) <= 30, f"line too long: {line!r}"


def test_wrapper_flushes_the_trailing_partial_word() -> None:
    """A stream ending mid-word must still emit it — finish() is not optional."""
    w = tui.StreamWrapper(text_width=80)
    streamed = w.feed("hello wor")
    assert "wor" not in streamed          # buffered, boundary not seen yet
    assert w.finish() .strip() == "wor"   # released on finish


def test_wrapper_honours_explicit_newlines() -> None:
    out = _run(["line one\nline two"], text_width=80)
    assert out.count("\n") == 1


def test_wrapper_applies_the_indent() -> None:
    out = _run(["alpha beta"], text_width=80, indent="    ")
    assert out.startswith("    ")


# ── layout helpers ───────────────────────────────────────────────────────────

def test_visible_len_ignores_ansi() -> None:
    assert tui._visible_len("plain") == 5
    assert tui._visible_len("\033[1mbold\033[0m") == 4
    assert tui._visible_len("\033[1;36maccent\033[0m") == 6


def test_box_is_rectangular() -> None:
    """Every row the same width, or the borders visibly stagger."""
    rendered = tui.box(["short", "a much longer line here"], title="Title")
    rows = rendered.split("\n")
    widths = {tui._visible_len(r) for r in rows}
    assert len(widths) == 1, f"ragged box: {widths}"


def test_box_survives_ansi_in_content() -> None:
    rendered = tui.box([tui.bold("bold"), tui.dim("dim")], title="T")
    widths = {tui._visible_len(r) for r in rendered.split("\n")}
    assert len(widths) == 1


def test_width_is_clamped_to_something_usable() -> None:
    w = tui.width()
    assert 40 <= w <= 100


def test_kv_aligns() -> None:
    a = tui.kv("key", "value", 10)
    b = tui.kv("longerkey", "value", 10)
    assert a.index("value") == b.index("value")


def test_spinner_is_a_noop_without_a_tty(capsys) -> None:
    """Under pytest stdout is captured, so COLOUR is False and the spinner
    must not emit escape codes — a progress animation in a log file is noise."""
    with tui.Spinner("loading") as sp:
        pass
    out = capsys.readouterr().out
    assert "\033[" not in out
    assert sp.elapsed >= 0
