"""Tests for swlp.core.speculative — NgramDrafter, verify_greedy, SpeculativeConfig.

Also covers build_runner() dispatch to SpeculativeRunner.
"""
from __future__ import annotations

import pytest

from swlp.config import load_config
from swlp.core.speculative import (
    NgramDrafter,
    SpeculativeConfig,
    verify_greedy,
)
from swlp.runner.base import build_runner
from swlp.runner.speculative import SpeculativeRunner

# ── NgramDrafter ──────────────────────────────────────────────────────────────

def test_drafter_proposes_repeated_ngram():
    """The last n-gram recurs earlier — its continuation should be proposed."""
    drafter = NgramDrafter(SpeculativeConfig(ngram_size=2, max_draft=3))
    # pattern [1, 2] first appears at index 0, followed by 3, 4, 5.
    tokens = [1, 2, 3, 4, 5, 9, 9, 1, 2]
    assert drafter.propose(tokens) == [3, 4, 5]


def test_drafter_returns_empty_when_no_match():
    drafter = NgramDrafter(SpeculativeConfig(ngram_size=2, max_draft=4))
    tokens = [1, 2, 3, 4, 5, 6]   # trailing [5, 6] never occurs earlier
    assert drafter.propose(tokens) == []


def test_drafter_returns_empty_when_sequence_too_short():
    drafter = NgramDrafter(SpeculativeConfig(ngram_size=3, max_draft=4))
    assert drafter.propose([1, 2, 3]) == []   # need > ngram_size tokens
    assert drafter.propose([]) == []


def test_drafter_respects_max_draft_cap():
    drafter = NgramDrafter(SpeculativeConfig(ngram_size=1, max_draft=2))
    # pattern [7] first appears at index 0, followed by 1, 2, 3, 4...
    tokens = [7, 1, 2, 3, 4, 5, 7]
    proposed = drafter.propose(tokens)
    assert proposed == [1, 2]
    assert len(proposed) <= 2


def test_drafter_max_draft_zero_returns_empty():
    drafter = NgramDrafter(SpeculativeConfig(ngram_size=2, max_draft=0))
    assert drafter.propose([1, 2, 3, 1, 2]) == []


def test_drafter_picks_most_recent_match():
    """When the pattern recurs multiple times, the most recent one wins."""
    drafter = NgramDrafter(SpeculativeConfig(ngram_size=1, max_draft=1))
    # pattern [5] occurs at indices 0 and 2; most recent earlier is index 2 -> 8.
    tokens = [5, 3, 5, 8, 5]
    assert drafter.propose(tokens) == [8]


# ── verify_greedy ─────────────────────────────────────────────────────────────

def test_verify_all_accepted_appends_bonus():
    """Every draft token matches — accept all K plus the bonus token."""
    draft = [10, 11, 12]
    picks = [10, 11, 12, 99]   # K+1 picks; last is the bonus
    new_tokens, n_accepted = verify_greedy(draft, picks)
    assert new_tokens == [10, 11, 12, 99]
    assert n_accepted == 3


def test_verify_first_token_mismatch():
    """First draft token rejected — emit only the correction."""
    draft = [10, 11, 12]
    picks = [77, 11, 12, 99]
    new_tokens, n_accepted = verify_greedy(draft, picks)
    assert new_tokens == [77]
    assert n_accepted == 0


def test_verify_middle_mismatch_partial_accept():
    """Accept the matching prefix, then the correction at the first mismatch."""
    draft = [10, 11, 12, 13]
    picks = [10, 11, 55, 13, 99]
    new_tokens, n_accepted = verify_greedy(draft, picks)
    assert new_tokens == [10, 11, 55]
    assert n_accepted == 2


def test_verify_empty_draft_emits_one_token():
    """No draft — a verification step still produces exactly one token."""
    new_tokens, n_accepted = verify_greedy([], [42])
    assert new_tokens == [42]
    assert n_accepted == 0


def test_verify_new_tokens_length_invariant():
    """len(new_tokens) == n_accepted + 1 for every outcome."""
    for draft, picks in [
        ([1, 2, 3], [1, 2, 3, 4]),
        ([1, 2, 3], [9, 2, 3, 4]),
        ([1, 2, 3], [1, 9, 3, 4]),
        ([], [7]),
    ]:
        new_tokens, n_accepted = verify_greedy(draft, picks)
        assert len(new_tokens) == n_accepted + 1


def test_verify_rejects_bad_picks_length():
    with pytest.raises(ValueError):
        verify_greedy([1, 2, 3], [1, 2, 3])   # need K+1 = 4 picks


# ── SpeculativeConfig ─────────────────────────────────────────────────────────

def test_speculative_config_defaults():
    cfg = SpeculativeConfig()
    assert cfg.ngram_size == 3
    assert cfg.max_draft == 8


def test_drafter_clamps_invalid_config():
    """ngram_size < 1 and max_draft < 0 are clamped to safe values."""
    drafter = NgramDrafter(SpeculativeConfig(ngram_size=0, max_draft=-5))
    assert drafter.ngram_size == 1
    assert drafter.max_draft == 0


# ── build_runner dispatch ─────────────────────────────────────────────────────

def test_build_runner_returns_speculative_runner():
    config = load_config(None)
    config.runtime.backend = "speculative"
    runner = build_runner(config)
    assert isinstance(runner, SpeculativeRunner)
    assert runner.backend == "speculative"
