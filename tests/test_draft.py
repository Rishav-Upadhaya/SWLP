"""Tests for swlp.runner.draft — DraftModelDrafter, ensure_same_tokenizer,
and the Phase 21 config/CLI wiring.

Uses a tiny randomly-initialized Qwen2 model (no downloads). The key property
under test: drafting through the persistent KV cache (with common-prefix crop
self-healing) produces exactly the same tokens as a fresh no-cache greedy loop.
"""
from __future__ import annotations

import argparse

import pytest
import torch

from swlp.config import load_config
from swlp.runner.draft import DraftModelDrafter, ensure_same_tokenizer

_VOCAB = 64


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(7)
    cfg = Qwen2Config(
        vocab_size=_VOCAB,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    model = Qwen2ForCausalLM(cfg)
    model.eval()
    return model


def _greedy_reference(model, tokens: list[int], k: int) -> list[int]:
    """No-cache argmax continuation — ground truth for the cached drafter."""
    ids = list(tokens)
    with torch.no_grad():
        for _ in range(k):
            logits = model(input_ids=torch.tensor([ids], dtype=torch.long)).logits
            ids.append(int(logits[0, -1, :].argmax().item()))
    return ids[len(tokens):]


# ── DraftModelDrafter ─────────────────────────────────────────────────────────


def test_propose_matches_nocache_reference(tiny_model) -> None:
    drafter = DraftModelDrafter(tiny_model, torch.device("cpu"), max_draft=4)
    tokens = [1, 2, 3, 4, 5]
    assert drafter.propose(tokens) == _greedy_reference(tiny_model, tokens, 4)


def test_propose_returns_exactly_k_tokens(tiny_model) -> None:
    drafter = DraftModelDrafter(tiny_model, torch.device("cpu"), max_draft=6)
    assert len(drafter.propose([3, 1, 4])) == 6


def test_max_draft_zero_returns_empty(tiny_model) -> None:
    drafter = DraftModelDrafter(tiny_model, torch.device("cpu"), max_draft=0)
    assert drafter.propose([1, 2, 3]) == []


def test_empty_tokens_returns_empty(tiny_model) -> None:
    drafter = DraftModelDrafter(tiny_model, torch.device("cpu"), max_draft=4)
    assert drafter.propose([]) == []


def test_cache_reuse_after_partial_accept(tiny_model) -> None:
    """Rejected drafts are cropped; the next proposal matches a fresh drafter."""
    drafter = DraftModelDrafter(tiny_model, torch.device("cpu"), max_draft=4)
    tokens = [1, 2, 3, 4, 5]
    draft = drafter.propose(tokens)
    # Simulate verification: 2 accepted, then a correction token that differs
    # from the rejected third draft token.
    correction = (draft[2] + 1) % _VOCAB
    confirmed = [*tokens, *draft[:2], correction]
    assert drafter.propose(confirmed) == _greedy_reference(tiny_model, confirmed, 4)


def test_cache_reuse_after_full_accept(tiny_model) -> None:
    """All drafts accepted + bonus token; the unfed last draft token is refed."""
    drafter = DraftModelDrafter(tiny_model, torch.device("cpu"), max_draft=3)
    tokens = [9, 8, 7]
    # The greedy continuation is what a greedy target would accept in full;
    # token k+1 plays the bonus token.
    continuation = _greedy_reference(tiny_model, tokens, 4)
    drafter.propose(tokens)
    confirmed = [*tokens, *continuation]
    assert drafter.propose(confirmed) == _greedy_reference(tiny_model, confirmed, 3)


def test_prefix_divergence_resets_cache(tiny_model) -> None:
    """A completely different sequence (new generation) self-heals via reset."""
    drafter = DraftModelDrafter(tiny_model, torch.device("cpu"), max_draft=4)
    drafter.propose([1, 2, 3, 4, 5])
    other = [40, 41, 42]
    assert drafter.propose(other) == _greedy_reference(tiny_model, other, 4)


def test_repeated_propose_is_deterministic(tiny_model) -> None:
    """Proposing twice for the same confirmed sequence yields identical drafts."""
    drafter = DraftModelDrafter(tiny_model, torch.device("cpu"), max_draft=4)
    tokens = [5, 6, 7, 8]
    assert drafter.propose(tokens) == drafter.propose(tokens)


# ── adaptive draft length (AIMD) ──────────────────────────────────────────────


def test_observe_poor_acceptance_halves_k(tiny_model) -> None:
    drafter = DraftModelDrafter(tiny_model, torch.device("cpu"), max_draft=8)
    drafter.observe(proposed=8, accepted=0)
    assert drafter.current_draft == 4
    drafter.observe(proposed=4, accepted=1)  # 1/4 < half → halve again
    assert drafter.current_draft == 2


def test_observe_floor_is_one(tiny_model) -> None:
    drafter = DraftModelDrafter(tiny_model, torch.device("cpu"), max_draft=8)
    for _ in range(6):
        drafter.observe(proposed=drafter.current_draft, accepted=0)
    assert drafter.current_draft == 1


def test_observe_full_acceptance_doubles_back_to_max(tiny_model) -> None:
    drafter = DraftModelDrafter(tiny_model, torch.device("cpu"), max_draft=8)
    for _ in range(5):
        drafter.observe(proposed=drafter.current_draft, accepted=0)
    assert drafter.current_draft == 1
    drafter.observe(proposed=1, accepted=1)   # full accept → 2
    drafter.observe(proposed=2, accepted=2)   # → 4
    drafter.observe(proposed=4, accepted=4)   # → 8 (capped at max)
    drafter.observe(proposed=8, accepted=8)
    assert drafter.current_draft == 8


def test_observe_majority_acceptance_increments(tiny_model) -> None:
    drafter = DraftModelDrafter(tiny_model, torch.device("cpu"), max_draft=8)
    drafter.observe(proposed=8, accepted=0)   # → 4
    drafter.observe(proposed=4, accepted=2)   # half accepted → +1
    assert drafter.current_draft == 5


def test_observe_zero_proposed_is_noop(tiny_model) -> None:
    drafter = DraftModelDrafter(tiny_model, torch.device("cpu"), max_draft=8)
    drafter.observe(proposed=0, accepted=0)
    assert drafter.current_draft == 8


def test_propose_respects_shrunken_k(tiny_model) -> None:
    drafter = DraftModelDrafter(tiny_model, torch.device("cpu"), max_draft=8)
    drafter.observe(proposed=8, accepted=0)
    drafter.observe(proposed=4, accepted=0)
    assert drafter.current_draft == 2
    assert len(drafter.propose([1, 2, 3])) == 2


# ── ensure_same_tokenizer ─────────────────────────────────────────────────────


class _StubTokenizer:
    def __init__(self, vocab: dict[str, int]) -> None:
        self._vocab = vocab

    def get_vocab(self) -> dict[str, int]:
        return self._vocab


def test_same_vocab_passes() -> None:
    vocab = {"a": 0, "b": 1}
    ensure_same_tokenizer(_StubTokenizer(vocab), _StubTokenizer(dict(vocab)))


def test_different_vocab_raises() -> None:
    with pytest.raises(RuntimeError, match="tokenizer"):
        ensure_same_tokenizer(
            _StubTokenizer({"a": 0, "b": 1}), _StubTokenizer({"a": 0, "c": 1})
        )


# ── config + CLI wiring ───────────────────────────────────────────────────────


def test_draft_model_config_default_empty() -> None:
    config = load_config(None)
    assert config.runtime.swlp_draft_model == ""


def test_draft_model_env_override(monkeypatch) -> None:
    monkeypatch.setenv("SWLP_DRAFT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    config = load_config(None)
    assert config.runtime.swlp_draft_model == "Qwen/Qwen2.5-0.5B-Instruct"


def _namespace(**overrides) -> argparse.Namespace:
    base = {"backend": None, "quant": None, "shard_dir": None, "draft_model": None}
    base.update(overrides)
    return argparse.Namespace(**base)
