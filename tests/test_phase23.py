"""Unit tests for the Phase 23 components (core/phase23.py).

Round-2 gap fix: EarlyExitDetector, LayerPruner, PreallocBuffer, and
ActivationCache previously had zero direct test coverage.
"""
from __future__ import annotations

import torch

from swlp.core.phase23 import (
    ActivationCache,
    EarlyExitDetector,
    LayerPruner,
    PreallocBuffer,
)


def _peaked_logits(vocab: int, peak: int) -> torch.Tensor:
    """Logits with a dominant token — low normalized entropy."""
    logits = torch.full((1, 1, vocab), -10.0)
    logits[0, 0, peak] = 10.0
    return logits


def test_early_exit_low_entropy_exits_after_min_layer() -> None:
    det = EarlyExitDetector(threshold=0.5, min_layer=8)
    logits = _peaked_logits(100, 7)
    for layer in range(8):
        assert not det.should_exit(layer, logits), f"exited before min_layer ({layer})"
    assert det.should_exit(8, logits), "low-entropy logits should exit at min_layer"


def test_early_exit_high_entropy_never_exits() -> None:
    det = EarlyExitDetector(threshold=0.5, min_layer=0)
    logits = torch.zeros(1, 1, 100)  # uniform → maximal entropy
    assert not any(det.should_exit(lay, logits) for lay in range(30))


def test_layer_pruner_light_skips_ten_percent() -> None:
    pruner = LayerPruner(mode="light")
    importance = [1.0] * 90 + [0.0] * 10  # last 10 layers least important
    pruner.set_profile(importance)
    skipped = [i for i in range(100) if pruner.should_skip(i)]
    assert len(skipped) == 10
    assert set(skipped) == set(range(90, 100))


def test_layer_pruner_aggressive_and_off() -> None:
    pruner = LayerPruner(mode="aggressive")
    pruner.set_profile([0.0] * 25 + [1.0] * 75)
    assert sum(1 for i in range(100) if pruner.should_skip(i)) == 25
    assert not any(LayerPruner(mode="off").should_skip(i) for i in range(10))


def test_prealloc_buffer_append_and_view() -> None:
    buf = PreallocBuffer(max_tokens=8, device=torch.device("cpu"))
    tok = torch.tensor([[5]], dtype=torch.long)
    out = buf.append(tok)
    assert out.shape == (1, 1)
    assert int(out[0, -1]) == 5
    # The exposed tensor is the FILLED view, not the whole buffer.
    assert buf.tensor.shape == (1, 1)
    buf.append(torch.tensor([[6]], dtype=torch.long))
    assert buf.tensor.shape == (1, 2)
    assert buf.tensor[0].tolist() == [5, 6]


def test_activation_cache_roundtrip_and_eviction() -> None:
    cache = ActivationCache(max_entries=2)
    ids_a = [1, 2, 3, 4]
    states: list[dict[str, torch.Tensor] | None] = [None, None]
    key = cache.store(ids_a, states)
    got = cache.lookup(ids_a)
    assert got is not None
    assert got[0] == key
    assert got[1] is states
    assert cache.lookup([9, 9]) is None
    cache.store([5, 6], states)
    cache.store([7, 8], states)  # evicts the LRU entry ([1,2,3,4])
    assert cache.lookup(ids_a) is None
    assert cache.lookup([7, 8]) is not None
