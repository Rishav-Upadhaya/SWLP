"""Tests for core/decode_cache.py (ActivationCache, PreallocBuffer)."""
from __future__ import annotations

import torch

from swlp.core.decode_cache import ActivationCache, PreallocBuffer


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
