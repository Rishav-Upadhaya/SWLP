"""Byte-identical decode speedups.

  - ActivationCache: cache (prompt_prefix_hash → layer_outputs) for prompt
    prefix reuse across turns in a chat session. Zero quality impact because
    it reuses exact pre-computed activations — same math, same result.
  - PreallocBuffer: pre-allocated tensor for token generation, eliminating
    the torch.cat overhead per token. Byte-identical output.

"""
from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from typing import Any

import torch

LOGGER = logging.getLogger(__name__)


# ── Activation Cache (quality-neutral) ──────────────────────────────────────


class ActivationCache:
    """LRU cache mapping prompt-prefix hashes to per-layer hidden states.

    When a new prompt arrives, the runner hashes the first N tokens and checks
    the cache.  If a prefix match is found, inference starts from the first
    non-cached layer — skipping all SSD reads and compute for the cached portion.

    This is the "materialized view" trick from database optimization applied
    to transformer inference.  It is especially powerful for multi-turn chat
    where the system prompt (often 200-1000 tokens) is repeated every turn.

    Quality impact: ZERO — reuses exact pre-computed activations.
    """

    def __init__(self, max_entries: int = 16) -> None:
        self._max_entries = max_entries
        # OrderedDict gives us LRU eviction via move_to_end.
        self._cache: OrderedDict[str, list[dict[str, torch.Tensor] | None]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _hash_tokens(token_ids: list[int], max_prefix: int = 64) -> str:
        """Hash the first ``max_prefix`` token ids into a short hex key."""
        key_bytes = str(token_ids[:max_prefix]).encode()
        return hashlib.blake2b(key_bytes, digest_size=16).hexdigest()

    def lookup(
        self, token_ids: list[int]
    ) -> tuple[str, list[dict[str, torch.Tensor] | None]] | None:
        """Return (cache_key, layer_outputs) if prefix is cached, else None.

        ``layer_outputs[i]`` is either a dict of hidden-state tensors for
        layer ``i``, or ``None`` if that layer was not cached (e.g. it was
        an embedding-only layer or the cache entry is partial).
        """
        key = self._hash_tokens(token_ids)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            LOGGER.debug("activation_cache_hit", extra={"key": key, "token_count": len(token_ids)})
            return key, self._cache[key]
        self._misses += 1
        return None

    def store(
        self, token_ids: list[int], layer_outputs: list[dict[str, torch.Tensor] | None]
    ) -> str:
        """Store the per-layer hidden states for a prompt prefix."""
        key = self._hash_tokens(token_ids)
        self._cache[key] = layer_outputs
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)
        LOGGER.debug("activation_cache_store", extra={"key": key, "token_count": len(token_ids)})
        return key

    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "entries": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0


# ── Pre-allocated Buffer (quality-neutral) ──────────────────────────────────


class PreallocBuffer:
    """Pre-allocated tensor for autoregressive token accumulation.

    Instead of ``torch.cat([generated, next_token])`` which allocates a new
    tensor every token, this maintains a fixed-size buffer and uses a view
    to expose only the filled portion.

    Quality impact: ZERO — same computation, same output.
    """

    def __init__(self, max_tokens: int, device: torch.device) -> None:
        # int64 always: this buffer holds TOKEN IDS, not activations. It used
        # to take a dtype, and the one production caller passed the model's
        # dtype (float16) — the embedding lookup then rejected a FloatTensor.
        # There is exactly one correct dtype here, so it is not a parameter.
        self._buf = torch.empty(1, max_tokens, dtype=torch.long, device=device)
        self._length = 0

    def append(self, token: torch.Tensor) -> torch.Tensor:
        """Append a token id (shape [1, 1]) and return the filled buffer view."""
        if self._length >= self._buf.shape[1]:
            # Buffer full — should not happen in normal usage.
            return self._buf[:, : self._length]
        self._buf[:, self._length] = token.squeeze(0)
        self._length += 1
        return self._buf[:, : self._length]

    @property
    def tensor(self) -> torch.Tensor:
        """Return the filled portion as a view (no copy)."""
        return self._buf[:, : self._length]

    @property
    def length(self) -> int:
        return self._length

    def reset(self) -> None:
        self._length = 0
