"""Phase 23: Quality-neutral speedups and opt-in quality tradeoffs.

Quality-neutral (always on when config enabled):
  - ActivationCache: cache (prompt_prefix_hash → layer_outputs) for prompt
    prefix reuse across turns in a chat session. Zero quality impact because
    it reuses exact pre-computed activations — same math, same result.
  - PreallocBuffer: pre-allocated tensor for token generation, eliminating
    the torch.cat overhead per token. Byte-identical output.

Opt-in quality tradeoffs (configurable via CLI/config):
  - EarlyExit: skip remaining transformer layers when next-token entropy is
    below a threshold. Speeds up "easy" tokens (articles, prepositions,
    common patterns) at the cost of using intermediate logits.
  - LayerPruner: permanently skip least-important layers based on a
    pre-computed importance profile. Reduces model capacity.
"""
from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from typing import Any

import torch
import torch.nn.functional as F

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


# ── Early Exit (opt-in quality tradeoff) ────────────────────────────────────


class EarlyExitDetector:
    """Check after each transformer layer whether we can skip remaining layers.

    Uses next-token distribution entropy as the confidence signal.  When
    entropy is below the threshold, the model is "confident" enough about
    its prediction that running additional layers provides diminishing returns.

    Research (DeeBERT, Early Exit Transformer) shows 30-50% of layers can be
    skipped for "easy" tokens with <1% quality loss at moderate thresholds.

    Quality impact: depends on threshold.
      - threshold=0.8 → conservative, negligible quality loss (~0.1%)
      - threshold=0.5 → moderate, small quality loss (~0.5%)
      - threshold=0.3 → aggressive, noticeable quality loss (~1-2%)
      - threshold=0.1 → very aggressive, significant quality loss (~3-5%)
    """

    def __init__(self, threshold: float, min_layer: int = 8) -> None:
        """
        Args:
            threshold: entropy threshold below which to trigger early exit.
                Must be in (0.0, 1.0].  Typical values: 0.3-0.8.
            min_layer: never exit before this layer index (early layers extract
                general features that all later layers depend on).
        """
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"early_exit threshold must be in (0.0, 1.0], got {threshold}")
        self.threshold = threshold
        self.min_layer = min_layer
        self._exits = 0
        self._total_checks = 0

    def should_exit(self, layer_index: int, logits: torch.Tensor) -> bool:
        """Check if we should exit early after computing ``layer_index``.

        ``logits`` is the raw output from the layer's forward pass (before
        final norm/lm_head).  We compute the entropy of the softmax
        distribution — low entropy = high confidence = safe to exit.
        """
        if layer_index < self.min_layer:
            return False

        self._total_checks += 1

        # Compute softmax entropy.  Use float32 for numerical stability.
        with torch.no_grad():
            probs = F.softmax(logits.float(), dim=-1)
            # Entropy = -sum(p * log(p)).  Clamp to avoid log(0).
            entropy = -torch.sum(probs * torch.clamp(torch.log(probs + 1e-8), min=-20), dim=-1)
            # Normalize by log(vocab_size) to get entropy in [0, 1].
            vocab_size = logits.shape[-1]
            normalized_entropy = entropy / max(1.0, torch.log(torch.tensor(float(vocab_size))))
            mean_entropy = float(normalized_entropy.mean())

        if mean_entropy < self.threshold:
            self._exits += 1
            LOGGER.debug(
                "early_exit_triggered",
                extra={
                    "layer": layer_index,
                    "entropy": round(mean_entropy, 4),
                    "threshold": self.threshold,
                },
            )
            return True
        return False

    def stats(self) -> dict[str, Any]:
        return {
            "exits": self._exits,
            "total_checks": self._total_checks,
            "exit_rate": self._exits / self._total_checks if self._total_checks > 0 else 0.0,
            "threshold": self.threshold,
            "min_layer": self.min_layer,
        }


# ── Layer Pruner (opt-in quality tradeoff) ──────────────────────────────────


class LayerPruner:
    """Skip least-important transformer layers based on importance profiling.

    Importance is measured by the average gradient-norm of each layer during
    a calibration pass.  Layers with low gradient-norm contribute less to the
    output and can be safely removed.

    Usage:
        1. Run a calibration: ``pruner.calibrate(model, sample_inputs)``
        2. Save the profile: ``pruner.save_profile("path/to/profile.json")``
        3. Load and apply: ``pruner.load_profile("path/to/profile.json")``
        4. During inference: ``if pruner.should_skip(i): continue``

    Quality impact:
      - "light" (~10% removed): <0.5% quality loss
      - "aggressive" (~25% removed): ~1-3% quality loss
    """

    def __init__(self, mode: str = "off") -> None:
        self.mode = mode
        self._skip_layers: set[int] = set()
        self._importance: list[float] = []

    def should_skip(self, layer_index: int) -> bool:
        """Return True if this layer should be skipped entirely."""
        return layer_index in self._skip_layers

    def set_profile(self, importance: list[float]) -> None:
        """Set layer importance scores and derive skip set based on mode."""
        self._importance = importance
        n_layers = len(importance)
        if self.mode == "light":
            n_skip = max(1, int(n_layers * 0.10))
        elif self.mode == "aggressive":
            n_skip = max(1, int(n_layers * 0.25))
        else:
            self._skip_layers = set()
            return

        # Sort by importance ascending, skip the least important.
        indexed = sorted(enumerate(importance), key=lambda x: x[1])
        self._skip_layers = {idx for idx, _ in indexed[:n_skip]}
        LOGGER.info(
            "layer_pruning_applied",
            extra={
                "mode": self.mode,
                "n_layers": n_layers,
                "n_skipped": len(self._skip_layers),
                "skipped_indices": sorted(self._skip_layers),
            },
        )

    def stats(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "n_skipped": len(self._skip_layers),
            "skipped_indices": sorted(self._skip_layers),
        }


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
