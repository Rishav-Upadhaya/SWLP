"""Draft-model speculative decoding (Phase 21).

``DraftModelDrafter`` wraps a small **resident** causal LM (e.g.
Qwen2.5-0.5B-Instruct, ~1 GB FP16) that proposes up to K greedy continuation
tokens per sweep. The streamed target model then verifies all K in a single
disk sweep via ``runner/speculative.py`` — exactly the Phase 5 machinery, with
this drafter replacing prompt-lookup n-gram matching.

Why a draft model: ``NgramDrafter`` only fires when the trailing n-gram recurs
in the history, so it gets ~0% acceptance on *novel* text. A small same-family
model drafts useful tokens on every step, typically 60–80% acceptance against
a greedy target — the per-sweep disk cost of the big model is then amortised
over several tokens on all workloads, not just repetitive ones.

Constraints:
- The draft model must share the target model's tokenizer **exactly**
  (``ensure_same_tokenizer``) — verification compares token IDs.
- Output remains byte-identical to plain greedy SWLP: every drafted token is
  still verified by the target; drafting changes throughput only.

The drafter keeps its own ``DynamicCache`` across sweeps and self-heals on
rejection: ``propose()`` crops the cache back to the longest common prefix of
what it has cached and the confirmed sequence, so the caller never manages
drafter state.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch

LOGGER = logging.getLogger(__name__)


def ensure_same_tokenizer(target_tokenizer: Any, draft_tokenizer: Any) -> None:
    """Raise unless the draft tokenizer's vocab is identical to the target's.

    Speculative verification compares raw token IDs, so any vocab divergence
    silently destroys acceptance (or worse, corrupts output framing). A full
    vocab-dict comparison is a one-time O(vocab) check at load.
    """
    if target_tokenizer.get_vocab() != draft_tokenizer.get_vocab():
        raise RuntimeError(
            "Draft model tokenizer does not match the target model tokenizer. "
            "Draft-model speculative decoding requires an identical vocabulary "
            "(same model family, e.g. Qwen2.5-0.5B drafting for Qwen2.5-14B). "
            "Pick a draft model from the target's family or unset "
            "swlp_draft_model / SWLP_DRAFT_MODEL."
        )


def load_draft_model(
    model_id: str,
    cache_dir: Path,
    dtype: torch.dtype,
    device: torch.device,
    trust_remote_code: bool = False,
) -> tuple[Any, Any]:
    """Load the small draft model fully resident on ``device``.

    Returns ``(model, tokenizer)``. The tokenizer is returned so the caller can
    run ``ensure_same_tokenizer`` against the target's tokenizer.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    start = time.perf_counter()
    LOGGER.info(
        "draft_model_loading",
        extra={"draft_model": model_id, "device": device.type, "dtype": str(dtype)},
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=str(cache_dir), trust_remote_code=trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=str(cache_dir),
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)
    LOGGER.info(
        "draft_model_loaded",
        extra={
            "draft_model": model_id,
            "load_seconds": round(time.perf_counter() - start, 2),
            "parameters_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        },
    )
    return model, tokenizer


class DraftModelDrafter:
    """Greedy autoregressive token proposer backed by a resident draft model.

    Same ``propose(tokens) -> list[int]`` interface as ``NgramDrafter``. Drafts
    are raw argmax picks — draft choices only affect the acceptance rate, never
    correctness, since the target verifies every token.

    Adaptive draft length: the caller reports each sweep's outcome via
    ``observe(proposed, accepted)`` and the drafter adjusts K multiplicatively
    (AIMD) — full acceptance doubles K, majority acceptance bumps it by one,
    poor acceptance halves it (floor 1). On low-agreement text the per-sweep
    drafting overhead therefore collapses to ~2 small-model forwards instead
    of ``max_draft + 1``, capping the worst-case regression vs plain decoding,
    while predictable text recovers the full draft depth within a few sweeps.
    """

    def __init__(self, model: Any, device: torch.device, max_draft: int) -> None:
        self._model = model
        self._device = device
        self._max_k = max(0, int(max_draft))
        self._k = self._max_k
        self._cache: Any | None = None
        # Token IDs whose KV the cache currently covers (in order).
        self._cached_ids: list[int] = []

    @property
    def max_draft(self) -> int:
        return self._max_k

    @property
    def current_draft(self) -> int:
        return self._k

    def observe(self, proposed: int, accepted: int) -> None:
        """Adapt the draft length to the last sweep's acceptance (AIMD)."""
        if proposed <= 0 or self._max_k == 0:
            return
        if accepted >= proposed:
            self._k = min(self._k * 2, self._max_k)
        elif accepted * 2 >= proposed:
            self._k = min(self._k + 1, self._max_k)
        else:
            self._k = max(1, self._k // 2)

    def propose(self, tokens: list[int]) -> list[int]:
        """Return up to K greedy draft tokens continuing ``tokens``.

        Args:
            tokens: all confirmed token IDs so far (prompt + completion).
        """
        if self._k == 0 or not tokens:
            return []
        self._sync_cache(tokens)
        drafted: list[int] = []
        with torch.no_grad():
            delta = tokens[len(self._cached_ids):]
            input_ids = torch.tensor([delta], device=self._device, dtype=torch.long)
            for _ in range(self._k):
                out = self._model(
                    input_ids=input_ids, past_key_values=self._cache, use_cache=True
                )
                self._cache = out.past_key_values
                next_id = int(out.logits[0, -1, :].argmax().item())
                drafted.append(next_id)
                input_ids = torch.tensor([[next_id]], device=self._device, dtype=torch.long)
        # The final drafted token was never fed forward, so the cache covers
        # everything confirmed plus all drafts except the last.
        self._cached_ids = [*tokens, *drafted[:-1]]
        return drafted

    def _sync_cache(self, tokens: list[int]) -> None:
        """Crop the cache to the longest common prefix with ``tokens``.

        Capped at ``len(tokens) - 1`` so the next forward always feeds at least
        the final confirmed token (whose logits pick the first draft).
        """
        limit = min(len(self._cached_ids), len(tokens) - 1)
        prefix = 0
        while prefix < limit and self._cached_ids[prefix] == tokens[prefix]:
            prefix += 1
        if prefix == len(self._cached_ids):
            return  # cache is already a strict prefix — nothing to crop
        if prefix == 0:
            self._cache = None
            self._cached_ids = []
            return
        try:
            self._cache.crop(prefix)
            self._cached_ids = self._cached_ids[:prefix]
        except Exception:
            LOGGER.exception("draft_cache_crop_failed", extra={"keep_len": prefix})
            self._cache = None
            self._cached_ids = []
