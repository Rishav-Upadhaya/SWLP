"""Speculative-decoding runner for SWLP (Phases 5, 21, 29).

``SpeculativeRunner`` replaces the one-token-per-disk-sweep decode loop: a
drafter proposes up to K tokens and the streamed target verifies all K in one
disk sweep, so throughput rises with the acceptance rate. Drafters, by config:
``swlp_mtp`` → ``MtpDrafter`` (the checkpoint's own MTP head, ``mtp.py``);
``swlp_draft_model`` → ``DraftModelDrafter`` (small resident same-tokenizer
model); otherwise ``NgramDrafter`` (prompt lookup, zero extra RAM).

Output is **bit-identical to greedy SWLP** — every drafted token is greedily
verified; speculation changes throughput only. Rejected tokens are rolled back
with ``DynamicCache.crop()``, or for hybrid Gated-DeltaNet caches by exact
state recomputation (``hybrid_rollback.py``). KV compression (Phase 2) is
mutually exclusive with this path: its compressed cache cannot be cheaply rolled back.
"""
from __future__ import annotations

import logging
import time
from contextlib import nullcontext

import torch

from ..config import AppConfig
from ..core.speculative import NgramDrafter, SpeculativeConfig, verify_greedy
from .arch import ArchAdapter
from .draft import (
    AdaptiveDraftLength,
    DraftModelDrafter,
    ensure_same_tokenizer,
    load_draft_model,
)
from .hybrid_rollback import is_hybrid_cache, record_linear_attention, rollback_hybrid_cache
from .mtp import MtpDrafter, MtpHead, load_mtp_head
from .swlp import SWLPRunner

LOGGER = logging.getLogger(__name__)


class SpeculativeRunner(SWLPRunner):
    backend = "speculative"

    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)
        self._draft_model: torch.nn.Module | None = None
        self._mtp_head: MtpHead | None = None

    def load(self) -> float:
        """Load the target, then the resident drafter weights (when configured).

        Drafter weights load here so their cost lands in load time, not in the
        generation timing — keeps tok/s comparable with the other backends.
        """
        elapsed = super().load()
        started = time.perf_counter()
        if self.config.runtime.swlp_mtp:
            self._load_mtp_head()
        elif self.config.runtime.swlp_draft_model.strip():
            self._load_draft_model()
        return elapsed + time.perf_counter() - started

    def _build_drafter(self) -> NgramDrafter | DraftModelDrafter | MtpDrafter:
        """Pick the drafter: MTP head, resident draft model, else n-gram.

        Drafter weights are loaded once and kept resident across generations
        (chat turns reuse them); the drafter wrapper itself is per-generation
        state (``DraftModelDrafter``'s KV cache self-heals via common-prefix
        crop, so a fresh wrapper just costs one cheap drafter prefill).
        """
        spec_cfg = self._spec_config()
        draft_id = self.config.runtime.swlp_draft_model.strip()
        if self.config.runtime.swlp_mtp:
            if draft_id:
                raise ValueError(
                    "swlp_mtp (--mtp) and swlp_draft_model (--draft-model) are "
                    "mutually exclusive — pick one drafter"
                )
            return MtpDrafter(self._load_mtp_head(), self.model, self.device, spec_cfg.max_draft)
        if not draft_id:
            return NgramDrafter(spec_cfg)
        return DraftModelDrafter(self._load_draft_model(), self.device, spec_cfg.max_draft)

    def _load_draft_model(self) -> torch.nn.Module:
        """The resident draft model, loaded once and tokenizer-checked."""
        if self._draft_model is None:
            model, draft_tokenizer = load_draft_model(
                self.config.runtime.swlp_draft_model.strip(),
                self.config.cache.cache_dir,
                self.dtype,
                self.device,
                self.config.model.trust_remote_code,
            )
            ensure_same_tokenizer(self.tokenizer, draft_tokenizer)
            self._draft_model = model
        return self._draft_model

    def _load_mtp_head(self) -> MtpHead:
        """The target's MTP head, loaded once from the shard dir and kept resident."""
        if self._mtp_head is None:
            assert self.model is not None
            shard_dir = self.config.runtime.shard_dir
            if shard_dir is None:
                raise ValueError("swlp_mtp needs a shard dir holding mtp.safetensors")
            self._mtp_head = load_mtp_head(
                shard_dir, self.model.config.get_text_config(), self.dtype, self.device
            )
        return self._mtp_head

    def _make_past_state(self, adapter: ArchAdapter, num_layers: int):
        """Force a plain, rollback-capable cache (the adapter's past state).

        The Phase 2 ``CompressedDynamicCache`` cannot be cheaply rolled back on
        rejection, so ``kv_compression`` is bypassed here — loudly (Phase 14).
        """
        if self.config.runtime.kv_compression:
            LOGGER.warning(
                "speculative_kv_compression_ignored",
                extra={"reason": "CompressedDynamicCache cannot be rolled back during "
                       "speculative verification; use --backend swlp for KV compression."},
            )
        return adapter.init_past_state(self.model, num_layers)

    def _spec_config(self) -> SpeculativeConfig:
        rt = self.config.runtime
        return SpeculativeConfig(
            ngram_size=max(1, int(rt.swlp_spec_ngram)),
            max_draft=max(0, int(rt.swlp_spec_max_draft)),
        )

    def _verify_step(
        self,
        adapter: ArchAdapter,
        scheduler,
        ctx,
        generated: torch.Tensor,
        draft: list[int],
    ) -> tuple[list[int], int, torch.Tensor]:
        """Run one speculative verification sweep.

        Feeds ``[last_token, *draft]`` through the streamed target model in a
        single disk sweep, greedily verifies the draft, rolls the cache back to
        the accepted prefix, and returns ``(new_tokens, n_accepted, hidden)``
        where ``hidden`` is the post-final-norm target state at the
        ``1 + n_accepted`` kept positions (the MTP drafter's input).

        Hybrid (Gated-DeltaNet) caches cannot be cropped: their recurrent state
        is recorded during the sweep and recomputed for the kept prefix.
        """
        assert self.model is not None
        device = self.device
        last_token = generated[:, -1:]
        if draft:
            draft_tensor = torch.tensor(
                [draft], device=device, dtype=generated.dtype
            )
            verify_input = torch.cat([last_token, draft_tensor], dim=-1)
        else:
            verify_input = last_token

        # KV cache holds every token except the most recent one (fed now).
        cache_len_before = int(generated.shape[-1]) - 1
        step_ctx = adapter.prepare_step(
            self.model, verify_input, ctx.past_state, device, cache_len_before
        )
        recording = (
            record_linear_attention(self.model)
            if draft and is_hybrid_cache(ctx.past_state)
            else nullcontext()
        )
        with recording as record:
            step_ctx.hidden_states = self._run_blocks(adapter, step_ctx, scheduler)
        hidden = adapter.final_norm(self.model, step_ctx.hidden_states)
        logits = self.model.lm_head(hidden)  # [1, K+1, vocab]

        # Greedy pick at each verified position, using the optimistic drafted
        # prefix so the per-position selection (and any repetition penalty)
        # matches what plain greedy SWLP would compute had the draft held.
        target_picks: list[int] = []
        for i in range(int(logits.shape[1])):
            prefix = generated
            if i > 0:
                prefix = torch.cat(
                    [
                        generated,
                        torch.tensor(
                            [draft[:i]], device=device, dtype=generated.dtype
                        ),
                    ],
                    dim=-1,
                )
            pick = self._select_next(logits[:, i, :], prefix)
            target_picks.append(int(pick.item()))

        new_tokens, n_accepted = verify_greedy(draft, target_picks)

        # Roll the KV cache back to the confirmed prefix:
        #   keep = (tokens before last) + last_token + n_accepted draft tokens.
        keep_len = cache_len_before + 1 + n_accepted
        cache_len_after = cache_len_before + int(verify_input.shape[-1])
        if keep_len < cache_len_after:
            try:
                if record is not None:
                    rollback_hybrid_cache(
                        ctx.past_state, record, keep=1 + n_accepted,
                        fed=int(verify_input.shape[-1]),
                    )
                else:
                    ctx.past_state.crop(keep_len)
            except Exception as exc:
                self.degrade(
                    f"speculative_cache_rollback_failed(keep={keep_len}, "
                    f"cache={cache_len_after}): {exc}",
                    exc,
                )
        return new_tokens, n_accepted, hidden[:, : 1 + n_accepted]

    def _generate_remaining(
        self,
        adapter: ArchAdapter,
        scheduler,
        ctx,
        generated: torch.Tensor,
    ) -> torch.Tensor:
        """Speculative decode loop — one disk sweep verifies up to K tokens."""
        assert self.model is not None
        spec_cfg = self._spec_config()
        drafter = self._build_drafter()
        if isinstance(drafter, MtpDrafter):
            # Prefill states pair with the token that follows each position.
            # Known limit: chunked prefill only leaves the last chunk's states
            # here, so the MTP head sees a shorter history (affects acceptance only).
            prefill = adapter.final_norm(self.model, ctx.hidden_states)
            drafter.extend(prefill, generated[0, -int(prefill.shape[1]):].tolist())
        max_new = int(self.config.generation.max_new_tokens)
        eos = (
            int(self.tokenizer.eos_token_id)
            if self.tokenizer is not None and self.tokenizer.eos_token_id is not None
            else None
        )
        # On entry, exactly one token has been generated after the prompt.
        prompt_len = int(generated.shape[-1]) - 1

        steps = 0
        total_draft = 0
        total_accepted = 0

        with torch.no_grad():
            while int(generated.shape[-1]) - prompt_len < max_new:
                remaining = max_new - (int(generated.shape[-1]) - prompt_len)
                draft = drafter.propose(generated[0].tolist())
                # A step always emits one target token, so cap the draft at
                # remaining - 1 to never overshoot max_new_tokens.
                draft = draft[: max(0, remaining - 1)]

                new_tokens, n_accepted, kept_hidden = self._verify_step(
                    adapter, scheduler, ctx, generated, draft
                )
                steps += 1
                total_draft += len(draft)
                total_accepted += n_accepted
                if isinstance(drafter, AdaptiveDraftLength):
                    drafter.observe(len(draft), n_accepted)
                if isinstance(drafter, MtpDrafter):
                    drafter.extend(kept_hidden, new_tokens)

                stop = False
                if eos is not None and eos in new_tokens:
                    new_tokens = new_tokens[: new_tokens.index(eos) + 1]
                    stop = True
                self._emit_tokens(new_tokens)  # stream every accepted token
                new_tensor = torch.tensor(
                    [new_tokens], device=self.device, dtype=generated.dtype
                )
                generated = torch.cat([generated, new_tensor], dim=-1)
                if stop:
                    break

        accept_rate = (total_accepted / total_draft) if total_draft else 0.0
        tokens_generated = int(generated.shape[-1]) - prompt_len
        tokens_per_sweep = (tokens_generated / steps) if steps else 0.0
        LOGGER.info(
            "speculative_stats",
            extra={
                "speculative_steps": steps,
                "draft_tokens_proposed": total_draft,
                "draft_tokens_accepted": total_accepted,
                "acceptance_rate": round(accept_rate, 4),
                "tokens_per_sweep": round(tokens_per_sweep, 3),
                "drafter": type(drafter).__name__,
                "draft_model": self.config.runtime.swlp_draft_model or None,
                "mtp": self.config.runtime.swlp_mtp,
                "ngram_size": spec_cfg.ngram_size,
                "max_draft": spec_cfg.max_draft,
            },
        )
        return generated
