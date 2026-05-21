"""Speculative-decoding runner for SWLP (Phase 5).

``SpeculativeRunner`` is a ``SWLPRunner`` that replaces the
one-token-per-disk-sweep decode loop with **prompt-lookup speculative
decoding**: an n-gram drafter proposes up to K continuation tokens, and the
streamed target model verifies all K in a single 32-layer disk sweep. Accepted
tokens are amortised over that one sweep, so throughput rises with the
acceptance rate.

Output is **bit-identical to greedy SWLP** — every proposed token is greedily
verified by the target model; speculation changes throughput only, never
correctness.

KV cache: this path uses a plain ``DynamicCache`` whose ``crop()`` rolls the
cache back when draft tokens are rejected. KV compression (Phase 2) is mutually
exclusive with speculative decoding — see ``docs/phase5_design_decisions.md``.
"""
from __future__ import annotations

import logging

import torch

from ..core.speculative import NgramDrafter, SpeculativeConfig, verify_greedy
from .arch import ArchAdapter
from .swlp import SWLPRunner

LOGGER = logging.getLogger(__name__)


class SpeculativeRunner(SWLPRunner):
    backend = "speculative"

    def _make_past_state(self, adapter: ArchAdapter, num_layers: int):
        """Force a plain, rollback-capable KV cache.

        Speculative verification crops the KV cache when draft tokens are
        rejected; that needs ``DynamicCache.crop()``. The Phase 2
        ``CompressedDynamicCache`` cannot be cheaply rolled back on rejection,
        so the speculative path always uses the adapter's plain past state.

        Phase 14 — mutual-exclusion warning: if the caller also set
        ``kv_compression=True``, log a warning so they know compression is
        silently bypassed on this path rather than experiencing silent behaviour
        differences.
        """
        if self.config.runtime.kv_compression:
            LOGGER.warning(
                "speculative_kv_compression_ignored",
                extra={
                    "reason": (
                        "CompressedDynamicCache cannot be cheaply rolled back "
                        "during speculative verification; KV compression is "
                        "disabled for this run. Use --backend swlp to enable it."
                    )
                },
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
    ) -> tuple[list[int], int]:
        """Run one speculative verification sweep.

        Feeds ``[last_token, *draft]`` through the streamed target model in a
        single disk sweep, greedily verifies the draft, crops the KV cache to
        the accepted prefix, and returns ``(new_tokens, n_accepted)``.
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
                ctx.past_state.crop(keep_len)
            except Exception:
                LOGGER.exception(
                    "speculative_cache_crop_failed",
                    extra={"keep_len": keep_len, "cache_len": cache_len_after},
                )
        return new_tokens, n_accepted

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
        drafter = NgramDrafter(spec_cfg)
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

                new_tokens, n_accepted = self._verify_step(
                    adapter, scheduler, ctx, generated, draft
                )
                steps += 1
                total_draft += len(draft)
                total_accepted += n_accepted

                stop = False
                if eos is not None and eos in new_tokens:
                    new_tokens = new_tokens[: new_tokens.index(eos) + 1]
                    stop = True
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
                "ngram_size": spec_cfg.ngram_size,
                "max_draft": spec_cfg.max_draft,
            },
        )
        return generated
