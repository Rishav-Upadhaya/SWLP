"""Batched streaming inference for ``SWLPRunner`` (Phase 10).

``run_batch()`` processes N prompts through **one disk sweep per decode step**
instead of one sweep per prompt — FlexGen's "column-wise execution" trick. Each
streamed layer is materialized once and the whole batch passes through it, so
the disk read is amortized across all N sequences. Aggregate throughput scales
with batch size while every sequence's output stays bit-identical to a batch-1
run (greedy decode is row-independent).

This lives outside ``swlp.py`` deliberately: ``SWLPRunner.run()`` stays the
single-sequence path that ``SpeculativeRunner`` overrides via
``_generate_remaining``; batching is a separate decode strategy that reuses the
runner's setup helpers (``load``, ``_build_scheduler``, ``_run_blocks``, …)
without disturbing that override seam.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import psutil
import torch

from ..core.kv_cache import KVCacheManager
from ..metrics import RunMetrics, RunResult
from .arch import ArchAdapter, get_adapter

if TYPE_CHECKING:
    from .swlp import SWLPRunner

LOGGER = logging.getLogger(__name__)


def _select_batch(
    runner: SWLPRunner, logits: torch.Tensor, generated: torch.Tensor
) -> torch.Tensor:
    """Pick the next token for every row of a ``[N, vocab]`` logits tensor.

    Greedy decode (the default) is row-independent, so each row's argmax is
    identical to what a batch-1 run would select — this is what keeps batched
    output bit-exact. The repetition penalty, when enabled, is applied per row.
    """
    penalty = runner.config.generation.repetition_penalty
    if penalty != 1.0:
        for row in range(logits.shape[0]):
            for token_id in set(int(t) for t in generated[row].tolist()):
                score = logits[row, token_id]
                logits[row, token_id] = score * penalty if score < 0 else score / penalty
    return runner._select_next_token(logits)


def _build_batch_metrics(
    runner: SWLPRunner,
    *,
    load_seconds: float,
    generate_seconds: float,
    input_tokens: int,
    generated_tokens: int,
    aggregate_tps: float,
    peak_rss_bytes: int,
    profile: bool,
) -> RunMetrics:
    """Per-sequence metrics. ``throughput`` is the batch aggregate (the figure
    Phase 10 measures); ``generated_tokens`` is this sequence's own count."""
    assert runner.model is not None
    return RunMetrics(
        model_id=runner.config.model.model_id,
        backend=runner.backend,
        device=runner.device.type,
        load_seconds=load_seconds,
        input_tokens=input_tokens,
        output_tokens=input_tokens + generated_tokens,
        generate_seconds=generate_seconds,
        total_seconds=load_seconds + generate_seconds,
        throughput_tokens_per_second=aggregate_tps,
        generated_tokens=generated_tokens,
        ram_peak_bytes=peak_rss_bytes if profile else None,
    )


def _extend_mask(attn_mask: torch.Tensor) -> torch.Tensor:
    """Append a column of ones — the newly generated token always attends.
    Cumsum is per-row, so a frozen (finished) row never affects active rows."""
    ones = torch.ones((attn_mask.shape[0], 1), dtype=attn_mask.dtype, device=attn_mask.device)
    return torch.cat([attn_mask, ones], dim=-1)


def _decode_completion(tokenizer, row_ids: torch.Tensor, prompt_len: int) -> tuple[str, int]:
    """Decode the generated tail of one sequence; return (text, token_count).

    Left-padding puts the prompt in the leading columns, so generated tokens
    are everything from ``prompt_len`` onward. ``skip_special_tokens`` drops the
    EOS / pad tokens used to keep finished sequences in lockstep with the batch.
    """
    gen_ids = row_ids[prompt_len:].tolist()
    eos = tokenizer.eos_token_id
    count = len(gen_ids)
    if eos is not None and eos in gen_ids:
        count = gen_ids.index(eos) + 1
    text = tokenizer.decode(gen_ids[:count], skip_special_tokens=True)
    return text, count


def run_batch(
    runner: SWLPRunner, prompts: list[str], profile: bool = False
) -> list[RunResult]:
    """Run a batch of prompts through the SWLP streaming path in lockstep."""
    if not prompts:
        return []

    torch.manual_seed(runner.config.generation.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(runner.config.generation.seed)

    memory_tracker = psutil.Process()
    peak_rss = 0
    runner._check_streaming_feasible()

    scheduler = None
    try:
        load_seconds = runner.load()
        peak_rss = max(peak_rss, memory_tracker.memory_info().rss)
        model, tokenizer = runner.model, runner.tokenizer
        assert model is not None and tokenizer is not None

        adapter: ArchAdapter = get_adapter(getattr(model.config, "model_type", ""))
        for module in adapter.device_modules(model):
            module.to(runner.device)
        blocks = adapter.get_blocks(model)
        scheduler = runner._build_scheduler(blocks)
        if hasattr(scheduler, "load_resident_layers"):
            scheduler.load_resident_layers()

        try:
            runner.kv_manager = KVCacheManager(
                budget_bytes=runner._resolve_kv_budget_bytes(adapter, len(blocks)),
                compression=bool(runner.config.runtime.kv_compression),
                compression_level=runner._resolve_compression_level(),
                tiering=bool(runner.config.runtime.kv_tiering),
                device=runner.device,
            )
        except Exception:
            LOGGER.exception("kv_manager_init_failed")
            runner.kv_manager = KVCacheManager()

        # ── batched, left-padded tokenization ──────────────────────────────
        # Left padding keeps every sequence's newest token in the last column,
        # so a single `generated[:, -1:]` slice feeds the decode step.
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        prev_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        finally:
            tokenizer.padding_side = prev_side
        input_ids = encoded["input_ids"].to(runner.device)
        attn_mask = encoded["attention_mask"].to(runner.device)
        n_seq, prompt_len = input_ids.shape
        real_prompt_tokens = [int(m.sum()) for m in attn_mask]

        past_state = runner._make_past_state(adapter, len(blocks))
        max_new = max(int(runner.config.generation.max_new_tokens), 0)
        eos_id = tokenizer.eos_token_id
        pad_id = int(tokenizer.pad_token_id)

        gen_start = time.perf_counter()
        with torch.no_grad():
            ctx = adapter.prepare_step(model, input_ids, past_state, runner.device, 0, attn_mask)
            ctx.hidden_states = runner._run_blocks(adapter, ctx, scheduler)
            hidden = adapter.final_norm(model, ctx.hidden_states)
            generated = input_ids
            finished = torch.zeros(n_seq, dtype=torch.bool, device=runner.device)

            if max_new > 0:
                logits = model.lm_head(hidden)[:, -1, :]
                next_tok = _select_batch(runner, logits, generated)
                generated = torch.cat([generated, next_tok], dim=-1)
                attn_mask = _extend_mask(attn_mask)
                if eos_id is not None:
                    finished |= next_tok.squeeze(-1) == eos_id

                for _ in range(max_new - 1):
                    if bool(finished.all()):
                        break
                    step = adapter.prepare_step(
                        model, generated[:, -1:], past_state, runner.device, 0, attn_mask
                    )
                    step.hidden_states = runner._run_blocks(adapter, step, scheduler)
                    hidden = adapter.final_norm(model, step.hidden_states)
                    logits = model.lm_head(hidden)[:, -1, :]
                    next_tok = _select_batch(runner, logits, generated)
                    # Freeze finished sequences with pad so the batch stays in lockstep.
                    next_tok = torch.where(
                        finished.unsqueeze(-1), torch.full_like(next_tok, pad_id), next_tok
                    )
                    generated = torch.cat([generated, next_tok], dim=-1)
                    attn_mask = _extend_mask(attn_mask)
                    if eos_id is not None:
                        finished |= next_tok.squeeze(-1) == eos_id

        generate_seconds = time.perf_counter() - gen_start
        peak_rss = max(peak_rss, memory_tracker.memory_info().rss)

        completions: list[tuple[str, int]] = [
            _decode_completion(tokenizer, generated[i], prompt_len) for i in range(n_seq)
        ]
        total_generated = sum(count for _, count in completions)
        aggregate_tps = total_generated / generate_seconds if generate_seconds > 0 else 0.0

        results: list[RunResult] = []
        for i, (text, count) in enumerate(completions):
            metrics = _build_batch_metrics(
                runner,
                load_seconds=load_seconds,
                generate_seconds=generate_seconds,
                input_tokens=real_prompt_tokens[i],
                generated_tokens=count,
                aggregate_tps=aggregate_tps,
                peak_rss_bytes=peak_rss,
                profile=profile,
            )
            results.append(RunResult(prompt=prompts[i], completion=text.lstrip(), metrics=metrics))

        LOGGER.info(
            "swlp_batch_finished",
            extra={
                "batch_size": n_seq,
                "total_generated_tokens": total_generated,
                "aggregate_tps": round(aggregate_tps, 3),
                "generate_seconds": round(generate_seconds, 2),
            },
        )
        return results

    finally:
        try:
            runner._cleanup_resources(scheduler)
        except Exception:
            LOGGER.exception("batch_cleanup_failed")
