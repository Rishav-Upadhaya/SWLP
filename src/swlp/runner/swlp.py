from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import psutil
import torch

from ..config import AppConfig
from ..core.compressed_cache import CompressedDynamicCache
from ..core.decode_cache import ActivationCache, PreallocBuffer
from ..core.kv_cache import KVCacheManager
from ..core.pipeline_model import pipeline_ratio_from_metrics
from ..core.profiler import LayerProfiler
from ..core.scheduler import PrefetchError, SchedulerConfig, ThreadedScheduler
from ..core.streaming import StreamingScheduler, has_shards
from ..metrics import RunResult
from .arch import ArchAdapter, LlamaLikeAdapter, get_adapter
from .hf import HuggingFaceRunner
from .load import load_from_shards, load_full_model
from .swlp_setup import SWLPSetupMixin

LOGGER = logging.getLogger(__name__)

# Tokens of trailing prompt context kept as the incremental-detokenizer anchor.
# The anchor preserves SentencePiece boundary behaviour (▁word → " word") and
# multi-byte merges across the prompt/completion border; 8 tokens is far more
# context than any practical tokenizer needs to detokenize locally.
_DETOK_ANCHOR_TOKENS = 8


def _process_memory_bytes(proc: psutil.Process) -> int:
    """RSS plus MPS driver memory. On macOS, Metal allocations (streamed layer
    weights, lm_head) are not in RSS — RSS alone under-reported a 10 GB
    process as 2 GB while it was swapping."""
    rss = int(proc.memory_info().rss)
    if torch.backends.mps.is_available():
        rss += int(torch.mps.driver_allocated_memory())
    return rss


class SWLPRunner(SWLPSetupMixin, HuggingFaceRunner):
    backend = "swlp"

    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)
        # Byte-identical decode speedups (initialized on first run()).
        self._activation_cache: ActivationCache | None = None
        self._prealloc_buf: PreallocBuffer | None = None
        # Self-calibration: measured ratio from a previous run in this process.
        self._measured_pipeline_ratio: float | None = None
        self._last_residency_decision: dict | None = None
        # Optional prefix KV cache (set by the chat REPL).
        self._prefix_cache = None
        # MoE expert scheduler (built in run() when the model is MoE).
        self._expert_sched = None
        # Per-run scratch state. Declared here, never conditionally created, so
        # a typo raises AttributeError instead of silently disabling a feature.
        self._trace: list[dict] = []
        self._token_callback: Callable[[str], None] | None = None
        self._stream_ids: list[int] | None = None
        self._stream_prev_text: str = ""
        self._profile_prints: dict | None = None
        self._trace_output: str | None = None
        self._last_trace_path: str | None = None

    def set_prefix_cache(self, cache) -> None:
        """Enable prefix KV reuse across run() calls (chat sessions)."""
        self._prefix_cache = cache

    def _build_scheduler(self, blocks):
        sched_config = SchedulerConfig(
            window_size=max(1, self.config.runtime.swlp_window_size),
            prefetch_depth=max(1, self.config.runtime.swlp_prefetch_depth),
            prefetch=self.config.runtime.swlp_prefetch,
        )
        shard_dir = self.config.runtime.shard_dir
        if shard_dir is not None and has_shards(shard_dir):
            LOGGER.info("swlp_streaming_from_shards", extra={"shard_dir": str(shard_dir)})
            resident_count = self._resolve_resident_count(len(blocks), shard_dir)
            volumes = [
                Path(v.strip())
                for v in str(self.config.runtime.swlp_shard_volumes).split(",")
                if v.strip()
            ]
            if volumes:
                LOGGER.info(
                    "swlp_shard_volumes", extra={"volumes": [str(v) for v in volumes]}
                )
            sched = StreamingScheduler(
                blocks, self.device, sched_config, shard_dir,
                resident_count=resident_count,
                direct_io=self._resolve_direct_io(shard_dir),
                extra_volumes=volumes,
            )
            sched.profiler = LayerProfiler(enabled=True)
            return sched
        return ThreadedScheduler(blocks, self.device, sched_config)

    def _cleanup_resources(self, scheduler=None) -> None:
        try:
            if scheduler is not None:
                try:
                    scheduler.cleanup()
                except Exception:
                    LOGGER.exception("scheduler_cleanup_failed")
            expert_sched = self._expert_sched
            if expert_sched is not None:
                try:
                    LOGGER.info("expert_cache_stats", extra=expert_sched.stats())
                except Exception:
                    LOGGER.exception("expert_stats_failed")
                try:
                    expert_sched.cleanup()
                except Exception:
                    LOGGER.exception("expert_scheduler_cleanup_failed")
                self._expert_sched = None
            if hasattr(self, "kv_manager") and self.kv_manager is not None:
                try:
                    self.kv_manager.clear()
                except Exception:
                    LOGGER.exception("kv_manager_clear_failed")
            self.model = None
            # Release the Metal buffer cache: on unified memory the cache
            # competes with the next run's weights for the same physical RAM.
            try:
                if hasattr(torch, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception as exc:
                self.degrade(f"mps_empty_cache_failed: {exc}", exc)
        except Exception:
            LOGGER.exception("cleanup_resources_failed")

    def load(self) -> float:
        # Idempotent: if the model is already in memory (e.g. run_chat pre-loaded it
        # before the REPL starts) skip the reload so successive run() / stream_tokens()
        # calls don't double-load — a 14 GB full-model reload while the first copy is
        # still resident causes an immediate OOM on 16 GB machines.
        if self.model is not None:
            return 0.0
        started = time.perf_counter()
        shard_dir = self.config.runtime.shard_dir
        # Auto-shard on first run if shard_dir doesn't have shards yet.
        if shard_dir is not None:
            self._auto_shard_if_needed(Path(shard_dir))
        if shard_dir is not None and has_shards(shard_dir):
            self.model, self.tokenizer = load_from_shards(self, Path(shard_dir))
        else:
            self.model, self.tokenizer = load_full_model(self)
        self._trace: list[dict] = []
        return time.perf_counter() - started

    def _resolve_compression_level(self) -> int:
        """zlib level: 0 means 'store only'. If compression is on but level is
        unset (0), fall back to 6 — the standard balanced default."""
        level = int(self.config.runtime.kv_compression_level)
        if self.config.runtime.kv_compression and level <= 0:
            return 6
        return level

    def _resolve_kv_budget_bytes(self, adapter: ArchAdapter, num_layers: int) -> int:
        """Resolve the KV memory budget. A configured value of <= 0 triggers
        auto-calculation from detected hardware and the streaming window."""
        configured_mb = int(self.config.runtime.kv_memory_budget_mb)
        if configured_mb > 0:
            return configured_mb * 1024 * 1024
        from ..hardware.detect import detect_hardware, kv_budget_recommendation

        layer_weight_mb = 0.0
        shard_dir = self.config.runtime.shard_dir
        if shard_dir is not None and has_shards(shard_dir):
            from ..model.shard import load_manifest

            layer_weight_mb = load_manifest(Path(shard_dir)).layer_weight_mb
        budget_mb = kv_budget_recommendation(
            detect_hardware(),
            window_size=max(1, self.config.runtime.swlp_window_size),
            layer_weight_mb=layer_weight_mb,
            num_layers=num_layers,
        )
        LOGGER.info("swlp_kv_budget_auto", extra={"budget_mb": budget_mb})
        return budget_mb * 1024 * 1024

    def build_kv_manager(self, adapter: ArchAdapter, num_layers: int) -> KVCacheManager:
        """Construct the run's KVCacheManager from config — the single place.

        Both the single-sequence path (``run``/``stream_tokens``) and the
        batched path (``runner/batch.py``) call this, so a new KV setting can
        never reach one and silently miss the other. On failure a default
        manager is returned and every discarded setting is named in the log.
        """
        try:
            return KVCacheManager(
                budget_bytes=self._resolve_kv_budget_bytes(adapter, num_layers),
                compression=bool(self.config.runtime.kv_compression),
                compression_level=self._resolve_compression_level(),
                tiering=bool(self.config.runtime.kv_tiering),
                device=self.device,
                kv_window=max(0, int(self.config.runtime.kv_window)),
                kv_quant=str(self.config.runtime.kv_quant),
                disk_dir=self.config.runtime.kv_disk_dir,
            )
        except Exception:
            LOGGER.exception("kv_manager_init_failed")
            # Loud fallback: a default KVCacheManager silently discards the
            # configured budget/compression/tiering — say exactly what was lost.
            LOGGER.warning(
                "kv_manager_fallback_defaults",
                extra={
                    "discarded_budget_mb": int(self.config.runtime.kv_memory_budget_mb),
                    "discarded_compression": bool(self.config.runtime.kv_compression),
                    "discarded_tiering": bool(self.config.runtime.kv_tiering),
                    "discarded_kv_quant": str(self.config.runtime.kv_quant),
                    "discarded_kv_window": int(self.config.runtime.kv_window),
                    "discarded_disk_dir": str(self.config.runtime.kv_disk_dir or ""),
                },
            )
            self.degradations.append("kv_manager_fallback_defaults")
            return KVCacheManager()

    def _make_past_state(self, adapter: ArchAdapter, num_layers: int):
        """Build the per-run KV cache. Llama-family + kv_compression -> a
        CompressedDynamicCache; otherwise the adapter's default past state."""
        if isinstance(adapter, LlamaLikeAdapter) and self.config.runtime.kv_compression:
            assert self.model is not None
            LOGGER.info("swlp_kv_compressed_cache_enabled")
            return CompressedDynamicCache(self.model.config, self.kv_manager)
        return adapter.init_past_state(self.model, num_layers)

    def _run_blocks(
        self,
        adapter: ArchAdapter,
        ctx,
        scheduler,
        token_index: int = -1,
    ) -> torch.Tensor:
        assert self.model is not None
        blocks = adapter.get_blocks(self.model)
        window = scheduler.config.window_size
        # prefetch_depth extends the lookahead beyond the window when the disk
        # can serve more parallel reads than compute consumes; at most
        # ``lookahead`` device-ready layers are ever in flight, so RAM stays
        # bounded at (lookahead + 1) x layer size.
        lookahead = max(window, scheduler.config.prefetch_depth)

        prof = getattr(scheduler, "profiler", None)
        if prof and token_index >= 0:
            prof.begin_token(token_index)

        # ── warmup: kick off parallel SSD reads for the first L layers ───────
        # Fires the worker pool on layers 0..L-1 simultaneously before compute
        # begins.  Without this, layer 0 is always a cold synchronous read
        # (~63 ms on M5 NVMe) while later layers prefetch one-at-a-time inside
        # the loop.
        for warm_idx in range(min(lookahead, len(blocks))):
            try:
                scheduler.prefetch(warm_idx)
            except PrefetchError as exc:
                self.degrade(f"warmup_prefetch_failed(layer={warm_idx}): {exc}", exc)
                try:
                    scheduler.disable_prefetch()
                except Exception as inner:
                    self.degrade(f"disable_prefetch_failed: {inner}", inner)
                break

        for layer_index, block in enumerate(blocks):
            trace_entry: dict = {"layer": layer_index}

            # MoE layers — prefetch experts predicted for this layer.
            expert_sched = self._expert_sched
            if expert_sched is not None:
                try:
                    expert_sched.prepare_layer(layer_index)
                except Exception as exc:
                    self.degrade(f"expert_prepare_failed(layer={layer_index}): {exc}", exc)

            # ── prefetch: keep the next L layers in flight ───────────────────
            try:
                trace_entry["prefetch_start"] = time.perf_counter()
                for ahead in range(1, lookahead + 1):
                    scheduler.prefetch(layer_index + ahead)
                trace_entry["prefetch_enqueued"] = time.perf_counter()
            except PrefetchError as exc:
                self.degrade(f"prefetch_failed(layer={layer_index}): {exc}", exc)
                try:
                    scheduler.disable_prefetch()
                except Exception as inner:
                    self.degrade(f"disable_prefetch_failed: {inner}", inner)

            try:
                block = scheduler.ensure(layer_index)
            except PrefetchError as exc:
                self.degrade(f"ensure_failed(layer={layer_index}): {exc}", exc)
                try:
                    scheduler.disable_prefetch()
                except Exception as inner:
                    self.degrade(f"disable_prefetch_failed: {inner}", inner)
                block = scheduler.ensure(layer_index)

            trace_entry["compute_start"] = time.perf_counter()
            if prof:
                prof.begin_compute(layer_index)

            hidden_states, _ = adapter.call_block(block, ctx, layer_index)
            ctx.hidden_states = hidden_states
            trace_entry["compute_end"] = time.perf_counter()
            if prof:
                prof.end_compute(layer_index)

            # ── evict: free the layer immediately after compute ───────────────
            scheduler.evict(layer_index)
            trace_entry["evict_time"] = time.perf_counter()
            self._trace.append(trace_entry)

            # Compressed-cache path: compress this layer's KV immediately
            # (DynamicCache subclasses mutate in place; GPT-2 and plain
            # Llama-like caches need no per-layer handling here).
            if isinstance(ctx.past_state, CompressedDynamicCache):
                try:
                    ctx.past_state.compress_layer(layer_index)
                except Exception as exc:
                    self.degrade(f"kv_compress_failed(layer={layer_index}): {exc}", exc)

        if prof and token_index >= 0:
            prof.end_token()

        return ctx.hidden_states

    def _select_next(self, logits: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
        logits = self._apply_repetition_penalty(
            logits, generated, self.config.generation.repetition_penalty
        )
        return self._select_next_token(logits)

    def _generate_remaining(
        self,
        adapter: ArchAdapter,
        scheduler,
        ctx,
        generated: torch.Tensor,
    ) -> torch.Tensor:
        """Autoregressive decode loop after the first token.

        One disk sweep (``_run_blocks``) per token. Overridable so alternative
        decoding strategies (e.g. ``SpeculativeRunner``) can replace just the
        loop while reusing all of ``run()``'s setup, metrics, and teardown.

        On entry ``generated`` holds the prompt plus exactly one generated token,
        and ``ctx.past_state`` is the KV cache populated by the prefill sweep.
        """
        assert self.model is not None
        # Use pre-allocated buffer when enabled (avoids torch.cat per token).
        prealloc: PreallocBuffer | None = self._prealloc_buf
        if prealloc is not None:
            # Copy prompt+first token into the pre-allocated buffer.
            seq_len = generated.shape[-1]
            prealloc._buf[:, :seq_len] = generated
            prealloc._length = seq_len
        _token_counter = 1  # prefill was token 0
        for _ in range(max(self.config.generation.max_new_tokens - 1, 0)):
            # Use prealloc buffer view if available, else raw tensor.
            gen_view = prealloc.tensor if prealloc is not None else generated
            position_offset = int(gen_view.shape[-1] - 1)
            ctx = adapter.prepare_step(
                self.model,
                gen_view[:, -1:],
                ctx.past_state,
                self.device,
                position_offset,
            )
            ctx.hidden_states = self._run_blocks(
                adapter,
                ctx,
                scheduler,
                token_index=_token_counter,
            )
            _token_counter += 1
            hidden_states = adapter.final_norm(self.model, ctx.hidden_states)
            logits = self.model.lm_head(hidden_states)[:, -1, :]
            next_token = self._select_next(logits, gen_view)
            # Append to pre-allocated buffer or fall back to torch.cat.
            if prealloc is not None:
                generated = prealloc.append(next_token)
            else:
                generated = torch.cat([generated, next_token], dim=-1)
            next_id = int(next_token.item())
            self._emit_tokens([next_id])
            if self.tokenizer is not None and self.tokenizer.eos_token_id is not None:
                if next_id == int(self.tokenizer.eos_token_id):
                    break
        return generated

    def _emit_tokens(self, ids: list[int]) -> None:
        """Stream newly generated ids to ``stream_tokens`` (no-op otherwise).

        Incremental detokenization: decode a short prompt anchor plus the
        generated ids and diff against the previous decode — the anchor keeps
        SentencePiece space-prefixed tokens right at the prompt boundary. Shared
        by the one-token loop and the speculative loop (several ids per sweep).
        """
        cb = self._token_callback
        if cb is None or self.tokenizer is None or self._stream_ids is None:
            return
        self._stream_ids.extend(ids)
        text = self.tokenizer.decode(self._stream_ids, skip_special_tokens=True)
        delta = text[len(self._stream_prev_text):]
        if delta:
            cb(delta)
        self._stream_prev_text = text

    def stream_tokens(self, prompt: str, max_tokens: int = 512) -> Iterator[str]:
        """Yield decoded tokens one at a time, streaming through the sliding window.

        Bridges the callback-based ``_token_callback`` hook in ``_generate_remaining``
        to a Python generator via a ``queue.Queue`` + daemon thread, so the full
        ``run()`` setup/teardown path is reused without duplication.
        """
        token_queue: queue.Queue[str | None] = queue.Queue()

        def _on_token(text: str) -> None:
            token_queue.put(text)

        original_max = self.config.generation.max_new_tokens
        self.config.generation.max_new_tokens = max_tokens

        def _run() -> None:
            self._token_callback: Callable[[str], None] | None = _on_token
            try:
                self.run(prompt)
            finally:
                self._token_callback = None
                self.config.generation.max_new_tokens = original_max
                token_queue.put(None)  # sentinel — generation finished

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        while True:
            item = token_queue.get()
            if item is None:
                break
            yield item
        t.join()

    def run_batch(self, prompts: list[str], profile: bool = False) -> list[RunResult]:
        """Run a batch of prompts in lockstep — one disk sweep per decode step
        amortized across all sequences (Phase 10). See ``runner/batch.py``."""
        from .batch import run_batch as _run_batch

        return _run_batch(self, prompts, profile=profile)

    def run(self, prompt: str, profile: bool = False) -> RunResult:
        self.degradations.clear()  # per-run ledger, not per-process
        torch.manual_seed(self.config.generation.seed)

        memory_tracker = psutil.Process()
        peak_rss_bytes = 0

        # Pre-flight: refuse cleanly if the model cannot fit even with streaming,
        # rather than crashing partway through inference.
        self._check_streaming_feasible()

        scheduler = None
        try:
            load_seconds = self.load()
            peak_rss_bytes = max(peak_rss_bytes, _process_memory_bytes(memory_tracker))

            assert self.model is not None
            assert self.tokenizer is not None

            model_type = getattr(self.model.config, "model_type", "")
            adapter = get_adapter(model_type)
            LOGGER.info(
                "swlp_adapter_selected",
                extra={"model_type": model_type, "adapter": adapter.name},
            )

            for module in adapter.device_modules(self.model):
                module.to(self.device)

            blocks = adapter.get_blocks(self.model)
            scheduler = self._build_scheduler(blocks)

            # ── initialize quality-neutral speedups ────────────────
            rt = self.config.runtime
            # Activation cache: reused across calls in chat mode, so build it
            # once and keep it. The guard is `is None`, not `not hasattr` —
            # __init__ declares the attribute, so a hasattr guard here is
            # always False and the cache would never be built at all.
            if rt.swlp_activation_cache and self._activation_cache is None:
                self._activation_cache = ActivationCache(
                    max_entries=rt.swlp_activation_cache_max_entries
                )
            elif not rt.swlp_activation_cache:
                self._activation_cache = None
            # Pre-allocated buffer: avoids a torch.cat per token.
            if rt.swlp_prealloc_buffer:
                if self._prealloc_buf is None:
                    # Max tokens: generate budget + a generous margin.
                    max_tok = self.config.generation.max_new_tokens + 2048
                    self._prealloc_buf = PreallocBuffer(
                        max_tokens=max_tok,
                        device=self.device,
                    )
            else:
                self._prealloc_buf = None
            # Pre-load resident layers once before inference starts.
            if hasattr(scheduler, "load_resident_layers"):
                scheduler.load_resident_layers()

            # Set up profiler hardware metadata.
            prof = getattr(scheduler, "profiler", None)
            if prof is not None:
                from ..core.profiler import collect_hardware_metadata
                hw = collect_hardware_metadata(
                    device_type=self.device.type,
                    model_id=self.config.model.model_id,
                    num_layers=len(blocks),
                    model_size_mb=0.0,
                    quantization="",
                    window_size=scheduler.config.window_size,
                    prefetch_depth=scheduler.config.prefetch_depth,
                    worker_count=getattr(scheduler, "_pool", None)
                        and scheduler._pool._max_workers
                        or 0,
                    direct_io=getattr(scheduler, "_direct_io", False),
                    resident_count=getattr(scheduler, "_resident_count", 0),
                )
                prof.set_hardware(hw)

            self.kv_manager = self.build_kv_manager(adapter, len(blocks))

            preprocess_start = time.perf_counter()
            encoded = self.tokenizer(prompt, return_tensors="pt")
            input_ids = encoded["input_ids"].to(self.device)
            preprocess_seconds = time.perf_counter() - preprocess_start
            peak_rss_bytes = max(peak_rss_bytes, _process_memory_bytes(memory_tracker))
            prompt_tokens = int(input_ids.shape[-1])

            past_state = self._make_past_state(adapter, len(blocks))

            # Prefix KV reuse — seed the cache with the longest
            # cached prefix and feed only the suffix. Lossless: identical
            # prefixes produce bitwise-identical KV under greedy decode.
            prefix_hit = None
            prefix_cache = self._prefix_cache
            # Full prompt ids, captured BEFORE any prefix slicing: the store
            # below must label snapshots with the complete sequence, or a
            # post-hit snapshot's ids would cover only its KV suffix.
            full_prompt_ids = input_ids[0].tolist()
            if (
                prefix_cache is not None
                and self.backend == "swlp"
                and isinstance(adapter, LlamaLikeAdapter)
            ):
                from transformers.cache_utils import DynamicCache as _DC

                # Exact-DynamicCache only; the speculative path stays out of
                # scope v1 by wiring (chat attaches the cache only for the
                # plain swlp backend), not by this type check alone.
                if type(past_state) is _DC:
                    hit = prefix_cache.lookup(
                        full_prompt_ids, min_length=16, max_length=prompt_tokens
                    )
                    if hit is not None:
                        prefix_cache.seed_dynamic_cache(past_state, hit, self.device)
                        prefix_hit = hit
                        input_ids = input_ids[:, hit.length:]
                        LOGGER.info(
                            "prefix_cache_hit",
                            extra={"prefix_tokens": hit.length,
                                   "suffix_tokens": prompt_tokens - hit.length},
                        )

            generation_start = time.perf_counter()
            first_token_start = None
            first_token_end = None

            with torch.no_grad():
                base_offset = prefix_hit.length if prefix_hit is not None else 0
                chunk = max(0, int(self.config.runtime.swlp_prefill_chunk))
                seq_len = int(input_ids.shape[-1])
                if chunk > 0 and seq_len > chunk:
                    # Chunked prefill: sweep the prompt in slices,
                    # KV accumulating across chunks. Lossless — causal
                    # attention over the prefix cache; bounds activation RAM.
                    offset = base_offset
                    for start in range(0, seq_len, chunk):
                        piece = input_ids[:, start:start + chunk]
                        ctx = adapter.prepare_step(
                            self.model, piece, past_state, self.device, offset
                        )
                        ctx.hidden_states = self._run_blocks(
                            adapter, ctx, scheduler, token_index=0
                        )
                        offset += int(piece.shape[-1])
                else:
                    ctx = adapter.prepare_step(
                        self.model, input_ids, past_state, self.device, base_offset
                    )
                    ctx.hidden_states = self._run_blocks(
                        adapter, ctx, scheduler, token_index=0
                    )
                hidden_states = adapter.final_norm(self.model, ctx.hidden_states)
                # First_token_start marks the end of the prefill sweep
                # (all input tokens have been processed).  Everything before this
                # point is prefill; everything after is argmax + decode.
                first_token_start = time.perf_counter()
                generated = input_ids
                if self.config.generation.max_new_tokens > 0:
                    logits = self.model.lm_head(hidden_states)[:, -1, :]
                    next_token = self._select_next(logits, input_ids)
                    first_token_end = time.perf_counter()
                    generated = torch.cat([input_ids, next_token], dim=-1)
                    # Emit first generated token via streaming callback.
                    # Must happen before _generate_remaining so token #1 is not
                    # silently dropped from the stream.  Also initialise
                    # _stream_prev_text so _generate_remaining's incremental
                    # decoder starts from the right baseline.
                    _stream_cb: Callable[[str], None] | None = getattr(
                        self, "_token_callback", None
                    )
                    if _stream_cb is not None and self.tokenizer is not None:
                        _baseline = self.tokenizer.decode(
                            input_ids[0].tolist(), skip_special_tokens=True
                        )
                        _after_first = self.tokenizer.decode(
                            generated[0].tolist(), skip_special_tokens=True
                        )
                        _first_delta = _after_first[len(_baseline):]
                        if _first_delta:
                            _stream_cb(_first_delta)
                        # Seed the incremental detokenizer: trailing prompt
                        # anchor + the first generated id (see
                        # _generate_remaining for the diffing contract).
                        _anchor = input_ids[0, -_DETOK_ANCHOR_TOKENS:].tolist()
                        self._stream_ids: list[int] | None = [
                            *_anchor,
                            int(next_token.item()),
                        ]
                        self._stream_prev_text: str = self.tokenizer.decode(
                            self._stream_ids, skip_special_tokens=True
                        )
                    else:
                        self._stream_ids = None
                        self._stream_prev_text = ""
                    generated = self._generate_remaining(adapter, scheduler, ctx, generated)
            peak_rss_bytes = max(peak_rss_bytes, _process_memory_bytes(memory_tracker))

            # Persist this run's KV as a reusable prefix snapshot.
            # Same gating as the lookup path (Llama-like + exact DynamicCache).
            # Snapshot ids = full sequence (any seeded prefix +
            # this run's prompt ids + generated ids), aligned to the fed-KV
            # length by store().
            if (
                prefix_cache is not None
                and past_state is not None
                and isinstance(adapter, LlamaLikeAdapter)
            ):
                from transformers.cache_utils import DynamicCache as _DC

                if type(past_state) is _DC:
                    try:
                        layers = getattr(past_state, "layers", [])
                        cache_len = int(layers[0].keys.shape[-2]) if layers else 0
                        if cache_len > 0:
                            base_ids = prefix_hit.ids if prefix_hit is not None else []
                            full_ids = [*base_ids, *generated[0].tolist()]
                            prefix_cache.store(
                                full_ids[:cache_len],
                                prefix_cache.extract_layers(past_state, len(blocks)),
                            )
                    except Exception:
                        LOGGER.exception("prefix_cache_store_failed")

            # After a prefix hit, `generated` holds only suffix+new tokens —
            # restore the full sequence so completion decoding and the
            # output/generated token metrics see prompt-inclusive ids.
            if prefix_hit is not None:
                suffix_len = prompt_tokens - prefix_hit.length
                new_ids = generated[0].tolist()[suffix_len:]
                generated = torch.tensor(
                    [full_prompt_ids + new_ids],
                    dtype=torch.long,
                    device=generated.device,
                )

            completion_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            completion = (
                completion_text[len(prompt):].lstrip()
                if completion_text.startswith(prompt)
                else completion_text
            )

            generate_seconds = time.perf_counter() - generation_start
            total_seconds = load_seconds + preprocess_seconds + generate_seconds
            peak_rss_bytes = max(peak_rss_bytes, _process_memory_bytes(memory_tracker))
            output_tokens = int(generated.shape[-1])
            generated_tokens = max(output_tokens - prompt_tokens, 0)


            metrics = self._build_metrics(
                adapter=adapter,
                profile=profile,
                load_seconds=load_seconds,
                preprocess_seconds=preprocess_seconds,
                generate_seconds=generate_seconds,
                total_seconds=total_seconds,
                generation_start=generation_start,
                first_token_start=first_token_start,
                first_token_end=first_token_end,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                generated_tokens=generated_tokens,
                peak_rss_bytes=peak_rss_bytes,
            )
            return RunResult(prompt=prompt, completion=completion, metrics=metrics)

        except Exception as exc:
            LOGGER.exception("swlp_run_failed", extra={"error": str(exc)})
            try:
                self._cleanup_resources(scheduler)
            except Exception:
                pass
            if self.config.runtime.swlp_fallback_to_baseline:
                LOGGER.warning("swlp_fallback_to_baseline_on_error", extra={"error": str(exc)})
                return HuggingFaceRunner(self.config).run(prompt, profile=profile)
            raise
        finally:
            # Dump pipeline profiler traces if available.
            try:
                if (
                    scheduler is not None
                    and hasattr(scheduler, "profiler")
                    and scheduler.profiler is not None
                ):
                    prof = scheduler.profiler
                    prof.finalize()
                    metrics = prof.pipeline_metrics()
                    measured_ratio = pipeline_ratio_from_metrics(
                        avg_read_ms=metrics.avg_read_ms,
                        avg_deserialize_ms=metrics.avg_deserialize_ms,
                        avg_upload_ms=metrics.avg_upload_ms,
                        avg_compute_ms=metrics.avg_compute_ms,
                    )
                    self._update_measured_pipeline_ratio(measured_ratio)
                    LOGGER.info(
                        "swlp_pipeline_ratio_calibrated",
                        extra={
                            "pipeline_ratio": round(measured_ratio, 2),
                            "smoothed_pipeline_ratio": round(
                                self._measured_pipeline_ratio or measured_ratio,
                                2,
                            ),
                            "avg_read_ms": round(metrics.avg_read_ms, 2),
                            "avg_deserialize_ms": round(metrics.avg_deserialize_ms, 2),
                            "avg_upload_ms": round(metrics.avg_upload_ms, 2),
                            "avg_compute_ms": round(metrics.avg_compute_ms, 2),
                        },
                    )
                    # Printing and the trace dump are opt-in (profile=True or
                    # SWLP_PROFILE=1): they used to print into every chat
                    # answer and write layer_traces.json into the cwd.
                    if profile or self.config.runtime.profile:
                        prints = self._profile_prints or {
                            "timeline": True, "detail": False, "summary": True,
                        }
                        if prints.get("timeline"):
                            prof.print_timeline()
                        if prints.get("detail"):
                            prof.print_layer_detail()
                        if prints.get("summary"):
                            prof.print_pipeline_summary()
                        dump_path = str(self._trace_output or "layer_traces.json")
                        prof.dump(dump_path)
                        self._last_trace_path = dump_path
            except Exception as exc:
                self.degrade(f"profiler_dump_failed: {exc}", exc)
            try:
                self._cleanup_resources(scheduler)
            except Exception:
                pass
