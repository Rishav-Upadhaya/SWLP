from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import psutil
import torch

from ..core.compressed_cache import CompressedDynamicCache
from ..core.kv_cache import KVCacheManager
from ..core.scheduler import PrefetchError, SchedulerConfig, SWLPScheduler, ThreadedScheduler
from ..core.streaming import StreamingScheduler, has_shards
from ..metrics import RunMetrics, RunResult
from .arch import ArchAdapter, GPT2Adapter, LlamaLikeAdapter, get_adapter
from .hf import HuggingFaceRunner
from .load import load_from_shards, load_full_model

LOGGER = logging.getLogger(__name__)


class SWLPRunner(HuggingFaceRunner):
    backend = "swlp"

    def _resolve_resident_count(self, num_blocks: int, shard_dir) -> int:
        """Compute how many layers to keep permanently resident.

        Respects swlp_residency: "auto" triggers ResidencyPlanner,
        "off" returns 0, and an integer string returns that exact count.
        """
        residency_cfg = str(self.config.runtime.swlp_residency).strip().lower()
        if residency_cfg == "off":
            return 0
        if residency_cfg.lstrip("-").isdigit():
            return max(0, min(int(residency_cfg), num_blocks))

        # "auto" — ask ResidencyPlanner.
        from ..core.residency import plan_residency
        from ..hardware.detect import detect_hardware

        hw = detect_hardware()
        total_bytes = int(hw.memory_gb * 1024 * 1024 * 1024)
        layer_bytes = 0
        if shard_dir is not None and has_shards(shard_dir):
            try:
                from ..model.shard import load_manifest
                layer_bytes = int(load_manifest(shard_dir).layer_weight_mb * 1024 * 1024)
            except Exception:
                LOGGER.exception("residency_manifest_read_failed")
        plan = plan_residency(total_bytes, layer_bytes, num_blocks)
        LOGGER.info(
            "swlp_residency_plan",
            extra={
                "resident_count": plan.resident_count,
                "streaming_count": plan.streaming_count,
                "resident_gb": round(plan.resident_bytes / 1e9, 2),
                "streaming_gb_per_token": round(plan.streaming_bytes / 1e9, 2),
            },
        )
        return plan.resident_count

    def _check_streaming_feasible(self) -> None:
        """Abort early with a clear message if the model cannot run even with
        streaming — i.e. the always-resident modules plus one layer window do
        not fit in RAM. Prevents a confusing mid-inference OOM crash.
        """
        shard_dir = self.config.runtime.shard_dir
        if shard_dir is None or not has_shards(shard_dir):
            return
        from ..hardware.detect import detect_hardware, streaming_fits_in_memory
        from ..model.shard import load_manifest

        manifest = load_manifest(shard_dir)
        shard_path = Path(shard_dir)
        resident_bytes = 0
        for name in (manifest.embed_file, manifest.lm_head_file):
            shard_file = shard_path / name
            if shard_file.is_file():
                resident_bytes += shard_file.stat().st_size
        window = max(1, self.config.runtime.swlp_window_size)
        window_bytes = int(window * manifest.layer_weight_mb * 1024 * 1024)
        hw = detect_hardware()
        if not streaming_fits_in_memory(resident_bytes, window_bytes, hw):
            raise RuntimeError(
                f"Model {manifest.model_id} cannot run even with layer streaming "
                f"on this machine: resident modules (embeddings + lm_head, "
                f"~{resident_bytes / 1e9:.1f} GB) plus a {window}-layer window "
                f"(~{window_bytes / 1e9:.1f} GB) exceed available RAM "
                f"(~{hw.memory_gb:.0f} GB total). Use a machine with more RAM, a "
                f"smaller window (swlp_window_size), or a smaller model."
            )

    def _build_scheduler(self, blocks):
        sched_config = SchedulerConfig(
            window_size=max(1, self.config.runtime.swlp_window_size),
            prefetch_depth=max(1, self.config.runtime.swlp_prefetch_depth),
            prefetch=self.config.runtime.swlp_prefetch,
            double_buffer=self.config.runtime.swlp_double_buffer,
            pin_memory=self.config.runtime.swlp_pin_memory,
        )
        shard_dir = self.config.runtime.shard_dir
        if shard_dir is not None and has_shards(shard_dir):
            LOGGER.info("swlp_streaming_from_shards", extra={"shard_dir": str(shard_dir)})
            resident_count = self._resolve_resident_count(len(blocks), shard_dir)
            return StreamingScheduler(
                blocks, self.device, sched_config, shard_dir,
                resident_count=resident_count,
            )
        if self.device.type == "cuda" and torch.cuda.is_available():
            return SWLPScheduler(blocks, self.device, sched_config)
        return ThreadedScheduler(blocks, self.device, sched_config)

    def _apply_tuning_profile(self) -> None:
        profile_path = os.getenv("SWLP_TUNING_FILE", "swlp_tuning.json")
        if not os.path.exists(profile_path):
            return
        try:
            with open(profile_path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            LOGGER.exception("failed_load_tuning_profile")
            return
        device_key = self.device.type if hasattr(self, "device") else "_default"
        profile = data.get(device_key) or data.get("_default")
        if not profile:
            return
        try:
            rw = self.config.runtime
            if "swlp_window_size" in profile:
                rw.swlp_window_size = int(profile["swlp_window_size"])
            if "swlp_prefetch_depth" in profile:
                rw.swlp_prefetch_depth = int(profile["swlp_prefetch_depth"])
            if "swlp_prefetch" in profile:
                rw.swlp_prefetch = bool(profile["swlp_prefetch"])
            if "swlp_pin_memory" in profile:
                rw.swlp_pin_memory = bool(profile["swlp_pin_memory"])
            if "swlp_double_buffer" in profile:
                rw.swlp_double_buffer = bool(profile["swlp_double_buffer"])
            LOGGER.info("applied_tuning_profile", extra={"device": device_key, "profile": profile})
        except Exception:
            LOGGER.exception("apply_tuning_profile_failed")

    def _cleanup_resources(self, scheduler=None) -> None:
        try:
            if scheduler is not None:
                try:
                    scheduler.cleanup()
                except Exception:
                    LOGGER.exception("scheduler_cleanup_failed")
            if hasattr(self, "kv_manager") and self.kv_manager is not None:
                try:
                    self.kv_manager.clear()
                except Exception:
                    LOGGER.exception("kv_manager_clear_failed")
            try:
                self.model = None
            except Exception:
                pass
            try:
                if self.device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception:
            LOGGER.exception("cleanup_resources_failed")

    def _auto_shard_if_needed(self, shard_dir: Path) -> None:
        """Auto-shard the model when ``shard_dir`` is specified but has no shards.

        Removes the "download first" friction (Phase 14): users can point
        ``--shard-dir`` at a non-existent directory and SWLP will download the
        model and split it automatically on first run.
        """
        if has_shards(shard_dir):
            return
        model_id = self.config.model.model_id
        if not model_id:
            return
        LOGGER.info(
            "auto_shard_start",
            extra={"model_id": model_id, "shard_dir": str(shard_dir)},
        )
        print(f"\nAuto-sharding {model_id} → {shard_dir}")
        print("This is a one-time operation. Large models may take 20–60 min.\n")
        from ..model.shard import shard_model_by_layer

        cache_dir = str(self.config.cache.cache_dir) if self.config.cache.cache_dir else None
        manifest = shard_model_by_layer(model_id, shard_dir, cache_dir=cache_dir)
        LOGGER.info(
            "auto_shard_complete",
            extra={
                "shard_dir": str(shard_dir),
                "num_layers": manifest.num_layers,
                "total_gb": round(manifest.total_weight_mb / 1024, 2),
            },
        )
        print(f"\n✓  Auto-shard complete: {manifest.num_layers} layers in {shard_dir}\n")

    def load(self) -> float:
        # Idempotent: if the model is already in memory (e.g. run_chat pre-loaded it
        # before the REPL starts) skip the reload so successive run() / stream_tokens()
        # calls don't double-load — a 14 GB full-model reload while the first copy is
        # still resident causes an immediate OOM on 16 GB machines.
        if self.model is not None:
            return 0.0
        started = time.perf_counter()
        shard_dir = self.config.runtime.shard_dir
        # Phase 14: auto-shard on first run if shard_dir doesn't have shards yet.
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
    ) -> torch.Tensor:
        assert self.model is not None
        blocks = adapter.get_blocks(self.model)
        is_gpt2 = isinstance(adapter, GPT2Adapter)
        window = scheduler.config.window_size

        # ── warmup: kick off parallel SSD reads for the first W layers ───────
        # Fires window_size background threads simultaneously so layers 0..W-1
        # are all loading from disk in parallel before compute begins.  Without
        # this, layer 0 is always a cold synchronous read (~63 ms on M5 NVMe)
        # while layers 1..W-1 are prefetched one-at-a-time inside the loop.
        for warm_idx in range(min(window, len(blocks))):
            try:
                scheduler.prefetch(warm_idx)
            except PrefetchError as exc:
                LOGGER.warning(
                    "swlp_warmup_prefetch_failed",
                    extra={"layer": warm_idx, "error": str(exc)},
                )
                try:
                    scheduler.disable_prefetch()
                except Exception:
                    LOGGER.exception("disable_prefetch_failed")
                break

        for layer_index, block in enumerate(blocks):
            trace_entry: dict = {"layer": layer_index}

            # ── prefetch: refill the window slot we're about to vacate ───────
            # Load the layer exactly W positions ahead so it arrives in CPU RAM
            # while the current layer is being computed.  Together with the
            # warmup above this keeps W layers in flight at all times:
            #   [computed & evicted] … [on device] [in CPU RAM ×(W-1)] [loading]
            try:
                trace_entry["prefetch_start"] = time.perf_counter()
                scheduler.prefetch(layer_index + window)
                trace_entry["prefetch_enqueued"] = time.perf_counter()
            except PrefetchError as exc:
                LOGGER.warning(
                    "swlp_prefetch_failed", extra={"layer": layer_index, "error": str(exc)}
                )
                try:
                    scheduler.disable_prefetch()
                except Exception:
                    LOGGER.exception("disable_prefetch_failed")

            try:
                block = scheduler.ensure(layer_index)
            except PrefetchError as exc:
                LOGGER.warning(
                    "swlp_ensure_failed", extra={"layer": layer_index, "error": str(exc)}
                )
                try:
                    scheduler.disable_prefetch()
                except Exception:
                    pass
                block = scheduler.ensure(layer_index)

            trace_entry["compute_start"] = time.perf_counter()
            if is_gpt2 and hasattr(self, "kv_manager") and ctx.past_state is not None:
                got = self.kv_manager.get(layer_index, self.device)
                if got is not None:
                    ctx.past_state[layer_index] = got

            hidden_states, present = adapter.call_block(block, ctx, layer_index)
            ctx.hidden_states = hidden_states
            trace_entry["compute_end"] = time.perf_counter()

            if is_gpt2:
                ctx.past_state[layer_index] = present
                if hasattr(self, "kv_manager"):
                    try:
                        self.kv_manager.set(layer_index, present)
                    except Exception:
                        LOGGER.exception("kv_set_failed", extra={"layer": layer_index})

            # ── evict: free the layer immediately after compute ───────────────
            # Each transformer layer is visited exactly once per token, so there
            # is no reason to hold it on device any longer.  Immediate eviction
            # keeps MPS memory at ≈1 layer at a time (minimising unified-memory
            # bus pressure) and is consistent with the W-ahead prefetch above.
            scheduler.evict(layer_index)
            trace_entry["evict_time"] = time.perf_counter()
            if hasattr(self, "_trace"):
                self._trace.append(trace_entry)

            if is_gpt2 and hasattr(self, "kv_manager"):
                try:
                    self.kv_manager.maybe_evict(layer_index)
                except Exception:
                    LOGGER.exception("kv_evict_failed", extra={"layer": layer_index})

            # Llama/Mistral path: compress this layer's KV immediately — it is
            # not read again until the next token step reaches the same layer.
            if not is_gpt2 and isinstance(ctx.past_state, CompressedDynamicCache):
                try:
                    ctx.past_state.compress_layer(layer_index)
                except Exception:
                    LOGGER.exception("kv_compress_failed", extra={"layer": layer_index})

        return ctx.hidden_states

    def _build_metrics(
        self,
        *,
        adapter: ArchAdapter,
        profile: bool,
        load_seconds: float,
        preprocess_seconds: float,
        generate_seconds: float,
        total_seconds: float,
        generation_start: float,
        first_token_start: float | None,
        first_token_end: float | None,
        prompt_tokens: int,
        output_tokens: int,
        generated_tokens: int,
        vram_peak_bytes: int | None,
        peak_rss_bytes: int,
    ) -> RunMetrics:
        assert self.model is not None
        kv_bytes_per_token = adapter.estimate_kv_bytes_per_token(self.model, self.dtype)
        kv_required_bytes = kv_bytes_per_token * (prompt_tokens + generated_tokens)
        kv_budget_bytes = self.config.runtime.kv_memory_budget_mb * 1024 * 1024
        if kv_budget_bytes > 0 and kv_required_bytes > kv_budget_bytes:
            LOGGER.warning(
                "swlp_kv_budget_exceeded",
                extra={"required_bytes": kv_required_bytes, "budget_bytes": kv_budget_bytes},
            )
        kv_stats = self.kv_manager.stats() if hasattr(self, "kv_manager") else {}
        # Phase 15: prefill_seconds = time from generation_start to first_token_start
        # (the forward sweep over all input tokens).
        # time_to_first_token_seconds = user-perceived TTFT = prefill + argmax.
        prefill_seconds: float | None = (
            (first_token_start - generation_start)
            if first_token_start is not None
            else None
        )
        ttft: float | None = (
            (first_token_end - generation_start)
            if first_token_end is not None
            else None
        )
        return RunMetrics(
            model_id=self.config.model.model_id,
            backend=self.backend,
            device=self.device.type,
            load_seconds=load_seconds,
            input_tokens=prompt_tokens,
            output_tokens=output_tokens,
            preprocess_seconds=preprocess_seconds if profile else None,
            forward_seconds=None,
            generate_seconds=generate_seconds,
            total_seconds=total_seconds,
            prefill_seconds=prefill_seconds,
            time_to_first_token_seconds=ttft,
            per_token_latency_seconds=None,
            throughput_tokens_per_second=(
                generated_tokens / generate_seconds if generate_seconds > 0 else None
            ),
            generated_tokens=generated_tokens,
            vram_peak_bytes=vram_peak_bytes if profile else None,
            ram_peak_bytes=peak_rss_bytes if profile else None,
            kv_cache_entries=kv_stats.get("entries"),
            kv_cache_device_bytes=kv_stats.get("device_bytes"),
            kv_cache_host_bytes=kv_stats.get("host_bytes"),
            kv_cache_compressed_bytes=kv_stats.get("compressed_bytes"),
            kv_cache_total_bytes=kv_stats.get("total_bytes"),
            kv_cache_peak_device_bytes=kv_stats.get("peak_device_bytes"),
            kv_cache_peak_host_bytes=kv_stats.get("peak_host_bytes"),
            kv_cache_peak_total_bytes=kv_stats.get("peak_total_bytes"),
            kv_cache_compressions=kv_stats.get("compressions"),
            kv_cache_decompressions=kv_stats.get("decompressions"),
            kv_cache_offloads=kv_stats.get("offloads"),
            kv_cache_moves_to_device=kv_stats.get("moves_to_device"),
            kv_cache_budget_bytes=kv_stats.get("budget_bytes"),
            kv_cache_device_budget_bytes=kv_stats.get("device_budget_bytes"),
            kv_cache_budget_violations=kv_stats.get("budget_violations"),
        )

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
        cb: Callable[[str], None] | None = getattr(self, "_token_callback", None)
        # Use incremental decoding so SentencePiece space-prefixed tokens (▁word →
        # " word") are preserved.  Decoding a single token ID in isolation strips the
        # leading space; decoding the full sequence and diffing against the previous
        # length produces the correct text including any whitespace.
        _prev_text: str = getattr(self, "_stream_prev_text", "")
        for _ in range(max(self.config.generation.max_new_tokens - 1, 0)):
            position_offset = int(generated.shape[-1] - 1)
            ctx = adapter.prepare_step(
                self.model,
                generated[:, -1:],
                ctx.past_state,
                self.device,
                position_offset,
            )
            ctx.hidden_states = self._run_blocks(adapter, ctx, scheduler)
            hidden_states = adapter.final_norm(self.model, ctx.hidden_states)
            logits = self.model.lm_head(hidden_states)[:, -1, :]
            next_token = self._select_next(logits, generated)
            generated = torch.cat([generated, next_token], dim=-1)
            if cb is not None and self.tokenizer is not None:
                _new_text = self.tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
                _delta = _new_text[len(_prev_text):]
                if _delta:
                    cb(_delta)
                _prev_text = _new_text
            if self.tokenizer is not None and self.tokenizer.eos_token_id is not None:
                if int(next_token.item()) == int(self.tokenizer.eos_token_id):
                    break
        return generated

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
        torch.manual_seed(self.config.generation.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.generation.seed)

        memory_tracker = psutil.Process()
        peak_rss_bytes = 0
        if profile and self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        # Pre-flight: refuse cleanly if the model cannot fit even with streaming,
        # rather than crashing partway through inference.
        self._check_streaming_feasible()

        scheduler = None
        try:
            load_seconds = self.load()
            peak_rss_bytes = max(peak_rss_bytes, memory_tracker.memory_info().rss)

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

            # Pre-load resident layers once before inference starts.
            if hasattr(scheduler, "load_resident_layers"):
                scheduler.load_resident_layers()

            try:
                self.kv_manager = KVCacheManager(
                    budget_bytes=self._resolve_kv_budget_bytes(adapter, len(blocks)),
                    compression=bool(self.config.runtime.kv_compression),
                    compression_level=self._resolve_compression_level(),
                    tiering=bool(self.config.runtime.kv_tiering),
                    device=self.device,
                    kv_window=max(0, int(self.config.runtime.kv_window)),
                    kv_quant=str(self.config.runtime.kv_quant),
                )
            except Exception:
                LOGGER.exception("kv_manager_init_failed")
                self.kv_manager = KVCacheManager()

            preprocess_start = time.perf_counter()
            encoded = self.tokenizer(prompt, return_tensors="pt")
            input_ids = encoded["input_ids"].to(self.device)
            preprocess_seconds = time.perf_counter() - preprocess_start
            peak_rss_bytes = max(peak_rss_bytes, memory_tracker.memory_info().rss)
            prompt_tokens = int(input_ids.shape[-1])

            past_state = self._make_past_state(adapter, len(blocks))

            generation_start = time.perf_counter()
            first_token_start = None
            first_token_end = None

            with torch.no_grad():
                ctx = adapter.prepare_step(self.model, input_ids, past_state, self.device, 0)
                ctx.hidden_states = self._run_blocks(adapter, ctx, scheduler)
                hidden_states = adapter.final_norm(self.model, ctx.hidden_states)
                # Phase 15: first_token_start marks the end of the prefill sweep
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
                        self._stream_prev_text: str = _after_first
                    else:
                        self._stream_prev_text = ""
                    generated = self._generate_remaining(adapter, scheduler, ctx, generated)

            completion_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            completion = (
                completion_text[len(prompt):].lstrip()
                if completion_text.startswith(prompt)
                else completion_text
            )

            generate_seconds = time.perf_counter() - generation_start
            total_seconds = load_seconds + preprocess_seconds + generate_seconds
            peak_rss_bytes = max(peak_rss_bytes, memory_tracker.memory_info().rss)
            output_tokens = int(generated.shape[-1])
            generated_tokens = max(output_tokens - prompt_tokens, 0)

            vram_peak_bytes = (
                int(torch.cuda.max_memory_allocated(self.device))
                if self.device.type == "cuda" and torch.cuda.is_available()
                else None
            )

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
                vram_peak_bytes=vram_peak_bytes,
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
            try:
                self._cleanup_resources(scheduler)
            except Exception:
                pass
