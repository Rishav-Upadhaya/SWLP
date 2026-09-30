"""Setup, resolution, and metrics assembly for :class:`SWLPRunner`.

Split out of ``runner/swlp.py`` purely to keep that file navigable: this is
the once-per-run half (resolve the residency plan, check feasibility, decide
direct-I/O, assemble RunMetrics), while ``swlp.py`` keeps the per-token hot
path (``_run_blocks``, ``_generate_remaining``, ``run``).

It is a mixin rather than a set of free functions so that every existing
``self._resolve_*`` / ``runner._check_*`` call site — including the ones in
``runner/batch.py``, ``tests/test_streaming.py`` and
``scripts/research/swlp_tune.py`` — keeps working untouched. ``SWLPRunner``
remains the single public type.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import psutil
import torch

from ..core.residency import build_residency_decision
from ..core.streaming import has_shards, resolve_direct_io
from ..hardware.detect import detect_hardware, streaming_fits_in_memory
from ..metrics import RunMetrics
from .arch import ArchAdapter

LOGGER = logging.getLogger(__name__)


class SWLPSetupMixin:
    """Once-per-run setup and reporting for :class:`SWLPRunner`."""

    def _resolve_dtype(self) -> torch.dtype:
        """``auto`` computes in the shards' own dtype (bf16 shards → bf16).

        The base rule (fp16 on MPS) is kept for runs without a manifest; a
        bf16 shard dir under an fp16 skeleton would mix dtypes per layer.
        """
        shard_dir = self.config.runtime.shard_dir
        if (
            self.config.runtime.dtype.lower() == "auto"
            and shard_dir is not None
            and has_shards(shard_dir)
        ):
            from ..model.shard import load_manifest

            return getattr(torch, load_manifest(shard_dir).weight_dtype)
        return super()._resolve_dtype()

    def _update_measured_pipeline_ratio(self, measured_ratio: float) -> None:
        """Update measured ratio with smoothing and sanity bounds."""
        if measured_ratio <= 0 or not (measured_ratio < float("inf")):
            LOGGER.warning("swlp_pipeline_ratio_invalid", extra={"value": measured_ratio})
            return
        bounded = max(0.1, min(20.0, measured_ratio))
        if self._measured_pipeline_ratio is None:
            self._measured_pipeline_ratio = bounded
            return
        alpha = 0.35
        self._measured_pipeline_ratio = (
            (1.0 - alpha) * self._measured_pipeline_ratio + alpha * bounded
        )

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

        # "auto" — ask ResidencyPlanner with startup hardware calibration.
        from ..core.residency import estimate_pipeline_ratio
        from ..hardware.detect import detect_hardware

        hw = detect_hardware()
        total_bytes = int(hw.memory_gb * 1024 * 1024 * 1024)
        free_bytes = int(psutil.virtual_memory().available)
        layer_bytes = 0
        if shard_dir is not None and has_shards(shard_dir):
            try:
                from ..model.shard import load_manifest
                layer_bytes = int(load_manifest(shard_dir).layer_weight_mb * 1024 * 1024)
            except Exception:
                LOGGER.exception("residency_manifest_read_failed")

        estimated_ratio = estimate_pipeline_ratio(
            layer_weight_bytes=layer_bytes,
            ssd_bandwidth_gbps=hw.ssd_bandwidth_gbps,
        )
        ratio_source = "measured" if self._measured_pipeline_ratio is not None else "estimated"
        pipeline_ratio = self._measured_pipeline_ratio or estimated_ratio

        decision = build_residency_decision(
            total_memory_bytes=total_bytes,
            free_memory_bytes=free_bytes,
            layer_weight_bytes=layer_bytes,
            num_layers=num_blocks,
            pipeline_ratio=pipeline_ratio,
            ratio_source=ratio_source,
        )
        self._last_residency_decision = {
            "resident_count": decision.resident_count,
            "streaming_count": decision.streaming_count,
            "resident_gb": round(decision.resident_bytes / 1e9, 2),
            "streaming_gb_per_token": round(decision.streaming_bytes / 1e9, 2),
            "pipeline_ratio": round(decision.pipeline_ratio, 2),
            "estimated_ratio": round(estimated_ratio, 2),
            "ratio_source": decision.ratio_source,
            "memory_budget_gb": round(decision.memory_budget_bytes / 1e9, 2),
            "available_gb": round(decision.available_bytes / 1e9, 2),
            "estimated_resident": decision.estimated_resident_count,
            "memory_clamp_applied": decision.memory_clamp_applied,
            "confidence_score": round(decision.confidence_score, 3),
            "confidence_level": decision.confidence_level,
            "reasoning": decision.reasoning,
        }
        LOGGER.info(
            "swlp_residency_plan",
            extra=self._last_residency_decision,
        )
        return decision.resident_count

    def _check_streaming_feasible(self) -> None:
        """Abort early with a clear message if the model cannot run even with
        streaming — i.e. the always-resident modules plus one layer window do
        not fit in RAM. Prevents a confusing mid-inference OOM crash.
        """
        shard_dir = self.config.runtime.shard_dir
        if shard_dir is None or not has_shards(shard_dir):
            return
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

    def _resolve_direct_io(self, shard_dir) -> bool:
        """Resolve the SWLP_DIRECT_IO policy ("auto" | "on" | "off") for this run."""
        total_bytes = 0
        try:
            from ..model.shard import load_manifest

            total_bytes = int(load_manifest(Path(shard_dir)).total_weight_mb * 1024 * 1024)
        except Exception:
            LOGGER.exception("direct_io_manifest_read_failed")
        available = int(psutil.virtual_memory().available)
        direct_io = resolve_direct_io(
            self.config.runtime.swlp_direct_io, total_bytes, available
        )
        LOGGER.info(
            "swlp_direct_io_resolved",
            extra={
                "mode": self.config.runtime.swlp_direct_io,
                "direct_io": direct_io,
                "model_gb": round(total_bytes / 1e9, 2),
                "available_gb": round(available / 1e9, 2),
            },
        )
        return direct_io

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
            LOGGER.info("applied_tuning_profile", extra={"device": device_key, "profile": profile})
        except Exception:
            LOGGER.exception("apply_tuning_profile_failed")

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
        # Prefill_seconds = time from generation_start to first_token_start
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
            # Decode-only latency: tokens after the first, over the time after TTFT.
            per_token_latency_seconds=(
                (generate_seconds - ttft) / (generated_tokens - 1)
                if ttft is not None and generated_tokens > 1
                else None
            ),
            throughput_tokens_per_second=(
                generated_tokens / generate_seconds if generate_seconds > 0 else None
            ),
            generated_tokens=generated_tokens,
            degradations=list(self.degradations) or None,
            degradation_count=len(self.degradations),
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
