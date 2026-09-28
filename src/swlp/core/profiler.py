"""Fine-grained per-layer profiler for SWLP streaming pipeline.

Records absolute timestamps for every stage of a layer's lifecycle:

    Prefetch Submit → SSD Read → Deserialize → Upload MPS → ensure()
    → Compute → Evict

The goal: answer "where does one token spend its time?" quantitatively.

Usage:
    profiler = LayerProfiler(enabled=True)
    # In StreamingScheduler.prefetch():
    profiler.record_prefetch_submit(idx)
    # In _read_and_deserialize():
    profiler.begin_read(idx)
    profiler.end_read(idx)
    profiler.begin_deserialize(idx)
    profiler.end_deserialize(idx)
    # In _upload_to_device():
    profiler.begin_upload(idx)
    profiler.end_upload(idx)
    # In ensure():
    profiler.begin_ensure(idx)
    profiler.end_ensure(idx, wait, status)
    # In _run_blocks():
    profiler.begin_compute(idx)
    profiler.end_compute(idx)
    # In evict():
    profiler.begin_evict(idx)
    profiler.end_evict(idx)
    # At end of run:
    profiler.finalize()  # computes pipeline metrics
    profiler.dump("layer_traces.json")
"""

from __future__ import annotations

import json
import logging
import platform
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import torch

LOGGER = logging.getLogger(__name__)


@dataclass
class LayerTrace:
    """Per-layer trace with absolute timestamps (seconds from perf_counter)."""

    layer: int

    # Stage 0: Prefetch submitted to worker pool
    prefetch_submit: float = 0.0

    # Stage 1: SSD Read
    read_start: float = 0.0
    read_end: float = 0.0

    # Stage 2: Deserialization (safetensors parse / torch.load / dequant)
    deserialize_start: float = 0.0
    deserialize_end: float = 0.0

    # Stage 3: Upload to MPS (host->device copy)
    upload_start: float = 0.0
    upload_end: float = 0.0

    # Stage 4: ensure() entry/exit
    ensure_enter: float = 0.0
    ensure_exit: float = 0.0

    # Stage 5: Ready for compute (ensure() returned)
    ready_time: float = 0.0

    # Stage 6: GPU Forward pass
    compute_start: float = 0.0
    compute_end: float = 0.0

    # Stage 7: Eviction
    evict_start: float = 0.0
    evict_end: float = 0.0

    # Memory snapshots (bytes)
    rss_at_read: int = 0
    rss_at_upload: int = 0
    rss_at_evict: int = 0

    # Pipeline metrics
    ensure_wait: float = 0.0  # time blocked in ensure() waiting for prefetch
    overlap_status: str = ""  # "hit" | "wait" | "miss" | "resident" | "sync"

    def durations(self) -> dict[str, float]:
        """Return duration of each stage in milliseconds."""
        return {
            "prefetch_to_read_ms": (self.read_start - self.prefetch_submit) * 1000,
            "read_ms": (self.read_end - self.read_start) * 1000,
            "deserialize_ms": (self.deserialize_end - self.deserialize_start) * 1000,
            "upload_ms": (self.upload_end - self.upload_start) * 1000,
            "ensure_wait_ms": self.ensure_wait * 1000,
            "ready_to_compute_ms": (self.compute_start - self.ready_time) * 1000
            if self.ready_time > 0 and self.compute_start > 0
            else 0.0,
            "compute_ms": (self.compute_end - self.compute_start) * 1000,
            "evict_ms": (self.evict_end - self.evict_start) * 1000,
            "total_lifecycle_ms": (self.evict_end - self.prefetch_submit) * 1000
            if self.prefetch_submit > 0 and self.evict_end > 0
            else 0.0,
        }

    def to_dict(self) -> dict:
        """Serialize to dict with durations."""
        d = {
            "layer": self.layer,
            "prefetch_submit": self.prefetch_submit,
            "read_start": self.read_start,
            "read_end": self.read_end,
            "deserialize_start": self.deserialize_start,
            "deserialize_end": self.deserialize_end,
            "upload_start": self.upload_start,
            "upload_end": self.upload_end,
            "ensure_enter": self.ensure_enter,
            "ensure_exit": self.ensure_exit,
            "ready_time": self.ready_time,
            "compute_start": self.compute_start,
            "compute_end": self.compute_end,
            "evict_start": self.evict_start,
            "evict_end": self.evict_end,
            "rss_at_read": self.rss_at_read,
            "rss_at_upload": self.rss_at_upload,
            "rss_at_evict": self.rss_at_evict,
            "ensure_wait": self.ensure_wait,
            "overlap_status": self.overlap_status,
        }
        d.update(self.durations())
        return d


@dataclass
class PipelineMetrics:
    """System-level metrics computed from layer traces."""

    num_layers: int = 0
    num_tokens: int = 0

    # Per-stage statistics (all in ms)
    avg_read_ms: float = 0.0
    avg_deserialize_ms: float = 0.0
    avg_upload_ms: float = 0.0
    avg_compute_ms: float = 0.0
    avg_evict_ms: float = 0.0
    avg_ensure_wait_ms: float = 0.0
    max_ensure_wait_ms: float = 0.0
    total_ensure_wait_ms: float = 0.0

    # Pipeline stall tracking
    pipeline_stall_count: int = 0
    prefetch_hits: int = 0
    prefetch_waits: int = 0
    prefetch_misses: int = 0
    prefetch_total: int = 0
    prefetch_hit_rate: float = 0.0

    # GPU efficiency
    gpu_busy_time_ms: float = 0.0
    gpu_idle_time_ms: float = 0.0
    gpu_efficiency: float = 0.0

    # Memory
    peak_rss_bytes: int = 0
    avg_rss_bytes: int = 0
    total_ssd_bytes_read: int = 0

    # Timing
    total_pipeline_time_ms: float = 0.0
    wall_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _collect_durations(traces: list[LayerTrace], fn) -> list[float]:
    return [v for v in (fn(t) for t in traces) if v > 0]


def compute_pipeline_metrics(
    traces: list[LayerTrace],
    wall_time: float = 0.0,
    num_tokens: int = 1,
) -> PipelineMetrics:
    """Compute system-level metrics from layer traces."""
    if not traces:
        return PipelineMetrics()

    read_ms = _collect_durations(traces, lambda t: (t.read_end - t.read_start) * 1000)
    deser_ms = _collect_durations(
        traces, lambda t: (t.deserialize_end - t.deserialize_start) * 1000
    )
    upload_ms = _collect_durations(traces, lambda t: (t.upload_end - t.upload_start) * 1000)
    compute_ms = _collect_durations(traces, lambda t: (t.compute_end - t.compute_start) * 1000)
    evict_ms = _collect_durations(traces, lambda t: (t.evict_end - t.evict_start) * 1000)
    wait_ms = [t.ensure_wait * 1000 for t in traces if t.ensure_wait > 0]

    total_compute = sum(compute_ms)
    total_wait = sum(wait_ms)
    total_pipeline = total_compute + total_wait

    rss_values = [t.rss_at_read for t in traces if t.rss_at_read > 0]
    peak_rss = max(rss_values) if rss_values else 0
    avg_rss = int(sum(rss_values) / len(rss_values)) if rss_values else 0

    overlap = _count_overlap(traces)
    hits = overlap.get("hit", 0)
    waits = overlap.get("wait", 0)
    misses = overlap.get("misses", 0) + overlap.get("miss", 0)
    prefetch_total = hits + waits + misses
    prefetch_hit_rate = hits / prefetch_total if prefetch_total > 0 else 0.0

    # Estimate GPU idle time: time when compute is not running
    # Between consecutive compute phases, the GPU is idle
    gpu_idle = 0.0
    sorted_by_compute = sorted(
        [t for t in traces if t.compute_start > 0 and t.compute_end > 0],
        key=lambda t: t.compute_start,
    )
    for i in range(1, len(sorted_by_compute)):
        gap = sorted_by_compute[i].compute_start - sorted_by_compute[i - 1].compute_end
        if gap > 0:
            gpu_idle += gap * 1000

    return PipelineMetrics(
        num_layers=len(traces),
        num_tokens=num_tokens,
        avg_read_ms=_mean(read_ms),
        avg_deserialize_ms=_mean(deser_ms),
        avg_upload_ms=_mean(upload_ms),
        avg_compute_ms=_mean(compute_ms),
        avg_evict_ms=_mean(evict_ms),
        avg_ensure_wait_ms=_mean(wait_ms),
        max_ensure_wait_ms=max(wait_ms) if wait_ms else 0.0,
        total_ensure_wait_ms=total_wait,
        pipeline_stall_count=misses,
        prefetch_hits=hits,
        prefetch_waits=waits,
        prefetch_misses=misses,
        prefetch_total=prefetch_total,
        prefetch_hit_rate=prefetch_hit_rate,
        gpu_busy_time_ms=total_compute,
        gpu_idle_time_ms=gpu_idle,
        gpu_efficiency=total_compute / total_pipeline if total_pipeline > 0 else 0.0,
        peak_rss_bytes=peak_rss,
        avg_rss_bytes=avg_rss,
        total_ssd_bytes_read=0,  # caller must fill from manifest
        total_pipeline_time_ms=total_pipeline,
        wall_time_ms=wall_time * 1000 if wall_time > 0 else total_pipeline,
    )


@dataclass
class QueueSnapshot:
    """Queue depth snapshot at a scheduling event."""

    timestamp: float
    event: str  # "prefetch_submit" | "ensure_enter" | "ensure_exit" | "evict"
    read_queue_depth: int = 0
    upload_queue_depth: int = 0
    ready_layers: int = 0
    active_reads: int = 0
    active_uploads: int = 0
    rss_bytes: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HardwareMetadata:
    """Hardware and environment metadata for trace export."""

    device_type: str = ""
    chip_name: str = ""
    memory_gb: float = 0.0
    unified_memory: bool = False
    macos_version: str = ""
    pytorch_version: str = ""
    python_version: str = ""
    mps_enabled: bool = False

    model_id: str = ""
    model_size_mb: float = 0.0
    num_layers: int = 0
    quantization: str = ""
    window_size: int = 0
    prefetch_depth: int = 0
    worker_count: int = 0
    direct_io: bool = False
    resident_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class LayerProfiler:
    """Collects LayerTraces across all layers for one or more token passes.

    Thread-safe: worker threads call begin_read/begin_deserialize etc.,
    main thread calls record_ready/begin_compute/end_compute.
    """

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._traces: dict[int, LayerTrace] = {}
        self._multi_token_traces: list[list[LayerTrace]] = []
        self._current_token_traces: dict[int, LayerTrace] = {}
        self._baseline_rss = _get_rss()
        self._hardware = HardwareMetadata()
        self._queue_snapshots: list[QueueSnapshot] = []
        self._token_index = 0
        self._start_time = 0.0
        self._end_time = 0.0

    def set_hardware(self, hw: HardwareMetadata) -> None:
        self._hardware = hw

    def _get_trace(self, layer: int) -> LayerTrace:
        if layer not in self._current_token_traces:
            self._current_token_traces[layer] = LayerTrace(layer=layer)
        return self._current_token_traces[layer]

    # ── Token boundary ───────────────────────────────────────────────────

    def begin_token(self, token_index: int) -> None:
        """Mark the start of a new token's scheduling pass."""
        if not self._enabled:
            return
        self._token_index = token_index
        self._current_token_traces.clear()
        if token_index == 0:
            self._start_time = time.perf_counter()

    def end_token(self) -> None:
        """Mark the end of a token's scheduling pass."""
        if not self._enabled:
            return
        traces = list(self._current_token_traces.values())
        self._multi_token_traces.append(traces)
        # Merge into the flat trace dict (last token wins for per-layer view)
        for t in traces:
            self._traces[t.layer] = t

    # ── Stage 0: Prefetch Submit ─────────────────────────────────────────

    def record_prefetch_submit(self, layer: int) -> None:
        if not self._enabled:
            return
        t = self._get_trace(layer)
        t.prefetch_submit = time.perf_counter()

    # ── Stage 1: SSD Read ────────────────────────────────────────────────

    def begin_read(self, layer: int) -> None:
        if not self._enabled:
            return
        t = self._get_trace(layer)
        t.read_start = time.perf_counter()
        t.rss_at_read = _get_rss()

    def end_read(self, layer: int) -> None:
        if not self._enabled:
            return
        self._get_trace(layer).read_end = time.perf_counter()

    # ── Stage 2: Deserialization ─────────────────────────────────────────

    def begin_deserialize(self, layer: int) -> None:
        if not self._enabled:
            return
        self._get_trace(layer).deserialize_start = time.perf_counter()

    def end_deserialize(self, layer: int) -> None:
        if not self._enabled:
            return
        self._get_trace(layer).deserialize_end = time.perf_counter()

    # ── Stage 3: Upload to MPS ───────────────────────────────────────────

    def begin_upload(self, layer: int) -> None:
        if not self._enabled:
            return
        t = self._get_trace(layer)
        t.upload_start = time.perf_counter()
        t.rss_at_upload = _get_rss()

    def end_upload(self, layer: int) -> None:
        if not self._enabled:
            return
        self._get_trace(layer).upload_end = time.perf_counter()

    # ── Stage 4: ensure() ────────────────────────────────────────────────

    def begin_ensure(self, layer: int) -> None:
        if not self._enabled:
            return
        self._get_trace(layer).ensure_enter = time.perf_counter()

    def end_ensure(self, layer: int, wait: float = 0.0, status: str = "") -> None:
        if not self._enabled:
            return
        t = self._get_trace(layer)
        t.ensure_exit = time.perf_counter()
        t.ready_time = t.ensure_exit
        t.ensure_wait = wait
        t.overlap_status = status

    def record_ready(self, layer: int, ensure_wait: float = 0.0, overlap_status: str = "") -> None:
        """Legacy API: alias for end_ensure."""
        self.end_ensure(layer, ensure_wait, overlap_status)

    # ── Stage 5: Compute ─────────────────────────────────────────────────

    def begin_compute(self, layer: int) -> None:
        if not self._enabled:
            return
        self._get_trace(layer).compute_start = time.perf_counter()

    def end_compute(self, layer: int) -> None:
        if not self._enabled:
            return
        self._get_trace(layer).compute_end = time.perf_counter()

    # ── Stage 6: Eviction ────────────────────────────────────────────────

    def begin_evict(self, layer: int) -> None:
        if not self._enabled:
            return
        t = self._get_trace(layer)
        t.evict_start = time.perf_counter()
        t.rss_at_evict = _get_rss()

    def end_evict(self, layer: int) -> None:
        if not self._enabled:
            return
        self._get_trace(layer).evict_end = time.perf_counter()

    # ── Queue Snapshot Logging ───────────────────────────────────────────

    def record_queue_snapshot(
        self,
        event: str,
        read_queue_depth: int = 0,
        upload_queue_depth: int = 0,
        ready_layers: int = 0,
        active_reads: int = 0,
        active_uploads: int = 0,
    ) -> None:
        if not self._enabled:
            return
        self._queue_snapshots.append(
            QueueSnapshot(
                timestamp=time.perf_counter(),
                event=event,
                read_queue_depth=read_queue_depth,
                upload_queue_depth=upload_queue_depth,
                ready_layers=ready_layers,
                active_reads=active_reads,
                active_uploads=active_uploads,
                rss_bytes=_get_rss(),
            )
        )

    # ── Reporting ────────────────────────────────────────────────────────

    def get_traces(self) -> list[LayerTrace]:
        """Return traces sorted by layer index (flat view, last token wins)."""
        return [self._traces[k] for k in sorted(self._traces)]

    def finalize(self) -> None:
        """Mark end of profiling session."""
        self._end_time = time.perf_counter()

    def summary(self) -> dict:
        """Aggregate statistics across all layers."""
        traces = self.get_traces()
        if not traces:
            return {}

        read_times = _collect_durations(traces, lambda t: (t.read_end - t.read_start) * 1000)
        deserialize_times = _collect_durations(
            traces, lambda t: (t.deserialize_end - t.deserialize_start) * 1000
        )
        upload_times = _collect_durations(traces, lambda t: (t.upload_end - t.upload_start) * 1000)
        compute_times = _collect_durations(
            traces, lambda t: (t.compute_end - t.compute_start) * 1000
        )
        evict_times = _collect_durations(traces, lambda t: (t.evict_end - t.evict_start) * 1000)
        wait_times = [t.ensure_wait * 1000 for t in traces if t.ensure_wait > 0]

        total_gpu_time = sum(compute_times)
        total_wait_time = sum(wait_times)
        total_pipeline_time = total_gpu_time + total_wait_time

        return {
            "num_layers": len(traces),
            "num_tokens": len(self._multi_token_traces),
            "read_ms": {
                "mean": _mean(read_times),
                "max": max(read_times) if read_times else 0,
                "p95": _p95(read_times),
            },
            "deserialize_ms": {
                "mean": _mean(deserialize_times),
                "max": max(deserialize_times) if deserialize_times else 0,
                "p95": _p95(deserialize_times),
            },
            "upload_ms": {
                "mean": _mean(upload_times),
                "max": max(upload_times) if upload_times else 0,
                "p95": _p95(upload_times),
            },
            "compute_ms": {
                "mean": _mean(compute_times),
                "max": max(compute_times) if compute_times else 0,
                "p95": _p95(compute_times),
            },
            "evict_ms": {
                "mean": _mean(evict_times),
                "max": max(evict_times) if evict_times else 0,
                "p95": _p95(evict_times),
            },
            "ensure_wait_ms": {
                "mean": _mean(wait_times),
                "max": max(wait_times) if wait_times else 0,
                "total": total_wait_time,
            },
            "gpu_efficiency": (
                total_gpu_time / total_pipeline_time if total_pipeline_time > 0 else 0.0
            ),
            "overlap_status_counts": _count_overlap(traces),
            "baseline_rss_bytes": self._baseline_rss,
        }

    def pipeline_metrics(self) -> PipelineMetrics:
        """Compute pipeline-level metrics from all traces."""
        wall_time = self._end_time - self._start_time if self._end_time > 0 else 0.0
        return compute_pipeline_metrics(
            self.get_traces(),
            wall_time=wall_time,
            num_tokens=len(self._multi_token_traces),
        )

    def dump(self, path: str | Path) -> None:
        """Dump all traces + summary + hardware metadata to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metrics = self.pipeline_metrics()
        data = {
            "hardware": self._hardware.to_dict(),
            "pipeline_metrics": metrics.to_dict(),
            "summary": self.summary(),
            "traces": [t.to_dict() for t in self.get_traces()],
            "token_traces": [
                [t.to_dict() for t in token] for token in self._multi_token_traces
            ],
            "queue_snapshots": [s.to_dict() for s in self._queue_snapshots],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        LOGGER.info(
            "profiler_dumped",
            extra={
                "path": str(path),
                "layers": len(self._traces),
                "tokens": len(self._multi_token_traces),
            },
        )

    def print_timeline(self, max_layers: int = 32) -> None:
        """Print a simple text timeline for visual inspection."""
        traces = self.get_traces()[:max_layers]
        if not traces:
            print("No traces recorded.")
            return

        # Find global time range
        read_starts = [t.read_start for t in traces if t.read_start > 0]
        evict_ends = [t.evict_end for t in traces if t.evict_end > 0]
        if not read_starts and not evict_ends:
            compute_starts = [t.compute_start for t in traces if t.compute_start > 0]
            compute_ends = [t.compute_end for t in traces if t.compute_end > 0]
            if not compute_starts:
                print("No compute traces recorded.")
                return
            t_min = min(compute_starts)
            t_max = max(compute_ends) if compute_ends else t_min + 1.0
        else:
            t_min = min(read_starts) if read_starts else min(
                t.compute_start for t in traces if t.compute_start > 0
            )
            t_max = max(evict_ends) if evict_ends else max(
                t.compute_end for t in traces if t.compute_end > 0
            )
        if t_max <= t_min:
            t_max = t_min + 1.0

        total_span = (t_max - t_min) * 1000
        print(f"\n{'='*80}")
        print(f"PIPELINE TIMELINE — {len(traces)} layers, {total_span:.1f} ms total")
        print(f"{'='*80}")
        print(
            f"{'Layer':>6} | {'Read':>8} | {'Deser':>8} | {'Upload':>8} | {'Ready->Comp':>10} | "
            f"{'Compute':>8} | {'Evict':>8} | {'Wait':>6} | Status"
        )
        print(f"{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*10}")

        for t in traces:
            read_ms = (t.read_end - t.read_start) * 1000 if t.read_end > 0 else 0
            deser_ms = (
                (t.deserialize_end - t.deserialize_start) * 1000 if t.deserialize_end > 0 else 0
            )
            upload_ms = (t.upload_end - t.upload_start) * 1000 if t.upload_end > 0 else 0
            ready_to_comp = (
                (t.compute_start - t.ready_time) * 1000
                if t.ready_time > 0 and t.compute_start > 0
                else 0
            )
            compute_ms = (t.compute_end - t.compute_start) * 1000 if t.compute_end > 0 else 0
            evict_ms = (t.evict_end - t.evict_start) * 1000 if t.evict_end > 0 else 0
            wait_ms = t.ensure_wait * 1000

            print(
                f"  L{t.layer:>3} | {read_ms:>7.1f} | {deser_ms:>7.1f} | {upload_ms:>7.1f} | "
                f"{ready_to_comp:>9.1f} | {compute_ms:>7.1f} | {evict_ms:>7.1f} | "
                f"{wait_ms:>5.1f} | {t.overlap_status}"
            )

        print(f"{'='*80}\n")

    def print_layer_detail(self, max_layers: int = 32) -> None:
        """Print per-layer block format."""
        traces = self.get_traces()[:max_layers]
        if not traces:
            print("No traces recorded.")
            return
        print(f"\n{'='*50}")
        print(f"LAYER DETAIL — {len(traces)} layers")
        print(f"{'='*50}")
        for t in traces:
            d = t.durations()
            print(f"\n  Layer {t.layer}")
            print(f"    Read:        {d['read_ms']:>7.1f} ms")
            print(f"    Deserialize: {d['deserialize_ms']:>7.1f} ms")
            print(f"    Upload:      {d['upload_ms']:>7.1f} ms")
            print(f"    Wait:        {d['ensure_wait_ms']:>7.1f} ms")
            print(f"    Compute:     {d['compute_ms']:>7.1f} ms")
            print(f"    Evict:       {d['evict_ms']:>7.1f} ms")
            print(f"    Status:      {t.overlap_status or 'unknown'}")
        print(f"\n{'='*50}\n")

    def print_token_summary(self) -> None:
        """Print aggregated multi-token summary with per-stage statistics."""
        traces = self.get_traces()
        if not traces:
            print("No traces recorded.")
            return

        def _collect(fn):
            vals = [fn(t) for t in traces]
            return [v for v in vals if v > 0]

        read_ms = _collect(lambda t: (t.read_end - t.read_start) * 1000)
        deser_ms = _collect(lambda t: (t.deserialize_end - t.deserialize_start) * 1000)
        upload_ms = _collect(lambda t: (t.upload_end - t.upload_start) * 1000)
        compute_ms = _collect(lambda t: (t.compute_end - t.compute_start) * 1000)
        evict_ms = _collect(lambda t: (t.evict_end - t.evict_start) * 1000)
        wait_ms = _collect(lambda t: t.ensure_wait * 1000)

        def _stats(vals: list[float]) -> str:
            if not vals:
                return "  (no data)"
            s = sorted(vals)
            mean = sum(s) / len(s)
            med = s[len(s) // 2]
            p95 = s[int(len(s) * 0.95)] if len(s) > 1 else s[0]
            mx = s[-1]
            return f"  mean={mean:>7.1f}  med={med:>7.1f}  p95={p95:>7.1f}  max={mx:>7.1f} ms"

        overlap = _count_overlap(traces)
        total_compute = sum(compute_ms)
        total_wait = sum(wait_ms)
        total = total_compute + total_wait
        gpu_eff = total_compute / total if total > 0 else 0.0

        print(f"\n{'='*70}")
        print(f"TOKEN SUMMARY — {len(traces)} layer-traces across all tokens")
        print(f"{'='*70}")
        print(f"  Read:       {_stats(read_ms)}")
        print(f"  Deserialize:{_stats(deser_ms)}")
        print(f"  Upload:     {_stats(upload_ms)}")
        print(f"  Compute:    {_stats(compute_ms)}")
        print(f"  Evict:      {_stats(evict_ms)}")
        print(f"  Wait:       {_stats(wait_ms)}")
        print()
        print(f"  GPU efficiency: {gpu_eff*100:.1f}%")
        print(f"  Overlap: {overlap}")
        print(f"{'='*70}\n")

    def print_pipeline_summary(self) -> None:
        """Print a comprehensive pipeline summary."""
        metrics = self.pipeline_metrics()
        print(f"\n{'='*70}")
        print("PIPELINE METRICS")
        print(f"{'='*70}")
        print(f"  Layers:               {metrics.num_layers}")
        print(f"  Tokens:               {metrics.num_tokens}")
        print()
        print("  Per-stage averages (ms):")
        print(f"    SSD Read:           {metrics.avg_read_ms:>7.1f}")
        print(f"    Deserialize:        {metrics.avg_deserialize_ms:>7.1f}")
        print(f"    Upload to device:   {metrics.avg_upload_ms:>7.1f}")
        print(f"    Compute:            {metrics.avg_compute_ms:>7.1f}")
        print(f"    Evict:              {metrics.avg_evict_ms:>7.1f}")
        print(f"    ensure() wait:      {metrics.avg_ensure_wait_ms:>7.1f}")
        print()
        print(
            f"  ensure() wait:        max={metrics.max_ensure_wait_ms:.1f} ms  "
            f"total={metrics.total_ensure_wait_ms:.1f} ms"
        )
        print(f"  Pipeline stalls:      {metrics.pipeline_stall_count}")
        print()
        print("  Prefetch overlap:")
        print(f"    Hits:               {metrics.prefetch_hits}")
        print(f"    Waits:              {metrics.prefetch_waits}")
        print(f"    Misses:             {metrics.prefetch_misses}")
        print(f"    Hit rate:           {metrics.prefetch_hit_rate*100:.1f}%")
        print()
        print(f"  GPU efficiency:       {metrics.gpu_efficiency*100:.1f}%")
        print(f"  GPU busy:             {metrics.gpu_busy_time_ms:.1f} ms")
        print(f"  GPU idle (est):       {metrics.gpu_idle_time_ms:.1f} ms")
        print()
        print(f"  Peak RSS:             {metrics.peak_rss_bytes / 1e9:.2f} GB")
        print(f"  Avg RSS:              {metrics.avg_rss_bytes / 1e9:.2f} GB")
        print(f"  Wall time:            {metrics.wall_time_ms:.1f} ms")
        print(f"{'='*70}\n")


# ── Helpers ────────────────────────────────────────────────────────────────

def _get_rss() -> int:
    """Instantaneous process RSS in bytes.

    psutil gives a true point-in-time reading; ``resource.ru_maxrss`` is a
    monotone high-water mark, which made per-stage "snapshots" grow even when
    memory was freed between stages. The Process handle is cached —
    constructing it per call measurably inflates profiler overhead.
    """
    try:
        return int(_psutil_process().memory_info().rss)
    except Exception:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        except Exception:
            return 0


@lru_cache(maxsize=1)
def _psutil_process():
    import psutil

    return psutil.Process()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = int(len(sorted_v) * 0.95)
    return sorted_v[min(idx, len(sorted_v) - 1)]


def _count_overlap(traces: list[LayerTrace]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in traces:
        status = t.overlap_status or "unknown"
        counts[status] = counts.get(status, 0) + 1
    return counts


def collect_hardware_metadata(
    device_type: str = "",
    model_id: str = "",
    num_layers: int = 0,
    model_size_mb: float = 0.0,
    quantization: str = "",
    window_size: int = 0,
    prefetch_depth: int = 0,
    worker_count: int = 0,
    direct_io: bool = False,
    resident_count: int = 0,
) -> HardwareMetadata:
    """Collect hardware and environment metadata."""
    import sys as _sys

    hw = HardwareMetadata(
        device_type=device_type,
        macos_version=platform.mac_ver()[0] if hasattr(platform, "mac_ver") else "",
        pytorch_version=torch.__version__,
        python_version=f"{_sys.version_info.major}.{_sys.version_info.minor}.{_sys.version_info.micro}",
        model_id=model_id,
        num_layers=num_layers,
        model_size_mb=model_size_mb,
        quantization=quantization,
        window_size=window_size,
        prefetch_depth=prefetch_depth,
        worker_count=worker_count,
        direct_io=direct_io,
        resident_count=resident_count,
    )

    try:
        if device_type == "mps":
            hw.mps_enabled = bool(
                getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            )
            hw.unified_memory = True
    except Exception:
        pass

    try:
        import psutil
        hw.memory_gb = psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass

    if device_type == "mps":
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0 and result.stdout.strip():
                hw.chip_name = result.stdout.strip()
        except Exception:
            pass

    return hw
