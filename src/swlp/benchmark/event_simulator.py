"""Discrete-event simulator for SWLP scheduling strategies — the *policy lab*.

    See the "which of the three simulators" note in core/simulator.py. In short:
    this one holds the workload fixed and varies the *strategy*; core/simulator.py
    holds the strategy fixed and varies the *settings*; benchmark/simulator.py is
    closed-form arithmetic with no event loop. Not duplicates — different axes.

Models the full layer lifecycle (SSD → Deserialize → Upload → Compute → Evict)
without loading real models.  Enables comparing scheduling strategies under
identical workloads before touching the real system.

Usage:
    from swlp.benchmark.event_simulator import (
        EventSimulator, SimConfig, LayerTimings,
        SlidingWindowStrategy, ResidentCacheStrategy,
        AdaptiveResidentStrategy, DynamicSchedulerStrategy,
    )

    config = SimConfig(num_layers=32, layer_weight_mb=450, ram_budget_mb=16000)
    timings = LayerTimings(read_ms=12, deserialize_ms=8, upload_ms=5, compute_ms=40, evict_ms=0.5)

    for Strategy in [SlidingWindowStrategy, ResidentCacheStrategy, ...]:
        strategy = Strategy(config=config, timings=timings)
        sim = EventSimulator(config, timings, strategy)
        result = sim.simulate(num_tokens=10)
        print(result.summary())
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# ── Constants ────────────────────────────────────────────────────────────────

_MS = 0.001  # 1 ms in seconds


# ── Enums ────────────────────────────────────────────────────────────────────

class LayerState(Enum):
    """Lifecycle state of a single transformer layer."""
    UNLOADED = "unloaded"
    READING = "reading"
    DESERIALIZING = "deserializing"
    UPLOADING = "uploading"
    READY = "ready"
    COMPUTING = "computing"
    EVICTING = "evicting"


class EventType(Enum):
    """Events that advance the simulation."""
    PREFETCH_START = "prefetch_start"
    READ_DONE = "read_done"
    DESER_DONE = "deser_done"
    UPLOAD_DONE = "upload_done"
    ENSURE = "ensure"
    COMPUTE_START = "compute_start"
    COMPUTE_DONE = "compute_done"
    EVICT_DONE = "evict_done"


class OverlapStatus(Enum):
    """Why ensure() blocked or didn't."""
    HIT = "hit"          # prefetch done before ensure
    WAIT = "wait"        # prefetch still running
    MISS = "miss"        # no prefetch at all
    RESIDENT = "resident"  # loaded from CPU-RAM cache


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass(slots=True)
class LayerTimings:
    """Per-layer timing profile (all in milliseconds)."""
    read_ms: float = 12.0
    deserialize_ms: float = 8.0
    upload_ms: float = 5.0
    compute_ms: float = 40.0
    evict_ms: float = 0.5

    def read_s(self) -> float:
        return self.read_ms * _MS

    def deserialize_s(self) -> float:
        return self.deserialize_ms * _MS

    def upload_s(self) -> float:
        return self.upload_ms * _MS

    def compute_s(self) -> float:
        return self.compute_ms * _MS

    def ssd_to_ready_s(self) -> float:
        """Total time from read start to ready (SSD path)."""
        return self.read_s() + self.deserialize_s() + self.upload_s()

    def ssd_to_ready_ms(self) -> float:
        return self.ssd_to_ready_s() / _MS


@dataclass(slots=True)
class SimConfig:
    """Simulation parameters."""
    num_layers: int = 32
    layer_weight_mb: float = 450.0
    ram_budget_mb: float = 16000.0
    window_size: int = 4
    prefetch_depth: int = 2
    # Max concurrent SSD reads (thread pool size).
    max_ssd_concurrency: int = 2
    # Max concurrent MPS uploads.
    max_upload_concurrency: int = 2
    # SSD read jitter: multiply read_ms by uniform(1-jitter, 1+jitter).
    read_jitter: float = 0.1
    # Compute jitter.
    compute_jitter: float = 0.05
    # Seed for reproducible jitter.
    seed: int = 42


@dataclass(slots=True)
class LayerTrace:
    """Per-layer trace from simulation."""
    layer: int
    read_start: float = 0.0
    read_end: float = 0.0
    deserialize_start: float = 0.0
    deserialize_end: float = 0.0
    upload_start: float = 0.0
    upload_end: float = 0.0
    ensure_time: float = 0.0
    compute_start: float = 0.0
    compute_end: float = 0.0
    evict_start: float = 0.0
    evict_end: float = 0.0
    ensure_wait_ms: float = 0.0
    overlap_status: str = ""
    token: int = 0  # which token this belongs to

    def durations_ms(self) -> dict[str, float]:
        return {
            "read_ms": (self.read_end - self.read_start) / _MS if self.read_end > 0 else 0.0,
            "deserialize_ms": (
                (self.deserialize_end - self.deserialize_start) / _MS
                if self.deserialize_end > 0
                else 0.0
            ),
            "upload_ms": (
                (self.upload_end - self.upload_start) / _MS if self.upload_end > 0 else 0.0
            ),
            "compute_ms": (
                (self.compute_end - self.compute_start) / _MS if self.compute_end > 0 else 0.0
            ),
            "evict_ms": (self.evict_end - self.evict_start) / _MS if self.evict_end > 0 else 0.0,
            "ensure_wait_ms": self.ensure_wait_ms,
        }


@dataclass(slots=True)
class SimulationResult:
    """Aggregated results from a full simulation run."""
    strategy_name: str
    num_layers: int
    num_tokens: int
    total_seconds: float
    per_token_ms: float
    throughput_toks_per_sec: float
    # Averages across all layers × tokens
    avg_read_ms: float
    avg_deserialize_ms: float
    avg_upload_ms: float
    avg_compute_ms: float
    avg_evict_ms: float
    avg_ensure_wait_ms: float
    # Overlap efficiency
    overlap_hit_rate: float
    overlap_hits: int
    overlap_waits: int
    overlap_misses: int
    overlap_residents: int
    # Resource utilization
    gpu_busy_pct: float
    ssd_busy_pct: float
    ram_peak_layers: int
    # I/O intensity
    gb_per_token: float = 0.0  # GB read from SSD per generated token
    # Per-token breakdown
    per_token_seconds: list[float] = field(default_factory=list)
    # All layer traces
    traces: list[LayerTrace] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"{'='*70}",
            f"SIMULATION: {self.strategy_name}",
            f"{'='*70}",
            f"Layers: {self.num_layers}  |  Tokens: {self.num_tokens}",
            f"Total: {self.total_seconds:.3f}s  |  Per-token: {self.per_token_ms:.1f}ms  |  "
            f"{self.throughput_toks_per_sec:.2f} tok/s",
            f"I/O:   {self.gb_per_token:.2f} GB/token",
            "",
            f"{'Per-Layer Averages':}",
            f"  Read:       {self.avg_read_ms:>7.1f} ms",
            f"  Deserialize:{self.avg_deserialize_ms:>7.1f} ms",
            f"  Upload:     {self.avg_upload_ms:>7.1f} ms",
            f"  Compute:    {self.avg_compute_ms:>7.1f} ms",
            f"  Evict:      {self.avg_evict_ms:>7.1f} ms",
            f"  Wait:       {self.avg_ensure_wait_ms:>7.1f} ms",
            "",
            f"{'Overlap Efficiency':}",
            f"  Hit rate:   {self.overlap_hit_rate*100:.1f}%",
            f"  Hits: {self.overlap_hits}  |  Waits: {self.overlap_waits}  |  "
            f"Misses: {self.overlap_misses}  |  Resident: {self.overlap_residents}",
            "",
            f"{'Resource Utilization':}",
            f"  GPU busy:   {self.gpu_busy_pct*100:.1f}%",
            f"  SSD busy:   {self.ssd_busy_pct*100:.1f}%",
            f"  RAM peak:   {self.ram_peak_layers} layers",
            f"{'='*70}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON output."""
        return {
            "strategy": self.strategy_name,
            "num_layers": self.num_layers,
            "num_tokens": self.num_tokens,
            "total_seconds": self.total_seconds,
            "per_token_ms": self.per_token_ms,
            "throughput_toks_per_sec": self.throughput_toks_per_sec,
            "avg_read_ms": self.avg_read_ms,
            "avg_deserialize_ms": self.avg_deserialize_ms,
            "avg_upload_ms": self.avg_upload_ms,
            "avg_compute_ms": self.avg_compute_ms,
            "avg_evict_ms": self.avg_evict_ms,
            "avg_ensure_wait_ms": self.avg_ensure_wait_ms,
            "overlap_hit_rate": self.overlap_hit_rate,
            "overlap_hits": self.overlap_hits,
            "overlap_waits": self.overlap_waits,
            "overlap_misses": self.overlap_misses,
            "overlap_residents": self.overlap_residents,
            "gpu_busy_pct": self.gpu_busy_pct,
            "ssd_busy_pct": self.ssd_busy_pct,
            "ram_peak_layers": self.ram_peak_layers,
            "gb_per_token": self.gb_per_token,
            "per_token_seconds": self.per_token_seconds,
        }

    def save(self, path: str | Path) -> None:
        """Save results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))


# ── Scheduling Strategies ────────────────────────────────────────────────────

class SchedulingStrategy(ABC):
    """Base class for scheduling strategies.

    Subclasses implement the policy decisions: what to prefetch, when to evict,
    and how to handle residency.
    """

    def __init__(self, config: SimConfig, timings: LayerTimings) -> None:
        self.config = config
        self.timings = timings

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def resident_layers(self) -> set[int]:
        """Layers permanently in CPU-RAM (never evicted, no SSD read)."""
        ...

    @abstractmethod
    def should_prefetch(
        self, current_layer: int, token: int, loaded: set[int], in_flight: set[int]
    ) -> list[int]:
        """Return list of layer indices to prefetch ahead of current_layer."""
        ...

    @abstractmethod
    def should_evict(self, current_layer: int, token: int, loaded: set[int]) -> set[int]:
        """Return set of layer indices to evict after computing current_layer."""
        ...


class SlidingWindowStrategy(SchedulingStrategy):
    """Current SWLP strategy: fixed sliding window.

    Layers 0..W-1 are in the window. After computing layer i, evict i and
    prefetch i+W. At most W layers in RAM.
    """

    @property
    def name(self) -> str:
        return "SlidingWindow"

    def resident_layers(self) -> set[int]:
        return set()

    def should_prefetch(
        self, current_layer: int, token: int, loaded: set[int], in_flight: set[int]
    ) -> list[int]:
        prefetch = []
        for ahead in range(1, self.config.prefetch_depth + 1):
            idx = current_layer + ahead
            if idx < self.config.num_layers and idx not in loaded and idx not in in_flight:
                prefetch.append(idx)
        return prefetch

    def should_evict(self, current_layer: int, token: int, loaded: set[int]) -> set[int]:
        return {current_layer}


class ResidentCacheStrategy(SchedulingStrategy):
    """Fixed resident cache: first R layers stay in CPU-RAM permanently.

    Non-resident layers stream from SSD as in the sliding window.
    """

    def __init__(self, config: SimConfig, timings: LayerTimings, resident_count: int = 0) -> None:
        super().__init__(config, timings)
        if resident_count <= 0:
            # Auto-compute: fit as many layers as possible in RAM
            available_mb = config.ram_budget_mb - 6144  # OS + working reserve (6 GB)
            resident_count = max(0, int(available_mb * 0.75 / config.layer_weight_mb))
            resident_count = min(resident_count, config.num_layers)
        self._resident_count = resident_count

    @property
    def name(self) -> str:
        return f"ResidentCache(R={self._resident_count})"

    def resident_layers(self) -> set[int]:
        return set(range(self._resident_count))

    def should_prefetch(
        self, current_layer: int, token: int, loaded: set[int], in_flight: set[int]
    ) -> list[int]:
        prefetch = []
        for ahead in range(1, self.config.prefetch_depth + 1):
            idx = current_layer + ahead
            if idx < self.config.num_layers and idx not in loaded and idx not in in_flight:
                if idx >= self._resident_count:  # resident layers don't need SSD
                    prefetch.append(idx)
        return prefetch

    def should_evict(self, current_layer: int, token: int, loaded: set[int]) -> set[int]:
        # Never evict resident layers
        if current_layer < self._resident_count:
            return set()
        return {current_layer}


class AdaptiveResidentStrategy(SchedulingStrategy):
    """Adaptive resident cache: resident size adapts to available RAM.

    Like ResidentCache but dynamically adjusts based on actual memory pressure.
    Also prefetches more aggressively when GPU is idle.
    """

    def __init__(self, config: SimConfig, timings: LayerTimings) -> None:
        super().__init__(config, timings)
        available_mb = config.ram_budget_mb - 6144
        self._base_resident = max(0, int(available_mb * 0.75 / config.layer_weight_mb))
        self._base_resident = min(self._base_resident, config.num_layers)
        # Adaptive: if compute >> SSD, increase streaming depth
        if timings.compute_s() > timings.ssd_to_ready_s() * 2:
            self._adaptive_depth = min(
                config.prefetch_depth + 2, config.num_layers - self._base_resident
            )
        else:
            self._adaptive_depth = config.prefetch_depth

    @property
    def name(self) -> str:
        return f"AdaptiveResident(R={self._base_resident},D={self._adaptive_depth})"

    def resident_layers(self) -> set[int]:
        return set(range(self._base_resident))

    def should_prefetch(
        self, current_layer: int, token: int, loaded: set[int], in_flight: set[int]
    ) -> list[int]:
        prefetch = []
        for ahead in range(1, self._adaptive_depth + 1):
            idx = current_layer + ahead
            if idx < self.config.num_layers and idx not in loaded and idx not in in_flight:
                if idx >= self._base_resident:
                    prefetch.append(idx)
        return prefetch

    def should_evict(self, current_layer: int, token: int, loaded: set[int]) -> set[int]:
        if current_layer < self._base_resident:
            return set()
        return {current_layer}


class DynamicSchedulerStrategy(SchedulingStrategy):
    """Dynamic scheduler: decides resident size and prefetch depth per-token.

    Observes the previous token's GPU idle ratio and adjusts:
    - If GPU idle > 30%: increase prefetch depth (SSD is the bottleneck)
    - If GPU idle < 10%: decrease prefetch depth (compute is the bottleneck)
    - Resident size stays fixed (set at init based on RAM).
    """

    def __init__(self, config: SimConfig, timings: LayerTimings) -> None:
        super().__init__(config, timings)
        available_mb = config.ram_budget_mb - 6144
        self._resident_count = max(0, int(available_mb * 0.75 / config.layer_weight_mb))
        self._resident_count = min(self._resident_count, config.num_layers)
        self._prefetch_depth = config.prefetch_depth
        self._min_depth = 1
        self._max_depth = min(config.num_layers - self._resident_count, 8)

    @property
    def name(self) -> str:
        return f"DynamicScheduler(R={self._resident_count})"

    def resident_layers(self) -> set[int]:
        return set(range(self._resident_count))

    def adjust_from_gpu_idle(self, gpu_idle_pct: float) -> None:
        """Called after each token to adapt prefetch depth."""
        if gpu_idle_pct > 0.30:
            self._prefetch_depth = min(self._prefetch_depth + 1, self._max_depth)
        elif gpu_idle_pct < 0.10:
            self._prefetch_depth = max(self._prefetch_depth - 1, self._min_depth)

    def should_prefetch(
        self, current_layer: int, token: int, loaded: set[int], in_flight: set[int]
    ) -> list[int]:
        prefetch = []
        for ahead in range(1, self._prefetch_depth + 1):
            idx = current_layer + ahead
            if idx < self.config.num_layers and idx not in loaded and idx not in in_flight:
                if idx >= self._resident_count:
                    prefetch.append(idx)
        return prefetch

    def should_evict(self, current_layer: int, token: int, loaded: set[int]) -> set[int]:
        if current_layer < self._resident_count:
            return set()
        return {current_layer}


# ── Simulator ────────────────────────────────────────────────────────────────

class EventSimulator:
    """Discrete-event simulator for SWLP scheduling.

    Models the two-stage pipeline (SSD pool → upload pool) with configurable
    concurrency, jitter, and scheduling strategies.
    """

    def __init__(
        self,
        config: SimConfig,
        timings: LayerTimings,
        strategy: SchedulingStrategy,
    ) -> None:
        self.config = config
        self.timings = timings
        self.strategy = strategy

        # State
        self._layer_state: dict[int, LayerState] = {
            i: LayerState.UNLOADED for i in range(config.num_layers)
        }
        self._layer_events: dict[int, list[tuple[float, EventType]]] = {
            i: [] for i in range(config.num_layers)
        }
        self._traces: list[LayerTrace] = []
        self._time = 0.0
        self._resident = strategy.resident_layers()

        # Resource tracking
        self._ssd_busy_until: list[float] = []  # per-slot busy times
        self._upload_busy_until: list[float] = []
        self._gpu_busy_until = 0.0
        self._ram_peak = 0
        self._ram_current = 0

        # Overlap tracking
        self._hits = 0
        self._waits = 0
        self._misses = 0
        self._residents = 0

        # In-flight tracking
        self._ssd_in_flight: set[int] = set()
        self._upload_in_flight: set[int] = set()
        self._all_in_flight: set[int] = set()

        # Pending completions: (time, layer, stage)
        self._pending: list[tuple[float, int, str]] = []

        # RNG for jitter
        import random
        self._rng = random.Random(config.seed)

        # Pre-allocate SSD slots
        self._ssd_busy_until = [0.0] * config.max_ssd_concurrency
        self._upload_busy_until = [0.0] * config.max_upload_concurrency

    def _jittered(self, base_ms: float, jitter: float) -> float:
        """Apply jitter to a timing value."""
        if jitter <= 0:
            return base_ms * _MS
        factor = 1.0 + self._rng.uniform(-jitter, jitter)
        return base_ms * factor * _MS

    def _find_ssd_slot(self) -> int | None:
        """Find an available SSD slot, or None if all busy."""
        earliest = min(self._ssd_busy_until)
        if earliest <= self._time:
            return self._ssd_busy_until.index(earliest)
        return None

    def _find_upload_slot(self) -> int | None:
        """Find an available upload slot, or None if all busy."""
        earliest = min(self._upload_busy_until)
        if earliest <= self._time:
            return self._upload_busy_until.index(earliest)
        return None

    def _start_read(self, layer: int) -> None:
        """Begin SSD read for a layer."""
        slot = self._find_ssd_slot()
        if slot is None:
            return  # all SSD slots busy — will retry

        read_time = self._jittered(self.timings.read_ms, self.config.read_jitter)
        self._ssd_busy_until[slot] = self._time + read_time

        self._layer_state[layer] = LayerState.READING
        self._layer_events[layer].append((self._time, EventType.PREFETCH_START))
        self._ssd_in_flight.add(layer)
        self._all_in_flight.add(layer)

        # Schedule read completion
        self._pending.append((self._time + read_time, layer, "read"))

    def _start_deserialize(self, layer: int) -> None:
        """Begin deserialization (immediately after read)."""
        deser_time = self._jittered(self.timings.deserialize_ms, 0.02)
        self._layer_state[layer] = LayerState.DESERIALIZING
        self._layer_events[layer].append((self._time, EventType.READ_DONE))

        # Deserialization is CPU-only, happens inline (no slot contention)
        self._pending.append((self._time + deser_time, layer, "deser"))

    def _start_upload(self, layer: int) -> None:
        """Begin MPS upload (after deserialization)."""
        slot = self._find_upload_slot()
        if slot is None:
            # All upload slots busy — schedule for when earliest slot frees
            earliest = min(self._upload_busy_until)
            self._pending.append((earliest, layer, "upload_retry"))
            return

        upload_time = self._jittered(self.timings.upload_ms, 0.05)
        self._upload_busy_until[slot] = self._time + upload_time

        self._layer_state[layer] = LayerState.UPLOADING
        self._layer_events[layer].append((self._time, EventType.DESER_DONE))
        self._ssd_in_flight.discard(layer)

        self._pending.append((self._time + upload_time, layer, "upload"))

    def _start_compute(self, layer: int, token: int) -> None:
        """Begin GPU compute for a layer."""
        compute_time = self._jittered(self.timings.compute_ms, self.config.compute_jitter)

        self._layer_state[layer] = LayerState.COMPUTING
        self._gpu_busy_until = self._time + compute_time
        self._layer_events[layer].append((self._time, EventType.COMPUTE_START))

        # Record trace
        trace = LayerTrace(layer=layer, token=token)
        events = self._layer_events[layer]
        for t, evt in events:
            if evt == EventType.PREFETCH_START:
                trace.read_start = t
            elif evt == EventType.READ_DONE:
                trace.read_end = t
                trace.deserialize_start = t
            elif evt == EventType.DESER_DONE:
                trace.deserialize_end = t
                trace.upload_start = t
            elif evt == EventType.UPLOAD_DONE:
                trace.upload_end = t
            elif evt == EventType.COMPUTE_START:
                trace.compute_start = t
        self._traces.append(trace)

        self._pending.append((self._time + compute_time, layer, "compute"))

    def _finish_compute(self, layer: int, token: int) -> None:
        """Compute done — evict and move on."""
        self._layer_events[layer].append((self._time, EventType.COMPUTE_DONE))
        # Update trace
        for tr in reversed(self._traces):
            if tr.layer == layer and tr.token == token:
                tr.compute_end = self._time
                break

        # Evict
        self._start_evict(layer, token)

    def _start_evict(self, layer: int, token: int) -> None:
        """Begin eviction."""
        evict_time = self._jittered(self.timings.evict_ms, 0.1)
        self._layer_state[layer] = LayerState.EVICTING
        self._layer_events[layer].append((self._time, EventType.EVICT_DONE))
        # Record evict_start on the matching trace (match by layer only — evict is per-layer)
        for tr in reversed(self._traces):
            if tr.layer == layer:
                tr.evict_start = self._time
                break

        self._pending.append((self._time + evict_time, layer, "evict"))

    def _finish_evict(self, layer: int, token: int) -> None:
        """Eviction complete — layer is now UNLOADED."""
        self._layer_state[layer] = LayerState.UNLOADED
        self._layer_events[layer] = []  # reset events for next token
        self._all_in_flight.discard(layer)
        self._upload_in_flight.discard(layer)
        self._ram_current -= 1
        # Update trace (match by layer only — evict is per-layer)
        for tr in reversed(self._traces):
            if tr.layer == layer:
                tr.evict_end = self._time
                break

    def _ensure_layer(self, layer: int, token: int) -> None:
        """Compute thread needs this layer now — determine overlap status."""
        state = self._layer_state[layer]

        trace = None
        for tr in reversed(self._traces):
            if tr.layer == layer and tr.token == token:
                trace = tr
                break

        if layer in self._resident:
            # Resident: apply from CPU RAM (fast)
            self._residents += 1
            if trace:
                trace.ensure_time = self._time
                trace.ensure_wait_ms = 0.0
                trace.overlap_status = OverlapStatus.RESIDENT.value
            return

        if (
            state == LayerState.UPLOADING
            or state == LayerState.READING
            or state == LayerState.DESERIALIZING
        ):
            # Prefetch still running — must wait
            self._waits += 1
            if trace:
                trace.ensure_time = self._time
                trace.overlap_status = OverlapStatus.WAIT.value
        elif state == LayerState.UNLOADED:
            # No prefetch was running — sync fallback
            self._misses += 1
            if trace:
                trace.ensure_time = self._time
                trace.overlap_status = OverlapStatus.MISS.value
        else:
            # Already ready (hit)
            self._hits += 1
            if trace:
                trace.ensure_time = self._time
                trace.overlap_status = OverlapStatus.HIT.value

    def simulate(self, num_tokens: int = 10) -> SimulationResult:
        """Run the simulation for N tokens.

        Each token: sweep all layers 0..num_layers-1.
        """
        self._time = 0.0
        self._traces = []
        self._hits = self._waits = self._misses = self._residents = 0
        self._ram_peak = self._ram_current = 0
        token_times: list[float] = []

        # Initialize resident layers
        for idx in self._resident:
            self._layer_state[idx] = LayerState.UNLOADED  # will be "ensure"d without SSD
            self._ram_current += 1
        self._ram_peak = max(self._ram_peak, self._ram_current)

        for token in range(num_tokens):
            token_start = self._time

            # Reset layer events for this token
            for i in range(self.config.num_layers):
                self._layer_events[i] = []

            # Dynamic scheduler: adjust from previous token's GPU idle
            if isinstance(self.strategy, DynamicSchedulerStrategy) and token > 0:
                prev_token_traces = [tr for tr in self._traces if tr.token == token - 1]
                if prev_token_traces:
                    total_time = token_times[-1] if token_times else 1.0
                    gpu_time = sum(
                        tr.compute_end - tr.compute_start
                        for tr in prev_token_traces if tr.compute_start > 0 and tr.compute_end > 0
                    )
                    gpu_idle = max(0, 1.0 - gpu_time / total_time) if total_time > 0 else 0.5
                    self.strategy.adjust_from_gpu_idle(gpu_idle)

            # Warmup: prefetch first `prefetch_depth` layers
            for ahead in range(min(self.config.prefetch_depth, self.config.num_layers)):
                if ahead not in self._resident:
                    self._start_read(ahead)

            # Process each layer
            for layer in range(self.config.num_layers):
                # Trigger prefetch for upcoming layers
                to_prefetch = self.strategy.should_prefetch(
                    layer, token, self._loaded_set(), self._all_in_flight
                )
                for idx in to_prefetch:
                    if self._layer_state[idx] == LayerState.UNLOADED:
                        self._start_read(idx)
                        self._ram_current += 1
                        self._ram_peak = max(self._ram_peak, self._ram_current)

                # Ensure layer is ready
                self._ensure_layer(layer, token)

                # Process pending events up to this point
                self._process_pending_until(self._time)

                # Start compute
                self._start_compute(layer, token)

                # Process pending events (compute takes time, other things finish)
                self._process_pending_until(self._gpu_busy_until)

                # Finish compute
                self._finish_compute(layer, token)

                # Evict
                to_evict = self.strategy.should_evict(layer, token, self._loaded_set())
                for idx in to_evict:
                    if self._layer_state[idx] not in (LayerState.UNLOADED, LayerState.EVICTING):
                        self._start_evict(idx, token)

                # Process remaining events
                self._process_pending_until(self._time + 0.001)

            # Wait for any remaining in-flight work
            self._process_all_pending()

            token_times.append(self._time - token_start)

        # Compute ensure_wait_ms for each trace
        for tr in self._traces:
            if tr.overlap_status == OverlapStatus.WAIT.value:
                # Wait time = time between ensure and when the layer actually became ready
                ready_time = tr.upload_end if tr.upload_end > 0 else tr.compute_start
                if ready_time > tr.ensure_time:
                    tr.ensure_wait_ms = (ready_time - tr.ensure_time) / _MS
            elif tr.overlap_status == OverlapStatus.MISS.value:
                # Sync fallback: wait = total SSD+deser+upload time from ensure
                tr.ensure_wait_ms = self.timings.ssd_to_ready_ms()

        return self._build_result(num_tokens, token_times)

    def _loaded_set(self) -> set[int]:
        """Return set of layers currently on device."""
        return {
            i for i, s in self._layer_state.items()
            if s in (LayerState.READY, LayerState.COMPUTING)
        }

    def _process_pending_until(self, deadline: float) -> None:
        """Process pending events up to deadline."""
        while self._pending:
            self._pending.sort(key=lambda x: x[0])
            t, layer, stage = self._pending[0]
            if t > deadline:
                break
            self._pending.pop(0)
            self._time = max(self._time, t)

            if stage == "read":
                self._start_deserialize(layer)
            elif stage == "deser":
                self._start_upload(layer)
            elif stage == "upload_retry":
                self._start_upload(layer)
            elif stage == "upload":
                self._layer_state[layer] = LayerState.READY
                self._layer_events[layer].append((self._time, EventType.UPLOAD_DONE))
                self._upload_in_flight.add(layer)
                self._all_in_flight.discard(layer)
                # Update trace upload_end
                for tr in reversed(self._traces):
                    if tr.layer == layer and tr.upload_end == 0 and tr.upload_start > 0:
                        tr.upload_end = self._time
                        break
            elif stage == "compute":
                pass  # handled by caller
            elif stage == "evict":
                self._finish_evict(layer, 0)  # token doesn't matter for evict timing

    def _process_all_pending(self) -> None:
        """Process all remaining pending events."""
        while self._pending:
            self._pending.sort(key=lambda x: x[0])
            t, layer, stage = self._pending.pop(0)
            self._time = max(self._time, t)

            if stage == "read":
                self._start_deserialize(layer)
            elif stage == "deser":
                self._start_upload(layer)
            elif stage == "upload_retry":
                self._start_upload(layer)
            elif stage == "upload":
                self._layer_state[layer] = LayerState.READY
                self._layer_events[layer].append((self._time, EventType.UPLOAD_DONE))
                self._upload_in_flight.add(layer)
                self._all_in_flight.discard(layer)
            elif stage == "evict":
                self._finish_evict(layer, 0)

    def _build_result(self, num_tokens: int, token_times: list[float]) -> SimulationResult:
        """Build aggregated simulation result."""
        total = sum(token_times)
        per_token_ms = (total / num_tokens) / _MS if num_tokens > 0 else 0.0
        throughput = num_tokens / total if total > 0 else 0.0

        # Aggregate trace durations
        read_times = []
        deser_times = []
        upload_times = []
        compute_times = []
        evict_times = []
        wait_times = []

        for tr in self._traces:
            d = tr.durations_ms()
            if d["read_ms"] > 0:
                read_times.append(d["read_ms"])
            if d["deserialize_ms"] > 0:
                deser_times.append(d["deserialize_ms"])
            if d["upload_ms"] > 0:
                upload_times.append(d["upload_ms"])
            if d["compute_ms"] > 0:
                compute_times.append(d["compute_ms"])
            if d["evict_ms"] > 0:
                evict_times.append(d["evict_ms"])
            wait_times.append(d["ensure_wait_ms"])

        def _mean(lst: list[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        # Resource utilization
        total_wall = total if total > 0 else 1.0
        gpu_busy = sum(compute_times) * _MS / total_wall
        ssd_busy = sum(read_times) * _MS / total_wall

        # I/O intensity: total layer weight read / tokens generated
        total_layer_weight_mb = self.config.num_layers * self.config.layer_weight_mb
        gb_per_token = total_layer_weight_mb / 1024.0  # each token reads all layers once

        total_overlap = self._hits + self._waits + self._misses + self._residents
        hit_rate = self._hits / total_overlap if total_overlap > 0 else 0.0

        return SimulationResult(
            strategy_name=self.strategy.name,
            num_layers=self.config.num_layers,
            num_tokens=num_tokens,
            total_seconds=total,
            per_token_ms=per_token_ms,
            throughput_toks_per_sec=throughput,
            avg_read_ms=_mean(read_times),
            avg_deserialize_ms=_mean(deser_times),
            avg_upload_ms=_mean(upload_times),
            avg_compute_ms=_mean(compute_times),
            avg_evict_ms=_mean(evict_times),
            avg_ensure_wait_ms=_mean(wait_times),
            overlap_hit_rate=hit_rate,
            overlap_hits=self._hits,
            overlap_waits=self._waits,
            overlap_misses=self._misses,
            overlap_residents=self._residents,
            gpu_busy_pct=min(1.0, gpu_busy),
            ssd_busy_pct=min(1.0, ssd_busy),
            ram_peak_layers=self._ram_peak,
            gb_per_token=gb_per_token,
            per_token_seconds=token_times,
            traces=self._traces,
        )


# ── Convenience ──────────────────────────────────────────────────────────────

def compare_strategies(
    config: SimConfig,
    timings: LayerTimings,
    num_tokens: int = 10,
    strategies: list[SchedulingStrategy] | None = None,
) -> list[SimulationResult]:
    """Run all strategies and return results for comparison."""
    if strategies is None:
        strategies = [
            SlidingWindowStrategy(config, timings),
            ResidentCacheStrategy(config, timings),
            AdaptiveResidentStrategy(config, timings),
            DynamicSchedulerStrategy(config, timings),
        ]

    results = []
    for strategy in strategies:
        sim = EventSimulator(config, timings, strategy)
        result = sim.simulate(num_tokens=num_tokens)
        results.append(result)

    return results


def print_comparison(results: list[SimulationResult]) -> None:
    """Print a side-by-side comparison table."""
    print(f"\n{'='*90}")
    print(f"STRATEGY COMPARISON — {results[0].num_layers} layers, {results[0].num_tokens} tokens")
    print(f"{'='*90}")

    header = (
        f"{'Strategy':<30} | {'tok/s':>7} | {'ms/tok':>7} | "
        f"{'GPU%':>6} | {'SSD%':>6} | {'Hit%':>6} | {'Wait':>6} | {'RAM':>4}"
    )
    print(header)
    print(f"{'-'*30}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*4}")

    for r in results:
        print(
            f"{r.strategy_name:<30} | {r.throughput_toks_per_sec:>7.2f} | "
            f"{r.per_token_ms:>7.1f} | {r.gpu_busy_pct*100:>5.1f}% | "
            f"{r.ssd_busy_pct*100:>5.1f}% | {r.overlap_hit_rate*100:>5.1f}% | "
            f"{r.avg_ensure_wait_ms:>5.1f} | {r.ram_peak_layers:>4}"
        )

    print(f"{'='*90}\n")
