"""Discrete-event pipeline simulator for SWLP — the *tuner*.

    Which of the three simulators is this?
      core/simulator.py            ← YOU ARE HERE. One fixed strategy (the real
                                     StreamingScheduler), varying hardware and
                                     window/prefetch/resident knobs. Emits the
                                     same LayerTrace shape as the real profiler,
                                     so core/analyzer.py can read either.
                                     Used by `swlp sim` and core/sweep.py.
      benchmark/event_simulator.py   Many *strategies* (SlidingWindow,
                                     ResidentCache, Adaptive, Dynamic) under one
                                     fixed workload. Answers "which policy?",
                                     not "which setting?". Used by the research
                                     scripts and the paper.
      benchmark/simulator.py         No events at all — closed-form bottleneck
                                     arithmetic for `swlp simulate` scenarios.

    They look similar and are not duplicates: the axis of variation differs.
    Do not merge them without a test proving the merged model reproduces both.

Models hardware resources (SSD, upload engine, GPU, RAM) with explicit
queues and contention.  Produces the same LayerTrace output as the real
profiler, but with realistic pipeline stalls and GPU idle periods.

The simulator processes events chronologically.  Each event represents a
state transition: SSD read finishes, upload finishes, GPU starts/finishes
compute, etc.  Resources are modeled as bounded worker pools — when all
workers are busy, new tasks queue and wait, creating the stalls that
determine real-world throughput.

Usage:
    from swlp.core.simulator import SimulatorConfig, simulate

    config = SimulatorConfig(
        num_layers=32,
        layer_size_mb=512,
        ram_capacity_gb=16,
        window_size=2,
        prefetch_depth=4,
        worker_count=2,
        compute_time_ms=50,
        ssd_read_latency_ms=30,
        upload_latency_ms=10,
    )
    result = simulate(config)
    result.print_timeline()
    result.dump("sim_traces.json")
"""

from __future__ import annotations

import heapq
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .profiler import LayerTrace, compute_pipeline_metrics

# ── Configuration ──────────────────────────────────────────────────────────


@dataclass
class SimulatorConfig:
    """Configuration for the scheduler simulator."""

    num_layers: int = 32
    layer_size_mb: float = 512
    ram_capacity_gb: float = 16.0
    window_size: int = 2
    prefetch_depth: int = 4
    worker_count: int = 2
    compute_time_ms: float = 50.0
    ssd_read_latency_ms: float = 30.0
    upload_latency_ms: float = 10.0
    eviction_latency_ms: float = 5.0
    num_tokens: int = 10
    use_resident_cache: bool = False
    resident_count: int = 0
    # Memory pressure: when RAM usage exceeds this fraction, upload latency increases
    memory_pressure_threshold: float = 0.8  # 80% of RAM
    memory_pressure_penalty: float = 0.5  # 50% increase per 10% over threshold

    def to_dict(self) -> dict:
        return asdict(self)


# ── Events ─────────────────────────────────────────────────────────────────


@dataclass(order=True)
class Event:
    """A discrete event in the simulation.  Ordered by time_ms for the heap."""

    time_ms: float
    seq: int = field(compare=False)  # tiebreaker
    event_type: str = field(compare=False)
    layer: int = field(compare=False)
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)


# ── Resource pools ─────────────────────────────────────────────────────────


class _ResourcePool:
    """Bounded worker pool with queue.

    Tracks active workers and pending queue.  When all workers are busy,
    tasks queue and start at the earliest worker-free time.
    """

    def __init__(self, name: str, worker_count: int) -> None:
        self.name = name
        self.worker_count = worker_count
        self._busy_until: list[float] = []  # sorted heap of worker-free times
        self._active: dict[int, float] = {}  # layer -> end_time

    def acquire(self, now: float, duration_ms: float) -> tuple[float, bool]:
        """Try to acquire a worker.  Returns (start_time, was_delayed).

        If a worker is free, starts immediately.  Otherwise waits for the
        earliest free worker.
        """
        if len(self._busy_until) < self.worker_count:
            # Worker available
            start = now
            heapq.heappush(self._busy_until, now + duration_ms)
            return start, False
        else:
            # All busy — wait for earliest
            earliest = heapq.heappop(self._busy_until)
            start = max(now, earliest)
            heapq.heappush(self._busy_until, start + duration_ms)
            return start, True

    def start(self, layer: int, end_time: float) -> None:
        self._active[layer] = end_time

    def complete(self, layer: int) -> None:
        self._active.pop(layer, None)



# ── Simulation state ───────────────────────────────────────────────────────


class _SimState:
    """Mutable state shared across event handlers."""

    def __init__(self, config: SimulatorConfig) -> None:
        self.cfg = config
        self.current_time_ms = 0.0
        self.seq = 0
        self.events: list[Event] = []
        self.traces: dict[int, LayerTrace] = {}
        self.token_traces: list[list[LayerTrace]] = []

        # Resources
        self.ssd_pool = _ResourcePool("ssd", config.worker_count)
        self.upload_pool = _ResourcePool("upload", config.worker_count)

        # Layer state
        self.loaded: set[int] = set()          # layers with data on device
        self.resident: set[int] = set()        # layers permanently resident
        self.reading: set[int] = set()         # layers with SSD read in progress
        self.uploading: set[int] = set()       # layers with upload in progress
        self.queued_for_read: set[int] = set() # layers in SSD queue
        self.queued_for_upload: set[int] = set()

        # Ready queue: layers ready for GPU compute, in order
        self.ready_queue: list[int] = []
        self._computing: set[int] = set()  # layers currently being computed
        self.gpu_busy_until = 0.0

        # Prefetch tracking: which layers have been submitted for prefetch
        self.prefetched: set[int] = set()

        # RAM tracking
        self.peak_ram_mb = 0.0

        # GPU idle tracking
        self.gpu_idle_ms = 0.0
        self.gpu_busy_ms = 0.0

        # Resident layers
        if config.use_resident_cache:
            for i in range(min(config.resident_count, config.num_layers)):
                self.resident.add(i)
                self.loaded.add(i)

        # Current token
        self.current_token = 0
        self.current_layer = 0
        self.token_traces_this: list[LayerTrace] = []

    def _next_seq(self) -> int:
        self.seq += 1
        return self.seq

    def _add_event(self, time_ms: float, etype: str, layer: int, **kw) -> None:
        heapq.heappush(self.events, Event(time_ms, self._next_seq(), etype, layer, kw))

    def _get_trace(self, layer: int) -> LayerTrace:
        if layer not in self.traces:
            self.traces[layer] = LayerTrace(layer=layer)
        return self.traces[layer]

    def _update_ram(self) -> None:
        ram = len(self.loaded) * self.cfg.layer_size_mb
        self.peak_ram_mb = max(self.peak_ram_mb, ram)

    def _effective_upload_ms(self) -> float:
        """Calculate upload latency with tiered memory pressure.

        Three regions on Apple Silicon unified memory:

        Region 1: usage < 65%  → no penalty
        Region 2: 65% ≤ usage < 85% → linear penalty (allocator cost, cache pressure)
        Region 3: usage ≥ 85% → exponential penalty (compression, page migration, swap)
        """
        ram_mb = len(self.loaded) * self.cfg.layer_size_mb
        capacity_mb = self.cfg.ram_capacity_gb * 1024
        usage = ram_mb / capacity_mb if capacity_mb > 0 else 0

        # Region 1: no pressure
        if usage < 0.65:
            return self.cfg.upload_latency_ms

        # Region 2: linear penalty (65%–85%)
        if usage < 0.85:
            over = usage - 0.65
            fraction = over / 0.20  # 0→1 across this region
            penalty = 1.0 + fraction * 2.0  # up to 3x at 85%
            return self.cfg.upload_latency_ms * penalty

        # Region 3: exponential penalty (85%+)
        over = usage - 0.85
        # Exponential: at 90% → ~5x, at 95% → ~25x, at 100% → ~125x
        penalty = 3.0 * (5.0 ** (over / 0.15))
        return self.cfg.upload_latency_ms * penalty

    def _advance_layer(self) -> None:
        """Move to the next layer in the current token."""
        self.current_layer += 1
        if self.current_layer >= self.cfg.num_layers:
            self._finish_token()
        else:
            # Check if next layer is resident and ready
            self._check_resident_ready()
            # Check if next layer is already ready
            _try_start_compute(self)

    def _finish_token(self) -> None:
        """Complete the current token and start the next one."""
        self.token_traces.append(list(self.token_traces_this))
        self.token_traces_this = []
        self.current_token += 1
        if self.current_token < self.cfg.num_tokens:
            self.current_layer = 0
            self._start_token()
        else:
            self._finish_simulation()

    def _start_token(self) -> None:
        """Begin a new token pass."""
        # Check if current layer is resident and ready immediately
        self._check_resident_ready()
        # Prefetch non-resident layers within pipeline window
        window = max(self.cfg.window_size, self.cfg.prefetch_depth)
        for i in range(self.cfg.num_layers):
            if i - self.current_layer > window:
                break
            self._submit_prefetch(i)

    def _submit_prefetch(self, layer: int) -> None:
        """Submit a layer for background prefetch if not already loaded/queued."""
        if layer in self.loaded or layer in self.prefetched or layer in self.resident:
            return
        if layer >= self.cfg.num_layers:
            return
        self.prefetched.add(layer)
        self._add_event(self.current_time_ms, "prefetch_submit", layer)

    def _check_resident_ready(self) -> None:
        """If current layer is resident, mark it ready immediately."""
        if self.current_layer in self.resident and self.current_layer in self.loaded:
            # Only apply if not already in ready queue or computing
            if (self.current_layer not in self.ready_queue and
                self.current_layer not in self._computing):
                self._add_event(self.current_time_ms, "resident_apply", self.current_layer)

    def _finish_simulation(self) -> None:
        """Mark simulation complete."""
        self._add_event(self.current_time_ms, "simulation_end", -1)

    def check_invariants(self) -> None:
        """Assert layer state consistency. Catches FSM bugs early."""
        for layer in range(self.cfg.num_layers):
            # Resident layers must be loaded
            if layer in self.resident:
                assert layer in self.loaded, f"L{layer}: resident but not loaded"
            # Can't be reading and loaded
            if layer in self.reading:
                assert layer not in self.loaded, f"L{layer}: reading and loaded"
                assert layer not in self.resident, f"L{layer}: reading and resident"
            # Can't be uploading and resident
            if layer in self.uploading:
                assert layer not in self.resident, f"L{layer}: uploading and resident"
            # Can't be in ready_queue and computing
            if layer in self._computing:
                assert layer not in self.ready_queue, f"L{layer}: computing and in ready_queue"
            # Prefetched exactly once (or not at all)
            # (multiple prefetch_submit events for same layer = bug)


# ── Event handlers ─────────────────────────────────────────────────────────


def _handle_prefetch_submit(state: _SimState, layer: int) -> None:
    """A prefetch was submitted — start SSD read if worker available."""
    if layer in state.loaded or layer in state.resident:
        return
    if layer in state.reading or layer in state.queued_for_read:
        return

    trace = state._get_trace(layer)
    trace.prefetch_submit = state.current_time_ms / 1000

    # Try to acquire SSD worker
    start, delayed = state.ssd_pool.acquire(state.current_time_ms, state.cfg.ssd_read_latency_ms)
    if delayed:
        state.queued_for_read.add(layer)
        state._add_event(start, "ssd_read_start", layer)
    else:
        state.reading.add(layer)
        trace.read_start = start / 1000
        end = start + state.cfg.ssd_read_latency_ms
        state.ssd_pool.start(layer, end)
        state._add_event(end, "ssd_read_end", layer)
        state._add_event(start, "ssd_read_start", layer)


def _handle_ssd_read_start(state: _SimState, layer: int) -> None:
    """SSD read started."""
    if layer in state.queued_for_read:
        state.queued_for_read.discard(layer)
        state.reading.add(layer)
        trace = state._get_trace(layer)
        trace.read_start = state.current_time_ms / 1000
        end = state.current_time_ms + state.cfg.ssd_read_latency_ms
        state.ssd_pool.start(layer, end)
        state._add_event(end, "ssd_read_end", layer)


def _handle_ssd_read_end(state: _SimState, layer: int) -> None:
    """SSD read completed — submit upload and prefetch ahead."""
    state.ssd_pool.complete(layer)
    state.reading.discard(layer)

    trace = state._get_trace(layer)
    trace.read_end = state.current_time_ms / 1000
    trace.deserialize_start = state.current_time_ms / 1000
    trace.deserialize_end = state.current_time_ms / 1000  # negligible

    # Submit upload
    state._add_event(state.current_time_ms, "upload_submit", layer)

    # SSD worker is now free — prefetch next layer
    _prefetch_ahead(state, layer)


def _handle_upload_submit(state: _SimState, layer: int) -> None:
    """Upload submitted — acquire upload worker."""
    if layer in state.uploading or layer in state.queued_for_upload:
        return

    trace = state._get_trace(layer)
    trace.upload_start = state.current_time_ms / 1000

    upload_ms = state._effective_upload_ms()
    start, delayed = state.upload_pool.acquire(state.current_time_ms, upload_ms)
    if delayed:
        state.queued_for_upload.add(layer)
        state._add_event(start, "upload_start", layer)
    else:
        state.uploading.add(layer)
        trace.upload_start = start / 1000
        end = start + upload_ms
        state.upload_pool.start(layer, end)
        state._add_event(end, "upload_end", layer)
        state._add_event(start, "upload_start", layer)


def _handle_upload_start(state: _SimState, layer: int) -> None:
    """Upload started."""
    if layer in state.queued_for_upload:
        state.queued_for_upload.discard(layer)
        state.uploading.add(layer)
        trace = state._get_trace(layer)
        trace.upload_start = state.current_time_ms / 1000
        upload_ms = state._effective_upload_ms()
        end = state.current_time_ms + upload_ms
        state.upload_pool.start(layer, end)
        state._add_event(end, "upload_end", layer)


def _handle_upload_end(state: _SimState, layer: int) -> None:
    """Upload completed — layer is ready for compute."""
    state.upload_pool.complete(layer)
    state.uploading.discard(layer)
    state.prefetched.discard(layer)

    trace = state._get_trace(layer)
    trace.upload_end = state.current_time_ms / 1000
    trace.ready_time = state.current_time_ms / 1000

    state.loaded.add(layer)
    state._update_ram()

    # Add to ready queue if not already there
    if layer not in state.ready_queue:
        state.ready_queue.append(layer)
        state.ready_queue.sort()

    # Try to start compute for the current layer
    _try_start_compute(state)


def _handle_resident_apply(state: _SimState, layer: int) -> None:
    """Resident layer applied — ready for compute.

    Even resident layers have a small apply latency (CPU→MPS copy),
    which increases under memory pressure.
    """
    trace = state._get_trace(layer)
    apply_ms = state._effective_upload_ms() * 0.3  # resident apply is ~30% of full upload
    trace.upload_start = state.current_time_ms / 1000
    trace.upload_end = (state.current_time_ms + apply_ms) / 1000
    trace.ready_time = (state.current_time_ms + apply_ms) / 1000
    trace.overlap_status = "resident"

    state._add_event(state.current_time_ms + apply_ms, "resident_apply_end", layer)


def _handle_resident_apply_end(state: _SimState, layer: int) -> None:
    """Resident layer finished applying — add to ready queue."""
    if layer not in state.ready_queue:
        state.ready_queue.append(layer)
        state.ready_queue.sort()
    _try_start_compute(state)


def _try_start_compute(state: _SimState) -> None:
    """Try to start compute on the current layer if GPU is free and layer is ready."""
    if state.current_layer not in state.ready_queue:
        return
    if state.gpu_busy_until > state.current_time_ms:
        return  # GPU still busy

    layer = state.current_layer
    state.ready_queue.remove(layer)

    trace = state._get_trace(layer)
    trace.compute_start = state.current_time_ms / 1000

    # Track GPU idle time (gap between previous compute end and this start)
    if state.gpu_busy_until > 0 and state.current_time_ms > state.gpu_busy_until:
        state.gpu_idle_ms += state.current_time_ms - state.gpu_busy_until

    # Calculate wait time: how long the layer was ready before GPU could start
    ready_time = trace.ready_time * 1000  # convert back to ms
    if ready_time > 0 and state.current_time_ms > ready_time:
        wait_ms = state.current_time_ms - ready_time
        trace.ensure_wait = wait_ms / 1000
        # Classify overlap
        if wait_ms < 1.0:
            trace.overlap_status = "hit"  # ready before GPU free
        else:
            trace.overlap_status = "wait"  # had to wait
    elif state.gpu_busy_until == 0:
        trace.overlap_status = "miss"  # no prefetch was running
    else:
        trace.overlap_status = "hit"

    state.gpu_busy_until = state.current_time_ms + state.cfg.compute_time_ms
    state.gpu_busy_ms += state.cfg.compute_time_ms
    state._computing.add(layer)

    state._add_event(state.gpu_busy_until, "compute_end", layer)


def _handle_compute_end(state: _SimState, layer: int) -> None:
    """Compute completed — evict if not resident, advance to next layer."""
    state._computing.discard(layer)
    trace = state._get_trace(layer)
    trace.compute_end = state.current_time_ms / 1000

    # Evict if not resident
    if layer not in state.resident:
        trace.evict_start = state.current_time_ms / 1000
        state._add_event(state.current_time_ms + state.cfg.eviction_latency_ms, "evict_end", layer)
        state.loaded.discard(layer)
        state._update_ram()
        # Submit prefetches for upcoming layers
        _prefetch_ahead(state, layer)
    else:
        trace.evict_start = state.current_time_ms / 1000
        trace.evict_end = state.current_time_ms / 1000
        # Still submit prefetches
        _prefetch_ahead(state, layer)

    state.token_traces_this.append(trace)
    state._advance_layer()


def _handle_evict_end(state: _SimState, layer: int) -> None:
    """Eviction completed."""
    trace = state._get_trace(layer)
    trace.evict_end = state.current_time_ms / 1000


def _prefetch_ahead(state: _SimState, last_computed: int) -> None:
    """Submit prefetches for layers ahead of current position."""
    lookahead = max(state.cfg.window_size, state.cfg.prefetch_depth)
    for ahead in range(1, lookahead + 1):
        next_idx = last_computed + ahead
        if next_idx < state.cfg.num_layers:
            state._submit_prefetch(next_idx)


# ── Main simulation loop ───────────────────────────────────────────────────


@dataclass
class SimResult:
    """Result of a simulation run."""

    config: SimulatorConfig
    traces: list[LayerTrace]
    events: list[Event]
    wall_time_ms: float
    tokens_processed: int
    peak_ram_mb: float
    gpu_idle_ms: float = 0.0
    gpu_busy_ms: float = 0.0
    resident_count: int = 0

    def dump(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metrics = compute_pipeline_metrics(
            self.traces, wall_time=self.wall_time_ms / 1000, num_tokens=self.tokens_processed
        )
        data = {
            "config": self.config.to_dict(),
            "pipeline_metrics": metrics.to_dict(),
            "traces": [t.to_dict() for t in self.traces],
            "events": [
                {
                    "time_ms": e.time_ms,
                    "event_type": e.event_type,
                    "layer": e.layer,
                    "metadata": e.metadata,
                }
                for e in self.events
            ],
            "wall_time_ms": self.wall_time_ms,
            "tokens_processed": self.tokens_processed,
            "peak_ram_mb": self.peak_ram_mb,
            "gpu_idle_ms": self.gpu_idle_ms,
            "gpu_busy_ms": self.gpu_busy_ms,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def print_timeline(self, max_layers: int = 32) -> None:
        traces = self.traces[:max_layers]
        if not traces:
            print("No traces recorded.")
            return

        print(f"\n{'='*90}")
        print(f"PIPELINE TIMELINE — {len(traces)} layers, {self.wall_time_ms:.1f} ms total")
        print(f"{'='*90}")
        print(
            f"{'Layer':>6} | {'Read':>8} | {'Upload':>8} | {'Ready':>8} | "
            f"{'Compute':>8} | {'Evict':>8} | {'Wait':>6} | Status"
        )
        print(f"{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*10}")

        for t in traces:
            d = t.durations()
            print(
                f"  L{t.layer:>3} | {d['read_ms']:>7.1f} | {d['upload_ms']:>7.1f} | "
                f"{d['ready_to_compute_ms']:>7.1f} | {d['compute_ms']:>7.1f} | "
                f"{d['evict_ms']:>7.1f} | {d['ensure_wait_ms']:>5.1f} | {t.overlap_status}"
            )

        print(f"{'='*90}\n")

    def print_summary(self) -> None:
        metrics = compute_pipeline_metrics(
            self.traces, wall_time=self.wall_time_ms / 1000, num_tokens=self.tokens_processed
        )
        print(f"\n{'='*70}")
        print("SIMULATION SUMMARY")
        print(f"{'='*70}")
        print(
            f"  Config: {self.config.num_layers} layers, {self.config.layer_size_mb:.0f} MB/layer"
        )
        print(f"  Window: {self.config.window_size}, Prefetch: {self.config.prefetch_depth}")
        print(f"  Workers: {self.config.worker_count}")
        print(f"  Tokens: {self.tokens_processed}, Wall time: {self.wall_time_ms:.1f} ms")
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
        print(f"  GPU busy:             {self.gpu_busy_ms:.1f} ms")
        print(f"  GPU idle (est):       {self.gpu_idle_ms:.1f} ms")
        print()
        print(f"  Peak RAM:             {self.peak_ram_mb:.0f} MB")
        print(f"  Resident layers:      {self.resident_count}")
        print(f"{'='*70}\n")


# ── Main entry point ──────────────────────────────────────────────────────


def simulate(config: SimulatorConfig) -> SimResult:
    """Run the discrete-event scheduler simulation.

    Models the exact same pipeline as StreamingScheduler with resource
    contention: SSD reads compete for workers, uploads compete for workers,
    and the GPU can only compute one layer at a time.
    """
    state = _SimState(config)

    # Start first token
    state._start_token()

    # Process events until simulation ends
    all_events: list[Event] = []
    while state.events:
        event = heapq.heappop(state.events)
        all_events.append(event)

        if event.event_type == "simulation_end":
            state.current_time_ms = event.time_ms
            break

        # Advance simulation time
        state.current_time_ms = max(state.current_time_ms, event.time_ms)

        # Dispatch
        if event.event_type == "prefetch_submit":
            _handle_prefetch_submit(state, event.layer)
        elif event.event_type == "ssd_read_start":
            _handle_ssd_read_start(state, event.layer)
        elif event.event_type == "ssd_read_end":
            _handle_ssd_read_end(state, event.layer)
        elif event.event_type == "upload_submit":
            _handle_upload_submit(state, event.layer)
        elif event.event_type == "upload_start":
            _handle_upload_start(state, event.layer)
        elif event.event_type == "upload_end":
            _handle_upload_end(state, event.layer)
        elif event.event_type == "compute_end":
            _handle_compute_end(state, event.layer)
        elif event.event_type == "evict_end":
            _handle_evict_end(state, event.layer)
        elif event.event_type == "resident_apply":
            _handle_resident_apply(state, event.layer)
        elif event.event_type == "resident_apply_end":
            _handle_resident_apply_end(state, event.layer)

        # Verify layer state consistency after every event
        state.check_invariants()

    # Merge traces (last token wins for flat view)
    merged: dict[int, LayerTrace] = {}
    for token_list in state.token_traces:
        for t in token_list:
            merged[t.layer] = t

    return SimResult(
        config=config,
        traces=[merged[k] for k in sorted(merged)],
        events=sorted(all_events, key=lambda e: (e.time_ms, e.seq)),
        wall_time_ms=state.current_time_ms,
        tokens_processed=config.num_tokens,
        peak_ram_mb=state.peak_ram_mb,
        gpu_idle_ms=state.gpu_idle_ms,
        gpu_busy_ms=state.gpu_busy_ms,
        resident_count=len(state.resident),
    )
