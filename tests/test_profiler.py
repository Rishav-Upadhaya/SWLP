"""Validate that the LayerProfiler adds minimal overhead."""

import json
import sys
import time
from pathlib import Path

import pytest

from swlp.core.profiler import LayerProfiler


@pytest.mark.skipif(
    "coverage" in sys.modules or sys.gettrace() is not None,
    reason="a timing assertion is meaningless under coverage/a debugger",
)
def test_profiler_overhead() -> None:
    """Instrumentation must stay cheap relative to un-instrumented calls.

    Uses NO sleeps (scheduler noise swamped the previous timing test —
    flaky under load) and takes the MINIMUM of three runs per side: the
    minimum is the standard robust estimator for timing benchmarks, since
    contention only ever inflates a measurement. A real regression
    reproduces on every repeat; load does not survive the min().
    """
    import time as _time

    N_LAYERS = 24
    N_TOKENS = 2000
    REPEATS = 3

    def _cycle(prof: LayerProfiler) -> float:
        t0 = _time.perf_counter()
        for tok in range(N_TOKENS):
            if prof._enabled:
                prof.begin_token(tok)
            for i in range(N_LAYERS):
                prof.begin_read(i)
                prof.end_read(i)
                prof.begin_deserialize(i)
                prof.end_deserialize(i)
                prof.begin_upload(i)
                prof.end_upload(i)
                prof.record_ready(i, ensure_wait=0.0001, overlap_status="hit")
                prof.begin_compute(i)
                prof.end_compute(i)
                prof.begin_evict(i)
                prof.end_evict(i)
            if prof._enabled:
                prof.end_token()
        return _time.perf_counter() - t0

    prof_on = LayerProfiler(enabled=True)
    on_min = min(_cycle(prof_on) for _ in range(REPEATS))

    cycles = N_TOKENS * N_LAYERS
    us_per_cycle = on_min / cycles * 1e6
    print(f"instrumented min: {on_min:.4f}s  "
          f"{us_per_cycle:.2f} µs per layer cycle (11 bookkeeping calls)")
    # Measured ~6 µs per layer cycle on M-series (~0.5 µs/call). In
    # production every call brackets real work (ms-scale reads/compute), so
    # this is noise there; the bound catches genuine regressions (accidental
    # I/O, unbounded growth) that would show up as order-of-magnitude jumps.
    assert us_per_cycle < 15.0, (
        f"Profiler overhead too high: {us_per_cycle:.2f} µs per layer cycle"
    )


def test_profiler_durations():
    """Verify duration calculations are correct."""
    prof = LayerProfiler(enabled=True)

    # Simulate a single token
    prof.begin_token(0)
    prof.begin_read(0)
    time.sleep(0.01)
    prof.end_read(0)
    prof.begin_deserialize(0)
    time.sleep(0.005)
    prof.end_deserialize(0)
    prof.begin_upload(0)
    time.sleep(0.003)
    prof.end_upload(0)
    prof.record_ready(0, ensure_wait=0.001, overlap_status="hit")
    prof.begin_compute(0)
    time.sleep(0.02)
    prof.end_compute(0)
    prof.begin_evict(0)
    time.sleep(0.001)
    prof.end_evict(0)
    prof.end_token()

    traces = prof.get_traces()
    assert len(traces) == 1
    t = traces[0]

    durations = t.durations()
    assert durations["read_ms"] >= 9.0, f"read_ms too low: {durations['read_ms']}"
    assert durations["deserialize_ms"] >= 4.0, (
        f"deserialize_ms too low: {durations['deserialize_ms']}"
    )
    assert durations["upload_ms"] >= 2.0, f"upload_ms too low: {durations['upload_ms']}"
    assert durations["compute_ms"] >= 19.0, f"compute_ms too low: {durations['compute_ms']}"
    assert durations["ensure_wait_ms"] >= 0.9, (
        f"ensure_wait_ms too low: {durations['ensure_wait_ms']}"
    )
    print(f"Durations: {durations}")


def test_profiler_summary():
    """Verify summary statistics."""
    prof = LayerProfiler(enabled=True)

    prof.begin_token(0)
    for i in range(10):
        prof.begin_read(i)
        time.sleep(0.001)
        prof.end_read(i)
        prof.begin_compute(i)
        time.sleep(0.002)
        prof.end_compute(i)
    prof.end_token()

    summary = prof.summary()
    assert summary["num_layers"] == 10
    assert summary["read_ms"]["mean"] >= 0.5
    assert summary["compute_ms"]["mean"] >= 1.5
    print(f"Summary: {json.dumps(summary, indent=2)}")


def test_profiler_dump(tmp_path):
    """Verify JSON dump works."""
    prof = LayerProfiler(enabled=True)
    prof.begin_token(0)
    for i in range(5):
        prof.begin_read(i)
        prof.end_read(i)
        prof.begin_compute(i)
        prof.end_compute(i)
    prof.end_token()

    dump_path = tmp_path / "traces.json"
    prof.dump(dump_path)

    with open(dump_path) as f:
        data = json.load(f)

    assert "summary" in data
    assert "traces" in data
    assert len(data["traces"]) == 5
    print(f"Dumped {len(data['traces'])} traces to {dump_path}")


def test_print_layer_detail(capsys):
    """Verify print_layer_detail produces per-layer block output."""
    prof = LayerProfiler(enabled=True)
    prof.begin_token(0)
    for i in range(4):
        prof.begin_read(i)
        prof.end_read(i)
        prof.begin_deserialize(i)
        prof.end_deserialize(i)
        prof.begin_upload(i)
        prof.end_upload(i)
        prof.record_ready(i, ensure_wait=0.0001, overlap_status="hit")
        prof.begin_compute(i)
        prof.end_compute(i)
        prof.begin_evict(i)
        prof.end_evict(i)
    prof.end_token()

    prof.print_layer_detail()
    captured = capsys.readouterr()
    assert "LAYER DETAIL" in captured.out
    assert "Layer 0" in captured.out
    assert "Read:" in captured.out
    assert "Deserialize:" in captured.out
    assert "Upload:" in captured.out
    assert "Wait:" in captured.out
    assert "Compute:" in captured.out
    assert "Status:" in captured.out


def test_print_token_summary(capsys):
    """Verify print_token_summary aggregates across layers."""
    prof = LayerProfiler(enabled=True)
    # Simulate 3 tokens × 5 layers (unique layer indices per token)
    for token in range(3):
        prof.begin_token(token)
        for j in range(5):
            layer = token * 5 + j
            prof.begin_read(layer)
            prof.end_read(layer)
            prof.begin_compute(layer)
            prof.end_compute(layer)
        prof.end_token()

    prof.print_token_summary()
    captured = capsys.readouterr()
    assert "TOKEN SUMMARY" in captured.out
    assert "15 layer-traces" in captured.out
    assert "GPU efficiency" in captured.out
    assert "mean=" in captured.out
    assert "p95=" in captured.out


def test_print_layer_detail_empty(capsys):
    """print_layer_detail handles no traces gracefully."""
    prof = LayerProfiler(enabled=True)
    prof.print_layer_detail()
    captured = capsys.readouterr()
    assert "No traces recorded" in captured.out


if __name__ == "__main__":
    test_profiler_overhead()
    test_profiler_durations()
    test_profiler_summary()
    test_profiler_dump(tmp_path=Path("/tmp/swlp_profiler_test"))
    test_print_layer_detail()
    test_print_token_summary()
    test_print_layer_detail_empty()
    print("\nAll profiler tests passed!")
