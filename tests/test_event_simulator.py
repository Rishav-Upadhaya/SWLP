"""Tests for the discrete-event scheduler simulator."""
import json

from scripts.research.simtools.event_simulator import (
    AdaptiveResidentStrategy,
    DynamicSchedulerStrategy,
    EventSimulator,
    LayerTimings,
    ResidentCacheStrategy,
    SimConfig,
    SimulationResult,
    SlidingWindowStrategy,
    compare_strategies,
    print_comparison,
)


def _default_config(**overrides) -> SimConfig:
    defaults = dict(
        num_layers=32,       # type: ignore[arg-type]
        layer_weight_mb=450.0,
        ram_budget_mb=16000.0,
        window_size=4,       # type: ignore[arg-type]
        prefetch_depth=2,    # type: ignore[arg-type]
        max_ssd_concurrency=2,  # type: ignore[arg-type]
        max_upload_concurrency=2,  # type: ignore[arg-type]
        seed=42,             # type: ignore[arg-type]
    )
    defaults.update(overrides)
    return SimConfig(**defaults)


def _default_timings(**overrides) -> LayerTimings:
    defaults = dict(
        read_ms=12.0,
        deserialize_ms=8.0,
        upload_ms=5.0,
        compute_ms=40.0,
        evict_ms=0.5,
    )
    defaults.update(overrides)
    return LayerTimings(**defaults)


# ── Sliding Window ───────────────────────────────────────────────────────────

def test_sliding_window_produces_result():
    config = _default_config(num_layers=8)
    timings = _default_timings()
    strategy = SlidingWindowStrategy(config, timings)
    sim = EventSimulator(config, timings, strategy)
    result = sim.simulate(num_tokens=3)

    assert isinstance(result, SimulationResult)
    assert result.num_tokens == 3
    assert result.num_layers == 8
    assert result.total_seconds > 0
    assert result.per_token_ms > 0
    assert result.throughput_toks_per_sec > 0


def test_sliding_window_overlap():
    """With enough prefetch depth, most layers should be hits."""
    config = _default_config(num_layers=8, prefetch_depth=4)
    timings = _default_timings(compute_ms=40.0, read_ms=12.0)
    strategy = SlidingWindowStrategy(config, timings)
    sim = EventSimulator(config, timings, strategy)
    result = sim.simulate(num_tokens=5)

    # After warmup tokens, hit rate should be high
    assert result.overlap_hit_rate > 0.3


def test_sliding_window_no_resident_layers():
    config = _default_config()
    strategy = SlidingWindowStrategy(config, _default_timings())
    assert strategy.resident_layers() == set()


def test_sliding_window_evicts_all():
    config = _default_config()
    strategy = SlidingWindowStrategy(config, _default_timings())
    to_evict = strategy.should_evict(current_layer=5, token=0, loaded={3, 4, 5, 6})
    assert 5 in to_evict


# ── Resident Cache ───────────────────────────────────────────────────────────

def test_resident_cache_auto_count():
    config = _default_config(num_layers=32, ram_budget_mb=16000)
    timings = _default_timings()
    strategy = ResidentCacheStrategy(config, timings)
    # Should auto-compute resident count
    assert len(strategy.resident_layers()) > 0


def test_resident_cache_no_eviction_for_resident():
    config = _default_config(num_layers=8)
    timings = _default_timings()
    strategy = ResidentCacheStrategy(config, timings, resident_count=4)
    resident = strategy.resident_layers()
    assert resident == {0, 1, 2, 3}

    # Resident layers should not be evicted
    to_evict = strategy.should_evict(current_layer=2, token=0, loaded={0, 1, 2, 3})
    assert 2 not in to_evict

    # Non-resident layers should be evicted
    to_evict = strategy.should_evict(current_layer=5, token=0, loaded={4, 5, 6})
    assert 5 in to_evict


def test_resident_cache_no_ssd_for_resident():
    config = _default_config(num_layers=8)
    timings = _default_timings()
    strategy = ResidentCacheStrategy(config, timings, resident_count=4)
    # Should not prefetch resident layers
    to_prefetch = strategy.should_prefetch(
        current_layer=0, token=0, loaded=set(), in_flight=set()
    )
    assert all(idx >= 4 for idx in to_prefetch)


def test_resident_cache_result():
    config = _default_config(num_layers=16)
    timings = _default_timings()
    strategy = ResidentCacheStrategy(config, timings, resident_count=8)
    sim = EventSimulator(config, timings, strategy)
    result = sim.simulate(num_tokens=5)

    assert result.total_seconds > 0
    # Resident layers should show up in overlap stats
    assert result.overlap_residents > 0


# ── Adaptive Resident ────────────────────────────────────────────────────────

def test_adaptive_resident_auto():
    config = _default_config(num_layers=16)
    timings = _default_timings()
    strategy = AdaptiveResidentStrategy(config, timings)
    sim = EventSimulator(config, timings, strategy)
    result = sim.simulate(num_tokens=3)

    assert result.total_seconds > 0
    assert "AdaptiveResident" in result.strategy_name


def test_adaptive_increases_depth_when_compute_dominates():
    """When compute >> SSD, adaptive should increase prefetch depth."""
    # Use large model that won't fit entirely in RAM → resident < num_layers
    config = _default_config(num_layers=32, prefetch_depth=2,
                             layer_weight_mb=800.0, ram_budget_mb=16000)
    # compute is 60ms, SSD is 8+4+3=15ms → compute > 2× SSD
    timings = _default_timings(compute_ms=60.0, read_ms=8.0, deserialize_ms=4.0, upload_ms=3.0)
    strategy = AdaptiveResidentStrategy(config, timings)
    # The adaptive depth should be > prefetch_depth
    assert strategy._adaptive_depth > config.prefetch_depth


# ── Dynamic Scheduler ────────────────────────────────────────────────────────

def test_dynamic_scheduler_adapts():
    # Use large model so resident < num_layers and max_depth > 0
    config = _default_config(num_layers=32, layer_weight_mb=800.0, ram_budget_mb=16000)
    strategy = DynamicSchedulerStrategy(config, _default_timings())

    initial_depth = strategy._prefetch_depth

    # High GPU idle → increase depth
    strategy.adjust_from_gpu_idle(0.5)
    assert strategy._prefetch_depth > initial_depth

    # Low GPU idle → decrease depth
    strategy.adjust_from_gpu_idle(0.05)
    assert strategy._prefetch_depth < strategy._max_depth


def test_dynamic_scheduler_bounds():
    # Use large model so resident < num_layers
    config = _default_config(num_layers=32, layer_weight_mb=800.0, ram_budget_mb=16000)
    strategy = DynamicSchedulerStrategy(config, _default_timings())

    # Push to max
    for _ in range(20):
        strategy.adjust_from_gpu_idle(1.0)
    assert strategy._prefetch_depth <= strategy._max_depth

    # Push to min
    for _ in range(20):
        strategy.adjust_from_gpu_idle(0.0)
    assert strategy._prefetch_depth >= strategy._min_depth


# ── Comparison ───────────────────────────────────────────────────────────────

def test_compare_strategies():
    config = _default_config(num_layers=16)
    timings = _default_timings()
    results = compare_strategies(config, timings, num_tokens=5)

    assert len(results) == 4
    assert all(isinstance(r, SimulationResult) for r in results)
    assert all(r.total_seconds > 0 for r in results)
    # All strategies should have unique names
    names = [r.strategy_name for r in results]
    assert len(set(names)) == len(names)


def test_print_comparison(capsys):
    config = _default_config(num_layers=8)
    timings = _default_timings()
    results = compare_strategies(config, timings, num_tokens=3)
    print_comparison(results)
    captured = capsys.readouterr()
    assert "STRATEGY COMPARISON" in captured.out
    assert "SlidingWindow" in captured.out


# ── Serialization ────────────────────────────────────────────────────────────

def test_result_to_dict():
    config = _default_config(num_layers=8)
    timings = _default_timings()
    strategy = SlidingWindowStrategy(config, timings)
    sim = EventSimulator(config, timings, strategy)
    result = sim.simulate(num_tokens=3)

    d = result.to_dict()
    assert isinstance(d, dict)
    assert "strategy" in d
    assert "throughput_toks_per_sec" in d
    assert "per_token_seconds" in d


def test_result_save(tmp_path):
    config = _default_config(num_layers=8)
    timings = _default_timings()
    strategy = SlidingWindowStrategy(config, timings)
    sim = EventSimulator(config, timings, strategy)
    result = sim.simulate(num_tokens=3)

    path = tmp_path / "sim_result.json"
    result.save(path)

    data = json.loads(path.read_text())
    assert data["strategy"] == "SlidingWindow"
    assert data["num_tokens"] == 3


# ── Timing Sanity ────────────────────────────────────────────────────────────

def test_per_token_time_increases_with_layers():
    """More layers → more time per token."""
    timings = _default_timings()
    results = []
    for n in [8, 16, 32]:
        config = _default_config(num_layers=n)
        strategy = SlidingWindowStrategy(config, timings)
        sim = EventSimulator(config, timings, strategy)
        results.append(sim.simulate(num_tokens=3))

    # Per-token time should increase with layer count
    assert results[0].per_token_ms < results[1].per_token_ms < results[2].per_token_ms


def test_faster_ssd_improves_throughput():
    """Faster SSD reads → better throughput."""
    # Make SSD the bottleneck: short compute, low prefetch depth so overlap is partial
    config = _default_config(num_layers=32, prefetch_depth=1,
                             layer_weight_mb=800.0, ram_budget_mb=16000)

    slow = _default_timings(read_ms=20.0, compute_ms=10.0)
    fast = _default_timings(read_ms=5.0, compute_ms=10.0)

    sim_slow = EventSimulator(config, slow, SlidingWindowStrategy(config, slow))
    sim_fast = EventSimulator(config, fast, SlidingWindowStrategy(config, fast))

    r_slow = sim_slow.simulate(num_tokens=5)
    r_fast = sim_fast.simulate(num_tokens=5)

    assert r_fast.throughput_toks_per_sec > r_slow.throughput_toks_per_sec


def test_jitter_produces_varied_results():
    """Different seeds produce different timings."""
    # Use large model so SSD reads happen (not all resident)
    timings = _default_timings()

    results = []
    for seed in [1, 2, 3]:
        c = SimConfig(num_layers=32, layer_weight_mb=800.0, ram_budget_mb=16000,
                      seed=seed, read_jitter=0.2)
        strategy = SlidingWindowStrategy(c, timings)
        sim = EventSimulator(c, timings, strategy)
        results.append(sim.simulate(num_tokens=5))

    # Results should differ due to jitter
    times = [r.total_seconds for r in results]
    assert len(set(f"{t:.6f}" for t in times)) > 1


# ── Edge Cases ───────────────────────────────────────────────────────────────

def test_single_layer():
    config = _default_config(num_layers=1)
    timings = _default_timings()
    strategy = SlidingWindowStrategy(config, timings)
    sim = EventSimulator(config, timings, strategy)
    result = sim.simulate(num_tokens=3)
    assert result.total_seconds > 0


def test_one_token():
    config = _default_config(num_layers=8)
    timings = _default_timings()
    strategy = SlidingWindowStrategy(config, timings)
    sim = EventSimulator(config, timings, strategy)
    result = sim.simulate(num_tokens=1)
    assert result.total_seconds > 0
    assert len(result.per_token_seconds) == 1


def test_many_tokens():
    config = _default_config(num_layers=8)
    timings = _default_timings()
    strategy = SlidingWindowStrategy(config, timings)
    sim = EventSimulator(config, timings, strategy)
    result = sim.simulate(num_tokens=50)
    assert result.total_seconds > 0
    assert len(result.per_token_seconds) == 50


# ── Summary Output ───────────────────────────────────────────────────────────

def test_summary_string():
    config = _default_config(num_layers=8)
    timings = _default_timings()
    strategy = SlidingWindowStrategy(config, timings)
    sim = EventSimulator(config, timings, strategy)
    result = sim.simulate(num_tokens=3)

    summary = result.summary()
    assert "SlidingWindow" in summary
    assert "tok/s" in summary
    assert "GPU busy" in summary


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
