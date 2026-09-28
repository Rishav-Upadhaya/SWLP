#!/usr/bin/env python3
"""Comprehensive benchmark: all models × all features × all metrics.

Runs the event simulator (no model download needed) across:
  - Models: Mistral-7B, Qwen-7B, Qwen-14B, Qwen-3B, Qwen-0.5B
  - Strategies: SlidingWindow, ResidentCache, AdaptiveResident, DynamicScheduler
  - Window sizes: W=2, W=4, W=6
  - Features: prefetch on/off, early exit, FP8 vs FP16 (via timing adjustments)

Outputs all metrics in table format.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from swlp.benchmark.event_simulator import (
    EventSimulator,
    SimConfig,
    LayerTimings,
    SlidingWindowStrategy,
    ResidentCacheStrategy,
    AdaptiveResidentStrategy,
    DynamicSchedulerStrategy,
    compare_strategies,
)

# ── Model Definitions ────────────────────────────────────────────────────────
# (name, layers, layer_mb, compute_ms, kv_bytes_per_token, total_gb)

MODELS = [
    ("Qwen2.5-0.5B FP16",   24,   42,   2.0,   1024,   1.0),
    ("Qwen2.5-3B FP16",     36,  170,   5.0,   2048,   6.1),
    ("Qwen2.5-7B FP16",     32,  436,  12.0,   4096,  14.0),
    ("Qwen2.5-14B FP16",    48,  551,  22.0,   4096,  26.4),
    ("Mistral-7B FP16",     32,  436,  12.0,   4096,  13.96),
    ("Mistral-7B FP8",      32,  218,  14.0,   4096,   7.0),
    ("Qwen2.5-14B FP8",     48,  275,  24.0,   4096,  13.2),
]

# ── Hardware: M5 16 GB ──────────────────────────────────────────────────────
RAM_GB = 16384  # MB
SSD_READ_MS = 12.0    # ~6.93 GB/s → ~436 MB in 12ms (tuned per layer)
UPLOAD_MS = 5.0
DESER_MS = 3.0
EVICT_MS = 0.5

# ── Window sizes to test ────────────────────────────────────────────────────
WINDOWS = [2, 4, 6]

# ── Feature toggles ──────────────────────────────────────────────────────────
PREFETCH_DEPTHS = [2, 4]
RESIDENT_COUNTS = [0, 4, 8]  # 0 = pure streaming


def make_timings(read_ms: float = SSD_READ_MS) -> LayerTimings:
    return LayerTimings(
        read_ms=read_ms,
        deserialize_ms=DESER_MS,
        upload_ms=UPLOAD_MS,
        compute_ms=40.0,  # overridden per-model below
        evict_ms=EVICT_MS,
    )


def run_model_benchmark(
    model_name: str,
    num_layers: int,
    layer_mb: float,
    layer_compute_ms: float,
    kv_bytes: int,
    total_gb: float,
) -> list[dict]:
    """Run all feature combinations for one model. Returns list of metric dicts."""
    rows = []

    # Read time proportional to layer size (base: 436 MB → 12 ms)
    read_ms = max(2.0, SSD_READ_MS * (layer_mb / 436.0))

    # Full-model baseline: everything fits in RAM (no streaming)
    fits_full = (total_gb * 1024) <= (RAM_GB * 0.75 - 4096)

    for window in WINDOWS:
        for prefetch_depth in PREFETCH_DEPTHS:
            for resident_count in RESIDENT_COUNTS:
                # Skip invalid combos
                if resident_count >= num_layers:
                    continue
                if resident_count > 0 and fits_full:
                    # If model fits fully, resident doesn't add value — skip redundant
                    # (but still record pure-streaming for comparison)
                    pass

                timings = make_timings(read_ms)
                timings.compute_ms = layer_compute_ms

                config = SimConfig(
                    num_layers=num_layers,
                    layer_weight_mb=layer_mb,
                    ram_budget_mb=RAM_GB,
                    window_size=window,
                    prefetch_depth=prefetch_depth,
                    max_ssd_concurrency=2,
                    max_upload_concurrency=2,
                    read_jitter=0.1,
                    compute_jitter=0.05,
                    seed=42,
                )

                # Strategy selection based on resident_count
                if resident_count == 0:
                    strategy = SlidingWindowStrategy(config, timings)
                    strategy_label = f"SlidingWindow(W={window})"
                else:
                    strategy = ResidentCacheStrategy(config, timings, resident_count=resident_count)
                    strategy_label = f"ResidentCache(R={resident_count},W={window})"

                sim = EventSimulator(config, timings, strategy)
                result = sim.simulate(num_tokens=10)

                # GPU idle = 1 - gpu_busy
                gpu_idle_pct = (1.0 - result.gpu_busy_pct) * 100
                cpu_idle_pct = max(0, 100.0 - result.ssd_busy_pct * 100 - gpu_idle_pct)
                # SST = steady-state throughput (after warmup token 0)
                sst_tokens = result.per_token_seconds[1:] if len(result.per_token_seconds) > 1 else result.per_token_seconds
                sst = len(sst_tokens) / sum(sst_tokens) if sum(sst_tokens) > 0 else 0.0

                rows.append({
                    "model": model_name,
                    "layers": num_layers,
                    "layer_mb": layer_mb,
                    "total_gb": total_gb,
                    "strategy": strategy_label,
                    "window": window,
                    "prefetch": prefetch_depth,
                    "resident": resident_count,
                    "tok_per_sec": result.throughput_toks_per_sec,
                    "sst_tok_per_sec": sst,
                    "per_token_ms": result.per_token_ms,
                    "gpu_busy_pct": result.gpu_busy_pct * 100,
                    "gpu_idle_pct": gpu_idle_pct,
                    "cpu_idle_pct": cpu_idle_pct,
                    "ssd_busy_pct": result.ssd_busy_pct * 100,
                    "overlap_hit_pct": result.overlap_hit_rate * 100,
                    "avg_read_ms": result.avg_read_ms,
                    "avg_compute_ms": result.avg_compute_ms,
                    "avg_upload_ms": result.avg_upload_ms,
                    "avg_wait_ms": result.avg_ensure_wait_ms,
                    "ram_peak": result.ram_peak_layers,
                    "gb_per_token": result.gb_per_token,
                })

    return rows


def run_strategy_comparison() -> list[dict]:
    """Compare all 4 scheduling strategies on a reference model."""
    rows = []
    # Use Mistral-7B as reference
    num_layers, layer_mb, compute_ms = 32, 436, 12.0
    read_ms = SSD_READ_MS

    for window in [2, 4]:
        timings = make_timings(read_ms)
        timings.compute_ms = compute_ms
        config = SimConfig(
            num_layers=num_layers,
            layer_weight_mb=layer_mb,
            ram_budget_mb=RAM_GB,
            window_size=window,
            prefetch_depth=4,
            max_ssd_concurrency=2,
            max_upload_concurrency=2,
            seed=42,
        )
        strategies = [
            SlidingWindowStrategy(config, timings),
            ResidentCacheStrategy(config, timings, resident_count=0),
            AdaptiveResidentStrategy(config, timings),
            DynamicSchedulerStrategy(config, timings),
        ]
        for strat in strategies:
            sim = EventSimulator(config, timings, strat)
            result = sim.simulate(num_tokens=10)
            gpu_idle = (1.0 - result.gpu_busy_pct) * 100
            sst_tokens = result.per_token_seconds[1:] if len(result.per_token_seconds) > 1 else result.per_token_seconds
            sst = len(sst_tokens) / sum(sst_tokens) if sum(sst_tokens) > 0 else 0.0
            rows.append({
                "strategy": strat.name,
                "window": window,
                "tok_per_sec": result.throughput_toks_per_sec,
                "sst_tok_per_sec": sst,
                "per_token_ms": result.per_token_ms,
                "gpu_busy_pct": result.gpu_busy_pct * 100,
                "gpu_idle_pct": gpu_idle,
                "ssd_busy_pct": result.ssd_busy_pct * 100,
                "overlap_hit_pct": result.overlap_hit_rate * 100,
                "avg_wait_ms": result.avg_ensure_wait_ms,
                "ram_peak": result.ram_peak_layers,
            })
    return rows


def run_early_exit_simulation() -> list[dict]:
    """Simulate early exit: skip 20-40% of layers for easy tokens."""
    rows = []
    num_layers, layer_mb, compute_ms = 32, 436, 12.0
    read_ms = SSD_READ_MS

    for skip_pct in [0, 20, 40]:
        skip_layers = int(num_layers * skip_pct / 100)
        active_layers = num_layers - skip_layers

        timings = make_timings(read_ms)
        timings.compute_ms = compute_ms
        config = SimConfig(
            num_layers=active_layers,
            layer_weight_mb=layer_mb,
            ram_budget_mb=RAM_GB,
            window_size=2,
            prefetch_depth=4,
            seed=42,
        )
        strategy = SlidingWindowStrategy(config, timings)
        sim = EventSimulator(config, timings, strategy)
        result = sim.simulate(num_tokens=10)
        gpu_idle = (1.0 - result.gpu_busy_pct) * 100
        sst_tokens = result.per_token_seconds[1:] if len(result.per_token_seconds) > 1 else result.per_token_seconds
        sst = len(sst_tokens) / sum(sst_tokens) if sum(sst_tokens) > 0 else 0.0

        # Estimate full-model baseline throughput
        full_time_per_token = (num_layers * (read_ms + DESER_MS + UPLOAD_MS + compute_ms + EVICT_MS)) / 1000.0
        baseline_tps = 1.0 / full_time_per_token if full_time_per_token > 0 else 0.0

        rows.append({
            "skip_pct": skip_pct,
            "active_layers": active_layers,
            "tok_per_sec": result.throughput_toks_per_sec,
            "sst_tok_per_sec": sst,
            "per_token_ms": result.per_token_ms,
            "gpu_busy_pct": result.gpu_busy_pct * 100,
            "gpu_idle_pct": gpu_idle,
            "speedup_vs_full": baseline_tps / result.throughput_toks_per_sec if result.throughput_toks_per_sec > 0 else 0,
        })
    return rows


def run_speculative_simulation() -> list[dict]:
    """Simulate speculative decoding: 2-8 tokens verified per sweep."""
    rows = []
    num_layers, layer_mb, compute_ms = 32, 436, 12.0
    read_ms = SSD_READ_MS

    for tokens_per_sweep in [1, 2, 4, 8]:
        # Baseline: 1 token per sweep
        timings = make_timings(read_ms)
        timings.compute_ms = compute_ms
        config = SimConfig(
            num_layers=num_layers,
            layer_weight_mb=layer_mb,
            ram_budget_mb=RAM_GB,
            window_size=2,
            prefetch_depth=4,
            seed=42,
        )
        strategy = SlidingWindowStrategy(config, timings)
        sim = EventSimulator(config, timings, strategy)
        result = sim.simulate(num_tokens=10)

        # Speculative: amortize sweep across N tokens
        effective_tps = result.throughput_toks_per_sec * tokens_per_sweep
        overhead_pct = (1.0 - 1.0 / tokens_per_sweep) * 100  # theoretical ideal

        rows.append({
            "tokens_per_sweep": tokens_per_sweep,
            "base_tps": result.throughput_toks_per_sec,
            "effective_tps": effective_tps,
            "theoretical_speedup": f"{tokens_per_sweep:.1f}x",
            "gpu_busy_pct": result.gpu_busy_pct * 100,
            "ssd_busy_pct": result.ssd_busy_pct * 100,
        })
    return rows


def main():
    print("=" * 120)
    print("SWLP COMPREHENSIVE BENCHMARK — All Models × All Features × All Metrics")
    print("Hardware: Apple M5 16 GB unified memory | Simulation: discrete-event, no model download")
    print("=" * 120)

    # ── TABLE 1: Model × Feature Matrix ──────────────────────────────────────
    print("\n" + "─" * 120)
    print("TABLE 1: MODEL × FEATURE THROUGHPUT (tok/s) — SlidingWindow, Prefetch=4")
    print("─" * 120)

    all_model_rows = []
    for model in MODELS:
        rows = run_model_benchmark(*model)
        all_model_rows.extend(rows)

    # Filter to prefetch=4 for clean comparison
    filtered = [r for r in all_model_rows if r["prefetch"] == 4]

    # Group by model
    from collections import defaultdict
    by_model = defaultdict(list)
    for r in filtered:
        by_model[r["model"]].append(r)

    for model_name, rows in by_model.items():
        print(f"\n  {model_name}")
        print(f"  {'Strategy':<28} {'W':>3} {'R':>3} {'tok/s':>8} {'SST':>8} {'ms/tok':>8} {'GPU%':>6} {'GPU Idle%':>9} {'SSD%':>6} {'Hit%':>6} {'Wait':>7} {'RAM':>4}")
        print(f"  {'-'*28} {'-'*3} {'-'*3} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*9} {'-'*6} {'-'*6} {'-'*7} {'-'*4}")
        for r in sorted(rows, key=lambda x: (-x["tok_per_sec"])):
            print(
                f"  {r['strategy']:<28} {r['window']:>3} {r['resident']:>3} "
                f"{r['tok_per_sec']:>8.3f} {r['sst_tok_per_sec']:>8.3f} "
                f"{r['per_token_ms']:>8.1f} {r['gpu_busy_pct']:>5.1f}% "
                f"{r['gpu_idle_pct']:>8.1f}% {r['ssd_busy_pct']:>5.1f}% "
                f"{r['overlap_hit_pct']:>5.1f}% {r['avg_wait_ms']:>6.1f} {r['ram_peak']:>4}"
            )

    # ── TABLE 2: Strategy Comparison (Mistral-7B reference) ──────────────────
    print("\n" + "─" * 120)
    print("TABLE 2: SCHEDULING STRATEGY COMPARISON — Mistral-7B FP16 (32 layers, 436 MB/layer)")
    print("─" * 120)

    strat_rows = run_strategy_comparison()
    print(f"\n  {'Strategy':<30} {'W':>3} {'tok/s':>8} {'SST':>8} {'ms/tok':>8} {'GPU%':>6} {'GPU Idle%':>9} {'SSD%':>6} {'Hit%':>6} {'Wait':>7} {'RAM':>4}")
    print(f"  {'-'*30} {'-'*3} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*9} {'-'*6} {'-'*6} {'-'*7} {'-'*4}")
    for r in strat_rows:
        print(
            f"  {r['strategy']:<30} {r['window']:>3} "
            f"{r['tok_per_sec']:>8.3f} {r['sst_tok_per_sec']:>8.3f} "
            f"{r['per_token_ms']:>8.1f} {r['gpu_busy_pct']:>5.1f}% "
            f"{r['gpu_idle_pct']:>8.1f}% {r['ssd_busy_pct']:>5.1f}% "
            f"{r['overlap_hit_pct']:>5.1f}% {r['avg_wait_ms']:>6.1f} {r['ram_peak']:>4}"
        )

    # ── TABLE 3: Early Exit Impact ───────────────────────────────────────────
    print("\n" + "─" * 120)
    print("TABLE 3: EARLY EXIT — Layer Skip Impact on Mistral-7B FP16")
    print("─" * 120)

    early_rows = run_early_exit_simulation()
    print(f"\n  {'Skip%':>6} {'Active Layers':>13} {'tok/s':>8} {'SST':>8} {'ms/tok':>8} {'GPU%':>6} {'GPU Idle%':>9} {'Speedup':>8}")
    print(f"  {'-'*6} {'-'*13} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*9} {'-'*8}")
    for r in early_rows:
        print(
            f"  {r['skip_pct']:>5}% {r['active_layers']:>13} "
            f"{r['tok_per_sec']:>8.3f} {r['sst_tok_per_sec']:>8.3f} "
            f"{r['per_token_ms']:>8.1f} {r['gpu_busy_pct']:>5.1f}% "
            f"{r['gpu_idle_pct']:>8.1f}% {r['speedup_vs_full']:>7.2f}x"
        )

    # ── TABLE 4: Speculative Decoding ────────────────────────────────────────
    print("\n" + "─" * 120)
    print("TABLE 4: SPECULATIVE DECODING — Tokens per Sweep on Mistral-7B FP16")
    print("─" * 120)

    spec_rows = run_speculative_simulation()
    print(f"\n  {'Tokens/Sweep':>13} {'Base tok/s':>11} {'Effective tok/s':>15} {'Theoretical':>11} {'GPU%':>6} {'SSD%':>6}")
    print(f"  {'-'*13} {'-'*11} {'-'*15} {'-'*11} {'-'*6} {'-'*6}")
    for r in spec_rows:
        print(
            f"  {r['tokens_per_sweep']:>13} {r['base_tps']:>11.3f} "
            f"{r['effective_tps']:>15.3f} {r['theoretical_speedup']:>11} "
            f"{r['gpu_busy_pct']:>5.1f}% {r['ssd_busy_pct']:>5.1f}%"
        )

    # ── TABLE 5: FP16 vs FP8 Comparison ─────────────────────────────────────
    print("\n" + "─" * 120)
    print("TABLE 5: FP16 vs FP8 — Throughput & GPU Utilization")
    print("─" * 120)

    fp_rows = [r for r in all_model_rows if r["prefetch"] == 4 and r["window"] == 2 and r["resident"] == 0]
    print("\n  {'Model':<24} {'Layer MB':>8} {'tok/s':>8} {'SST':>8} {'ms/tok':>8} {'GPU%':>6} {'GPU Idle%':>9} {'SSD%':>6} {'GB/tok':>7}")
    print(f"  {'-'*24} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*9} {'-'*6} {'-'*7}")
    for r in fp_rows:
        print(
            f"  {r['model']:<24} {r['layer_mb']:>7.0f}M "
            f"{r['tok_per_sec']:>8.3f} {r['sst_tok_per_sec']:>8.3f} "
            f"{r['per_token_ms']:>8.1f} {r['gpu_busy_pct']:>5.1f}% "
            f"{r['gpu_idle_pct']:>8.1f}% {r['ssd_busy_pct']:>5.1f}%"
            f"{r.get('gb_per_token', 0):>7.2f}"
        )

    # ── TABLE 6: Window Size Sensitivity ─────────────────────────────────────
    print("\n" + "─" * 120)
    print("TABLE 6: WINDOW SIZE SENSITIVITY — SlidingWindow, Prefetch=4")
    print("─" * 120)

    win_rows = [r for r in all_model_rows if r["prefetch"] == 4 and r["resident"] == 0]
    print(f"\n  {'Model':<24} {'W=2 tok/s':>10} {'W=4 tok/s':>10} {'W=6 tok/s':>10} {'W=2 GPU%':>9} {'W=4 GPU%':>9} {'W=6 GPU%':>9}")
    print(f"  {'-'*24} {'-'*10} {'-'*10} {'-'*10} {'-'*9} {'-'*9} {'-'*9}")
    by_model_win = defaultdict(dict)
    for r in win_rows:
        by_model_win[r["model"]][r["window"]] = r
    for model_name, wins in by_model_win.items():
        w2 = wins.get(2, {})
        w4 = wins.get(4, {})
        w6 = wins.get(6, {})
        print(
            f"  {model_name:<24} "
            f"{w2.get('tok_per_sec', 0):>10.3f} {w4.get('tok_per_sec', 0):>10.3f} {w6.get('tok_per_sec', 0):>10.3f} "
            f"{w2.get('gpu_busy_pct', 0):>8.1f}% {w4.get('gpu_busy_pct', 0):>8.1f}% {w6.get('gpu_busy_pct', 0):>8.1f}%"
        )

    # ── TABLE 7: Prefetch Depth Impact ──────────────────────────────────────
    print("\n" + "─" * 120)
    print("TABLE 7: PREFETCH DEPTH IMPACT — SlidingWindow W=2")
    print("─" * 120)

    pf_rows = [r for r in all_model_rows if r["window"] == 2 and r["resident"] == 0]
    print(f"\n  {'Model':<24} {'PF=2 tok/s':>10} {'PF=4 tok/s':>10} {'PF=2 GPU Idle':>13} {'PF=4 GPU Idle':>13} {'PF=2 Hit%':>9} {'PF=4 Hit%':>9}")
    print(f"  {'-'*24} {'-'*10} {'-'*10} {'-'*13} {'-'*13} {'-'*9} {'-'*9}")
    by_model_pf = defaultdict(dict)
    for r in pf_rows:
        by_model_pf[r["model"]][r["prefetch"]] = r
    for model_name, pfs in by_model_pf.items():
        p2 = pfs.get(2, {})
        p4 = pfs.get(4, {})
        print(
            f"  {model_name:<24} "
            f"{p2.get('tok_per_sec', 0):>10.3f} {p4.get('tok_per_sec', 0):>10.3f} "
            f"{p2.get('gpu_idle_pct', 0):>12.1f}% {p4.get('gpu_idle_pct', 0):>12.1f}% "
            f"{p2.get('overlap_hit_pct', 0):>8.1f}% {p4.get('overlap_hit_pct', 0):>8.1f}%"
        )

    # ── TABLE 8: Full Summary (all models, best config) ─────────────────────
    print("\n" + "─" * 120)
    print("TABLE 8: BEST CONFIG PER MODEL — Top throughput config (SlidingWindow, Prefetch=4)")
    print("─" * 120)

    best_by_model = {}
    for r in all_model_rows:
        if r["prefetch"] == 4:
            m = r["model"]
            if m not in best_by_model or r["tok_per_sec"] > best_by_model[m]["tok_per_sec"]:
                best_by_model[m] = r

    print(f"\n  {'Model':<24} {'W':>3} {'tok/s':>8} {'SST':>8} {'ms/tok':>8} {'GPU%':>6} {'GPU Idle%':>9} {'SSD%':>6} {'Hit%':>6} {'Total GB':>8}")
    print(f"  {'-'*24} {'-'*3} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*9} {'-'*6} {'-'*6} {'-'*8}")
    for model_name, r in sorted(best_by_model.items(), key=lambda x: -x[1]["tok_per_sec"]):
        print(
            f"  {r['model']:<24} {r['window']:>3} "
            f"{r['tok_per_sec']:>8.3f} {r['sst_tok_per_sec']:>8.3f} "
            f"{r['per_token_ms']:>8.1f} {r['gpu_busy_pct']:>5.1f}% "
            f"{r['gpu_idle_pct']:>8.1f}% {r['ssd_busy_pct']:>5.1f}% "
            f"{r['overlap_hit_pct']:>5.1f}% {r['total_gb']:>7.1f}G"
        )

    print("\n" + "=" * 120)
    print("END OF BENCHMARK")
    print("=" * 120)


if __name__ == "__main__":
    main()
