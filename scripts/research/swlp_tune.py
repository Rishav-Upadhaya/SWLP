#!/usr/bin/env python3
"""Tuning harness for SWLP Phase 9.

Runs a parameter sweep over window sizes, prefetch depths and memory settings,
records timings and overlap metrics, and writes a recommended profile JSON.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean

from swlp.config import load_config
from swlp.runtime import build_runner


def analyze_trace(trace: list[dict]) -> dict:
    if not trace:
        return {"overlap_ratio": 0.0, "prefetch_success_rate": 0.0}
    overlaps = 0
    successes = 0
    for t in trace:
        # prefetch_enqueued before compute_start indicates overlap opportunity
        pe = t.get("prefetch_enqueued")
        cs = t.get("compute_start")
        if pe is None or cs is None:
            continue
        successes += 1
        if pe <= cs:
            overlaps += 1
    overlap_ratio = overlaps / successes if successes > 0 else 0.0
    return {"overlap_ratio": overlap_ratio, "prefetch_success_count": overlaps, "prefetch_attempts": successes}


def run_once(cfg, prompt, profile):
    cfg.runtime.backend = "swlp"
    runner = build_runner(cfg)
    # allow runner to pick up tuning profile if present
    if hasattr(runner, "_apply_tuning_profile"):
        try:
            runner._apply_tuning_profile()
        except Exception:
            pass
    t0 = time.perf_counter()
    res = runner.run(prompt, profile=profile)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    trace = getattr(runner, "_trace", None)
    trace_analysis = analyze_trace(trace or [])
    metrics = res.metrics.to_dict() if hasattr(res.metrics, "to_dict") else {}
    metrics.update({"elapsed": elapsed})
    return metrics, trace_analysis, trace


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="swlp_tuning_results.json")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config()
    prompt = args.prompt or cfg.generation.prompt

    # parameter grid (conservative defaults)
    window_sizes = [1, 2, 3]
    prefetch_depths = [1, 2, 4]
    pin_memory_opts = [False, True]
    double_buffer_opts = [False, True]

    results = {
        "device": cfg.runtime.device,
        "grid": [],
    }

    for w in window_sizes:
        for d in prefetch_depths:
            for pin in pin_memory_opts:
                for db in double_buffer_opts:
                    cfg.runtime.swlp_window_size = w
                    cfg.runtime.swlp_prefetch_depth = d
                    cfg.runtime.swlp_pin_memory = pin
                    cfg.runtime.swlp_double_buffer = db
                    cfg.runtime.swlp_prefetch = True
                    runs = []
                    for _ in range(args.repeat):
                        metrics, trace_analysis, trace = run_once(cfg, prompt, profile=args.profile)
                        runs.append({"metrics": metrics, "trace_analysis": trace_analysis})
                    avg_elapsed = mean(r["metrics"]["elapsed"] for r in runs)
                    avg_overlap = mean(r["trace_analysis"].get("overlap_ratio", 0.0) for r in runs)
                    grid_entry = {
                        "window_size": w,
                        "prefetch_depth": d,
                        "pin_memory": pin,
                        "double_buffer": db,
                        "avg_elapsed": avg_elapsed,
                        "avg_overlap": avg_overlap,
                        "runs": runs,
                    }
                    results["grid"].append(grid_entry)
                    print(f"Test w={w} d={d} pin={pin} db={db} -> elapsed={avg_elapsed:.3f}s overlap={avg_overlap:.3f}")

    # choose best by lowest elapsed, prefer higher overlap as tie-breaker
    best = min(results["grid"], key=lambda e: (e["avg_elapsed"], -e["avg_overlap"]))
    results["best"] = best

    # craft tuning profile per-device
    tune_profile = {
        cfg.runtime.device: {
            "swlp_window_size": int(best["window_size"]),
            "swlp_prefetch_depth": int(best["prefetch_depth"]),
            "swlp_prefetch": True,
            "swlp_pin_memory": bool(best["pin_memory"]),
            "swlp_double_buffer": bool(best["double_buffer"]),
        }
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    # also write recommended profile
    tune_path = Path("swlp_tuning.json")
    tune_path.write_text(json.dumps(tune_profile, indent=2), encoding="utf-8")
    print(f"Wrote tuning results to {out_path} and profile to {tune_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
