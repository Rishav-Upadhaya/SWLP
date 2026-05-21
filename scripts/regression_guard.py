#!/usr/bin/env python3
"""Repeatable performance regression guard with pass/fail thresholds.

Runs three workloads:
1. MockRunner (offline) – verification that infrastructure operates at speed.
2. HF Baseline (tiny-gpt2) – naive full-model inference baseline.
3. SWLP Runner (tiny-gpt2) – sliding-window pipeline inference.

Pass/fail conditions are enforced on:
- Time-to-first-token (TTFT)
- Throughput (tokens/second)
- Peak RSS RAM usage (MB)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import psutil

# Add src to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swlp import build_runner
from swlp.config import load_config
from swlp.logging import configure_logging

LOGGER = logging.getLogger(__name__)

# Documented thresholds for performance metrics
THRESHOLDS = {
    "mock": {
        "max_ttft_s": 0.2,
        "min_throughput_tps": 100.0,
        "max_rss_mb": 500.0,
    },
    "hf": {
        "max_ttft_s": 3.0,
        "min_throughput_tps": 20.0,
        "max_rss_mb": 1500.0,
    },
    "swlp": {
        "max_ttft_s": 3.0,
        "min_throughput_tps": 15.0,
        "max_rss_mb": 1500.0,
    },
}


def run_workload(backend: str, prompt: str) -> dict:
    """Run a single backend configuration and return its performance stats."""
    # Load config and adjust runtime options
    cfg = load_config()
    cfg.runtime.backend = backend
    cfg.runtime.profile = True
    cfg.runtime.log_level = "WARNING"
    cfg.generation.max_new_tokens = 32

    # If running on macOS, default to mps if available, otherwise cpu
    import torch
    if torch.backends.mps.is_available():
        cfg.runtime.device = "mps"
    elif torch.cuda.is_available():
        cfg.runtime.device = "cuda"
    else:
        cfg.runtime.device = "cpu"

    # Make sure we don't fall back to baseline silently for swlp
    if backend == "swlp":
        cfg.runtime.swlp_fallback_to_baseline = False

    LOGGER.info("Starting workload: backend=%s on %s", backend, cfg.runtime.device)
    
    # Track RAM usage
    process = psutil.Process()
    
    # Execute inference
    runner = build_runner(cfg)
    result = runner.run(prompt, profile=True)
    metrics = result.metrics
    rss_mb = process.memory_info().rss / (1024 * 1024)

    stats = {
        "backend": backend,
        "ttft_s": metrics.time_to_first_token_seconds or 0.0,
        "throughput_tps": metrics.throughput_tokens_per_second or 0.0,
        "rss_mb": rss_mb,
        "completion_len": len(result.completion),
    }
    
    LOGGER.info(
        "Completed workload %s: TTFT=%.3fs, Throughput=%.2f tok/s, RSS=%.1f MB",
        backend,
        stats["ttft_s"],
        stats["throughput_tps"],
        stats["rss_mb"],
    )
    
    return stats


def verify_stats(stats: dict) -> list[str]:
    """Compare measured stats against thresholds; return list of failures."""
    backend = stats["backend"]
    limits = THRESHOLDS[backend]
    failures = []

    # 1. TTFT check
    if stats["ttft_s"] > limits["max_ttft_s"]:
        failures.append(
            f"[{backend}] TTFT regression: measured {stats['ttft_s']:.3f}s "
            f"exceeds limit {limits['max_ttft_s']:.3f}s"
        )

    # 2. Throughput check
    if stats["throughput_tps"] < limits["min_throughput_tps"]:
        failures.append(
            f"[{backend}] Throughput regression: measured {stats['throughput_tps']:.2f} tok/s "
            f"is below limit {limits['min_throughput_tps']:.2f} tok/s"
        )

    # 3. Peak RSS memory check
    if stats["rss_mb"] > limits["max_rss_mb"]:
        failures.append(
            f"[{backend}] RSS memory regression: measured {stats['rss_mb']:.1f} MB "
            f"exceeds budget {limits['max_rss_mb']:.1f} MB"
        )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain in one sentence what makes a Macbook Air good for development.",
        help="Prompt to run.",
    )
    args = parser.parse_args()

    configure_logging("INFO", json_logs=False)
    LOGGER.info("Starting SWLP Performance Regression Guard")

    all_failures = []
    results = []

    for backend in ["mock", "hf", "swlp"]:
        try:
            stats = run_workload(backend, args.prompt)
            results.append(stats)
            failures = verify_stats(stats)
            if failures:
                all_failures.extend(failures)
        except Exception as exc:
            LOGGER.exception("Workload failed completely for backend %s", backend)
            all_failures.append(f"[{backend}] Workload crashed: {exc}")

    # Print summary table
    print("\n" + "=" * 80)
    header = (
        f"{'Backend':<12} | {'Metric':<15} | "
        f"{'Measured':<15} | {'Threshold':<15} | {'Status':<10}"
    )
    print(header)
    print("-" * 80)
    
    for stats in results:
        backend = stats["backend"]
        limits = THRESHOLDS[backend]
        
        # TTFT
        ttft_ok = stats["ttft_s"] <= limits["max_ttft_s"]
        print(
            f"{backend:<12} | {'TTFT':<15} | {stats['ttft_s']:>13.3f}s | "
            f"< {limits['max_ttft_s']:>11.3f}s | {'PASS' if ttft_ok else 'FAIL'}"
        )
        
        # Throughput
        tp_ok = stats["throughput_tps"] >= limits["min_throughput_tps"]
        print(
            f"{backend:<12} | {'Throughput':<15} | {stats['throughput_tps']:>11.2f} t/s | "
            f"> {limits['min_throughput_tps']:>9.2f} t/s | {'PASS' if tp_ok else 'FAIL'}"
        )
        
        # RSS
        rss_ok = stats["rss_mb"] <= limits["max_rss_mb"]
        print(
            f"{backend:<12} | {'Peak RSS':<15} | {stats['rss_mb']:>11.1f} MB | "
            f"< {limits['max_rss_mb']:>9.1f} MB | {'PASS' if rss_ok else 'FAIL'}"
        )
        print("-" * 80)

    print("=" * 80)

    if all_failures:
        print("\n\n❌ PERFORMANCE REGRESSION DETECTED!")
        for failure in all_failures:
            print(f"  - {failure}")
        print("=" * 80 + "\n")
        return 1
    
    print("\n\n✅ ALL PERFORMANCE CHECKS PASSED SUCCESSFULLY!\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
