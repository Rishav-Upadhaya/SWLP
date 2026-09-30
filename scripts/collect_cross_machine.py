#!/usr/bin/env python3
"""Cross-machine data collection for SWLP validation.

Runs on any machine to collect:
- Hardware profile (chip, RAM, SSD bandwidth)
- Pipeline ratio measurements for known models
- Best-observed resident count via sweep
- Prediction vs actual comparison

Usage:
    # Full sweep (all models, all resident counts)
    python scripts/collect_cross_machine.py --full

    # Quick check (just mistral-7b and qwen-14b)
    python scripts/collect_cross_machine.py

    # Specific model
    python scripts/collect_cross_machine.py --model mistral-7b

Output:
    experiments/cross_machine/{hardware_id}/{model}_{timestamp}.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def get_hardware_id() -> dict:
    """Detect hardware and return a machine-readable profile."""
    try:
        from swlp.hardware.detect import detect_hardware

        hw = detect_hardware()
        return {
            "chip_name": hw.chip_name,
            "memory_gb": hw.memory_gb,
            "unified_memory": hw.unified_memory,
            "device_type": hw.device_type,
            "ssd_bandwidth_gbps": hw.ssd_bandwidth_gbps,
            "preferred_backend": hw.preferred_backend,
        }
    except Exception as e:
        return {"error": str(e)}


def measure_pipeline_ratio(model_alias: str, hw: dict) -> dict:
    """Measure pipeline ratio for a model on this hardware."""
    try:
        from swlp.cli_doctor import _measure_pipeline_ratio
        from swlp.hardware.detect import HardwareInfo

        hw_info = HardwareInfo(
            chip_name=hw.get("chip_name", "unknown"),
            memory_gb=hw.get("memory_gb", 16),
            unified_memory=hw.get("unified_memory", True),
            device_type=hw.get("device_type", "mps"),
            ssd_bandwidth_gbps=hw.get("ssd_bandwidth_gbps", 3.0),
            preferred_backend=hw.get("preferred_backend", "mlx"),
        )
        result = _measure_pipeline_ratio(model_alias, hw_info)
        return result
    except Exception as e:
        return {"error": str(e)}


def run_sweep(model_alias: str, hw: dict, full: bool = False) -> list[dict]:
    """Run resident count sweep and find best-observed configuration."""
    try:
        from scripts.research.simtools.event_simulator import (
            EventSimulator,
            LayerTimings,
            SimConfig,
            SlidingWindowStrategy,
        )

        # Get model parameters
        from swlp.cli_doctor import KNOWN_FP16_GB, MODEL_LAYERS

        if model_alias not in MODEL_LAYERS:
            return [{"error": f"unknown model: {model_alias}"}]

        num_layers, layer_size_mb = MODEL_LAYERS[model_alias]
        KNOWN_FP16_GB.get(model_alias, 0)

        # Estimate timings from hardware
        bandwidth_mb_per_s = hw.get("ssd_bandwidth_gbps", 3.0) * 1024 / 8
        read_ms = (layer_size_mb / bandwidth_mb_per_s) * 1000 if bandwidth_mb_per_s > 0 else 30
        compute_ms = 30.0  # typical for this class

        # Run sweep
        resident_range = [0, 2, 4, 8, 12, 16] if full else [0, 4, 8]
        resident_range = [r for r in resident_range if r < num_layers]

        results = []
        for resident_count in resident_range:
            timings = LayerTimings(
                read_ms=read_ms,
                deserialize_ms=3.0,
                upload_ms=5.0,
                compute_ms=compute_ms,
                evict_ms=0.5,
            )
            config = SimConfig(
                num_layers=num_layers,
                layer_weight_mb=layer_size_mb,
                ram_budget_mb=hw.get("memory_gb", 16) * 1024,
                window_size=2,
                prefetch_depth=4,
                max_ssd_concurrency=2,
                max_upload_concurrency=2,
                seed=42,
            )
            strategy = SlidingWindowStrategy(config, timings)
            sim = EventSimulator(config, timings, strategy)
            result = sim.simulate(num_tokens=10)

            results.append(
                {
                    "resident_count": resident_count,
                    "throughput_tok_per_sec": result.throughput_toks_per_sec,
                    "gpu_busy_pct": result.gpu_busy_pct * 100,
                    "gpu_idle_pct": (1 - result.gpu_busy_pct) * 100,
                    "gb_per_token": result.gb_per_token,
                }
            )

        # Find best
        best = max(results, key=lambda r: r["throughput_tok_per_sec"])
        return {
            "sweep": results,
            "best_resident": best["resident_count"],
            "best_throughput": best["throughput_tok_per_sec"],
            "best_gpu_idle": best["gpu_idle_pct"],
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Collect cross-machine SWLP data")
    parser.add_argument("--model", default=None, help="Specific model to test")
    parser.add_argument("--full", action="store_true", help="Full sweep (all resident counts)")
    parser.add_argument(
        "--output-dir", default="experiments/cross_machine", help="Output directory"
    )
    args = parser.parse_args()

    # Detect hardware
    hw = get_hardware_id()
    hw_id = hw.get("chip_name", "unknown").replace(" ", "_").replace(",", "")
    ram_gb = int(hw.get("memory_gb", 0))
    print(f"Hardware: {hw.get('chip_name', '?')} {ram_gb}GB")

    # Models to test
    if args.model:
        models = [args.model]
    else:
        models = ["mistral-7b", "qwen-14b", "qwen-7b", "qwen-3b"]

    # Run collection
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = Path(args.output_dir) / f"{hw_id}_{ram_gb}GB"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for model in models:
        print(f"\n  Testing {model}...")

        # Measure pipeline ratio
        pipeline = measure_pipeline_ratio(model, hw)
        print(f"    Pipeline ratio: {pipeline.get('ratio', '?')}")

        # Run sweep
        sweep = run_sweep(model, hw, full=args.full)
        if isinstance(sweep, dict) and "error" not in sweep:
            print(
                f"    Best resident: R={sweep['best_resident']}, "
                f"{sweep['best_throughput']:.2f} tok/s"
            )
        else:
            print(f"    Error: {sweep}")

        result = {
            "hardware": hw,
            "model": model,
            "pipeline": pipeline,
            "sweep": sweep,
            "timestamp": timestamp,
        }
        all_results.append(result)

        # Save individual result
        model_file = output_dir / f"{model}_{timestamp}.json"
        model_file.write_text(json.dumps(result, indent=2, default=str))
        print(f"    Saved: {model_file}")

    # Save combined result
    combined_file = output_dir / f"all_models_{timestamp}.json"
    combined_file.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nCombined: {combined_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("COLLECTION SUMMARY")
    print("=" * 60)
    print(f"  Hardware: {hw.get('chip_name', '?')} {ram_gb}GB")
    print(f"  Models:   {len(models)}")
    print(f"  Output:   {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
