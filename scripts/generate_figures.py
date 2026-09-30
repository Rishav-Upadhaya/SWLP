#!/usr/bin/env python3
"""Generate paper figures from simulation and experiment data.

Produces:
1. Pipeline Ratio vs Best-Observed Resident Count
2. Throughput by Model Size (FP16)
3. GPU Idle vs Pipeline Ratio
4. GB/token by Model
5. Speculative Decoding Speedup
6. Early Exit Speedup vs Accuracy Tradeoff

Usage:
    python scripts/generate_figures.py --output-dir figures/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def generate_pipeline_ratio_vs_resident(output_dir: Path) -> None:
    """Figure 1: Pipeline Ratio vs Best-Observed Resident Count."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping figure generation.")
        return

    # Data from simulation (M5 16GB)
    ratios = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
    best_residents = [2, 2, 4, 8, 12, 16, 20]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ratios, best_residents, "o-", linewidth=2, markersize=8, color="#2563eb")
    ax.fill_between(
        ratios,
        [r - 1 for r in best_residents],
        [r + 1 for r in best_residents],
        alpha=0.2,
        color="#2563eb",
    )

    ax.set_xlabel("Pipeline Ratio", fontsize=12)
    ax.set_ylabel("Best-Observed Resident Layers", fontsize=12)
    ax.set_title("Pipeline Ratio Predicts Residency Requirement", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_xticks(ratios)
    ax.set_xticklabels([f"{r}" for r in ratios])

    # Annotate regions
    ax.axvspan(0.3, 1.0, alpha=0.1, color="red", label="Compute-bound")
    ax.axvspan(1.0, 15, alpha=0.1, color="green", label="I/O-bound")
    ax.legend(loc="upper left")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "pipeline_ratio_vs_resident.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "pipeline_ratio_vs_resident.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: pipeline_ratio_vs_resident.png/pdf")


def generate_throughput_by_model(output_dir: Path) -> None:
    """Figure 2: Throughput by Model Size (FP16, measured data)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    models = ["Qwen-0.5B", "Qwen-3B", "Qwen-7B", "Mistral-7B", "Qwen-14B"]
    swlp_tps = [8.39, 5.05, 2.49, 0.372, 0.194]
    mlx_int8 = [None, None, None, 16.0, None]
    mlx_int4 = [None, None, None, 27.9, 13.8]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    width = 0.25

    ax.bar(x - width, swlp_tps, width, label="SWLP FP16", color="#2563eb")
    mlx8_vals = [v if v else 0 for v in mlx_int8]
    bars2 = ax.bar(x, mlx8_vals, width, label="MLX int8", color="#059669")
    mlx4_vals = [v if v else 0 for v in mlx_int4]
    bars3 = ax.bar(x + width, mlx4_vals, width, label="MLX int4", color="#d97706")

    # Hide bars with 0 value
    for bar, val in zip(bars2, mlx_int8, strict=False):
        if val is None:
            bar.set_alpha(0)
    for bar, val in zip(bars3, mlx_int4, strict=False):
        if val is None:
            bar.set_alpha(0)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Throughput (tok/s)", fontsize=12)
    ax.set_title("Throughput by Model and Backend (M5 16GB)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_yscale("log")
    ax.set_ylim(0.1, 50)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "throughput_by_model.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "throughput_by_model.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: throughput_by_model.png/pdf")


def generate_gpu_idle_vs_ratio(output_dir: Path) -> None:
    """Figure 3: GPU Idle vs Pipeline Ratio."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Simulation data
    ratios = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    gpu_idle_w2 = [23.2, 12.0, 6.5, 4.3, 3.0, 2.4]
    gpu_idle_w4 = [23.2, 12.0, 6.5, 4.3, 3.0, 2.4]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ratios, gpu_idle_w2, "o-", linewidth=2, markersize=8, label="W=2", color="#2563eb")
    ax.plot(ratios, gpu_idle_w4, "s--", linewidth=2, markersize=8, label="W=4", color="#dc2626")

    ax.set_xlabel("Pipeline Ratio", fontsize=12)
    ax.set_ylabel("GPU Idle (%)", fontsize=12)
    ax.set_title("GPU Idle Decreases with Pipeline Ratio", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_xticks(ratios)
    ax.set_xticklabels([f"{r}" for r in ratios])

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "gpu_idle_vs_ratio.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "gpu_idle_vs_ratio.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: gpu_idle_vs_ratio.png/pdf")


def generate_gb_per_token(output_dir: Path) -> None:
    """Figure 4: GB/token by Model and Quantization."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    models = ["Qwen-0.5B", "Qwen-3B", "Qwen-7B", "Mistral-7B", "Qwen-14B"]
    gb_fp16 = [0.98, 5.98, 13.62, 13.62, 25.83]
    gb_fp8 = [None, None, None, 6.81, 12.89]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    width = 0.35

    ax.bar(x - width / 2, gb_fp16, width, label="FP16", color="#2563eb")
    gb8_vals = [v if v else 0 for v in gb_fp8]
    bars2 = ax.bar(x + width / 2, gb8_vals, width, label="FP8", color="#059669")

    for bar, val in zip(bars2, gb_fp8, strict=False):
        if val is None:
            bar.set_alpha(0)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("GB/token", fontsize=12)
    ax.set_title("Disk I/O Intensity by Model (GB read per generated token)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "gb_per_token.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "gb_per_token.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: gb_per_token.png/pdf")


def generate_speculative_speedup(output_dir: Path) -> None:
    """Figure 5: Speculative Decoding Speedup."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    # Measured data from docs/results.md
    workloads = ["Open-ended\n(novel text)", "Mildly\nrepetitive", "Repetition\nheavy"]
    speedups = [0.93, 1.15, 3.29]
    acceptance = [0, 100, 100]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    x = np.arange(len(workloads))
    bars = ax1.bar(x, speedups, 0.5, color=["#dc2626", "#d97706", "#059669"])
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Baseline")
    ax2.plot(
        x, acceptance, "o--", color="#7c3aed", linewidth=2, markersize=10, label="Acceptance %"
    )

    ax1.set_xlabel("Workload Type", fontsize=12)
    ax1.set_ylabel("Speedup vs Baseline", fontsize=12, color="#2563eb")
    ax2.set_ylabel("Acceptance Rate (%)", fontsize=12, color="#7c3aed")
    ax1.set_title("Speculative Decoding: Speedup vs Acceptance (Mistral-7B FP16)", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(workloads)
    ax1.set_ylim(0, 4)

    # Add value labels
    for bar, val in zip(bars, speedups, strict=False):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.2f}x",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "speculative_speedup.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "speculative_speedup.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: speculative_speedup.png/pdf")


def generate_early_exit_tradeoff(output_dir: Path) -> None:
    """Figure 6: Early Exit Speedup vs Layer Skip."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    skip_pct = [0, 20, 40]
    speedup = [1.0, 1.23, 1.60]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(skip_pct, speedup, "o-", linewidth=2, markersize=10, color="#2563eb")

    for x, y in zip(skip_pct, speedup, strict=False):
        ax.annotate(
            f"{y:.2f}x",
            (x, y),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xlabel("Layers Skipped (%)", fontsize=12)
    ax.set_ylabel("Speedup vs Full Model", fontsize=12)
    ax.set_title("Early Exit: Speedup vs Layer Skip (Mistral-7B FP16)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(skip_pct)
    ax.set_xticklabels([f"{s}%" for s in skip_pct])
    ax.set_ylim(0.8, 2.0)

    # Add note about accuracy
    ax.text(
        0.5,
        0.05,
        "Note: Accuracy impact not measured",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color="gray",
        style="italic",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "early_exit_tradeoff.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "early_exit_tradeoff.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: early_exit_tradeoff.png/pdf")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--output-dir", default="figures", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print(f"Generating figures to {output_dir}/")
    print()

    generate_pipeline_ratio_vs_resident(output_dir)
    generate_throughput_by_model(output_dir)
    generate_gpu_idle_vs_ratio(output_dir)
    generate_gb_per_token(output_dir)
    generate_speculative_speedup(output_dir)
    generate_early_exit_tradeoff(output_dir)

    print()
    print(f"Figures saved to {output_dir}/")
    print("Include in paper with: \\includegraphics{{figures/pipeline_ratio_vs_resident.pdf}}")


if __name__ == "__main__":
    main()
