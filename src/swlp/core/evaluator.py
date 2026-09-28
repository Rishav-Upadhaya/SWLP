"""Policy evaluator for SWLP scheduling research.

Runs the same workload across multiple scheduling policies and produces
a side-by-side comparison.  Answers: which policy is fastest, uses least
RAM, keeps GPU busiest, and has fewest stalls?

Usage:
    from swlp.core.evaluator import evaluate_policies, EvaluationConfig

    result = evaluate_policies(
        EvaluationConfig(
            num_layers=32,
            layer_size_mb=512,
            ram_gb=16,
            policies=["baseline", "resident_4", "resident_8"],
        )
    )
    result.print_table()
    result.to_json("eval_results.json")
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .profiler import compute_pipeline_metrics
from .simulator import SimResult, SimulatorConfig, simulate


@dataclass
class PolicyResult:
    """Result of evaluating a single policy."""

    # Policy identity
    policy_name: str
    policy_type: str  # "baseline" | "resident_cache" | "adaptive_window"
    resident_count: int
    window_size: int
    prefetch_depth: int

    # Input config
    num_layers: int
    layer_size_mb: float
    ram_gb: float
    num_tokens: int

    # Output metrics
    throughput_tokens_per_sec: float = 0.0
    wall_time_ms: float = 0.0
    gpu_efficiency: float = 0.0
    gpu_idle_pct: float = 0.0
    prefetch_hit_rate: float = 0.0
    avg_ensure_wait_ms: float = 0.0
    avg_read_ms: float = 0.0
    avg_upload_ms: float = 0.0
    peak_ram_mb: float = 0.0
    pipeline_stall_count: int = 0
    reloads_per_token: float = 0.0
    total_reloads: int = 0
    fits_in_ram: bool = False

    # Relative performance vs best
    throughput_vs_best: float = 1.0
    ram_vs_best: float = 1.0
    gpu_vs_best: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvaluationResult:
    """Complete evaluation result with all policies."""

    results: list[PolicyResult] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "config": self.config,
            "num_policies": len(self.results),
            "results": [r.to_dict() for r in self.results],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.results:
            return
        fieldnames = list(self.results[0].to_dict().keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.results:
                writer.writerow(r.to_dict())

    def print_table(self) -> None:
        """Print a formatted comparison table."""
        if not self.results:
            print("No evaluation results.")
            return

        print(f"\n{'='*100}")
        print("POLICY EVALUATION COMPARISON")
        print(f"{'='*100}")
        print(
            f"  {'Policy':<20} {'Tok/s':>7} {'GPU%':>5} {'Idle%':>5} "
            f"{'Hit%':>5} {'Wait':>6} {'Peak':>7} {'Stalls':>6} {'vs Best':>8}"
        )
        print(f"  {'-'*20} {'-'*7} {'-'*5} {'-'*5} {'-'*5} {'-'*6} {'-'*7} {'-'*6} {'-'*8}")

        best_throughput = max(r.throughput_tokens_per_sec for r in self.results)
        best_ram = min(r.peak_ram_mb for r in self.results)
        best_gpu = max(r.gpu_efficiency for r in self.results)

        for r in sorted(self.results, key=lambda x: x.throughput_tokens_per_sec, reverse=True):
            throughput_pct = (
                r.throughput_tokens_per_sec / best_throughput * 100 if best_throughput > 0 else 0
            )
            print(
                f"  {r.policy_name:<20} {r.throughput_tokens_per_sec:>7.2f} "
                f"{r.gpu_efficiency*100:>4.1f}% {r.gpu_idle_pct:>4.1f}% "
                f"{r.prefetch_hit_rate*100:>4.1f}% {r.avg_ensure_wait_ms:>5.1f} "
                f"{r.peak_ram_mb:>6.0f}M {r.pipeline_stall_count:>6} {throughput_pct:>6.1f}%"
            )

        print(
            f"\n  Best throughput: "
            f"{max(r.throughput_tokens_per_sec for r in self.results):.2f} tok/s"
        )
        print(f"  Best RAM usage: {best_ram:.0f} MB")
        print(f"  Best GPU efficiency: {best_gpu*100:.1f}%")

        # Cost model summary
        print(f"\n{'='*80}")
        print("COST MODEL ANALYSIS")
        print(f"{'='*80}")
        for r in sorted(self.results, key=lambda x: x.throughput_tokens_per_sec, reverse=True):
            reload_cost = r.total_reloads * (r.avg_read_ms + r.avg_upload_ms)
            save_cost = r.resident_count * r.avg_read_ms
            print(f"  {r.policy_name:<20}")
            print(f"    Resident layers:    {r.resident_count}")
            print(f"    Reloads per token:  {r.reloads_per_token:.2f}")
            print(
                f"    SSD reads saved:    {r.total_reloads} × {r.avg_read_ms:.1f}ms "
                f"= {reload_cost:.0f}ms"
            )
            print(
                f"    Resident overhead:  {r.resident_count} × {r.avg_read_ms:.1f}ms "
                f"= {save_cost:.0f}ms"
            )
            print(f"    Net benefit:        {reload_cost - save_cost:.0f}ms")
        print()


@dataclass
class EvaluationConfig:
    """Configuration for policy evaluation."""

    # Hardware
    num_layers: int = 32
    layer_size_mb: float = 512.0
    ram_gb: float = 16.0

    # Workload
    num_tokens: int = 10
    compute_time_ms: float = 50.0
    ssd_read_latency_ms: float = 30.0
    upload_latency_ms: float = 10.0
    eviction_latency_ms: float = 5.0
    worker_count: int = 2

    # Policies to evaluate
    policies: list[str] = field(default_factory=lambda: ["baseline", "resident_4", "resident_8"])

    # Base config
    base_window_size: int = 2
    base_prefetch_depth: int = 4


def _parse_policy(name: str, base: EvaluationConfig) -> tuple[str, str, SimulatorConfig]:
    """Parse a policy name into a SimulatorConfig.

    Returns (display_name, policy_type, config).
    """
    name_lower = name.lower().strip()

    if name_lower == "baseline":
        return (
            "Baseline",
            "baseline",
            SimulatorConfig(
                num_layers=base.num_layers,
                layer_size_mb=base.layer_size_mb,
                ram_capacity_gb=base.ram_gb,
                window_size=base.base_window_size,
                prefetch_depth=base.base_prefetch_depth,
                num_tokens=base.num_tokens,
                compute_time_ms=base.compute_time_ms,
                ssd_read_latency_ms=base.ssd_read_latency_ms,
                upload_latency_ms=base.upload_latency_ms,
                eviction_latency_ms=base.eviction_latency_ms,
                worker_count=base.worker_count,
                use_resident_cache=False,
                resident_count=0,
            ),
        )

    if name_lower.startswith("resident_"):
        try:
            count = int(name_lower.split("_", 1)[1])
        except (ValueError, IndexError):
            count = 4
        return (
            f"Resident Cache ({count})",
            "resident_cache",
            SimulatorConfig(
                num_layers=base.num_layers,
                layer_size_mb=base.layer_size_mb,
                ram_capacity_gb=base.ram_gb,
                window_size=base.base_window_size,
                prefetch_depth=base.base_prefetch_depth,
                num_tokens=base.num_tokens,
                compute_time_ms=base.compute_time_ms,
                ssd_read_latency_ms=base.ssd_read_latency_ms,
                upload_latency_ms=base.upload_latency_ms,
                eviction_latency_ms=base.eviction_latency_ms,
                worker_count=base.worker_count,
                use_resident_cache=True,
                resident_count=min(count, base.num_layers),
            ),
        )

    if name_lower.startswith("window_"):
        try:
            win = int(name_lower.split("_", 1)[1])
        except (ValueError, IndexError):
            win = 4
        return (
            f"Window ({win})",
            "window",
            SimulatorConfig(
                num_layers=base.num_layers,
                layer_size_mb=base.layer_size_mb,
                ram_capacity_gb=base.ram_gb,
                window_size=min(win, base.num_layers),
                prefetch_depth=base.base_prefetch_depth,
                num_tokens=base.num_tokens,
                compute_time_ms=base.compute_time_ms,
                ssd_read_latency_ms=base.ssd_read_latency_ms,
                upload_latency_ms=base.upload_latency_ms,
                eviction_latency_ms=base.eviction_latency_ms,
                worker_count=base.worker_count,
            ),
        )

    if name_lower.startswith("prefetch_"):
        try:
            depth = int(name_lower.split("_", 1)[1])
        except (ValueError, IndexError):
            depth = 6
        return (
            f"Prefetch ({depth})",
            "prefetch",
            SimulatorConfig(
                num_layers=base.num_layers,
                layer_size_mb=base.layer_size_mb,
                ram_capacity_gb=base.ram_gb,
                window_size=base.base_window_size,
                prefetch_depth=depth,
                num_tokens=base.num_tokens,
                compute_time_ms=base.compute_time_ms,
                ssd_read_latency_ms=base.ssd_read_latency_ms,
                upload_latency_ms=base.upload_latency_ms,
                eviction_latency_ms=base.eviction_latency_ms,
                worker_count=base.worker_count,
            ),
        )

    # Fallback: treat as baseline
    return (
        name,
        "unknown",
        SimulatorConfig(
            num_layers=base.num_layers,
            layer_size_mb=base.layer_size_mb,
            ram_capacity_gb=base.ram_gb,
            window_size=base.base_window_size,
            prefetch_depth=base.base_prefetch_depth,
            num_tokens=base.num_tokens,
            compute_time_ms=base.compute_time_ms,
            ssd_read_latency_ms=base.ssd_read_latency_ms,
            upload_latency_ms=base.upload_latency_ms,
            eviction_latency_ms=base.eviction_latency_ms,
            worker_count=base.worker_count,
        ),
    )


def _result_to_policy_result(
    result: SimResult,
    display_name: str,
    policy_type: str,
    base: EvaluationConfig,
) -> PolicyResult:
    """Extract metrics from a SimResult into a PolicyResult."""
    metrics = compute_pipeline_metrics(
        result.traces,
        wall_time=result.wall_time_ms / 1000,
        num_tokens=result.tokens_processed,
    )
    cfg = result.config
    fits = result.peak_ram_mb <= cfg.ram_capacity_gb * 1024
    throughput = (
        result.tokens_processed / (result.wall_time_ms / 1000) if result.wall_time_ms > 0 else 0.0
    )

    # Count reloads: layers that were loaded then evicted then loaded again
    # For resident layers: 0 reloads (they stay loaded)
    # For non-resident: each token reloads all non-resident layers
    non_resident = cfg.num_layers - cfg.resident_count
    total_reloads = non_resident * cfg.num_tokens

    return PolicyResult(
        policy_name=display_name,
        policy_type=policy_type,
        resident_count=cfg.resident_count,
        window_size=cfg.window_size,
        prefetch_depth=cfg.prefetch_depth,
        num_layers=cfg.num_layers,
        layer_size_mb=cfg.layer_size_mb,
        ram_gb=cfg.ram_capacity_gb,
        num_tokens=cfg.num_tokens,
        throughput_tokens_per_sec=throughput,
        wall_time_ms=result.wall_time_ms,
        gpu_efficiency=metrics.gpu_efficiency,
        gpu_idle_pct=(1.0 - metrics.gpu_efficiency) * 100,
        prefetch_hit_rate=metrics.prefetch_hit_rate,
        avg_ensure_wait_ms=metrics.avg_ensure_wait_ms,
        avg_read_ms=metrics.avg_read_ms,
        avg_upload_ms=metrics.avg_upload_ms,
        peak_ram_mb=result.peak_ram_mb,
        pipeline_stall_count=metrics.pipeline_stall_count,
        reloads_per_token=non_resident,
        total_reloads=total_reloads,
        fits_in_ram=fits,
    )


def evaluate_policies(config: EvaluationConfig) -> EvaluationResult:
    """Run all policies and return a comparison."""
    policy_results = []

    for policy_name in config.policies:
        display_name, policy_type, sim_config = _parse_policy(policy_name, config)
        result = simulate(sim_config)
        pr = _result_to_policy_result(result, display_name, policy_type, config)
        policy_results.append(pr)

    # Compute relative performance
    if policy_results:
        best_throughput = max(r.throughput_tokens_per_sec for r in policy_results)
        best_ram = min(r.peak_ram_mb for r in policy_results)
        best_gpu = max(r.gpu_efficiency for r in policy_results)
        for r in policy_results:
            r.throughput_vs_best = (
                r.throughput_tokens_per_sec / best_throughput if best_throughput > 0 else 0
            )
            r.ram_vs_best = best_ram / r.peak_ram_mb if r.peak_ram_mb > 0 else 0
            r.gpu_vs_best = r.gpu_efficiency / best_gpu if best_gpu > 0 else 0

    eval_config = {
        "num_layers": config.num_layers,
        "layer_size_mb": config.layer_size_mb,
        "ram_gb": config.ram_gb,
        "num_tokens": config.num_tokens,
        "policies": config.policies,
    }

    return EvaluationResult(results=policy_results, config=eval_config)
