"""Automated simulator sweeps for SWLP scheduling research.

Sweeps parameter ranges and produces structured datasets for analysis.
No model or GPU required — pure simulation.

Usage:
    from swlp.core.sweep import SweepConfig, run_sweep

    results = run_sweep(
        SweepConfig(
            num_layers=[32, 80],
            layer_size_mb=[512, 700],
            ram_gb=[16, 24, 32],
            window_size=[2, 4],
            prefetch_depth=[2, 4, 6],
            resident_count=[0, 4, 8],
        )
    )
    results.to_json("sweep_results.json")
    results.print_table()
"""

from __future__ import annotations

import csv
import itertools
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .profiler import compute_pipeline_metrics
from .simulator import SimResult, SimulatorConfig, simulate


@dataclass
class SweepPoint:
    """Result of a single simulation configuration point."""

    # Input parameters
    num_layers: int
    layer_size_mb: float
    ram_gb: float
    window_size: int
    prefetch_depth: int
    resident_count: int
    num_tokens: int
    compute_time_ms: float
    ssd_read_latency_ms: float
    upload_latency_ms: float
    eviction_latency_ms: float
    worker_count: int

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
    ram_utilization_pct: float = 0.0
    pipeline_stall_count: int = 0
    fits_in_ram: bool = False
    resident_layers_loaded: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SweepResult:
    """Complete sweep result with all configuration points."""

    points: list[SweepPoint] = field(default_factory=list)
    sweep_config: dict[str, Any] = field(default_factory=dict)

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "sweep_config": self.sweep_config,
            "num_points": len(self.points),
            "points": [p.to_dict() for p in self.points],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.points:
            return
        fieldnames = list(self.points[0].to_dict().keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for p in self.points:
                writer.writerow(p.to_dict())

    def print_table(self) -> None:
        """Print a formatted comparison table."""
        if not self.points:
            print("No sweep results.")
            return

        # Group by (num_layers, layer_size_mb) for readability
        groups: dict[tuple, list[SweepPoint]] = {}
        for p in self.points:
            key = (p.num_layers, p.layer_size_mb)
            groups.setdefault(key, []).append(p)

        for (nl, lsz), pts in groups.items():
            total_ram_needed = nl * lsz / 1024
            print(f"\n{'='*100}")
            print(f"  {nl} layers × {lsz:.0f} MB = {total_ram_needed:.1f} GB model")
            print(f"{'='*100}")
            print(
                f"  {'RAM':>5} {'Win':>4} {'Pref':>4} {'Res':>4} "
                f"{'Tok/s':>7} {'GPU%':>5} {'Idle%':>5} {'Hit%':>5} "
                f"{'Wait':>6} {'Peak':>7} {'Fits':>5}"
            )
            print(
                f"  {'-'*5} {'-'*4} {'-'*4} {'-'*4} {'-'*7} {'-'*5} {'-'*5} {'-'*5} "
                f"{'-'*6} {'-'*7} {'-'*5}"
            )

            for p in sorted(pts, key=lambda x: (x.ram_gb, x.window_size, x.resident_count)):
                fits = "yes" if p.fits_in_ram else "no"
                print(
                    f"  {p.ram_gb:>4.0f}G {p.window_size:>4} {p.prefetch_depth:>4} "
                    f"{p.resident_count:>4} "
                    f"{p.throughput_tokens_per_sec:>7.2f} {p.gpu_efficiency*100:>4.1f}% "
                    f"{p.gpu_idle_pct:>4.1f}% {p.prefetch_hit_rate*100:>4.1f}% "
                    f"{p.avg_ensure_wait_ms:>5.1f} {p.peak_ram_mb:>6.0f}M {fits:>5}"
                )


@dataclass
class SweepConfig:
    """Parameter ranges for a sweep.

    Each parameter can be a single value or a list. The sweep generates
    the Cartesian product of all parameter combinations.
    """

    num_layers: int | list[int] = 32
    layer_size_mb: float | list[float] = 512.0
    ram_gb: float | list[float] = 16.0
    window_size: int | list[int] = 2
    prefetch_depth: int | list[int] = 4
    resident_count: int | list[int] = 0
    num_tokens: int = 10
    compute_time_ms: float = 50.0
    ssd_read_latency_ms: float = 30.0
    upload_latency_ms: float = 10.0
    eviction_latency_ms: float = 5.0
    worker_count: int = 2

    def _to_list(self, val: Any) -> list:
        return val if isinstance(val, list) else [val]

    def iter_configs(self) -> list[SimulatorConfig]:
        """Generate all SimulatorConfig combinations."""
        params = {
            "num_layers": self._to_list(self.num_layers),
            "layer_size_mb": self._to_list(self.layer_size_mb),
            "ram_capacity_gb": self._to_list(self.ram_gb),
            "window_size": self._to_list(self.window_size),
            "prefetch_depth": self._to_list(self.prefetch_depth),
            "resident_count": self._to_list(self.resident_count),
        }
        keys = list(params.keys())
        configs = []
        for vals in itertools.product(*params.values()):
            kw = dict(zip(keys, vals, strict=False))
            # Resident count can't exceed num_layers
            kw["resident_count"] = min(kw["resident_count"], kw["num_layers"])
            # Window can't exceed num_layers
            kw["window_size"] = min(kw["window_size"], kw["num_layers"])
            configs.append(SimulatorConfig(
                num_layers=kw["num_layers"],
                layer_size_mb=kw["layer_size_mb"],
                ram_capacity_gb=kw["ram_capacity_gb"],
                window_size=kw["window_size"],
                prefetch_depth=kw["prefetch_depth"],
                resident_count=kw["resident_count"],
                use_resident_cache=kw["resident_count"] > 0,
                num_tokens=self.num_tokens,
                compute_time_ms=self.compute_time_ms,
                ssd_read_latency_ms=self.ssd_read_latency_ms,
                upload_latency_ms=self.upload_latency_ms,
                eviction_latency_ms=self.eviction_latency_ms,
                worker_count=self.worker_count,
            ))
        return configs


def _result_to_point(result: SimResult) -> SweepPoint:
    """Extract all metrics from a SimResult into a SweepPoint."""
    metrics = compute_pipeline_metrics(
        result.traces,
        wall_time=result.wall_time_ms / 1000,
        num_tokens=result.tokens_processed,
    )
    cfg = result.config
    total_ram_mb = cfg.num_layers * cfg.layer_size_mb
    fits = result.peak_ram_mb <= cfg.ram_capacity_gb * 1024

    throughput = (
        result.tokens_processed / (result.wall_time_ms / 1000) if result.wall_time_ms > 0 else 0.0
    )

    return SweepPoint(
        num_layers=cfg.num_layers,
        layer_size_mb=cfg.layer_size_mb,
        ram_gb=cfg.ram_capacity_gb,
        window_size=cfg.window_size,
        prefetch_depth=cfg.prefetch_depth,
        resident_count=cfg.resident_count,
        num_tokens=cfg.num_tokens,
        compute_time_ms=cfg.compute_time_ms,
        ssd_read_latency_ms=cfg.ssd_read_latency_ms,
        upload_latency_ms=cfg.upload_latency_ms,
        eviction_latency_ms=cfg.eviction_latency_ms,
        worker_count=cfg.worker_count,
        throughput_tokens_per_sec=throughput,
        wall_time_ms=result.wall_time_ms,
        gpu_efficiency=metrics.gpu_efficiency,
        gpu_idle_pct=(1.0 - metrics.gpu_efficiency) * 100,
        prefetch_hit_rate=metrics.prefetch_hit_rate,
        avg_ensure_wait_ms=metrics.avg_ensure_wait_ms,
        avg_read_ms=metrics.avg_read_ms,
        avg_upload_ms=metrics.avg_upload_ms,
        peak_ram_mb=result.peak_ram_mb,
        ram_utilization_pct=(result.peak_ram_mb / total_ram_mb * 100) if total_ram_mb > 0 else 0,
        pipeline_stall_count=metrics.pipeline_stall_count,
        fits_in_ram=fits,
        resident_layers_loaded=cfg.resident_count if cfg.use_resident_cache else 0,
    )


def run_sweep(config: SweepConfig) -> SweepResult:
    """Run all simulation configurations and return structured results."""
    configs = config.iter_configs()
    points = []
    for _i, sim_config in enumerate(configs):
        result = simulate(sim_config)
        point = _result_to_point(result)
        points.append(point)

    sweep_meta = {
        "total_configs": len(configs),
        "num_layers": (
            config.num_layers if isinstance(config.num_layers, int) else config.num_layers
        ),
        "layer_size_mb": (
            config.layer_size_mb
            if isinstance(config.layer_size_mb, float)
            else config.layer_size_mb
        ),
        "ram_gb": config.ram_gb if isinstance(config.ram_gb, float) else config.ram_gb,
        "window_size": (
            config.window_size if isinstance(config.window_size, int) else config.window_size
        ),
        "prefetch_depth": (
            config.prefetch_depth
            if isinstance(config.prefetch_depth, int)
            else config.prefetch_depth
        ),
        "resident_count": (
            config.resident_count
            if isinstance(config.resident_count, int)
            else config.resident_count
        ),
    }

    return SweepResult(points=points, sweep_config=sweep_meta)
