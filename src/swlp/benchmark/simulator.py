from __future__ import annotations

import csv
import json
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BYTES_PER_MB = 1024 * 1024


@dataclass(slots=True)
class Scenario:
    name: str
    num_layers: int
    layer_compute_ms: float
    layer_weight_mb: float
    window_sizes: list[int]
    context_tokens: int
    generate_tokens: int
    kv_bytes_per_token: int
    vram_capacity_mb: float
    ram_capacity_mb: float
    pcie_bandwidth_gbps: float
    ram_bandwidth_gbps: float
    disk_bandwidth_gbps: float
    disk_staging: bool
    overlap: bool


@dataclass(slots=True)
class SimulationResult:
    scenario: str
    window_size: int
    per_token_seconds: float
    time_to_first_token_seconds: float
    total_generate_seconds: float
    throughput_tokens_per_second: float
    compute_seconds_per_token: float
    transfer_seconds_per_token: float
    stall_seconds_per_token: float
    vram_peak_mb: float
    ram_peak_mb: float
    fits_vram: bool
    fits_ram: bool
    bottleneck: str
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_simulation_path(format: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory = Path("simulations")
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"swlp-sim-{timestamp}.{format}"


def load_scenario(path: Path) -> Scenario:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    data = payload.get("scenario", {})
    return Scenario(
        name=str(data["name"]),
        num_layers=int(data["num_layers"]),
        layer_compute_ms=float(data["layer_compute_ms"]),
        layer_weight_mb=float(data["layer_weight_mb"]),
        window_sizes=[int(value) for value in data["window_sizes"]],
        context_tokens=int(data["context_tokens"]),
        generate_tokens=int(data["generate_tokens"]),
        kv_bytes_per_token=int(data["kv_bytes_per_token"]),
        vram_capacity_mb=float(data["vram_capacity_mb"]),
        ram_capacity_mb=float(data["ram_capacity_mb"]),
        pcie_bandwidth_gbps=float(data["pcie_bandwidth_gbps"]),
        ram_bandwidth_gbps=float(data["ram_bandwidth_gbps"]),
        disk_bandwidth_gbps=float(data.get("disk_bandwidth_gbps", 0.0)),
        disk_staging=bool(data.get("disk_staging", False)),
        overlap=bool(data.get("overlap", True)),
    )


def _bandwidth_to_mb_per_s(gbps: float) -> float:
    return gbps * 1024 / 8


def _transfer_seconds_per_layer(scenario: Scenario) -> float:
    weight_mb = scenario.layer_weight_mb
    pcie_mb_per_s = _bandwidth_to_mb_per_s(scenario.pcie_bandwidth_gbps)
    pcie_seconds = weight_mb / pcie_mb_per_s if pcie_mb_per_s > 0 else float("inf")
    if not scenario.disk_staging:
        return pcie_seconds

    ram_mb_per_s = _bandwidth_to_mb_per_s(scenario.ram_bandwidth_gbps)
    ram_seconds = weight_mb / ram_mb_per_s if ram_mb_per_s > 0 else float("inf")
    disk_mb_per_s = _bandwidth_to_mb_per_s(scenario.disk_bandwidth_gbps)
    disk_seconds = weight_mb / disk_mb_per_s if disk_mb_per_s > 0 else float("inf")
    return disk_seconds + ram_seconds + pcie_seconds


def _compute_seconds_per_layer(scenario: Scenario) -> float:
    return scenario.layer_compute_ms / 1000.0


def _simulate_token_time(
    scenario: Scenario, window_size: int
) -> tuple[float, float, float, float]:
    compute_time = _compute_seconds_per_layer(scenario)
    transfer_time = _transfer_seconds_per_layer(scenario)
    total_layers = scenario.num_layers
    compute_end = 0.0
    transfer_available = 0.0
    compute_starts: list[float] = []

    for layer_index in range(total_layers):
        if scenario.overlap:
            if layer_index < window_size:
                prefetch_start = 0.0
            else:
                prefetch_start = compute_starts[layer_index - window_size]
            transfer_start = max(transfer_available, prefetch_start)
        else:
            transfer_start = max(transfer_available, compute_end)
        transfer_end = transfer_start + transfer_time
        transfer_available = transfer_end
        compute_start = max(compute_end, transfer_end)
        compute_starts.append(compute_start)
        compute_end = compute_start + compute_time

    per_token_seconds = compute_end
    compute_seconds = compute_time * total_layers
    transfer_seconds = transfer_time * total_layers
    if scenario.overlap:
        stall_seconds = max(per_token_seconds - max(compute_seconds, transfer_seconds), 0.0)
    else:
        stall_seconds = 0.0
    return per_token_seconds, compute_seconds, transfer_seconds, stall_seconds


def _estimate_memory_pressure(scenario: Scenario, window_size: int) -> tuple[float, float]:
    kv_mb = (
        scenario.kv_bytes_per_token * (scenario.context_tokens + scenario.generate_tokens)
    ) / BYTES_PER_MB
    vram_peak_mb = window_size * scenario.layer_weight_mb + kv_mb
    if scenario.disk_staging:
        ram_peak_mb = window_size * scenario.layer_weight_mb
    else:
        ram_peak_mb = 0.0
    return vram_peak_mb, ram_peak_mb


def _bottleneck(compute: float, transfer: float, stall: float, fits_vram: bool) -> str:
    if not fits_vram:
        return "memory"
    if stall > compute * 0.2 and transfer >= compute:
        return "transfer"
    return "compute"


def _recommendation(fits_vram: bool, throughput: float, bottleneck: str) -> str:
    if not fits_vram:
        return "not viable (VRAM overflow)"
    if throughput <= 0:
        return "not viable (zero throughput)"
    if bottleneck == "transfer":
        return "viable if overlap or bandwidth improves"
    return "viable"


def simulate_scenario(scenario: Scenario) -> list[SimulationResult]:
    results: list[SimulationResult] = []
    for window_size in scenario.window_sizes:
        per_token_seconds, compute_seconds, transfer_seconds, stall_seconds = _simulate_token_time(
            scenario, window_size
        )
        total_generate_seconds = per_token_seconds * scenario.generate_tokens
        throughput = (
            scenario.generate_tokens / total_generate_seconds
            if total_generate_seconds > 0
            else 0.0
        )
        vram_peak_mb, ram_peak_mb = _estimate_memory_pressure(scenario, window_size)
        fits_vram = vram_peak_mb <= scenario.vram_capacity_mb
        fits_ram = ram_peak_mb <= scenario.ram_capacity_mb
        bottleneck = _bottleneck(compute_seconds, transfer_seconds, stall_seconds, fits_vram)
        recommendation = _recommendation(fits_vram, throughput, bottleneck)
        results.append(
            SimulationResult(
                scenario=scenario.name,
                window_size=window_size,
                per_token_seconds=per_token_seconds,
                time_to_first_token_seconds=per_token_seconds,
                total_generate_seconds=total_generate_seconds,
                throughput_tokens_per_second=throughput,
                compute_seconds_per_token=compute_seconds,
                transfer_seconds_per_token=transfer_seconds,
                stall_seconds_per_token=stall_seconds,
                vram_peak_mb=vram_peak_mb,
                ram_peak_mb=ram_peak_mb,
                fits_vram=fits_vram,
                fits_ram=fits_ram,
                bottleneck=bottleneck,
                recommendation=recommendation,
            )
        )
    return results


def save_simulation(results: list[SimulationResult], output_path: Path, format: str) -> None:
    if format == "json":
        payload = {
            "schema_version": 1,
            "results": [result.to_dict() for result in results],
        }
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return

    if format == "csv":
        rows = [result.to_dict() for result in results]
        if not rows:
            output_path.write_text("", encoding="utf-8")
            return
        fieldnames = list(rows[0].keys())
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return

    raise ValueError(f"Unsupported format: {format}")
