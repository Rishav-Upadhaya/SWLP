from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("results", [])


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def load_results(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        return _load_json(path)
    if path.suffix.lower() == ".csv":
        return _load_csv(path)
    raise ValueError(f"Unsupported simulation format: {path.suffix}")


def _format_seconds(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}s"


def _format_rate(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f} tok/s"


def _format_mb(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1f} MB"


def print_simulation_report(path: Path) -> None:
    results = load_results(path)
    if not results:
        print("No simulation results found.")
        return

    scenario = results[0].get("scenario", "n/a")
    print(f"Simulation report for {path}")
    print(f"Scenario: {scenario} | Windows: {len(results)}")

    for result in results:
        window = result.get("window_size", "n/a")
        print(
            "W={window} | per-token={per_token} | throughput={throughput} | "
            "VRAM={vram} | RAM={ram} | bottleneck={bottleneck} | rec={rec}".format(
                window=window,
                per_token=_format_seconds(result.get("per_token_seconds")),
                throughput=_format_rate(result.get("throughput_tokens_per_second")),
                vram=_format_mb(result.get("vram_peak_mb")),
                ram=_format_mb(result.get("ram_peak_mb")),
                bottleneck=result.get("bottleneck", "n/a"),
                rec=result.get("recommendation", "n/a"),
            )
        )
