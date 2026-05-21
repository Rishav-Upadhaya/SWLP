from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("cases", [])


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def load_suite_cases(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        return _load_json(path)
    if path.suffix.lower() == ".csv":
        return _load_csv(path)
    raise ValueError(f"Unsupported suite format: {path.suffix}")


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        return float(value)
    return None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return None


def print_suite_report(path: Path) -> None:
    cases = load_suite_cases(path)
    if not cases:
        print("No suite results found.")
        return

    baseline_cases = [case for case in cases if case.get("backend") == "hf"]
    swlp_cases = [case for case in cases if case.get("backend") == "swlp"]
    baseline_success = [case for case in baseline_cases if not case.get("error")]
    swlp_success = [case for case in swlp_cases if not case.get("error")]

    def _avg_throughput(items):
        values = []
        for case in items:
            metrics = case.get("metrics") or {}
            value = _coerce_float(metrics.get("throughput_tokens_per_second"))
            if value is not None:
                values.append(value)
        return sum(values) / len(values) if values else None

    def _avg_metric(items, name):
        values = []
        for case in items:
            metrics = case.get("metrics") or {}
            value = _coerce_float(metrics.get(name))
            if value is not None:
                values.append(value)
        return sum(values) / len(values) if values else None

    baseline_tp = _avg_throughput(baseline_success)
    swlp_tp = _avg_throughput(swlp_success)
    baseline_vram = _avg_metric(baseline_success, "vram_peak_bytes")
    swlp_vram = _avg_metric(swlp_success, "vram_peak_bytes")
    baseline_ram = _avg_metric(baseline_success, "ram_peak_bytes")
    swlp_ram = _avg_metric(swlp_success, "ram_peak_bytes")

    best_case = None
    best_tp = -1.0
    for case in swlp_success:
        metrics = case.get("metrics") or {}
        tp = _coerce_float(metrics.get("throughput_tokens_per_second")) or 0.0
        quality = _coerce_float(case.get("quality_overlap")) or 0.0
        if quality < 0.5:
            continue
        if tp > best_tp:
            best_tp = tp
            best_case = case

    print(f"Suite report for {path}")
    print(f"Baseline runs: {len(baseline_cases)} | SWLP runs: {len(swlp_cases)}")
    print(f"Baseline avg throughput: {baseline_tp or 0.0:.2f} tok/s")
    print(f"SWLP avg throughput: {swlp_tp or 0.0:.2f} tok/s")
    if baseline_vram or swlp_vram:
        baseline_vram_mb = (baseline_vram or 0.0) / (1024 * 1024)
        swlp_vram_mb = (swlp_vram or 0.0) / (1024 * 1024)
        print(f"Avg VRAM: baseline={baseline_vram_mb:.1f} MB | swlp={swlp_vram_mb:.1f} MB")
    if baseline_ram or swlp_ram:
        baseline_ram_mb = (baseline_ram or 0.0) / (1024 * 1024)
        swlp_ram_mb = (swlp_ram or 0.0) / (1024 * 1024)
        print(f"Avg RAM: baseline={baseline_ram_mb:.1f} MB | swlp={swlp_ram_mb:.1f} MB")
    if best_case:
        print(
            "Best SWLP config: window={window} prefetch_depth={depth} "
            "prefetch={prefetch} double_buffer={double_buffer} kv_budget_mb={kv_budget}".format(
                window=best_case.get("window_size"),
                depth=best_case.get("prefetch_depth"),
                prefetch=_coerce_bool(best_case.get("prefetch_enabled")),
                double_buffer=_coerce_bool(best_case.get("double_buffer_enabled")),
                kv_budget=best_case.get("kv_memory_budget_mb"),
            )
        )
        print(
            "Best SWLP throughput: {tp:.2f} tok/s | quality overlap: {q:.2f}".format(
                tp=best_tp,
                q=_coerce_float(best_case.get("quality_overlap")) or 0.0,
            )
        )
    else:
        print("No viable SWLP configuration met the quality threshold.")
