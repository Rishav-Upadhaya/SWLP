from __future__ import annotations

import csv
import json
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..benchmark.run import METRIC_FIELDS


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}s"


def _format_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f} tok/s"


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / (1024 * 1024):.1f} MB"


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def _median(values: Iterable[float]) -> float | None:
    values = sorted(values)
    if not values:
        return None
    return values[len(values) // 2]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {"runs": payload}
    return payload


def _load_csv(path: Path) -> dict[str, Any]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {"runs": [dict(row) for row in reader]}


def load_benchmark(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        return _load_json(path)
    if path.suffix.lower() == ".csv":
        return _load_csv(path)
    raise ValueError(f"Unsupported benchmark format: {path.suffix}")


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        return float(value)
    return None


def _extract_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for record in records:
        metrics.append(record.get("metrics", record))
    return metrics


def _bound_label(preprocess: float | None, generate: float | None) -> str:
    if preprocess is None or generate is None:
        return "n/a"
    if preprocess > generate:
        return "transfer-bound"
    return "compute-bound"


def _stats(values: Iterable[float]) -> dict[str, float] | None:
    values = [float(value) for value in values]
    if not values:
        return None
    ordered = sorted(values)
    mean = sum(ordered) / len(ordered)
    median = ordered[len(ordered) // 2]
    std = statistics.pstdev(ordered)
    return {
        "mean": mean,
        "median": median,
        "min": ordered[0],
        "max": ordered[-1],
        "std": std,
    }


def _format_stat(stat: dict[str, float] | None) -> str:
    if not stat:
        return "n/a"
    return "{mean}/{median}/{min}/{max}/{std}".format(
        mean=_format_seconds(stat["mean"]),
        median=_format_seconds(stat["median"]),
        min=_format_seconds(stat["min"]),
        max=_format_seconds(stat["max"]),
        std=_format_seconds(stat["std"]),
    )


def _format_rate_stat(stat: dict[str, float] | None) -> str:
    if not stat:
        return "n/a"
    return "{mean}/{median}/{min}/{max}/{std}".format(
        mean=_format_rate(stat["mean"]),
        median=_format_rate(stat["median"]),
        min=_format_rate(stat["min"]),
        max=_format_rate(stat["max"]),
        std=_format_rate(stat["std"]),
    )


def print_report(path: Path) -> None:
    payload = load_benchmark(path)
    records = payload.get("runs", [])
    if not records:
        print("No benchmark runs found.")
        return
    metadata = payload.get("metadata", {})
    summary = payload.get("summary")

    metrics_records = _extract_metrics(records)
    load_values = [_coerce_float(m.get("load_seconds")) for m in metrics_records]
    preprocess_values = [_coerce_float(m.get("preprocess_seconds")) for m in metrics_records]
    generate_values = [_coerce_float(m.get("generate_seconds")) for m in metrics_records]
    ttft_values = [_coerce_float(m.get("time_to_first_token_seconds")) for m in metrics_records]
    throughput_values = [
        _coerce_float(m.get("throughput_tokens_per_second")) for m in metrics_records
    ]
    forward_values = [_coerce_float(m.get("forward_seconds")) for m in metrics_records]
    vram_values = [_coerce_float(m.get("vram_peak_bytes")) for m in metrics_records]
    ram_values = [_coerce_float(m.get("ram_peak_bytes")) for m in metrics_records]

    load_mean = _mean(v for v in load_values if v is not None)
    preprocess_mean = _mean(v for v in preprocess_values if v is not None)
    generate_mean = _mean(v for v in generate_values if v is not None)
    ttft_mean = _mean(v for v in ttft_values if v is not None)
    forward_mean = _mean(v for v in forward_values if v is not None)
    throughput_mean = _mean(v for v in throughput_values if v is not None)
    vram_peak = max((int(v) for v in vram_values if v is not None), default=None)
    ram_peak = max((int(v) for v in ram_values if v is not None), default=None)

    per_token_latencies: list[float] = []
    for metrics in metrics_records:
        values = metrics.get("per_token_latency_seconds")
        if isinstance(values, list):
            per_token_latencies.extend(float(value) for value in values)
    per_token_avg = _mean(per_token_latencies)
    per_token_p50 = _median(per_token_latencies)

    representative = metrics_records[0]
    model_id = representative.get("model_id", "n/a")
    backend = representative.get("backend", "n/a")
    device = representative.get("device", "n/a")

    print(f"Benchmark report for {path}")
    print(f"Runs: {len(records)} | Model: {model_id} | Backend: {backend} | Device: {device}")
    if metadata:
        prompt_sets = metadata.get("prompt_sets")
        warmup_runs = metadata.get("warmup_runs")
        cache_status = metadata.get("cache_status", {})
        if prompt_sets:
            print(f"Prompt sets: {', '.join(prompt_sets.keys())}")
        if warmup_runs is not None:
            print(f"Warm-up runs: {warmup_runs}")
        if cache_status:
            download = cache_status.get("download_seconds")
            verified = cache_status.get("verified")
            if verified is not None:
                print(f"Cache verified: {verified}")
            if isinstance(download, (int, float)) and download > 0:
                print(f"Download time (cold): {_format_seconds(float(download))}")
    print(
        f"Load: {_format_seconds(load_mean)}"
        f" | Preprocess: {_format_seconds(preprocess_mean)}"
        f" | Forward: {_format_seconds(forward_mean)}"
        f" | Generate: {_format_seconds(generate_mean)}"
    )
    print(
        f"TTFT: {_format_seconds(ttft_mean)}"
        f" | Per-token avg/p50:"
        f" {_format_seconds(per_token_avg)}/{_format_seconds(per_token_p50)}"
        f" | Throughput: {_format_rate(throughput_mean)}"
    )
    print(f"Peak VRAM: {_format_bytes(vram_peak)} | Peak RAM: {_format_bytes(ram_peak)}")
    print(f"Bound: {_bound_label(preprocess_mean, generate_mean)}")
    print("Metrics tracked: " + ", ".join(METRIC_FIELDS))

    if summary:
        overall = summary.get("overall", {})
        load_stat = overall.get("load_seconds")
        generate_stat = overall.get("generate_seconds")
        ttft_stat = overall.get("time_to_first_token_seconds")
        throughput_stat = overall.get("throughput_tokens_per_second")
    else:
        load_stat = _stats(v for v in load_values if v is not None)
        generate_stat = _stats(v for v in generate_values if v is not None)
        ttft_stat = _stats(v for v in ttft_values if v is not None)
        throughput_stat = _stats(v for v in throughput_values if v is not None)

    print(
        f"Stats (mean/median/min/max/std):"
        f" Load={_format_stat(load_stat)}"
        f" | Generate={_format_stat(generate_stat)}"
        f" | TTFT={_format_stat(ttft_stat)}"
        f" | Throughput={_format_rate_stat(throughput_stat)}"
    )
