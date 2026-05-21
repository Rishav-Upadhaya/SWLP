from __future__ import annotations

import csv
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..runner.base import execute_baseline

LOGGER = logging.getLogger(__name__)

METRIC_FIELDS = [
    "load_seconds",
    "preprocess_seconds",
    "forward_seconds",
    "generate_seconds",
    "total_seconds",
    "time_to_first_token_seconds",
    "per_token_latency_seconds",
    "per_token_latency_seconds_avg",
    "per_token_latency_seconds_p50",
    "throughput_tokens_per_second",
    "input_tokens",
    "output_tokens",
    "generated_tokens",
    "vram_peak_bytes",
    "ram_peak_bytes",
    "kv_cache_entries",
    "kv_cache_device_bytes",
    "kv_cache_host_bytes",
    "kv_cache_compressed_bytes",
    "kv_cache_total_bytes",
    "kv_cache_peak_device_bytes",
    "kv_cache_peak_host_bytes",
    "kv_cache_peak_total_bytes",
    "kv_cache_compressions",
    "kv_cache_decompressions",
    "kv_cache_offloads",
    "kv_cache_moves_to_device",
    "kv_cache_budget_bytes",
    "kv_cache_device_budget_bytes",
    "kv_cache_budget_violations",
]

PROMPT_SETS: dict[str, list[str]] = {
    "short": [
        "Summarize SWLP in one sentence.",
        "List two benefits of running models locally.",
        "Write a single-line tip for measuring latency.",
    ],
    "medium": [
        "Explain how a warm-up run affects benchmark stability in a short paragraph.",
        "Draft a concise checklist for verifying a model cache before timing.",
        "Summarize the difference between load time and generation time in 3-4 sentences.",
    ],
    "long": [
        (
            "Write a detailed, multi-paragraph explanation of why separating download time from "
            "model load time matters for reproducible benchmarks. "
            "Include a short checklist at the end."
        ),
        (
            "Provide a structured outline for a performance report"
            " that compares two model runtimes."
            " Include sections for setup, methodology, results, and risks."
        ),
        (
            "Describe a hypothetical benchmarking experiment for a local LLM runtime."
            " Cover the prompt set design, repeated runs, aggregation metrics,"
            " and variance handling."
        ),
    ],
}

SUMMARY_FIELDS = [
    "load_seconds",
    "preprocess_seconds",
    "forward_seconds",
    "generate_seconds",
    "total_seconds",
    "time_to_first_token_seconds",
    "per_token_latency_seconds_avg",
    "per_token_latency_seconds_p50",
    "throughput_tokens_per_second",
    "input_tokens",
    "output_tokens",
    "generated_tokens",
    "vram_peak_bytes",
    "ram_peak_bytes",
    "kv_cache_entries",
    "kv_cache_device_bytes",
    "kv_cache_host_bytes",
    "kv_cache_compressed_bytes",
    "kv_cache_total_bytes",
    "kv_cache_peak_device_bytes",
    "kv_cache_peak_host_bytes",
    "kv_cache_peak_total_bytes",
    "kv_cache_compressions",
    "kv_cache_decompressions",
    "kv_cache_offloads",
    "kv_cache_moves_to_device",
    "kv_cache_budget_bytes",
    "kv_cache_device_budget_bytes",
    "kv_cache_budget_violations",
]


@dataclass(slots=True)
class BenchmarkRecord:
    run_id: int
    timestamp: str
    prompt: str
    prompt_set: str
    config: dict[str, Any]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_benchmark_path(format: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory = Path("benchmarks")
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"baseline-{timestamp}.{format}"


@dataclass(slots=True)
class BenchmarkRun:
    records: list[BenchmarkRecord]
    metadata: dict[str, Any]
    summary: dict[str, Any]


def _resolve_prompt_sets(
    config: AppConfig, prompt: str | None, prompt_set: str | None
) -> list[tuple[str, list[str]]]:
    if prompt:
        return [("custom", [prompt])]
    if prompt_set is None:
        return [("default", [config.generation.prompt])]
    if prompt_set == "all":
        return [(name, prompts) for name, prompts in PROMPT_SETS.items()]
    if prompt_set in PROMPT_SETS:
        return [(prompt_set, PROMPT_SETS[prompt_set])]
    raise ValueError(f"Unknown prompt set: {prompt_set}")


def _verify_model_cache(config: AppConfig) -> dict[str, Any]:
    if config.runtime.backend.lower() == "mock":
        return {
            "verified": True,
            "skipped": True,
            "cache_dir": str(config.cache.cache_dir),
            "model_source": "mock",
        }
    model_source = config.model.local_model_path or config.model.model_id
    cache_dir = str(config.cache.cache_dir)
    if config.model.local_model_path:
        local_path = Path(model_source)
        if not local_path.exists():
            raise FileNotFoundError(f"Local model path not found: {local_path}")
        return {
            "verified": True,
            "download_seconds": 0.0,
            "cache_dir": cache_dir,
            "model_source": str(model_source),
            "local_path": True,
        }

    from huggingface_hub import snapshot_download

    try:
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError:
        from huggingface_hub import LocalEntryNotFoundError

    revision = config.model.revision or None
    verify_started = time.perf_counter()
    try:
        snapshot_download(
            repo_id=str(model_source),
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=True,
        )
        verify_seconds = time.perf_counter() - verify_started
        return {
            "verified": True,
            "download_seconds": 0.0,
            "verify_seconds": verify_seconds,
            "cache_dir": cache_dir,
            "model_source": str(model_source),
            "local_path": False,
        }
    except LocalEntryNotFoundError:
        if config.cache.offline:
            raise RuntimeError(
                "Model cache is incomplete and offline mode is enabled. "
                "Disable offline mode or pre-download the model."
            ) from None
        download_started = time.perf_counter()
        snapshot_download(
            repo_id=str(model_source),
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=False,
        )
        download_seconds = time.perf_counter() - download_started
        return {
            "verified": True,
            "download_seconds": download_seconds,
            "cache_dir": cache_dir,
            "model_source": str(model_source),
            "local_path": False,
            "cache_downloaded": True,
        }


def _stats(values: list[float]) -> dict[str, float] | None:
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
        "count": len(ordered),
    }


def _summarize(records: list[BenchmarkRecord]) -> dict[str, Any]:
    summary: dict[str, Any] = {"overall": {}, "by_prompt_set": {}}
    flattened = [_flatten_metrics(record.metrics) for record in records]

    def _build(rows: list[dict[str, Any]]) -> dict[str, Any]:
        stats: dict[str, Any] = {}
        for field in SUMMARY_FIELDS:
            values = []
            for row in rows:
                value = row.get(field)
                if value is not None:
                    values.append(float(value))
            stats[field] = _stats(values)
        return stats

    summary["overall"] = _build(flattened)
    prompts: dict[str, list[dict[str, Any]]] = {}
    for record, metrics in zip(records, flattened, strict=False):
        prompts.setdefault(record.prompt_set, []).append(metrics)
    summary["by_prompt_set"] = {key: _build(rows) for key, rows in prompts.items()}
    summary["runs"] = len(records)
    return summary


def _run_batched(
    config: AppConfig, prompts: list[str], batch_size: int, runs: int, warmup_runs: int
):
    """Stream a prompt set through ``SWLPRunner.run_batch`` in chunks of
    ``batch_size`` (Phase 10 — one disk sweep amortized across the batch).

    Yields ``RunResult`` objects; raises if the backend has no ``run_batch``.
    """
    from ..runner.base import build_runner

    runner = build_runner(config)
    if not hasattr(runner, "run_batch"):
        raise ValueError(
            f"--batch-size > 1 requires the swlp streaming backend; "
            f"got backend={config.runtime.backend!r}"
        )
    chunks = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]
    results = []
    for chunk in chunks:
        for _ in range(max(warmup_runs, 0)):
            runner.run_batch(chunk)
        for _ in range(runs):
            results.extend(runner.run_batch(chunk, profile=True))
    return results


def run_benchmark(
    config: AppConfig,
    prompt: str | None,
    runs: int,
    prompt_set: str | None,
    warmup_runs: int,
    batch_size: int = 1,
) -> BenchmarkRun:
    records: list[BenchmarkRecord] = []
    cache_status = _verify_model_cache(config)
    LOGGER.info("benchmark_cache_verified", extra=cache_status)
    prompt_sets = _resolve_prompt_sets(config, prompt, prompt_set)
    run_id = 1

    for set_name, prompts in prompt_sets:
        if batch_size > 1:
            for result in _run_batched(config, prompts, batch_size, runs, warmup_runs):
                records.append(
                    BenchmarkRecord(
                        run_id=run_id,
                        timestamp=datetime.now(UTC).isoformat(),
                        prompt=result.prompt,
                        prompt_set=set_name,
                        config=config.to_dict(),
                        metrics=result.metrics.to_dict(),
                    )
                )
                run_id += 1
            continue
        for prompt_text in prompts:
            for _ in range(max(warmup_runs, 0)):
                execute_baseline(config, prompt_text)
            for _ in range(runs):
                result = execute_baseline(config, prompt_text)
                record = BenchmarkRecord(
                    run_id=run_id,
                    timestamp=datetime.now(UTC).isoformat(),
                    prompt=result.prompt,
                    prompt_set=set_name,
                    config=config.to_dict(),
                    metrics=result.metrics.to_dict(),
                )
                records.append(record)
                run_id += 1

    summary = _summarize(records)
    metadata = {
        "prompt_sets": {name: prompts for name, prompts in prompt_sets},
        "warmup_runs": warmup_runs,
        "batch_size": batch_size,
        "cache_status": cache_status,
    }
    return BenchmarkRun(records=records, metadata=metadata, summary=summary)


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    flattened = dict(metrics)
    per_token = metrics.get("per_token_latency_seconds")
    if per_token:
        values = [float(value) for value in per_token]
        values.sort()
        flattened["per_token_latency_seconds_avg"] = sum(values) / len(values)
        flattened["per_token_latency_seconds_p50"] = values[len(values) // 2]
    flattened.pop("per_token_latency_seconds", None)
    return flattened


def save_benchmark(
    records: list[BenchmarkRecord],
    output_path: Path,
    format: str,
    metadata: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
) -> None:
    if format == "json":
        payload = {
            "schema_version": 2,
            "metadata": metadata or {},
            "summary": summary or {},
            "runs": [record.to_dict() for record in records],
        }
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return

    if format == "csv":
        rows: list[dict[str, Any]] = []
        for record in records:
            row = {
                "run_id": record.run_id,
                "timestamp": record.timestamp,
                "prompt": record.prompt,
                "prompt_set": record.prompt_set,
            }
            row.update(_flatten_metrics(record.metrics))
            rows.append(row)
        if not rows:
            output_path.write_text("", encoding="utf-8")
            return
        fieldnames = list(rows[0].keys())
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        if summary or metadata:
            summary_path = output_path.with_name(f"{output_path.stem}.summary.json")
            payload = {
                "schema_version": 2,
                "metadata": metadata or {},
                "summary": summary or {},
            }
            summary_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )
        return

    raise ValueError(f"Unsupported format: {format}")
