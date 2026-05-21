from __future__ import annotations

import csv
import json
import tomllib
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from ..config import AppConfig
from ..runner.base import execute_baseline


@dataclass(slots=True)
class SuiteConfig:
    name: str
    prompts: list[str]
    context_lengths: list[int]
    runs_per_case: int
    window_sizes: list[int]
    prefetch_depths: list[int]
    prefetch_enabled: list[bool]
    double_buffer_enabled: list[bool]
    kv_memory_budget_mb: list[int]
    kv_compression: list[bool]
    kv_tiering: list[bool]


@dataclass(slots=True)
class CaseResult:
    suite: str
    prompt: str
    context_length: int
    backend: str
    window_size: int | None
    prefetch_depth: int | None
    prefetch_enabled: bool | None
    double_buffer_enabled: bool | None
    kv_memory_budget_mb: int | None
    kv_compression: bool | None
    kv_tiering: bool | None
    kv_estimated_bytes_per_token: int | None
    metrics: dict[str, Any] | None
    completion: str | None
    quality_overlap: float | None
    error: str | None


@dataclass(slots=True)
class SuiteResult:
    suite: str
    created_at: str
    cases: list[CaseResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "created_at": self.created_at,
            "cases": [asdict(case) for case in self.cases],
        }


def load_suite(path: Path) -> SuiteConfig:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    data = payload.get("suite", {})
    return SuiteConfig(
        name=str(data["name"]),
        prompts=[str(item) for item in data["prompts"]],
        context_lengths=[int(value) for value in data["context_lengths"]],
        runs_per_case=int(data.get("runs_per_case", 1)),
        window_sizes=[int(value) for value in data.get("window_sizes", [1, 2, 4])],
        prefetch_depths=[int(value) for value in data.get("prefetch_depths", [1])],
        prefetch_enabled=[bool(value) for value in data.get("prefetch_enabled", [True])],
        double_buffer_enabled=[bool(value) for value in data.get("double_buffer_enabled", [True])],
        kv_memory_budget_mb=[int(value) for value in data.get("kv_memory_budget_mb", [512])],
        kv_compression=[bool(value) for value in data.get("kv_compression", [False])],
        kv_tiering=[bool(value) for value in data.get("kv_tiering", [False])],
    )


def default_suite_path(format: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory = Path("benchmarks")
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"suite-{timestamp}.{format}"


def _token_overlap(a: str, b: str) -> float:
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def _prepare_prompt(tokenizer, prompt: str, target_tokens: int) -> str:
    base_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(base_ids) >= target_tokens:
        return tokenizer.decode(base_ids[:target_tokens], skip_special_tokens=True)
    filler_ids = tokenizer.encode(" lorem", add_special_tokens=False)
    if not filler_ids:
        return prompt
    ids = list(base_ids)
    while len(ids) < target_tokens:
        remaining = target_tokens - len(ids)
        ids.extend(filler_ids[:remaining])
    return tokenizer.decode(ids, skip_special_tokens=True)


def _estimate_kv_bytes_per_token(model_config) -> int:
    n_layer = int(getattr(model_config, "n_layer", 0))
    n_embd = int(getattr(model_config, "n_embd", 0))
    dtype = getattr(model_config, "torch_dtype", None) or torch.float32
    bytes_per = torch.tensor([], dtype=dtype).element_size()
    return 2 * n_layer * n_embd * bytes_per


def run_suite(config: AppConfig, suite: SuiteConfig) -> SuiteResult:
    from transformers import AutoConfig, AutoTokenizer

    model_source = config.model.local_model_path or config.model.model_id
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        cache_dir=str(config.cache.cache_dir),
        trust_remote_code=config.model.trust_remote_code,
        revision=config.model.revision or None,
    )
    model_config = AutoConfig.from_pretrained(
        model_source,
        cache_dir=str(config.cache.cache_dir),
        trust_remote_code=config.model.trust_remote_code,
        revision=config.model.revision or None,
    )
    kv_bytes_per_token = _estimate_kv_bytes_per_token(model_config)

    results = SuiteResult(
        suite=suite.name,
        created_at=datetime.now(UTC).isoformat(),
    )

    for prompt in suite.prompts:
        for context_length in suite.context_lengths:
            prepared_prompt = _prepare_prompt(tokenizer, prompt, context_length)
            baseline_config = replace(
                config,
                runtime=replace(config.runtime, backend="hf", profile=True),
            )
            baseline_result = None
            baseline_error = None
            try:
                baseline_result = execute_baseline(baseline_config, prepared_prompt)
            except Exception as exc:
                baseline_error = str(exc)

            for _ in range(suite.runs_per_case):
                results.cases.append(
                    CaseResult(
                        suite=suite.name,
                        prompt=prepared_prompt,
                        context_length=context_length,
                        backend="hf",
                        window_size=None,
                        prefetch_depth=None,
                        prefetch_enabled=None,
                        double_buffer_enabled=None,
                        kv_memory_budget_mb=None,
                        kv_compression=None,
                        kv_tiering=None,
                        kv_estimated_bytes_per_token=kv_bytes_per_token,
                        metrics=baseline_result.metrics.to_dict() if baseline_result else None,
                        completion=baseline_result.completion if baseline_result else None,
                        quality_overlap=None,
                        error=baseline_error,
                    )
                )

            for window_size in suite.window_sizes:
                for prefetch_depth in suite.prefetch_depths:
                    for prefetch_enabled in suite.prefetch_enabled:
                        for double_buffer_enabled in suite.double_buffer_enabled:
                            for kv_budget in suite.kv_memory_budget_mb:
                                for kv_compression in suite.kv_compression:
                                    for kv_tiering in suite.kv_tiering:
                                        swlp_config = replace(
                                            config,
                                            runtime=replace(
                                                config.runtime,
                                                backend="swlp",
                                                profile=True,
                                                swlp_fallback_to_baseline=False,
                                                swlp_window_size=window_size,
                                                swlp_prefetch_depth=prefetch_depth,
                                                swlp_prefetch=prefetch_enabled,
                                                swlp_double_buffer=double_buffer_enabled,
                                                kv_memory_budget_mb=kv_budget,
                                                kv_compression=kv_compression,
                                                kv_tiering=kv_tiering,
                                            ),
                                        )
                                        swlp_result = None
                                        swlp_error = None
                                        try:
                                            swlp_result = execute_baseline(
                                                swlp_config, prepared_prompt
                                            )
                                        except Exception as exc:
                                            swlp_error = str(exc)
                                        quality = None
                                        if baseline_result and swlp_result:
                                            quality = _token_overlap(
                                                baseline_result.completion, swlp_result.completion
                                            )
                                        results.cases.append(
                                            CaseResult(
                                                suite=suite.name,
                                                prompt=prepared_prompt,
                                                context_length=context_length,
                                                backend="swlp",
                                                window_size=window_size,
                                                prefetch_depth=prefetch_depth,
                                                prefetch_enabled=prefetch_enabled,
                                                double_buffer_enabled=double_buffer_enabled,
                                                kv_memory_budget_mb=kv_budget,
                                                kv_compression=kv_compression,
                                                kv_tiering=kv_tiering,
                                                kv_estimated_bytes_per_token=kv_bytes_per_token,
                                                metrics=swlp_result.metrics.to_dict()
                                                if swlp_result
                                                else None,
                                                completion=swlp_result.completion
                                                if swlp_result
                                                else None,
                                                quality_overlap=quality,
                                                error=swlp_error,
                                            )
                                        )

    return results


def save_suite(result: SuiteResult, output_path: Path, format: str) -> None:
    if format == "json":
        payload = result.to_dict()
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return

    if format == "csv":
        rows = [asdict(case) for case in result.cases]
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
