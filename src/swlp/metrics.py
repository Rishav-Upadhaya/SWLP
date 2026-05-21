from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class RunResult:
    """Canonical output of every inference runner."""
    prompt: str
    completion: str
    metrics: RunMetrics


@dataclass(slots=True)
class RunMetrics:
    model_id: str
    backend: str
    device: str
    load_seconds: float
    input_tokens: int
    output_tokens: int
    preprocess_seconds: float | None = None
    forward_seconds: float | None = None
    generate_seconds: float | None = None
    total_seconds: float | None = None
    # Phase 15: prefill_seconds = time for the forward sweep over all input tokens
    # (from generation_start to first logit).  time_to_first_token_seconds is the
    # user-perceived TTFT = prefill_seconds + argmax_seconds.
    prefill_seconds: float | None = None
    time_to_first_token_seconds: float | None = None
    per_token_latency_seconds: Sequence[float] | None = field(default=None)
    throughput_tokens_per_second: float | None = None
    generated_tokens: int | None = None
    vram_peak_bytes: int | None = None
    ram_peak_bytes: int | None = None
    kv_cache_entries: int | None = None
    kv_cache_device_bytes: int | None = None
    kv_cache_host_bytes: int | None = None
    kv_cache_compressed_bytes: int | None = None
    kv_cache_total_bytes: int | None = None
    kv_cache_peak_device_bytes: int | None = None
    kv_cache_peak_host_bytes: int | None = None
    kv_cache_peak_total_bytes: int | None = None
    kv_cache_compressions: int | None = None
    kv_cache_decompressions: int | None = None
    kv_cache_offloads: int | None = None
    kv_cache_moves_to_device: int | None = None
    kv_cache_budget_bytes: int | None = None
    kv_cache_device_budget_bytes: int | None = None
    kv_cache_budget_violations: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
