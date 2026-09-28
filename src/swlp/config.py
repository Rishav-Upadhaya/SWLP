from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

_TRUE = {"1", "true", "yes", "on"}

# Env vars that do not follow the SWLP_<FIELD> rule (a leading "swlp_" is dropped).
_ENV_NAMES = {
    "local_model_path": "SWLP_MODEL_PATH",
    "offline": "SWLP_CACHE_OFFLINE",
    "swlp_fallback_to_baseline": "SWLP_FALLBACK_BASELINE",
    "kv_memory_budget_mb": "SWLP_KV_BUDGET_MB",
    "swlp_activation_cache_max_entries": "SWLP_ACTIVATION_CACHE_MAX",
    "mlx_prefill_step_size": "SWLP_MLX_PREFILL_STEP",
}


def env_name(field_name: str) -> str:
    return _ENV_NAMES.get(field_name, "SWLP_" + field_name.removeprefix("swlp_").upper())


def _coerce(kind: str, value: Any) -> Any:
    if kind == "Path":
        return Path(value).expanduser().resolve()
    if kind == "Path | None":
        return Path(value).expanduser().resolve() if value else None
    if kind == "str | None":
        return value or None
    if kind == "bool" and isinstance(value, str):
        return value.strip().lower() in _TRUE
    return {"bool": bool, "int": int, "float": float, "str": str}[kind](value)


def _section(cls: type[Any], values: dict[str, Any]) -> Any:
    """Build one config dataclass: TOML/default value, overridden by its env var.

    An empty env var counts as unset, except for str/bool fields where "" is a value.
    """
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        raw = os.getenv(env_name(f.name))
        use_env = raw is not None and (raw != "" or f.type in ("str", "bool"))
        kwargs[f.name] = _coerce(f.type, raw if use_env else values[f.name])
    return cls(**kwargs)


def _merge_dict(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(slots=True)
class ModelConfig:
    model_id: str = "sshleifer/tiny-gpt2"
    local_model_path: Path | None = None
    trust_remote_code: bool = False
    revision: str | None = None


@dataclass(slots=True)
class CacheConfig:
    cache_dir: Path = Path(".cache/hf")
    offline: bool = False


@dataclass(slots=True)
class GenerationConfig:
    max_new_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    repetition_penalty: float = 1.0
    seed: int = 42
    prompt: str = "Write a short, friendly welcome message for a local LLM runtime."


@dataclass(slots=True)
class RuntimeConfig:
    device: str = "auto"
    dtype: str = "auto"
    backend: str = "hf"
    allow_mock_fallback: bool = True
    log_level: str = "INFO"
    json_logs: bool = True
    profile: bool = False
    swlp_window_size: int = 2
    swlp_prefetch_depth: int = 2
    swlp_prefetch: bool = True
    swlp_double_buffer: bool = True
    swlp_pin_memory: bool = True
    swlp_fallback_to_baseline: bool = True
    # Re-raise hot-path failures instead of logging and degrading. Off by
    # default (a degraded answer beats no answer); on for clean benchmarks.
    swlp_strict: bool = False
    # KV cache control
    kv_memory_budget_mb: int = 512
    kv_compression: bool = False
    kv_compression_level: int = 0
    kv_tiering: bool = False
    # Phase 12: disk spill dir for KV overflow (None = no disk spill)
    kv_disk_dir: Path | None = None
    # Phase 16: sliding-window KV budget — keep only the most recent N token
    # positions of KV (0 = unbounded = Phase 1–15 behaviour).
    kv_window: int = 0
    # Layer sharding (Phase 1): path to pre-sharded per-layer files
    shard_dir: Path | None = None
    # Adaptive residency (Phase 4): "auto" | "off" | "<int>" layers
    swlp_residency: str = "auto"
    # Direct I/O policy (Phase 20): "auto" | "on" | "off".  "auto" bypasses the
    # OS page cache (F_NOCACHE) only when the model cannot fit in available
    # RAM; models that fit get page-cache residency for free.
    swlp_direct_io: str = "auto"
    # MoE expert streaming (Phase 25): RAM budget for the global expert slot
    # cache in MB. 0 = auto (a quarter of available RAM, capped at 4 GB).
    swlp_expert_cache_mb: int = 0
    # Expert prefetch mode: "predictive" (routing-history, default) | "lru" |
    # "off" (cache still works; nothing is prefetched).
    swlp_expert_prefetch: str = "predictive"
    # Multi-volume striping (Phase 26): comma-separated extra directories
    # holding layer shards. Layers are assigned round-robin across volumes so
    # parallel reads aggregate SSD bandwidth. Empty = single shard_dir.
    swlp_shard_volumes: str = ""
    # Chunked prefill (Phase 26): feed the prompt through the sweep in chunks
    # of this many tokens (0 = one full-prompt sweep). Bounds activation RAM
    # on very long prompts; lossless — causal attention over prefix KV.
    swlp_prefill_chunk: int = 0
    # Speculative decoding (Phase 5): prompt-lookup n-gram drafting
    swlp_spec_ngram: int = 3
    swlp_spec_max_draft: int = 16
    # Phase 21: optional resident draft model for SWLP speculative decoding
    # (alias or HF id). Must share the target model's tokenizer exactly.
    # Empty = n-gram prompt-lookup drafting (Phase 5 behaviour).
    swlp_draft_model: str = ""
    # MLX backend (Phase 8): native quantized compute on Apple Silicon.
    # "bf16" (lossless) | "int8" (near-lossless, default) | "int4" (fast tier)
    mlx_quant: str = "int8"
    # ── Apple Silicon tuning (see runner/mlx_tune.py for the rationale) ──
    # Raise the Metal wired-memory ceiling: auto | off | <MB>. macOS defaults
    # to ~70-75% of RAM, which strands several GB on a 16 GB machine.
    mlx_wired_limit: str = "auto"
    # KV-cache quantization: 0 = off, 4 or 8 bits. On unified memory 4-bit KV
    # is measured as FASTER than fp16 (bandwidth saved > kernel overhead),
    # so this is a throughput setting as much as a memory one.
    mlx_kv_bits: int = 0
    mlx_kv_group_size: int = 64
    # Keep the first N tokens of KV exact; quantize only beyond that.
    mlx_quantized_kv_start: int = 512
    # Draft tokens per speculative step. 4-6 is the measured sweet spot;
    # longer drafts raise mid-sequence rejection and waste compute.
    mlx_num_draft_tokens: int = 4
    # Prompt-chunk size during prefill. Bigger = lower TTFT, more transient RAM.
    mlx_prefill_step_size: int = 2048
    # Reuse the KV of the shared prefix across chat turns.
    mlx_prompt_cache: bool = True
    # Optional draft model for MLX speculative decoding (alias or HF id).
    # Must share the same tokenizer as the main model. Empty = disabled.
    mlx_draft_model: str = ""
    # Phase 18: INT4 KV quantization — off by default (lossy).
    # "none" = lossless (default) | "int4" = ~4× smaller KV, ~0.5–1% ppl cost.
    # Must be explicitly opt-in; always labelled in reports.
    kv_quant: str = "none"
    # ── Phase 23: Quality-neutral speedups ──────────────────────────────────
    # Activation cache: cache (prompt_prefix_hash → layer_outputs) for prompt
    # prefix reuse across turns in a chat session. Zero quality impact.
    swlp_activation_cache: bool = True
    swlp_activation_cache_max_entries: int = 16
    # Pre-allocated generate buffer: avoid torch.cat overhead per token.
    swlp_prealloc_buffer: bool = True
    # ── Phase 23: Opt-in quality tradeoffs (Phase 3 style) ─────────────────
    # Early exit: skip remaining layers when next-token entropy < threshold.
    # "off" = disabled (default) | float 0.0-1.0 = entropy threshold.
    # Lower threshold = more aggressive skipping = faster but more quality risk.
    # Typical: 0.5 = moderate, 0.3 = aggressive, 0.1 = very aggressive.
    swlp_early_exit: str = "off"
    # Layer pruning: remove least-important layers entirely.
    # "off" = disabled (default) | "light" = remove ~10% | "aggressive" = ~25%
    # Requires a one-time calibration run per model (profile=True first).
    swlp_layer_pruning: str = "off"
    # Adaptive precision: use FP16 for early layers, FP8/INT8 for later layers.
    # "off" = disabled (default) | "fp8_late" = FP8 for last 50% of layers
    # | "int8_late" = INT8 for last 50% of layers.
    swlp_adaptive_precision: str = "off"


@dataclass(slots=True)
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def to_dict(self) -> dict[str, Any]:
        def _jsonify(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {key: _jsonify(item) for key, item in value.items()}
            if isinstance(value, list):
                return [_jsonify(item) for item in value]
            return value

        return _jsonify(asdict(self))


def _default_config_path() -> Path:
    return Path("configs/default.toml")


def _load_toml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_config(config_path: str | Path | None = None) -> AppConfig:
    path = (
        Path(config_path).expanduser()
        if config_path
        else Path(os.getenv("SWLP_CONFIG", _default_config_path()))
    )
    file_config = _load_toml(path)

    config_data = _merge_dict(
        AppConfig().to_dict(),
        file_config,
    )

    return AppConfig(
        model=_section(ModelConfig, config_data["model"]),
        cache=_section(CacheConfig, config_data["cache"]),
        generation=_section(GenerationConfig, config_data["generation"]),
        runtime=_section(RuntimeConfig, config_data["runtime"]),
    )
