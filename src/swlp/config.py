from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_float(value: str | None, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _parse_int(value: str | None, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _parse_path(value: str | None, default: Path | None) -> Path | None:
    if value is None or value == "":
        return default
    return Path(value).expanduser().resolve()


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
    swlp_debug: bool = False
    swlp_fallback_to_baseline: bool = True
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
    # Speculative decoding (Phase 5): prompt-lookup n-gram drafting
    swlp_spec_ngram: int = 3
    swlp_spec_max_draft: int = 8
    # MLX backend (Phase 8): native quantized compute on Apple Silicon.
    # "bf16" (lossless) | "int8" (near-lossless, default) | "int4" (fast tier)
    mlx_quant: str = "int8"
    # Optional draft model for MLX speculative decoding (alias or HF id).
    # Must share the same tokenizer as the main model. Empty = disabled.
    mlx_draft_model: str = ""
    # Phase 18: INT4 KV quantization — off by default (lossy).
    # "none" = lossless (default) | "int4" = ~4× smaller KV, ~0.5–1% ppl cost.
    # Must be explicitly opt-in; always labelled in reports.
    kv_quant: str = "none"


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

    model = config_data["model"]
    cache = config_data["cache"]
    generation = config_data["generation"]
    runtime = config_data["runtime"]

    model_id = os.getenv("SWLP_MODEL_ID", model["model_id"])
    local_model_path = _parse_path(
        os.getenv("SWLP_MODEL_PATH"),
        Path(model["local_model_path"]).expanduser().resolve()
        if model["local_model_path"]
        else None,
    )
    trust_remote_code = _parse_bool(
        os.getenv("SWLP_TRUST_REMOTE_CODE"), bool(model["trust_remote_code"])
    )
    revision = os.getenv("SWLP_REVISION", model.get("revision") or None)

    cache_dir = _parse_path(
        os.getenv("SWLP_CACHE_DIR"),
        Path(cache["cache_dir"]).expanduser().resolve(),
    ) or Path(cache["cache_dir"]).expanduser().resolve()
    offline = _parse_bool(os.getenv("SWLP_CACHE_OFFLINE"), bool(cache["offline"]))

    max_new_tokens = _parse_int(
        os.getenv("SWLP_MAX_NEW_TOKENS"), int(generation["max_new_tokens"])
    )
    temperature = _parse_float(os.getenv("SWLP_TEMPERATURE"), float(generation["temperature"]))
    top_p = _parse_float(os.getenv("SWLP_TOP_P"), float(generation["top_p"]))
    do_sample = _parse_bool(os.getenv("SWLP_DO_SAMPLE"), bool(generation["do_sample"]))
    repetition_penalty = _parse_float(
        os.getenv("SWLP_REPETITION_PENALTY"), float(generation["repetition_penalty"])
    )
    seed = _parse_int(os.getenv("SWLP_SEED"), int(generation["seed"]))
    prompt = os.getenv("SWLP_PROMPT", str(generation["prompt"]))

    device = os.getenv("SWLP_DEVICE", runtime["device"])
    dtype = os.getenv("SWLP_DTYPE", runtime["dtype"])
    backend = os.getenv("SWLP_BACKEND", runtime["backend"])
    allow_mock_fallback = _parse_bool(
        os.getenv("SWLP_ALLOW_MOCK_FALLBACK"), bool(runtime["allow_mock_fallback"])
    )
    log_level = os.getenv("SWLP_LOG_LEVEL", runtime["log_level"])
    json_logs = _parse_bool(os.getenv("SWLP_JSON_LOGS"), bool(runtime["json_logs"]))
    profile = _parse_bool(os.getenv("SWLP_PROFILE"), bool(runtime["profile"]))
    swlp_window_size = _parse_int(
        os.getenv("SWLP_WINDOW_SIZE"), int(runtime.get("swlp_window_size", 2))
    )
    swlp_prefetch_depth = _parse_int(
        os.getenv("SWLP_PREFETCH_DEPTH"), int(runtime.get("swlp_prefetch_depth", swlp_window_size))
    )
    swlp_prefetch = _parse_bool(
        os.getenv("SWLP_PREFETCH"), bool(runtime.get("swlp_prefetch", True))
    )
    swlp_double_buffer = _parse_bool(
        os.getenv("SWLP_DOUBLE_BUFFER"), bool(runtime.get("swlp_double_buffer", True))
    )
    swlp_pin_memory = _parse_bool(
        os.getenv("SWLP_PIN_MEMORY"), bool(runtime.get("swlp_pin_memory", True))
    )
    swlp_debug = _parse_bool(os.getenv("SWLP_DEBUG"), bool(runtime.get("swlp_debug", False)))
    swlp_fallback_to_baseline = _parse_bool(
        os.getenv("SWLP_FALLBACK_BASELINE"), bool(runtime.get("swlp_fallback_to_baseline", True))
    )
    kv_memory_budget_mb = _parse_int(
        os.getenv("SWLP_KV_BUDGET_MB"), int(runtime.get("kv_memory_budget_mb", 512))
    )
    kv_compression = _parse_bool(
        os.getenv("SWLP_KV_COMPRESSION"), bool(runtime.get("kv_compression", False))
    )
    kv_compression_level = _parse_int(
        os.getenv("SWLP_KV_COMPRESSION_LEVEL"), int(runtime.get("kv_compression_level", 0))
    )
    kv_tiering = _parse_bool(os.getenv("SWLP_KV_TIERING"), bool(runtime.get("kv_tiering", False)))
    kv_disk_dir = _parse_path(
        os.getenv("SWLP_KV_DISK_DIR"),
        Path(runtime["kv_disk_dir"]).expanduser().resolve()
        if runtime.get("kv_disk_dir")
        else None,
    )
    shard_dir = _parse_path(
        os.getenv("SWLP_SHARD_DIR"),
        Path(runtime["shard_dir"]).expanduser().resolve() if runtime.get("shard_dir") else None,
    )
    swlp_residency = os.getenv("SWLP_RESIDENCY", runtime.get("swlp_residency", "auto"))
    swlp_spec_ngram = _parse_int(
        os.getenv("SWLP_SPEC_NGRAM"), int(runtime.get("swlp_spec_ngram", 3))
    )
    swlp_spec_max_draft = _parse_int(
        os.getenv("SWLP_SPEC_MAX_DRAFT"), int(runtime.get("swlp_spec_max_draft", 8))
    )
    mlx_quant = os.getenv("SWLP_MLX_QUANT", runtime.get("mlx_quant", "int8"))
    mlx_draft_model = os.getenv("SWLP_MLX_DRAFT_MODEL", runtime.get("mlx_draft_model", ""))
    kv_window = _parse_int(os.getenv("SWLP_KV_WINDOW"), int(runtime.get("kv_window", 0)))
    kv_quant = os.getenv("SWLP_KV_QUANT", runtime.get("kv_quant", "none"))

    return AppConfig(
        model=ModelConfig(
            model_id=model_id,
            local_model_path=local_model_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
        ),
        cache=CacheConfig(cache_dir=cache_dir, offline=offline),
        generation=GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            seed=seed,
            prompt=prompt,
        ),
        runtime=RuntimeConfig(
            device=device,
            dtype=dtype,
            backend=backend,
            allow_mock_fallback=allow_mock_fallback,
            log_level=log_level,
            json_logs=json_logs,
            profile=profile,
            swlp_window_size=swlp_window_size,
            swlp_prefetch_depth=swlp_prefetch_depth,
            swlp_prefetch=swlp_prefetch,
            swlp_double_buffer=swlp_double_buffer,
            swlp_pin_memory=swlp_pin_memory,
            swlp_debug=swlp_debug,
            swlp_fallback_to_baseline=swlp_fallback_to_baseline,
            kv_memory_budget_mb=kv_memory_budget_mb,
            kv_compression=kv_compression,
            kv_compression_level=kv_compression_level,
            kv_tiering=kv_tiering,
            kv_disk_dir=kv_disk_dir,
            kv_window=kv_window,
            shard_dir=shard_dir,
            swlp_residency=str(swlp_residency),
            swlp_spec_ngram=swlp_spec_ngram,
            swlp_spec_max_draft=swlp_spec_max_draft,
            mlx_quant=str(mlx_quant),
            mlx_draft_model=str(mlx_draft_model),
            kv_quant=str(kv_quant),
        ),
    )
