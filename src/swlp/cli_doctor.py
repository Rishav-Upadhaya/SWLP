"""``swlp doctor`` — what this Mac can run, how, and how to tune it.

One screen: the machine (chip, RAM, GPU working set, SSD), a table of models
with how each would run *here* and the exact command, and the Apple Silicon
tuning levers. Speeds are shown only where they were measured on an M5 16 GB.
"""
from __future__ import annotations

from . import ui
from .hardware.detect import HardwareInfo, detect_hardware, fits_in_memory

# Approximate FP16 disk size per alias (GB).
KNOWN_FP16_GB: dict[str, float] = {
    "smollm-360m": 0.72,
    "qwen-0.5b": 1.0,
    "qwen-1.5b": 3.1,
    "smollm-1.7b": 3.4,
    "qwen-3b": 6.2,
    "phi-3.5": 7.6,
    "mistral-7b": 14.0,
    "qwen-7b": 15.0,
    "qwen-14b": 26.0,
    "mistral-24b": 44.0,
}

# Layer count and per-layer MB (used by scripts/collect_cross_machine.py).
MODEL_LAYERS: dict[str, tuple[int, float]] = {
    "smollm-360m": (24, 30.0),
    "qwen-0.5b": (24, 42.0),
    "qwen-1.5b": (28, 110.0),
    "smollm-1.7b": (24, 142.0),
    "qwen-3b": (36, 172.0),
    "phi-3.5": (32, 238.0),
    "mistral-7b": (32, 416.0),
    "qwen-7b": (32, 469.0),
    "qwen-14b": (40, 650.0),
    "mistral-24b": (40, 1100.0),
}

# MoE models: as-shipped size and how SWLP runs them (expert streaming).
MOE_MODELS: dict[str, dict[str, object]] = {
    "gemma4-26b": {"disk_gb": 15.3, "precision": "4-bit", "prepared": True},
    "olmoe-7b": {"disk_gb": 13.8, "precision": "bf16", "prepared": False},
    "qwen3-30b-a3b": {"disk_gb": 61.0, "precision": "bf16", "prepared": False},
    "qwen3.6-35b": {"disk_gb": 72.0, "precision": "bf16", "prepared": False},
    "mixtral-8x7b": {"disk_gb": 93.0, "precision": "bf16", "prepared": False},
    "deepseek-v4-flash": {"disk_gb": 142.0, "precision": "mxfp4", "prepared": False},
}

# Steady-state decode measured on an Apple M5 16 GB (docs/ROADMAP.md,
# Phases 29–31). Everything else shows "—": no guesses in this table.
MEASURED_M5_16GB: dict[str, str] = {
    "gemma4-26b": "14.5",
    "olmoe-7b": "12.5",
    "qwen3.6-35b": "4.5–5.0",
}

_GB = 1024 ** 3


def recommend(alias: str, size_gb: float, hw: HardwareInfo,
              installed: frozenset[str] = frozenset()) -> tuple[str, str]:
    """``(how it runs on this machine, the command to type)``; installed models
    (``swlp models``) are ready to chat, the rest need ``swlp pull`` first."""
    if alias in MOE_MODELS:
        info = MOE_MODELS[alias]
        ready = info["prepared"] or alias in installed
        return (f"MoE experts · {info['precision']}",
                f"swlp chat {alias}" if ready else f"swlp pull {alias}")
    if alias in installed:
        return "layer streaming", f"swlp chat {alias}"
    mlx_ready = hw.unified_memory and hw.preferred_backend == "mlx"
    for quant, bytes_per_fp16 in (("int8", 0.5), ("int4", 0.25)):
        if mlx_ready and fits_in_memory(int(size_gb * bytes_per_fp16 * _GB), hw):
            return f"MLX {quant}", f"swlp chat {alias} -q {quant}"
    return "layer streaming", f"swlp pull {alias}"


def plan_rows(hw: HardwareInfo, only: str | None = None) -> list[list[str]]:
    """The "what you can run" table: model · size · how · speed · command."""
    from .cli_models import installed_models

    installed = frozenset(name for name, *_ in installed_models())
    sizes = {**KNOWN_FP16_GB, **{a: float(i["disk_gb"]) for a, i in MOE_MODELS.items()}}
    rows = []
    for alias, size in sorted(sizes.items(), key=lambda kv: kv[1]):
        if only and alias != only:
            continue
        how, cmd = recommend(alias, size, hw, installed)
        rows.append([alias, f"{size:.0f} GB" if size >= 1 else f"{size:.1f} GB", how,
                     MEASURED_M5_16GB.get(alias, "—"), cmd])
    return rows


def tuning_rows(hw: HardwareInfo) -> list[list[str]]:
    """Apple Silicon levers: the Metal wired ceiling first (resident vs swapping)."""
    if not hw.unified_memory:
        return []
    from .runner.mlx_tune import max_working_set_mb, sysctl_advice

    cap_mb = max_working_set_mb()
    rows = []
    if cap_mb is None:
        rows.append(["GPU memory", "MLX not installed — pip install 'swlp[apple]'"])
    else:
        pct = 100 * cap_mb / (hw.memory_gb * 1024)
        rows.append(["GPU working set", f"{cap_mb / 1024:.1f} GB ({pct:.0f}% of RAM)"])
    advice = sysctl_advice(hw.memory_gb)
    if advice:
        rows.append(["raise it (sudo)", advice.split("   #")[0]])
    rows += [
        ["faster KV cache", "SWLP_MLX_KV_BITS=4  (4-bit KV is faster here, not slower)"],
        ["speculative", "SWLP_DRAFT_MODEL=<small same-family model>  (1.9–2.1x)"],
        ["MoE expert cache", "SWLP_EXPERT_CACHE_MB=<MB>  (default: free RAM, GPU-capped)"],
    ]
    return rows


def print_doctor(model: str | None = None) -> None:
    import psutil

    hw = detect_hardware()
    free_gb = psutil.virtual_memory().available / _GB
    if not hw.unified_memory:
        mlx = "n/a (Apple Silicon only)"
    elif hw.preferred_backend == "mlx":
        mlx = "installed"
    else:
        mlx = "not installed — pip install 'swlp[apple]'"
    ui.header("swlp doctor", [
        ("chip", hw.chip_name),
        ("memory", f"{hw.memory_gb:.0f} GB unified · {free_gb:.1f} GB free now"),
        ("SSD", f"{hw.ssd_bandwidth_gbps:.1f} GB/s"),
        ("MLX", mlx),
    ])
    rows = plan_rows(hw, only=model)
    if model and not rows:
        ui.note(f"\n  {model}: no size on record — "
                f"try  swlp pull {model}  or  swlp chat {model} -q int4")
    else:
        ui.console.print()
        ui.table(["model", "size", "runs as", "M5 tok/s*", "command"], rows,
                 title="What you can run")
        ui.note("  * measured on an M5 16 GB (steady decode)  ·  after  swlp pull MODEL :  "
                "swlp chat MODEL")
    tuning = tuning_rows(hw)
    if tuning:
        ui.console.print()
        ui.table(["", ""], tuning, title="Tuning")
    ui.console.print()


def _measure_pipeline_ratio(alias: str, hw: HardwareInfo) -> dict:
    """Measure or estimate pipeline ratio for a model on this hardware."""
    from .core.pipeline_model import estimate_pipeline_ratio_from_hardware

    if alias not in MODEL_LAYERS:
        return {
            "ratio": 5.0,
            "source": "fallback",
            "layers": 0,
            "layer_size_mb": 0,
            "ssd_read_ms": 0,
            "deserialize_ms": 0,
            "upload_ms": 0,
            "compute_ms": 0,
        }

    num_layers, layer_size_mb = MODEL_LAYERS[alias]
    layer_bytes = int(layer_size_mb * 1024 * 1024)

    # Estimate from hardware specs
    ratio = estimate_pipeline_ratio_from_hardware(
        layer_weight_bytes=layer_bytes,
        ssd_bandwidth_gbps=hw.ssd_bandwidth_gbps,
        compute_ms=30.0,  # typical for this class of model
        deserialize_upload_ms=52.0,
    )

    # Estimate component timings
    bandwidth_mb_per_s = hw.ssd_bandwidth_gbps * 1024
    ssd_read_ms = (layer_size_mb / bandwidth_mb_per_s) * 1000 if bandwidth_mb_per_s > 0 else 100
    deserialize_ms = layer_size_mb * 0.1  # rough estimate: 10ms per 100MB
    upload_ms = layer_size_mb * 0.06  # rough estimate: 6ms per 100MB
    compute_ms = 30.0  # typical for 7B-class on MPS

    return {
        "ratio": ratio,
        "source": "estimated",
        "layers": num_layers,
        "layer_size_mb": layer_size_mb,
        "ssd_read_ms": ssd_read_ms,
        "deserialize_ms": deserialize_ms,
        "upload_ms": upload_ms,
        "compute_ms": compute_ms,
    }
