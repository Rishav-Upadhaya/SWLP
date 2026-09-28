"""``swlp doctor`` — hardware-aware scheduling diagnosis.

Probes hardware, measures pipeline characteristics, predicts best-observed
resident cache configuration, and explains every decision.

Architecture:
    Hardware Probe → Pipeline Model → Resident Policy → Residency Planner → Doctor Output
"""
from __future__ import annotations

from .cli_args import MODEL_ALIASES
from .hardware.detect import HardwareInfo, detect_hardware, fits_in_memory

# Approximate FP16 disk size per alias (GB)
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

# Approximate layer counts and sizes for known models
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

_GB = 1024 ** 3

# Phase 25/27: MoE models runnable via expert-selective streaming. Disk size is
# the as-shipped checkpoint; "active" params drive per-token streamed bytes.
# Miss-rate guidance follows FreeToken's measured LRU curve (arXiv:2608.16157):
# ~30% miss at ~4% of the expert pool cached, ~16% at ~11%.
MOE_MODELS: dict[str, dict[str, object]] = {
    "qwen3-30b-a3b": {
        "hf_id": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "disk_gb": 61.0, "active_b": 3.4, "experts": 128, "top_k": 8,
        "note": "best first MoE target — fine-grained routing, 3.4B active",
    },
    "mixtral-8x7b": {
        "hf_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "disk_gb": 93.0, "active_b": 12.9, "experts": 8, "top_k": 2,
        "note": "coarse top-2-of-8 routing streams ~1/4 of every layer",
    },
    "deepseek-v4-flash": {
        "hf_id": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "disk_gb": 142.0, "active_b": 13.0, "experts": 256, "top_k": 6,
        "note": "284B MXFP4 — feasibility class, not interactive speed",
    },
}


def recommend_command(alias: str, size_gb: float, hw: HardwareInfo) -> str:
    """Return the recommended ``swlp`` command for one model on this hardware."""
    if alias in MOE_MODELS:
        return f"swlp run {alias} --prompt \"...\"  # expert-streamed MoE"
    mlx_ready = hw.unified_memory and hw.preferred_backend == "mlx"
    if mlx_ready and fits_in_memory(int(size_gb / 2 * _GB), hw):
        return f"swlp run {alias} --backend mlx --quant int8 --prompt \"...\""
    return f"swlp run {alias} --prompt \"...\""


def _apple_tuning_lines(hw: HardwareInfo) -> list[str]:
    """Apple-specific levers, with the exact commands to pull them.

    This is the most actionable part of the report on a Mac: the Metal wired
    ceiling decides whether a model is resident or swapping, and swapping is
    the difference between usable and unusable.
    """
    if not hw.unified_memory:
        return []

    from .runner.mlx_tune import (
        max_working_set_mb,
        recommended_wired_limit_mb,
        sysctl_advice,
    )

    lines = ["APPLE SILICON TUNING", "─" * 72]
    cap_mb = max_working_set_mb()
    want_mb = recommended_wired_limit_mb(hw.memory_gb)

    if cap_mb is None:
        lines.append("  Metal:           unavailable (install 'swlp[apple]' for MLX)")
    else:
        pct = 100.0 * cap_mb / (hw.memory_gb * 1024)
        lines.append(f"  GPU wired cap:   {cap_mb / 1024:.1f} GB  ({pct:.0f}% of RAM)")
        wired_gb = min(cap_mb, want_mb) / 1024
        lines.append(f"  SWLP will wire:  {wired_gb:.1f} GB  (auto, per process)")

    advice = sysctl_advice(hw.memory_gb)
    if advice:
        lines.extend([
            "",
            "  The GPU ceiling is below what this machine could give it.",
            "  To raise it (needs sudo, resets on reboot, leaves macOS 4 GB):",
            f"    {advice.split('   #')[0]}",
        ])
    else:
        lines.append("  Wired ceiling:   already at the safe maximum — nothing to do")

    lines.extend([
        "",
        "  Throughput levers, highest payoff first:",
        "    --quant int4            ~2x over int8; 4x less weight traffic",
        "    --draft-model <small>   1.9-2.1x measured; same model family only",
        "    --kv-bits 4             4-bit KV is FASTER than fp16 here, not slower",
        "                            (decode is bandwidth-bound; arXiv:2605.05699)",
        "    --max-kv-size <n>       caps long-context RAM (lossy: drops oldest)",
        "",
    ])
    return lines


def _moe_advisory_lines(hw: HardwareInfo, free_ram_gb: float) -> list[str]:
    """Phase 27: expert-cache guidance for MoE streaming targets."""
    lines = [
        "",
        "MoE STREAMING (Phase 25 — expert-selective sweeps)",
        "─" * 72,
        f"  {'MODEL':<20}{'DISK':>7}{'ACTIVE':>8}{'CEIL t/s':>9}   ADVICE",
    ]
    # Mirrors the auto policy in runner/load.py (25% of available, 4 GB cap)
    # so the advice and the implementation cannot disagree.
    cache_mb = int(min(free_ram_gb * 0.25, 4.0) * 1024)
    for alias, info in sorted(MOE_MODELS.items(), key=lambda kv: kv[1]["disk_gb"]):
        active_gb = float(info["active_b"]) * 2.0  # FP16: 2 bytes/param, decimal GB
        ceiling = hw.ssd_bandwidth_gbps / active_gb if active_gb > 0 else 0.0
        lines.append(
            f"  {alias:<20}{info['disk_gb']:>5.0f} GB{info['active_b']:>6.1f}B"
            f"{ceiling:>9.2f}   {info['note']}"
        )
    lines += [
        "",
        f"  Expert cache: SWLP_EXPERT_CACHE_MB={cache_mb} matches the auto",
        "  policy (25% of free RAM, 4 GB cap). FreeToken-measured LRU miss",
        "  ≈30% at ~4% of the expert pool cached, ≈16% at ~11% — hit rate,",
        "  not raw bandwidth, dominates MoE tok/s.",
        "  SWLP_EXPERT_PREFETCH=predictive (default) records routing history.",
        "  Sweep budgets on your machine: scripts/research/moe_sweep.py",
    ]
    return lines


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


def _predict_resident(ratio: float, free_ram_gb: float, layer_size_mb: float) -> dict:
    """Predict best-observed resident count with confidence."""
    from .core.confidence import estimate_policy_confidence
    from .core.resident_policy import estimate_optimal_resident_count

    result = estimate_optimal_resident_count(ratio, free_ram_gb, layer_size_mb)
    confidence = estimate_policy_confidence(
        pipeline_ratio=ratio,
        free_ram_gb=free_ram_gb,
        measured_ratio=False,
        has_layer_size=True,
    )

    return {
        "resident_count": result.resident_count,
        "raw_float": result.raw_float,
        "memory_clamped": result.memory_clamped,
        "in_grid_region": result.in_grid_region,
        "confidence_score": confidence.score,
        "confidence_level": confidence.level,
        "reasons": confidence.reasons,
        "factors": confidence.factors,
    }


def doctor_lines(hw: HardwareInfo, model: str | None = None) -> list[str]:
    """Build the enhanced ``swlp doctor`` report."""
    memory_kind = "unified" if hw.unified_memory else "system"
    free_ram_gb = hw.memory_gb * 0.7  # rough estimate of free RAM

    lines = [
        "",
        "  ╔" + "═" * 68 + "╗",
        "  ║" + "SWLP DOCTOR".center(68) + "║",
        "  ║" + "what this Mac can run, and how fast".center(68) + "║",
        "  ╚" + "═" * 68 + "╝",
        "",
        "HARDWARE",
        "─" * 72,
        f"  Chip:            {hw.chip_name}",
        f"  Memory:          {hw.memory_gb:.1f} GB ({memory_kind})",
        f"  Device:          {hw.device_type}",
        f"  SSD Bandwidth:   ~{hw.ssd_bandwidth_gbps:.1f} GB/s",
    ]

    # Add MLX status
    mlx_ready = hw.unified_memory and hw.preferred_backend == "mlx"
    if hw.unified_memory:
        mlx_status = "installed" if mlx_ready else "not installed  (pip install 'swlp[apple]')"
    else:
        mlx_status = "n/a  (Apple Silicon only)"
    lines.append(f"  MLX:             {mlx_status}")

    # .swz compression advice from the measured Phase 22 crossover.
    from .codec import recommend_compression
    from .hardware.detect import _measured_ssd_bandwidth

    measured = _measured_ssd_bandwidth()
    if measured is not None:
        if recommend_compression(measured):
            lines.append(
                f"  .swz Shards:     recommended — measured {measured:.1f} GB/s < 3.5 GB/s"
                " crossover (`swlp compress-shards <dir>`)"
            )
        else:
            lines.append(
                f"  .swz Shards:     disk-only win — measured {measured:.1f} GB/s ≥ 3.5 GB/s"
                " crossover (~25% tok/s cost; use only for disk space)"
            )
    else:
        lines.append(
            "  .swz Shards:     unknown — measure SSD first"
            " (`python scripts/phase0_hardware_check.py`)"
        )
    lines.append("")

    # Analyze each model (or just the specified one)
    models_to_check = [model] if model else ["mistral-7b", "qwen-14b"]

    for alias in models_to_check:
        if alias not in KNOWN_FP16_GB:
            continue

        size_gb = KNOWN_FP16_GB[alias]
        pipeline = _measure_pipeline_ratio(alias, hw)
        prediction = _predict_resident(
            pipeline["ratio"], free_ram_gb, pipeline["layer_size_mb"]
        )

        # Expected speedup (rough estimate based on pipeline ratio)
        if pipeline["ratio"] > 3:
            expected_speedup = 1.0 + (pipeline["ratio"] - 3) * 0.05
            expected_speedup = min(expected_speedup, 3.0)
        else:
            expected_speedup = 1.0

        # Memory usage
        resident_gb = prediction["resident_count"] * pipeline["layer_size_mb"] / 1024
        grid_region = "inside" if prediction["in_grid_region"] else "outside (extrapolated)"

        lines.extend([
            f"MODEL: {alias.upper()}",
            "─" * 72,
            f"  Layers:          {pipeline['layers']}",
            f"  Layer size:      {pipeline['layer_size_mb']:.0f} MB",
            f"  Total size:      {size_gb:.1f} GB",
            "",
            "MEASURED PIPELINE",
            "─" * 72,
            f"  SSD Read:        {pipeline['ssd_read_ms']:.0f} ms",
            f"  Deserialize:     {pipeline['deserialize_ms']:.0f} ms",
            f"  Upload:          {pipeline['upload_ms']:.0f} ms",
            f"  Compute:         {pipeline['compute_ms']:.0f} ms",
            f"  Pipeline Ratio:  {pipeline['ratio']:.2f}",
            f"  Ratio source:    {pipeline['source']}",
            "",
            "PREDICTION",
            "─" * 72,
            f"  Resident Layers: {prediction['resident_count']}",
            f"  Expected Speedup: {expected_speedup:.2f}x",
            f"  Confidence:      {prediction['confidence_score']:.0%} "
            f"({prediction['confidence_level']})",
            f"  Memory Used:     {resident_gb:.1f} GB",
            f"  Grid Region:     {grid_region}",
            "",
            "CONFIDENCE BREAKDOWN",
            "─" * 72,
            f"  {'Factor':<30} {'Impact':>8}",
            f"  {'-'*30} {'-'*8}",
        ])
        for factor, impact in sorted(prediction["factors"].items(), key=lambda x: x[1]):
            sign = "+" if impact >= 0 else ""
            lines.append(f"  {factor:<30} {sign}{impact:.0%}")

        lines.extend([
            "",
            "REASONING",
            "─" * 72,
        ])
        for i, reason in enumerate(prediction["reasons"], 1):
            lines.append(f"  {i}. {reason}")

        lines.extend([
            "",
            "RECOMMENDATION",
            "─" * 72,
            f"  Use {prediction['resident_count']} resident layers for {alias}.",
            f"  Command: swlp run {alias}  # auto-shards; "
            f"--window {prediction['resident_count']} to pin residency",
            "",
        ])

    lines.extend(_apple_tuning_lines(hw))

    # Coverage summary
    lines.extend(_moe_advisory_lines(hw, free_ram_gb))
    lines.extend([
        "COMMANDS",
        "─" * 72,
        f"  {'MODEL':<14}{'FP16':>7}   COMMAND",
    ])
    for alias, size_gb in sorted(KNOWN_FP16_GB.items(), key=lambda kv: kv[1]):
        lines.append(f"  {alias:<14}{size_gb:>5.1f} GB  {recommend_command(alias, size_gb, hw)}")
    for alias, info in sorted(MOE_MODELS.items(), key=lambda kv: kv[1]["disk_gb"]):
        disk_gb = info["disk_gb"]
        lines.append(f"  {alias:<14}{disk_gb:>5.0f} GB  {recommend_command(alias, disk_gb, hw)}")

    lines.extend([
        "",
        "─" * 72,
        "Run ``swlp doctor <model>`` for a specific model diagnosis.",
        "Run ``swlp profile --shard-dir ./shards/<model> --max-tokens 3`` to validate.",
        "",
    ])

    return lines


def models_lines() -> list[str]:
    """Build the ``swlp models`` alias reference table."""
    lines = [
        "Model aliases  (any other HuggingFace id also works)",
        "─" * 72,
        f"  {'ALIAS':<14}{'FP16':>7}   HUGGINGFACE ID",
    ]
    for alias, hf_id in sorted(MODEL_ALIASES.items(), key=lambda kv: KNOWN_FP16_GB.get(kv[0], 0)):
        size_gb = KNOWN_FP16_GB.get(alias)
        size = f"{size_gb:>5.1f} GB" if size_gb is not None else f"{'—':>8}"
        lines.append(f"  {alias:<14}{size}  {hf_id}")
    lines += [
        "",
        "All aliases support FP16 streaming (--shard-dir).  MLX (--backend mlx)",
        "requires Apple Silicon.  Run  swlp doctor  for per-machine advice.",
    ]
    return lines


def print_doctor(model: str | None = None) -> None:
    """Detect hardware and print the doctor report."""
    hw = detect_hardware()
    for line in doctor_lines(hw, model):
        print(line)


def print_models() -> None:
    """Print the model alias reference."""
    for line in models_lines():
        print(line)
