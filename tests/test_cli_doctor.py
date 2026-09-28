"""Tests for cli_doctor.py — swlp doctor / swlp models.

Recommendation logic is tested with constructed HardwareInfo values so no
real hardware detection (or torch device queries) is needed.
"""
from __future__ import annotations

from swlp.cli_args import MODEL_ALIASES
from swlp.cli_doctor import (
    KNOWN_FP16_GB,
    doctor_lines,
    models_lines,
    recommend_command,
)
from swlp.hardware.detect import HardwareInfo


def _apple_m5_16gb(mlx: bool = True) -> HardwareInfo:
    return HardwareInfo(
        device_type="mps",
        unified_memory=True,
        memory_gb=16.0,
        ssd_bandwidth_gbps=6.5,
        preferred_backend="mlx" if mlx else "torch",
        chip_name="Apple M5",
    )


def _linux_cpu_64gb() -> HardwareInfo:
    return HardwareInfo(
        device_type="cpu",
        unified_memory=False,
        memory_gb=64.0,
        ssd_bandwidth_gbps=3.5,
        preferred_backend="torch",
        chip_name="x86_64",
    )


def test_too_big_model_recommends_streaming():
    cmd = recommend_command("mistral-24b", 44.0, _apple_m5_16gb())
    # One-command path: `swlp run` auto-shards, no explicit --shard-dir needed.
    assert cmd == 'swlp run mistral-24b --prompt "..."'


def test_apple_with_mlx_recommends_int8_when_it_fits():
    cmd = recommend_command("mistral-7b", 14.0, _apple_m5_16gb(mlx=True))
    assert "--backend mlx --quant int8" in cmd


def test_apple_without_mlx_falls_back_to_streaming_for_big_model():
    cmd = recommend_command("mistral-7b", 14.0, _apple_m5_16gb(mlx=False))
    assert "swlp run" in cmd
    assert "--backend" not in cmd  # streaming is the auto default


def test_big_ram_machine_recommends_plain_hf():
    cmd = recommend_command("mistral-7b", 14.0, _linux_cpu_64gb())
    assert "--shard-dir" not in cmd
    assert "--backend mlx" not in cmd


def test_doctor_lines_include_hardware_and_every_known_model():
    lines = "\n".join(doctor_lines(_apple_m5_16gb()))
    assert "Apple M5" in lines
    assert "16.0 GB" in lines
    for alias in KNOWN_FP16_GB:
        assert alias in lines


def test_doctor_lines_flag_missing_mlx():
    lines = "\n".join(doctor_lines(_apple_m5_16gb(mlx=False)))
    assert "swlp[apple]" in lines


def test_models_lines_list_every_alias_with_hf_id():
    lines = "\n".join(models_lines())
    for alias, hf_id in MODEL_ALIASES.items():
        assert alias in lines
        assert hf_id in lines


def test_models_command_via_cli(capsys):
    from swlp.cli import main

    assert main(["models"]) == 0
    out = capsys.readouterr().out
    assert "mistral-7b" in out
    assert "unsloth/mistral-7b-instruct-v0.2" in out


def test_shard_progress_renders_bar_and_final_newline(capsys):
    from swlp.cli import _shard_progress

    _shard_progress(1, 4, 436.0)
    mid = capsys.readouterr().out
    assert "layer   1/4" in mid
    assert not mid.endswith("\n")

    _shard_progress(4, 4, 436.0)
    end = capsys.readouterr().out
    assert end.endswith("\n")


def test_moe_advisory_lines_render():
    """Phase 27: the MoE advisory renders rows, ceiling math, and cache
    guidance consistent with the load.py auto policy (25% / 4 GB cap)."""
    from swlp.cli_doctor import MOE_MODELS, _moe_advisory_lines
    from swlp.hardware.detect import HardwareInfo

    hw = HardwareInfo(
        device_type="mps", unified_memory=True, memory_gb=16.0,
        ssd_bandwidth_gbps=6.93, preferred_backend="mlx", chip_name="M5",
    )
    lines = _moe_advisory_lines(hw, free_ram_gb=10.0)
    text = "\n".join(lines)
    for alias in MOE_MODELS:
        assert alias in text, f"missing advisory row for {alias}"
    # qwen3-30b-a3b: 3.4B active → 6.8 GB/token FP16 → 6.93/6.8 ≈ 1.02 t/s.
    assert "1.02" in text
    # free_ram 10 GB → 25% = 2.5 GB = 2560 MB (below the 4 GB cap).
    assert "SWLP_EXPERT_CACHE_MB=2560" in text
