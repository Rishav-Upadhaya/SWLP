"""swlp doctor: how each model runs on a given machine, and the command to type.

Constructed HardwareInfo values — no real hardware detection needed.
"""
from __future__ import annotations

from swlp.cli_doctor import (
    KNOWN_FP16_GB,
    MEASURED_M5_16GB,
    MOE_MODELS,
    plan_rows,
    recommend,
)
from swlp.hardware.detect import HardwareInfo


def _m5_16gb(mlx: bool = True) -> HardwareInfo:
    return HardwareInfo(device_type="mps", unified_memory=True, memory_gb=16.0,
                        ssd_bandwidth_gbps=6.9, preferred_backend="mlx" if mlx else "torch",
                        chip_name="Apple M5")


def test_small_model_runs_resident_on_mlx():
    how, cmd = recommend("mistral-7b", 14.0, _m5_16gb())
    assert how == "MLX int8" and cmd == "swlp chat mistral-7b -q int8"


def test_int4_when_int8_does_not_fit():
    how, cmd = recommend("qwen-14b", 26.0, _m5_16gb())
    assert how == "MLX int4" and cmd.endswith("-q int4")


def test_without_mlx_big_models_stream():
    how, cmd = recommend("mistral-7b", 14.0, _m5_16gb(mlx=False))
    assert how == "layer streaming" and cmd == "swlp pull mistral-7b"


def test_moe_needs_pull_unless_installed():
    assert recommend("qwen3-30b-a3b", 61.0, _m5_16gb())[1] == "swlp pull qwen3-30b-a3b"
    installed = frozenset({"qwen3-30b-a3b"})
    assert recommend("qwen3-30b-a3b", 61.0, _m5_16gb(), installed)[1] == "swlp chat qwen3-30b-a3b"
    assert recommend("gemma4-26b", 15.3, _m5_16gb())[1] == "swlp chat gemma4-26b"  # MLX repo


def test_plan_covers_every_model_and_only_measured_speeds(monkeypatch):
    monkeypatch.setattr("swlp.cli_models.installed_models", lambda: [])
    rows = {r[0]: r for r in plan_rows(_m5_16gb())}
    assert set(rows) == set(KNOWN_FP16_GB) | set(MOE_MODELS)
    for alias, row in rows.items():
        assert row[3] == MEASURED_M5_16GB.get(alias, "—")
    assert [r[0] for r in plan_rows(_m5_16gb(), only="gemma4-26b")] == ["gemma4-26b"]


def test_doctor_command_renders(capsys):
    from swlp.cli import main

    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "What you can run" in out and "gemma4-26b" in out
