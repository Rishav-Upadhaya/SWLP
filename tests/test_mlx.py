"""Tests for swlp.runner.mlx — MlxRunner config and wiring.

These cover construction, quality-dial validation, model-path resolution,
build_runner registration, and the new M5 optimisation knobs (draft model,
max_kv_size / kv_bits generation kwargs).  The actual MLX forward pass is not
exercised here — it needs a real model and is measured separately.
"""
from __future__ import annotations

import pytest

from swlp.config import AppConfig
from swlp.runner.base import build_runner
from swlp.runner.mlx import MlxRunner


def _mlx_config(quant: str = "int8") -> AppConfig:
    config = AppConfig()
    config.runtime.backend = "mlx"
    config.runtime.mlx_quant = quant
    config.model.model_id = "unsloth/mistral-7b-instruct-v0.2"
    return config


# ── basic construction ────────────────────────────────────────────────────────


def test_build_runner_returns_mlx_runner():
    runner = build_runner(_mlx_config())
    assert isinstance(runner, MlxRunner)
    assert runner.backend == "mlx"


@pytest.mark.parametrize("quant", ["bf16", "int8", "int4"])
def test_valid_quant_accepted(quant):
    runner = MlxRunner(_mlx_config(quant))
    assert runner.mlx_quant == quant


def test_invalid_quant_rejected():
    with pytest.raises(ValueError):
        MlxRunner(_mlx_config("int2"))


def test_quant_is_case_insensitive():
    runner = MlxRunner(_mlx_config("INT4"))
    assert runner.mlx_quant == "int4"


def test_bf16_resolves_to_model_id():
    runner = MlxRunner(_mlx_config("bf16"))
    assert runner._resolve_model_path() == "unsloth/mistral-7b-instruct-v0.2"


def test_local_model_path_takes_precedence_for_bf16():
    config = _mlx_config("bf16")
    config.model.local_model_path = "/tmp/local-model"
    runner = MlxRunner(config)
    assert runner._resolve_model_path() == "/tmp/local-model"


def test_tokenizer_starts_as_none():
    """run_chat() relies on runner.tokenizer being None before load()."""
    runner = MlxRunner(_mlx_config())
    assert runner.tokenizer is None
    assert runner._mlx_model is None


# ── draft model wiring ────────────────────────────────────────────────────────


def test_draft_model_default_is_empty():
    config = _mlx_config()
    assert config.runtime.mlx_draft_model == ""


def test_draft_model_stored_on_runner():
    """Draft model field is accessible; runner starts with no loaded draft."""
    runner = MlxRunner(_mlx_config())
    assert runner._mlx_draft_model is None


def test_draft_model_set_via_config():
    config = _mlx_config()
    config.runtime.mlx_draft_model = "HuggingFaceTB/SmolLM2-360M-Instruct"
    runner = MlxRunner(config)
    assert runner.config.runtime.mlx_draft_model == "HuggingFaceTB/SmolLM2-360M-Instruct"


# ── _gen_kwargs (max_kv_size / kv_bits) ──────────────────────────────────────


def test_gen_kwargs_default_is_prefill_chunking_only():
    """Defaults must be quality-neutral: a prefill chunk size and nothing else.
    KV quantization and the KV window both change behaviour, so neither is on
    unless asked for."""
    runner = MlxRunner(_mlx_config())
    assert runner._gen_kwargs() == {"prefill_step_size": 2048}


def test_gen_kwargs_max_kv_size_from_kv_window():
    config = _mlx_config()
    config.runtime.kv_window = 512
    runner = MlxRunner(config)
    assert runner._gen_kwargs()["max_kv_size"] == 512


def test_gen_kwargs_no_max_kv_size_when_kv_window_zero():
    config = _mlx_config()
    config.runtime.kv_window = 0
    runner = MlxRunner(config)
    assert "max_kv_size" not in runner._gen_kwargs()


def test_gen_kwargs_kv_bits_when_int4():
    config = _mlx_config()
    config.runtime.kv_quant = "int4"
    runner = MlxRunner(config)
    assert runner._gen_kwargs()["kv_bits"] == 4


def test_gen_kwargs_no_kv_bits_when_none():
    config = _mlx_config()
    config.runtime.kv_quant = "none"
    runner = MlxRunner(config)
    assert "kv_bits" not in runner._gen_kwargs()


def test_gen_kwargs_combined():
    config = _mlx_config()
    config.runtime.kv_window = 256
    config.runtime.kv_quant = "int4"
    runner = MlxRunner(config)
    kwargs = runner._gen_kwargs()
    assert kwargs["max_kv_size"] == 256
    assert kwargs["kv_bits"] == 4


# ── Apple Silicon tuning (runner/mlx_tune.py) ────────────────────────────────

def test_gen_kwargs_mlx_kv_bits_emits_group_and_start():
    """kv_bits never travels alone: group size and the exact-prefix start
    must accompany it or mlx-lm silently uses its own defaults."""
    config = _mlx_config()
    config.runtime.mlx_kv_bits = 4
    kwargs = MlxRunner(config)._gen_kwargs()
    assert kwargs["kv_bits"] == 4
    assert kwargs["kv_group_size"] == 64
    assert kwargs["quantized_kv_start"] == 512


def test_gen_kwargs_invalid_kv_bits_is_ignored_not_passed_through():
    config = _mlx_config()
    config.runtime.mlx_kv_bits = 3          # MLX supports 4 and 8 only
    assert "kv_bits" not in MlxRunner(config)._gen_kwargs()


def test_gen_kwargs_draft_tokens_only_with_a_draft_model():
    """num_draft_tokens without a draft model is meaningless — mlx-lm would
    ignore it, and emitting it hides a misconfiguration."""
    config = _mlx_config()
    config.runtime.mlx_num_draft_tokens = 6
    runner = MlxRunner(config)
    assert "num_draft_tokens" not in runner._gen_kwargs()
    runner._mlx_draft_model = object()      # pretend a draft model loaded
    assert runner._gen_kwargs()["num_draft_tokens"] == 6


def test_recommended_wired_limit_leaves_the_os_room():
    from swlp.runner.mlx_tune import recommended_wired_limit_mb

    # 16 GB machine: 4 GB reserved for macOS, so ~12 GB wired.
    assert recommended_wired_limit_mb(16.0) == 12 * 1024
    # Never exceed the safety fraction, however big the machine.
    assert recommended_wired_limit_mb(8.0) <= int(8.0 * 0.92 * 1024)
    # Tiny machines still get a positive, non-absurd number.
    assert recommended_wired_limit_mb(4.0) > 0


def test_apply_memory_tuning_off_is_a_no_op():
    from swlp.runner.mlx_tune import apply_memory_tuning

    report = apply_memory_tuning("off", 16.0)
    assert report.wired_limit_mb is None
    assert "untouched" in report.notes[0]


def test_apply_memory_tuning_rejects_garbage_without_raising():
    """A bad value must not take the process down — it is an optimisation."""
    from swlp.runner.mlx_tune import apply_memory_tuning

    report = apply_memory_tuning("not-a-number", 16.0)
    assert report.wired_limit_mb is None
    assert any("invalid" in n for n in report.notes)
