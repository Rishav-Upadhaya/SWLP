"""Tests for swlp.runner.mlx — MlxRunner config and wiring (Phase 8).

These cover construction, the quality-dial validation, model-path resolution
and build_runner registration. The actual MLX forward pass is not exercised
here — it needs a real model and is measured separately.
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
