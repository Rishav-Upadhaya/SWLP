"""Tests for swlp.model.quant — FP8 weight quantization (Phase 7)."""
from __future__ import annotations

import json

import pytest
import torch

from swlp.model.quant import (
    dequantize_layer_state,
    is_quantized,
    quantize_layer_state,
    requantize_shards,
)
from swlp.model.shard import load_manifest


def _layer_state() -> dict:
    """A small state dict mimicking one transformer block: 2-D weights + 1-D norms."""
    torch.manual_seed(0)
    return {
        "self_attn.q_proj.weight": torch.randn(64, 48, dtype=torch.float16),
        "mlp.down_proj.weight": torch.randn(48, 128, dtype=torch.float16),
        "input_layernorm.weight": torch.randn(48, dtype=torch.float16),
    }


def test_quantize_float16_is_passthrough():
    state = _layer_state()
    out = quantize_layer_state(state, scheme="float16")
    assert not is_quantized(out)
    assert set(out) == set(state)


def test_quantize_float8_marks_dict():
    out = quantize_layer_state(_layer_state(), scheme="float8")
    assert is_quantized(out)
    assert out["_swlp_quant"] == "float8"
    assert set(out["weights"]) == set(_layer_state())


def test_unknown_scheme_raises():
    with pytest.raises(ValueError):
        quantize_layer_state(_layer_state(), scheme="int7")


def test_fp8_2d_weight_roundtrip_near_lossless():
    state = _layer_state()
    deq = dequantize_layer_state(quantize_layer_state(state, scheme="float8"))
    for name in ("self_attn.q_proj.weight", "mlp.down_proj.weight"):
        orig, got = state[name], deq[name]
        assert got.dtype == torch.float16
        assert got.shape == orig.shape
        rel = (got.float() - orig.float()).abs().mean() / orig.float().abs().mean()
        # Per-output-channel FP8 keeps the relative L1 error well under 5%.
        assert rel < 0.05


def test_fp8_1d_tensor_kept_exact():
    state = _layer_state()
    deq = dequantize_layer_state(quantize_layer_state(state, scheme="float8"))
    # 1-D norm weights are stored FP16 verbatim — bit-exact.
    assert torch.equal(deq["input_layernorm.weight"], state["input_layernorm.weight"])


def test_dequantize_plain_dict_is_noop():
    state = _layer_state()
    assert dequantize_layer_state(state) is state


def test_fp8_storage_is_smaller():
    state = _layer_state()
    q = quantize_layer_state(state, scheme="float8")
    qw = q["weights"]["self_attn.q_proj.weight"]
    # FP8 payload is 1 byte/elem vs 2 for FP16.
    assert qw["data"].dtype == torch.float8_e4m3fn
    assert qw["data"].element_size() == 1


def _write_fp16_shard_dir(path):
    """Build a minimal 2-layer FP16 shard directory for requantize tests."""
    path.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        torch.save(_layer_state(), path / f"layer_{i:03d}.pt")
    torch.save({"embed_tokens": {"weight": torch.randn(8, 8, dtype=torch.float16)}},
               path / "embed.pt")
    torch.save({"weight": torch.randn(8, 8, dtype=torch.float16)}, path / "lm_head.pt")
    manifest = {
        "model_id": "test/model", "num_layers": 2, "layer_weight_mb": 1.0,
        "total_weight_mb": 2.0, "embed_file": "embed.pt",
        "lm_head_file": "lm_head.pt", "model_type": "llama",
    }
    (path / "shard_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_requantize_shards_writes_fp8_manifest(tmp_path):
    src, dst = tmp_path / "fp16", tmp_path / "fp8"
    _write_fp16_shard_dir(src)
    manifest = requantize_shards(src, dst, scheme="float8")

    assert manifest.weight_dtype == "float8"
    assert manifest.num_layers == 2
    on_disk = load_manifest(dst)
    assert on_disk.weight_dtype == "float8"
    # Phase 17: requantize_shards writes .safetensors layer shards.
    assert on_disk.shard_format == "safetensors"
    # embed / lm_head copied verbatim (.pt).
    assert (dst / "embed.pt").is_file() and (dst / "lm_head.pt").is_file()
    # Layer shards written as .safetensors.
    for i in range(2):
        assert (dst / f"layer_{i:03d}.safetensors").is_file()


def test_requantized_layer_dequantizes_close_to_source(tmp_path):
    src, dst = tmp_path / "fp16", tmp_path / "fp8"
    _write_fp16_shard_dir(src)
    requantize_shards(src, dst, scheme="float8")

    orig = torch.load(src / "layer_000.pt", weights_only=True)
    # Phase 17: FP8 output is .safetensors; use _load_safetensors_shard to
    # reconstruct the nested FP8 format, then dequantize.
    from swlp.core.streaming import _load_safetensors_shard
    fp8_state = _load_safetensors_shard(dst / "layer_000.safetensors")
    assert fp8_state.get("_swlp_quant") == "float8"
    deq = dequantize_layer_state(fp8_state)
    w = "self_attn.q_proj.weight"
    rel = (deq[w].float() - orig[w].float()).abs().mean() / orig[w].float().abs().mean()
    assert rel < 0.05
