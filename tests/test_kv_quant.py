"""Tests for swlp.core.kv_quant — INT4 KV tensor quantization (Phase 18)."""
import torch

from swlp.core.kv_quant import kv_dequantize_int4, kv_quantize_int4, kv_quantized_bytes


# ── kv_quantize_int4 ──────────────────────────────────────────────────────────


def test_quantize_returns_correct_shapes() -> None:
    """kv_quantize_int4 returns packed uint8 [b,h,s,d//2] and scales [b,h,s,1]."""
    t = torch.randn(1, 4, 8, 64)  # [batch, heads, seq, head_dim]
    packed, scales = kv_quantize_int4(t)
    assert packed.shape == (1, 4, 8, 32)  # head_dim // 2
    assert scales.shape == (1, 4, 8, 1)


def test_quantize_packed_dtype_is_uint8() -> None:
    """Packed tensor must be uint8."""
    t = torch.randn(1, 2, 4, 16)
    packed, _ = kv_quantize_int4(t)
    assert packed.dtype == torch.uint8


def test_quantize_scales_dtype_is_float16() -> None:
    """Scale tensor must be float16."""
    t = torch.randn(1, 2, 4, 16)
    _, scales = kv_quantize_int4(t)
    assert scales.dtype == torch.float16


def test_quantize_raises_for_odd_head_dim() -> None:
    """Raises ValueError when head_dim is odd."""
    import pytest

    t = torch.randn(1, 2, 4, 15)
    with pytest.raises(ValueError, match="even head_dim"):
        kv_quantize_int4(t)


def test_quantize_scales_positive() -> None:
    """All scale values must be strictly positive."""
    t = torch.randn(2, 4, 16, 32)
    _, scales = kv_quantize_int4(t)
    assert (scales > 0).all()


# ── kv_dequantize_int4 ────────────────────────────────────────────────────────


def test_dequantize_returns_float16() -> None:
    """kv_dequantize_int4 output dtype must be float16."""
    t = torch.randn(1, 2, 8, 32)
    packed, scales = kv_quantize_int4(t)
    out = kv_dequantize_int4(packed, scales)
    assert out.dtype == torch.float16


def test_dequantize_returns_correct_shape() -> None:
    """Shape after dequantize is [b, h, s, head_dim] (same as input)."""
    t = torch.randn(2, 4, 12, 64)
    packed, scales = kv_quantize_int4(t)
    out = kv_dequantize_int4(packed, scales)
    assert out.shape == t.shape


def test_round_trip_relative_error() -> None:
    """Round-trip relative L1 error is bounded for INT4 quantization.

    With only 14 signed levels the worst-case relative L1 error on iid normal
    data is ~1/7 ≈ 14%.  Structured transformer KV activations achieve much
    lower error (~0.5–2%) because they are heavy-tailed and the per-token absmax
    scale adapts.  We bound at 20% here — a generous budget for random noise.
    """
    t = torch.randn(1, 8, 32, 64).clamp(-3, 3)
    packed, scales = kv_quantize_int4(t)
    out = kv_dequantize_int4(packed, scales)
    rel_err = (out.float() - t.float()).abs().mean() / t.float().abs().mean().clamp(min=1e-8)
    assert float(rel_err) < 0.20, f"Relative L1 error too high: {rel_err:.4f}"


def test_round_trip_zero_tensor() -> None:
    """Zero tensors round-trip without NaN or Inf."""
    t = torch.zeros(1, 2, 4, 16)
    packed, scales = kv_quantize_int4(t)
    out = kv_dequantize_int4(packed, scales)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_round_trip_large_values() -> None:
    """Large-valued tensors (e.g., ±100) round-trip with relative error < 20%."""
    t = torch.randn(1, 2, 8, 32) * 100
    packed, scales = kv_quantize_int4(t)
    out = kv_dequantize_int4(packed, scales)
    rel_err = (out.float() - t.float()).abs().mean() / t.float().abs().mean().clamp(min=1e-8)
    assert float(rel_err) < 0.20


# ── kv_quantized_bytes ────────────────────────────────────────────────────────


def test_kv_quantized_bytes_smaller_than_fp16() -> None:
    """INT4 packed representation is ~4× smaller than FP16."""
    t = torch.randn(1, 8, 128, 64)
    fp16_bytes = t.element_size() * t.numel()  # 2 bytes each
    int4_bytes = kv_quantized_bytes(t)
    # Packed uint8 (d//2) + float16 scales (2 bytes per token):
    # ratio ≈ 4 for large head_dim (scales are negligible).
    assert int4_bytes < fp16_bytes / 3, (
        f"Expected INT4 < FP16/3; got INT4={int4_bytes}, FP16={fp16_bytes}"
    )


def test_kv_quantized_bytes_formula() -> None:
    """kv_quantized_bytes matches the manual formula."""
    t = torch.randn(1, 4, 8, 64)
    expected = (64 // 2) * (1 * 4 * 8) + (1 * 4 * 8) * 2
    assert kv_quantized_bytes(t) == expected
