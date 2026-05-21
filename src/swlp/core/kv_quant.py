"""INT4 KV cache quantization (Phase 18 — first lossy tier).

Quantize Key/Value tensors to 4-bit integers with per-token absmax scales.
Two INT4 values are packed per uint8 byte, giving a ~4× memory reduction vs
FP16.

⚠️  **This is the project's first deliberate quality compromise.**
INT4 KV quantization is **off by default** and must be explicitly enabled via
``SWLP_KV_QUANT=int4`` or ``kv_quant = "int4"`` in a config TOML.  Every
report that uses INT4 KV must label it as a lossy approximation and state the
measured perplexity cost — never silently.

Design
------
Per-token (per sequence-position, per head) absmax scale vectors are computed
from the FP16 activations.  The value is divided by the scale and rounded to
the nearest signed 4-bit integer in ``[−7, 7]``.  Two INT4 values are packed
into one uint8 byte (low nibble = even head-dim index, high nibble = odd);
the even/odd layout allows unpack via simple bitwise ops without any table
lookups.

Expected error:  relative L1 error ≈ 0.5–2% on typical transformer KV
activations.  Perplexity delta is workload-dependent but typically < 1 ppl
for medium-length context (see Phase 18 notes in CLAUDE.md).
"""
from __future__ import annotations

import torch

# Symmetric signed 4-bit range.  Using ±7 instead of [-8, 7] avoids the
# asymmetry artefact at -8; reconstruction error is dominated by the absmax
# scale granularity, not the ±1 asymmetry.
_INT4_BOUND: int = 7
# Bias applied before nibble packing: shift [-7, 7] → [1, 15] (all positive,
# fits in 4 bits).  0 is technically available but unused for simplicity.
_INT4_BIAS: int = 8


def kv_quantize_int4(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a KV tensor to packed INT4 with per-token absmax scales.

    Args:
        tensor: ``[batch, heads, seq_len, head_dim]`` FP16/BF16/FP32.
                ``head_dim`` must be even.

    Returns:
        ``(packed, scales)`` where:

        - ``packed``:  ``[batch, heads, seq_len, head_dim // 2]`` uint8 —
          two signed 4-bit values per byte (low nibble = even index, high = odd).
        - ``scales``:  ``[batch, heads, seq_len, 1]`` float16 — per-token
          absmax divided by ``_INT4_BOUND``.  Multiply by this to reconstruct.

    Raises:
        ValueError: if ``head_dim`` is odd.
    """
    head_dim = tensor.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError(
            f"kv_quantize_int4 requires even head_dim; got {head_dim}"
        )
    t = tensor.float()
    # Per-token absmax scale (shape: [..., seq_len, 1]).
    scales_raw = t.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    # Quantize: map to [-_INT4_BOUND, +_INT4_BOUND].
    q = (t / scales_raw * _INT4_BOUND).round().clamp(-_INT4_BOUND, _INT4_BOUND).to(torch.int8)
    # Shift to unsigned [1, 15] for nibble packing.
    q_u = (q + _INT4_BIAS).to(torch.uint8)
    even = q_u[..., 0::2]           # [..., head_dim // 2]
    odd  = q_u[..., 1::2]           # [..., head_dim // 2]
    packed = (even & 0xF) | ((odd & 0xF) << 4)
    # Store scale as float16; divide by _INT4_BOUND now so dequant is a
    # single multiply: x ≈ q_int * stored_scale.
    stored_scales = (scales_raw / _INT4_BOUND).to(torch.float16)
    return packed, stored_scales


def kv_dequantize_int4(
    packed: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Dequantize packed INT4 KV tensors back to FP16.

    Args:
        packed: ``[batch, heads, seq_len, head_dim // 2]`` uint8.
        scales: ``[batch, heads, seq_len, 1]`` float16 absmax scales
                (as returned by :func:`kv_quantize_int4`).

    Returns:
        ``[batch, heads, seq_len, head_dim]`` float16 tensor.
    """
    low  = (packed & 0xF).to(torch.int8)           # [..., head_dim // 2]
    high = ((packed >> 4) & 0xF).to(torch.int8)    # [..., head_dim // 2]
    # Reverse the +8 bias → signed int4 values in [-7, 7].
    low  = low  - _INT4_BIAS
    high = high - _INT4_BIAS
    # Interleave even/odd back to full head_dim.
    head_dim = packed.shape[-1] * 2
    shape = packed.shape[:-1] + (head_dim,)
    q = torch.empty(shape, dtype=torch.int8, device=packed.device)
    q[..., 0::2] = low
    q[..., 1::2] = high
    return (q.float() * scales.float()).to(torch.float16)


def kv_quantized_bytes(
    tensor: torch.Tensor,
) -> int:
    """Estimate on-disk/in-RAM bytes for INT4-quantised storage of ``tensor``.

    Returns the packed uint8 size plus float16 scale size.
    """
    batch_heads_seq = tensor.numel() // tensor.shape[-1]
    packed_bytes = (tensor.shape[-1] // 2) * batch_heads_seq  # uint8
    scale_bytes  = batch_heads_seq * 2  # float16 = 2 bytes per token
    return int(packed_bytes + scale_bytes)
