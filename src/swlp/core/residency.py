"""Adaptive residency planning for SWLP.

Given available memory and per-layer weight size, computes how many transformer
blocks can be kept permanently resident (loaded once, never evicted) vs. streamed
from disk on every token step.

This is pure computation — no I/O, no torch, no HardwareInfo import. The caller
(runner/swlp.py, which may import hardware/) is responsible for converting
HardwareInfo into total_memory_bytes before calling plan_residency().
"""
from __future__ import annotations

from dataclasses import dataclass

# Bytes reserved for the OS, background processes, and Metal driver overhead.
# On Apple Silicon unified memory, MPS tensors are locked in GPU memory and
# cannot be reclaimed by macOS under pressure — so the true OS reserve needs
# to be generous (4 GB covers macOS kernel + services + driver overhead).
_OS_RESERVE_BYTES: int = 4 * 1024 * 1024 * 1024  # 4 GB
# Bytes reserved for embeddings, final norm, lm_head, KV cache, activations,
# streaming layer slots (1-2 × layer_bytes), and PyTorch MPS internal buffers.
_WORKING_RESERVE_BYTES: int = 2 * 1024 * 1024 * 1024  # 2 GB
# Use 75 % of remaining headroom for resident layers to avoid Metal memory
# pressure (macOS compressor activates when free pages drop near zero).
_HEADROOM_FACTOR: float = 0.75


@dataclass(slots=True)
class ResidencyPlan:
    resident_count: int    # layers permanently loaded at startup
    streaming_count: int   # layers loaded/evicted per token
    resident_bytes: int    # total bytes locked in RAM
    streaming_bytes: int   # bytes streamed from disk per token
    available_bytes: int   # usable bytes after reserves


def plan_residency(
    total_memory_bytes: int,
    layer_weight_bytes: int,
    num_layers: int,
) -> ResidencyPlan:
    """Compute how many layers stay resident vs. stream.

    Args:
        total_memory_bytes: Total system/unified RAM in bytes.
        layer_weight_bytes: Size of one transformer block on disk (bytes).
        num_layers: Total number of transformer blocks in the model.

    Returns:
        ResidencyPlan with resident_count and streaming_count.
        If the model fits entirely, resident_count == num_layers and
        streaming_count == 0.
    """
    headroom = total_memory_bytes - _OS_RESERVE_BYTES - _WORKING_RESERVE_BYTES
    usable = max(0, int(headroom * _HEADROOM_FACTOR))

    if layer_weight_bytes <= 0 or num_layers <= 0:
        return ResidencyPlan(
            resident_count=0,
            streaming_count=num_layers,
            resident_bytes=0,
            streaming_bytes=0,
            available_bytes=usable,
        )

    total_model_bytes = layer_weight_bytes * num_layers

    # Partial residency is counter-productive: locking a subset of layers in
    # Python heap as CPU tensors evicts the OS page cache that the remaining
    # *streaming* layers depend on, triggering macOS memory compression and
    # making everything slower.  Only enable residency when the *full* model
    # fits within the safe usable budget — then all layers can be cached and
    # the streaming overhead drops to zero.
    if total_model_bytes > usable:
        return ResidencyPlan(
            resident_count=0,
            streaming_count=num_layers,
            resident_bytes=0,
            streaming_bytes=total_model_bytes,
            available_bytes=usable,
        )

    resident_count = min(num_layers, usable // layer_weight_bytes)
    streaming_count = num_layers - resident_count
    return ResidencyPlan(
        resident_count=resident_count,
        streaming_count=streaming_count,
        resident_bytes=resident_count * layer_weight_bytes,
        streaming_bytes=streaming_count * layer_weight_bytes,
        available_bytes=usable,
    )
