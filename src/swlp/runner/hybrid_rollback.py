"""Exact state rollback for hybrid (Gated-DeltaNet + attention) caches.

Speculative verification feeds ``[last_token, *draft]`` (K+1 tokens) through
every layer in one sweep, then must forget the rejected tail. Attention layers
just drop the tail of their K/V. Linear-attention layers cannot: their
recurrent state is a fixed-size summary of *all* tokens seen, and
``LinearAttentionLayer.crop`` refuses (``is_croppable`` is False once a
recurrent state exists). Left alone, the DeltaNet state silently absorbs the
rejected tokens and every later token is computed from a corrupted state.

The fix records, per linear-attention layer during the verify sweep, what is
needed to recompute the state for any accepted prefix:

- the causal-conv input ``x = cat(old_conv_state, new_inputs)`` — the conv
  state after keeping ``m`` new tokens is exactly ``x[..., m : m + kernel]``;
- the delta-rule inputs ``(q, k, v, g, beta)`` and a *clone* of the incoming
  recurrent state (the cache ``copy_``s the new state into the same tensor, so
  without a clone the pre-verify state is gone).

On rejection, the recurrent state is recomputed over the first ``m`` positions
with the token-by-token ``torch_recurrent_gated_delta_rule`` — the same kernel
plain decode uses — and copied back in place.

Recording works by temporarily wrapping the two module-level functions the
HF ``GatedDeltaNet.forward`` resolves at call time (``causal_conv1d_fn`` and
``torch_chunk_gated_delta_rule``). Calls arrive in layer order, one per
linear-attention layer per sweep.
"""
from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import ModuleType

import torch
from torch import nn
from transformers.cache_utils import Cache, LinearAttentionCacheLayerMixin

_CONV_FN = "causal_conv1d_fn"
_CHUNK_FN = "torch_chunk_gated_delta_rule"
_RECURRENT_FN = "torch_recurrent_gated_delta_rule"


@dataclass
class DeltaCall:
    """Inputs of one chunked gated-delta-rule call (one linear layer)."""

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    initial_state: torch.Tensor | None


@dataclass
class LinearStateRecord:
    """Everything a verify sweep fed the linear-attention layers, in layer order."""

    module: ModuleType
    conv_inputs: list[torch.Tensor] = field(default_factory=list)
    delta_calls: list[DeltaCall] = field(default_factory=list)


def is_hybrid_cache(cache: Cache) -> bool:
    """True when ``cache`` holds any linear-attention (recurrent) layer."""
    layers = getattr(cache, "layers", None) or []
    return any(isinstance(layer, LinearAttentionCacheLayerMixin) for layer in layers)


def _modeling_module(model: nn.Module) -> ModuleType:
    """The HF modeling module whose globals the DeltaNet forward resolves."""
    module = sys.modules[type(model).__module__]
    missing = [n for n in (_CONV_FN, _CHUNK_FN, _RECURRENT_FN) if not hasattr(module, n)]
    if missing:
        raise RuntimeError(
            f"{module.__name__} has no {missing}; hybrid-cache rollback supports "
            "Gated-DeltaNet models (qwen3_5) only"
        )
    return module


@contextmanager
def record_linear_attention(model: nn.Module) -> Iterator[LinearStateRecord]:
    """Record every linear layer's conv / delta-rule inputs inside the block.

    Not thread-safe: the wrapped functions are module globals. That is fine
    while blocks compute on one thread (scheduler threads only load weights).
    """
    module = _modeling_module(model)
    record = LinearStateRecord(module=module)
    conv_fn: Callable[..., torch.Tensor] = getattr(module, _CONV_FN)
    chunk_fn: Callable[..., tuple[torch.Tensor, torch.Tensor | None]] = getattr(
        module, _CHUNK_FN
    )

    def conv_recorder(hidden_states: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        record.conv_inputs.append(hidden_states)
        return conv_fn(hidden_states, *args, **kwargs)

    def chunk_recorder(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        initial = kwargs.get("initial_state")
        assert isinstance(kwargs["g"], torch.Tensor) and isinstance(kwargs["beta"], torch.Tensor)
        record.delta_calls.append(
            DeltaCall(
                query, key, value, kwargs["g"], kwargs["beta"],
                initial.clone() if isinstance(initial, torch.Tensor) else None,
            )
        )
        return chunk_fn(query, key, value, **kwargs)

    setattr(module, _CONV_FN, conv_recorder)
    setattr(module, _CHUNK_FN, chunk_recorder)
    try:
        yield record
    finally:
        setattr(module, _CONV_FN, conv_fn)
        setattr(module, _CHUNK_FN, chunk_fn)


def rollback_hybrid_cache(
    cache: Cache, record: LinearStateRecord, keep: int, fed: int
) -> None:
    """Rewind ``cache`` so it covers only the first ``keep`` of ``fed`` tokens.

    Attention layers drop their last ``fed - keep`` K/V positions; linear
    layers get their conv and recurrent state recomputed for the kept prefix.
    """
    remove = fed - keep
    if remove <= 0:
        return
    if not 1 <= keep <= fed:
        raise ValueError(f"keep={keep} out of range for fed={fed}")
    linear = [layer for layer in cache.layers if isinstance(layer, LinearAttentionCacheLayerMixin)]
    if len(record.conv_inputs) != len(linear) or len(record.delta_calls) != len(linear):
        raise RuntimeError(
            f"recorded {len(record.conv_inputs)} conv / {len(record.delta_calls)} delta "
            f"calls for {len(linear)} linear-attention layers — sweep was not recorded whole"
        )
    recurrent_fn = getattr(record.module, _RECURRENT_FN)
    for layer in cache.layers:
        if not isinstance(layer, LinearAttentionCacheLayerMixin):
            layer.crop(-remove)
    for layer, x, call in zip(linear, record.conv_inputs, record.delta_calls, strict=True):
        conv_state = layer.conv_states[0]
        kernel = int(conv_state.shape[-1])
        conv_state.copy_(x[..., keep : keep + kernel])
        _, state = recurrent_fn(
            call.query[:, :keep], call.key[:, :keep], call.value[:, :keep],
            g=call.g[:, :keep], beta=call.beta[:, :keep],
            initial_state=call.initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        layer.recurrent_states[0].copy_(state)
