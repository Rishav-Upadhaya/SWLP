"""CachedSwitchGLU — ``mlx_lm`` ``SwitchGLU`` drop-in over :class:`MlxExpertCache` (Phase 30).

Split from ``runner/mlx_expert_cache.py``: that module owns residency (policy +
I/O); this one owns the expert compute. See the cache module for the measured
rationale (per-expert arrays, LFU, workers-do-only-I/O).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .mlx_expert_cache import MlxExpertCache, Weights


class CachedSwitchGLU(nn.Module):
    """Drop-in for ``mlx_lm`` ``SwitchGLU``: ``(x [..., H], inds [..., K]) → [..., K, H]``.

    Reads routed experts from :class:`MlxExpertCache`. The routing indices are
    evaluated here (one small sync per layer) because the experts to read are
    only known once routing has run. When ``next_gate`` is set, the next MoE
    layer's router is applied to this layer's input to prefetch its likely
    experts while this layer computes (prediction only drives reads — outputs
    always use the true routing).
    """

    def __init__(self, layer: int, cache: MlxExpertCache, top_k: int,
                 activation: nn.Module | None = None,
                 quant: tuple[int, int, str] | None = None) -> None:
        super().__init__()
        self._layer = layer
        self._cache = cache
        self._top_k = top_k
        # The replaced SwitchGLU's own activation (SwiGLU, GeGLU, …), called as
        # act(x_up, x_gate). Bound method, not the Module (see _next_gate).
        self._act: Callable[[mx.array, mx.array], mx.array] = (
            activation.__call__ if activation is not None else _swiglu
        )
        # Quantized experts: (group_size, bits, mode) as plain scalars.
        self._q_group, self._q_bits, self._q_mode = quant if quant else (0, 0, "")
        # Plain int + bound method, never a tuple/Module: MLX's Module is a
        # dict and would adopt either as a child, duplicating the next
        # layer's router in this module's parameter tree.
        self._next_layer: int | None = None
        self._next_gate: Callable[[mx.array], mx.array] | None = None

    def set_next_router(self, layer: int, gate: nn.Module) -> None:
        self._next_layer = layer
        self._next_gate = gate.__call__

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        lead = x.shape[:-1]
        k = indices.shape[-1]
        flat_x = x.reshape(-1, x.shape[-1])
        flat_i = indices.reshape(-1, k)
        if self._next_gate is not None and self._next_layer is not None:
            logits = self._next_gate(flat_x)
            pred = mx.argpartition(logits, kth=-self._top_k, axis=-1)[..., -self._top_k:]
            mx.eval(flat_i, pred)
            self._cache.prefetch(self._next_layer, sorted(set(np.array(pred).ravel().tolist())))
        ids = np.array(flat_i)
        experts = self._cache.get_many(self._layer, sorted(set(ids.ravel().tolist())))
        n_tok = ids.shape[0]
        if n_tok == 1:
            ys = [self._expert_out(flat_x, experts[int(e)]) for e in ids[0]]
            out = mx.concatenate(ys, axis=0)[None]  # [1, K, H]
        else:
            out = _routed_outputs(flat_x, ids, experts, self._expert_out)
        return out.reshape(*lead, k, x.shape[-1])


    def _expert_out(self, x: mx.array, w: Weights) -> mx.array:
        """One expert: ``down(act(up x, gate x))`` — dense or quantized matmuls
        (``quantized_matmul`` dequantizes exactly as the stock ``gather_qmm``)."""
        if not self._q_bits:
            return self._act(x @ w["up"].T, x @ w["gate"].T) @ w["down"].T

        def qmm(inp: mx.array, proj: str) -> mx.array:
            return mx.quantized_matmul(
                inp, w[f"{proj}.weight"], w[f"{proj}.scales"], w.get(f"{proj}.biases"),
                transpose=True, group_size=self._q_group, bits=self._q_bits,
                mode=self._q_mode,
            )

        return qmm(self._act(qmm(x, "up"), qmm(x, "gate")), "down")


def _swiglu(x_up: mx.array, x_gate: mx.array) -> mx.array:
    return nn.silu(x_gate) * x_up


def _routed_outputs(x: mx.array, ids: np.ndarray, experts: dict[int, Weights],
                    expert_out: Callable[[mx.array, Weights], mx.array]) -> mx.array:
    """Prefill: each expert runs once over all tokens routed to it; rows are
    put back in (token, k) order with a single gather."""
    n_tok, k = ids.shape
    chunks: list[mx.array] = []
    order: list[np.ndarray] = []
    for eid, w in experts.items():
        tok, slot = np.nonzero(ids == eid)
        chunks.append(expert_out(x[mx.array(tok)], w))
        order.append(tok * k + slot)
    flat_pos = np.concatenate(order)
    inverse = np.empty_like(flat_pos)
    inverse[flat_pos] = np.arange(flat_pos.size)
    stacked = mx.concatenate(chunks, axis=0)
    return stacked[mx.array(inverse)].reshape(n_tok, k, -1)


class MmapEmbedding(nn.Module):
    """Token-embedding lookup from a CPU mmap-backed torch table.

    Replaces ``nn.Embedding`` for untied models so the vocabulary table never
    enters the Metal working set; only the looked-up rows are copied into MLX.
    The torch tensor is a plain attribute (not an ``mx.array``), so it is not a
    parameter and ``load_weights`` never materializes it.
    """

    def __init__(self, table: Any) -> None:
        super().__init__()
        self._table = table  # torch.Tensor, mmap-backed; keeps the mapping alive

    def __call__(self, ids: mx.array) -> mx.array:
        import torch

        idx = torch.from_numpy(np.array(ids, dtype=np.int64).reshape(-1))
        rows = self._table.index_select(0, idx)
        if rows.dtype == torch.bfloat16:
            out = mx.array(rows.view(torch.int16).numpy()).view(mx.bfloat16)
        else:
            out = mx.array(rows.numpy())
        return out.reshape(*ids.shape, rows.shape[-1])
