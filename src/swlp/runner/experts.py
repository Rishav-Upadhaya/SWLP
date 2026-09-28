"""Slot-cached fused-Experts replacement for MoE layers (Phase 25).

:class:`SwlpCachedExperts` swaps in for a transformers ≥5 fused ``Experts``
module (``…mlp.experts`` / ``…block_sparse_moe.experts``). Its ``forward``
replicates the reference loop — one_hot mask, ascending expert iteration,
fused gate_up linear + chunk, activation, down linear, routing-weight scale,
``index_add_`` — reading weights from a bounded slot cache, so results are
bit-identical to a fully materialized module while only the routed experts
ever occupy memory. Slot fills and evictions are driven by the global
:class:`swlp.runner.expert_scheduler.ExpertScheduler`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from .expert_scheduler import ExpertScheduler


def _act_fn_for(name: str):
    from transformers.activations import ACT2FN

    return ACT2FN[name]


class SwlpCachedExperts(nn.Module):
    """Drop-in fused-Experts replacement with a bounded slot cache."""

    # Marker consumed by core/streaming.py (which must not import runner/):
    # children flagged True are detached before block.to_empty(device="meta")
    # so layer eviction never destroys live expert slots.
    _swlp_preserve_on_meta = True

    def __init__(self, layer: int, num_experts: int, hidden: int, intermediate: int,
                 dtype: torch.dtype, device: torch.device, act_fn: str,
                 scheduler: ExpertScheduler, slots: int) -> None:
        super().__init__()
        self.layer = layer
        self.num_experts = num_experts
        self.hidden = hidden
        self.intermediate = intermediate
        self.act = _act_fn_for(act_fn)
        self._sched = scheduler
        self._slot_of: dict[int, int] = {}
        self._id_of: list[int | None] = [None] * max(1, slots)
        with torch.no_grad():
            self.gate_up_slots = nn.Parameter(
                torch.empty(max(1, slots), 2 * intermediate, hidden, dtype=dtype, device=device),
                requires_grad=False,
            )
            self.down_slots = nn.Parameter(
                torch.empty(max(1, slots), hidden, intermediate, dtype=dtype, device=device),
                requires_grad=False,
            )

    # ── slot bookkeeping ──────────────────────────────────────────────────

    @property
    def capacity(self) -> int:
        return self.gate_up_slots.shape[0]

    def slot_of(self, eid: int) -> int | None:
        return self._slot_of.get(eid)

    def has(self, eid: int) -> bool:
        return eid in self._slot_of

    def free_slots(self) -> int:
        return self.capacity - len(self._slot_of)

    def fill(self, eid: int, gate_up: torch.Tensor, down: torch.Tensor) -> None:
        slot = self._next_free_slot()
        with torch.no_grad():
            self.gate_up_slots.data[slot] = gate_up
            self.down_slots.data[slot] = down
        self._slot_of[eid] = slot
        self._id_of[slot] = eid

    def _next_free_slot(self) -> int:
        for slot, eid in enumerate(self._id_of):
            if eid is None:
                return slot
        raise RuntimeError("no free expert slot — evict before filling")

    def evict(self, eid: int) -> None:
        """Free the slot holding ``eid`` (weights zeroed to release refs)."""
        slot = self._slot_of.pop(eid, None)
        if slot is not None:
            self._id_of[slot] = None
            with torch.no_grad():
                self.gate_up_slots.data[slot].zero_()
                self.down_slots.data[slot].zero_()

    def evict_lru_local(self) -> None:
        """Evict the longest-resident expert of this layer (fallback path)."""
        if self._slot_of:
            self.evict(next(iter(self._slot_of)))

    def reset_slots(self) -> None:
        self._slot_of.clear()
        self._id_of = [None] * self.capacity

    def resize(self, new_cap: int) -> None:
        """Grow/shrink the slot arrays, preserving live experts (LRU-trimmed).

        Live experts are compacted into the lowest slots of the new arrays —
        high-index slots of the old arrays may hold live entries, so the copy
        iterates the full old occupancy map, not just its first ``new_cap``
        entries.
        """
        new_cap = max(1, int(new_cap))
        if new_cap == self.capacity:
            return
        while len(self._slot_of) > new_cap:
            self.evict_lru_local()
        old_gu, old_dn = self.gate_up_slots.data, self.down_slots.data
        old_ids = list(self._id_of)
        with torch.no_grad():
            self.gate_up_slots = nn.Parameter(
                torch.empty(new_cap, *old_gu.shape[1:], dtype=old_gu.dtype,
                            device=old_gu.device),
                requires_grad=False,
            )
            self.down_slots = nn.Parameter(
                torch.empty(new_cap, *old_dn.shape[1:], dtype=old_dn.dtype,
                            device=old_dn.device),
                requires_grad=False,
            )
            remap: dict[int, int] = {}
            for slot, eid in enumerate(old_ids):
                if eid is None or len(remap) >= new_cap:
                    continue
                self.gate_up_slots.data[len(remap)] = old_gu[slot]
                self.down_slots.data[len(remap)] = old_dn[slot]
                remap[eid] = len(remap)
        self._slot_of = remap
        self._id_of = [None] * new_cap
        for eid, slot in remap.items():
            self._id_of[slot] = eid

    # ── forward (bit-exact port of the fused Experts loop) ────────────────

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                top_k_index, num_classes=self.num_experts
            ).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        routed: list[int] = []
        for hit in expert_hit:
            eid = int(hit[0])
            if eid == self.num_experts:
                continue
            if eid not in routed:
                routed.append(eid)
        # Fan ALL misses out to the fetch pool before consuming — turns K
        # serial disk round-trips into ⌈K/workers⌉ overlapped reads.
        self._sched.prepare_set(self.layer, routed)
        for eid in routed:
            slot = self._sched.ensure(self.layer, eid)
            top_k_pos, token_idx = torch.where(expert_mask[eid])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(
                current_state, self.gate_up_slots.data[slot]
            ).chunk(2, dim=-1)
            current_hidden_states = self.act(gate) * up
            current_hidden_states = nn.functional.linear(
                current_hidden_states, self.down_slots.data[slot]
            )
            current_hidden_states = current_hidden_states * top_k_weights[
                token_idx, top_k_pos, None
            ]
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )
        self._sched.record_routing(self.layer, routed)
        return final_hidden_states
