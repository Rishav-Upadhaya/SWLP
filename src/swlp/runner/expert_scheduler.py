"""Global expert-cache scheduler for MoE streaming (Phase 25).

One :class:`ExpertScheduler` serves every layer's :class:`SwlpCachedExperts`
module: a single global LRU across ``(layer, expert)`` keys under a byte
budget, a small prefetch pool that stages predicted experts, and elastic
budget resizing at generation safe points. Prediction uses
:class:`swlp.core.moe_policy.RoutingHistory`; prefetch admission
path uses the FreeToken q-star split (arXiv:2608.16157), while unified memory —
no second execution site — admits every candidate.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ..core.moe_policy import RoutingHistory
from ..model.expert_bank import ExpertIndex, read_expert

if TYPE_CHECKING:
    from .experts import SwlpCachedExperts

LOGGER = logging.getLogger(__name__)

_STAGING_PER_WORKER = 2
_VALID_MODES = ("off", "lru", "predictive")


class ExpertScheduler:
    """Global recency-tracked expert cache with predictive prefetch."""

    def __init__(
        self,
        index: ExpertIndex,
        shard_dir: str | Path,
        device: torch.device,
        dtype: torch.dtype,
        budget_bytes: int,
        *,
        mode: str = "predictive",
        workers: int = 2,
    ) -> None:


        self.index = index
        self.shard_dir = Path(shard_dir)
        self.device = device
        self.dtype = dtype
        self.budget_bytes = max(0, int(budget_bytes))
        if mode not in _VALID_MODES:
            raise ValueError(
                f"invalid swlp_expert_prefetch mode {mode!r}; "
                f"expected one of {_VALID_MODES}"
            )
        self.mode = mode
        self._modules: dict[int, SwlpCachedExperts] = {}
        self._lru: deque[tuple[int, int]] = deque()
        self._staging: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._staging_order: deque[tuple[int, int]] = deque()
        self._inflight: set[tuple[int, int]] = set()
        self._lock = threading.Lock()
        # Signalled on every stage/eviction; lets ensure() block on a
        # specific key without sleep-polling (poll granularity dominated
        # ms-scale SSD reads in the first cut of this mechanism).
        self._stage_cv = threading.Condition(self._lock)
        self._workers = max(1, workers)
        self._pool = ThreadPoolExecutor(max_workers=self._workers,
                                        thread_name_prefix="swlp-expert")
        self._history = RoutingHistory()
        self._hits = 0
        self._misses = 0
        self._prefetch_hits = 0
        self._evictions = 0

    # ── wiring ────────────────────────────────────────────────────────────

    def register(self, layer: int, module: SwlpCachedExperts) -> None:
        self._modules[layer] = module

    def expert_bytes(self, layer: int) -> int:
        li = self.index.layers.get(layer)
        return li.expert_bytes() if li else 0

    def slot_cap(self, layer: int) -> int:
        """Per-layer slot capacity; budget_bytes=0 means unbounded (all experts)."""
        li = self.index.layers.get(layer)
        if li is None:
            return 0
        per_expert = li.expert_bytes()
        if per_expert <= 0 or self.budget_bytes <= 0:
            return li.num_experts
        per_layer_budget = self.budget_bytes // max(1, len(self.index.layers))
        return max(1, min(li.num_experts, per_layer_budget // per_expert))

    def set_budget(self, budget_bytes: int) -> None:
        """Elastic resize at a generation safe point (between sweeps)."""
        self.budget_bytes = max(0, int(budget_bytes))
        for layer, module in self._modules.items():
            module.resize(self.slot_cap(layer))
        LOGGER.info("expert_budget_resized", extra={"budget_bytes": self.budget_bytes})

    # ── cache paths ───────────────────────────────────────────────────────

    def _touch(self, layer: int, eid: int) -> None:
        key = (layer, eid)
        with self._lock:
            try:
                self._lru.remove(key)
            except ValueError:
                pass
            self._lru.append(key)

    def _evict_least_recent_in_layer(self, layer: int) -> None:
        """Free one slot of ``layer``'s module by evicting its least-recently-
        used expert (recency order: leftmost in the global LRU deque)."""
        module = self._modules.get(layer)
        if module is None:
            return
        with self._lock:
            for lay, eid in self._lru:
                if lay == layer and module.has(eid):
                    module.evict(eid)
                    self._evictions += 1
                    try:
                        self._lru.remove((lay, eid))
                    except ValueError:
                        pass
                    return
        # No recency record (e.g. budget=1, filled this step): fall back to
        # the module's own occupancy order.
        module.evict_lru_local()
        with self._lock:
            self._evictions += 1

    def _stage(self, key: tuple[int, int], gate_up: torch.Tensor, down: torch.Tensor) -> None:
        with self._stage_cv:
            self._staging[key] = (gate_up, down)
            self._staging_order.append(key)
            self._inflight.discard(key)
            while len(self._staging_order) > _STAGING_PER_WORKER * self._workers:
                old = self._staging_order.popleft()
                self._staging.pop(old, None)
            self._stage_cv.notify_all()

    def prefetch(self, layer: int, eid: int) -> None:
        key = (layer, eid)
        module = self._modules.get(layer)
        if module is None:
            return
        with self._lock:
            if module.has(eid) or key in self._staging or key in self._inflight:
                return
            self._inflight.add(key)

        def _job() -> None:
            try:
                gate_up, down = _load_expert(self, layer, eid)
                self._stage(key, gate_up, down)
            except Exception:
                LOGGER.exception("expert_prefetch_failed",
                                 extra={"layer": layer, "expert": eid})
                with self._lock:
                    self._inflight.discard(key)

        try:
            self._pool.submit(_job)
        except RuntimeError:
            with self._lock:
                self._inflight.discard(key)

    def prepare_set(self, layer: int, expert_ids) -> None:
        """Stage this token's routed experts CONCURRENTLY before consumption.

        ``SwlpCachedExperts.forward`` consumes experts one at a time; without
        staging, each miss is a serial disk round-trip on the compute thread.
        Fanning the whole routed set into the prefetch pool up front turns K
        serial reads into ⌈K/workers⌉ overlapped ones (SP-MoE-style batched
        I/O; workers stay few — MoE-Infinity shows concurrent fetches contend
        for bandwidth). Dedup vs cached/staged/in-flight keeps it idempotent.
        """
        if self.mode == "off" or layer not in self._modules:
            return
        for eid in set(int(e) for e in expert_ids):
            self.prefetch(layer, eid)

    def ensure(self, layer: int, eid: int) -> int:
        """Return the slot holding ``eid``, filling it synchronously on miss."""
        module = self._modules[layer]
        slot = module.slot_of(eid)
        if slot is not None:
            with self._lock:
                self._hits += 1
            self._touch(layer, eid)
            return slot
        with self._lock:
            self._misses += 1
        staged = None
        with self._stage_cv:
            if (layer, eid) in self._inflight:
                # prepare_set already dispatched a parallel fetch; wait for
                # THAT read instead of issuing a duplicate serial one.
                deadline = time.monotonic() + 5.0
                while (layer, eid) in self._inflight:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._stage_cv.wait(remaining)
                    if (layer, eid) in self._staging:
                        staged = self._staging.pop((layer, eid))
                        break
                if staged is None and (layer, eid) in self._staging:
                    staged = self._staging.pop((layer, eid))
        if staged is not None:
            with self._lock:
                self._prefetch_hits += 1
            gate_up, down = staged
        else:
            gate_up, down = _load_expert(self, layer, eid)
        while module.free_slots() == 0:
            self._evict_least_recent_in_layer(layer)
            if module.free_slots() == 0:
                break  # single-slot module holding the requested expert itself
        module.fill(eid, gate_up, down)
        self._touch(layer, eid)
        return module.slot_of(eid) or 0

    # ── prediction ────────────────────────────────────────────────────────

    def record_routing(self, layer: int, ids) -> None:
        if self.mode == "off":
            return
        self._history.record(layer, ids)
        if self.mode != "predictive" or layer + 1 not in self._modules:
            return
        for eid in self._history.predict(layer + 1, self._admit(len(ids))):
            self.prefetch(layer + 1, eid)

    def prepare_layer(self, layer: int) -> None:
        if self.mode != "predictive" or layer not in self._modules:
            return
        for eid in self._history.predict(layer, self._admit(1)):
            self.prefetch(layer, eid)

    def _admit(self, candidates: int) -> int:
        """How many missing experts to stage concurrently.

        On unified memory every miss is a read from the same pool at the same
        bandwidth, so there is no placement decision to make — only a queue
        depth. Stage what the workers can absorb, bounded by what is actually
        needed. (The CUDA build used FreeToken's q* split here; that models a
        host-bus/device-bus division that does not exist on Apple Silicon.)
        """
        return max(1, min(max(1, candidates), _STAGING_PER_WORKER * self._workers))

    def stats(self) -> dict[str, int | float]:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total else 0.0,
            "prefetch_hits": self._prefetch_hits,
            "evictions": self._evictions,
            "staged": len(self._staging),
        }

    def cleanup(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)
        self._staging.clear()
        self._staging_order.clear()
        self._history.reset()
        for module in self._modules.values():
            module.reset_slots()
        self._modules.clear()
        self._lru.clear()


def _load_expert(sched: ExpertScheduler, layer: int, eid: int) -> tuple[torch.Tensor, torch.Tensor]:
    parts = read_expert(sched.index, sched.shard_dir, layer, eid)
    gate, up, down = parts["gate"], parts["up"], parts["down"]
    if gate.dtype != sched.dtype:
        raise RuntimeError(
            f"expert bank dtype {gate.dtype} != compute dtype {sched.dtype} "
            f"(layer {layer}, expert {eid}); re-shard without quantization"
        )
    gate_up = torch.cat([gate, up], dim=0).to(device=sched.device, dtype=sched.dtype)
    down_d = down.to(device=sched.device, dtype=sched.dtype)
    return gate_up, down_d
