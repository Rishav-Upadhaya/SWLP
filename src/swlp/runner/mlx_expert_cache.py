"""MLX expert cache + cached SwitchGLU for MoE streaming on Apple Silicon (Phase 30).

Why MLX: the torch path (``runner/experts.py``) costs 3.84 ms/layer of kernel
launch + sync overhead at 100% cache hits (measured, Qwen3.6-35B-A3B dims,
top-8) — ~154 ms/token for 40 layers, a ~6 tok/s ceiling before any I/O.
A lazy per-expert MLX loop over the same work measures 24.9 ms/token
including the per-layer routing sync.

Why one small array per expert: a partial cache cannot be a stacked
``[slots, I, H]`` MLX array — ``slots[i] = w`` copies the whole array
(5.3 ms per matrix at 160 slots). Stack-then-``gather`` per layer is 2x slower
than the per-expert loop. So each cached expert is its own ``(gate, up, down)``
triple in one global byte-budgeted cache.

Why LFU, not LRU: a decode token sweeps top-k experts across every layer in
order, so the reuse distance of an expert is ~one full token of routed
experts. Below that size LRU evicts everything before it is reused —
replayed OLMoE routing traces: LRU 0.0% hits at 8% and 12% of experts cached
vs LFU 24.7% / 31.2% (Belady optimum 43.8% / 54.2%); LFU >= LRU at every size
up to 70%. Eviction pops a lazily-invalidated heap: O(log n) per miss.

Misses are whole-expert ranged ``pread`` s (F_NOCACHE — this cache *is* the
cache; a page-cache copy would hold the same bytes twice) fanned out over a
thread pool. Output matches ``mlx_lm`` ``SwitchGLU`` math: per (token, k),
``down(silu(gate x) * up x)``, returned as ``[..., K, H]`` for the caller's
score-weighted sum.
"""
from __future__ import annotations

import heapq
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import numpy as np

from ..core.shard_io import _set_nocache
from ..model.expert_bank import ExpertIndex, ExpertLayerIndex

Key = tuple[int, int]
# Full precision: {"gate" [I,H], "up" [I,H], "down" [H,I]}. Quantized (MLX
# checkpoints): {"<proj>.<part>"} with packed "weight" + "scales"/"biases".
Weights = dict[str, mx.array]
# slot → (raw bytes, safetensors dtype, rows, cols), staged by a worker thread.
Raw = dict[str, tuple[np.ndarray, str, int, int]]

# safetensors dtype → (numpy carrier dtype, mlx dtype). bf16 has no numpy
# dtype, so it travels as uint16 and is reinterpreted with a zero-copy view.
_DTYPES: dict[str, tuple[type, mx.Dtype]] = {
    "BF16": (np.uint16, mx.bfloat16),
    "F16": (np.float16, mx.float16),
    "F32": (np.float32, mx.float32),
    "U32": (np.uint32, mx.uint32),  # packed quantized weights
}

# Stale heap entries tolerated before a rebuild (lazy invalidation).
_HEAP_SLACK = 4096


class MlxExpertCache:
    """Global LFU cache of routed experts as MLX arrays, under a byte budget.

    Threading rule: worker threads only ``pread`` into NumPy buffers. Every
    MLX array is created, cached and evicted on the compute thread — MLX graph
    construction concurrent with ``mx.eval`` aborts the process.
    """

    def __init__(self, index: ExpertIndex, shard_dir: str | Path, budget_bytes: int,
                 workers: int) -> None:
        self.index = index
        self.shard_dir = Path(shard_dir)
        self.budget_bytes = max(0, int(budget_bytes))
        self._cache: dict[Key, Weights] = {}
        self._nbytes = 0
        # LFU state: access counts outlive eviction (frequency is a property of
        # the routing, not of residency); ties go to the least recently used.
        self._freq: dict[Key, int] = {}
        self._last: dict[Key, int] = {}
        self._tick = 0
        self._heap: list[tuple[int, int, Key]] = []
        self._inflight: dict[Key, Future[Raw]] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max(1, workers),
                                        thread_name_prefix="swlp-mlx-expert")
        self._fds: dict[str, int] = {}
        self.hits = 0
        self.prefetch_hits = 0
        self.misses = 0
        self.evictions = 0
        self.read_wait_seconds = 0.0  # compute thread blocked on expert reads
        # (group_size, bits): quantize full-precision experts as they are
        # cached (quantize-on-load; lossy, opt-in). None keeps them exact.
        self.quant: tuple[int, int] | None = None

    # ── public API (compute thread only) ──────────────────────────────────

    def get_many(self, layer: int, eids: list[int]) -> dict[int, Weights]:
        """Return all of ``eids`` for ``layer``; every miss is read in parallel."""
        self._absorb_finished()
        pending: dict[int, Future[Raw]] = {}
        out: dict[int, Weights] = {}
        with self._lock:
            for eid in eids:
                key = (layer, eid)
                w = self._cache.get(key)
                if w is not None:
                    self._touch(key)
                    self.hits += 1
                    out[eid] = w
                elif key in self._inflight:
                    self.prefetch_hits += 1
                    pending[eid] = self._inflight[key]
                else:
                    self.misses += 1
                    pending[eid] = self._submit(key)
        started = time.perf_counter()
        for eid, fut in pending.items():
            out[eid] = self._materialize((layer, eid), fut.result(), used=True)
        self.read_wait_seconds += time.perf_counter() - started
        return out

    def prefetch(self, layer: int, eids: list[int]) -> None:
        """Start background reads for experts not cached or already in flight."""
        with self._lock:
            for eid in eids:
                key = (layer, eid)
                if key not in self._cache and key not in self._inflight:
                    self._submit(key)

    def stats(self) -> dict[str, int | float]:
        total = self.hits + self.prefetch_hits + self.misses
        return {
            "hits": self.hits,
            "prefetch_hits": self.prefetch_hits,
            "misses": self.misses,
            "hit_rate": (self.hits + self.prefetch_hits) / total if total else 0.0,
            "evictions": self.evictions,
            "cached_experts": len(self._cache),
            "cached_bytes": self._nbytes,
            "read_wait_seconds": round(self.read_wait_seconds, 3),
        }

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
        self._inflight.clear()
        self._cache.clear()
        self._heap.clear()
        self._nbytes = 0

    # ── internals ─────────────────────────────────────────────────────────

    def _submit(self, key: Key) -> Future[Raw]:  # caller holds _lock
        fut = self._pool.submit(self._read, *key)
        self._inflight[key] = fut
        return fut

    def _absorb_finished(self) -> None:
        """Move completed prefetches (incl. mispredicted ones) into the LRU so
        staged NumPy buffers never pile up outside the byte budget."""
        with self._lock:
            done = [(k, f) for k, f in self._inflight.items() if f.done()]
        for key, fut in done:
            if fut.exception() is None:
                self._materialize(key, fut.result(), used=False)
            else:
                with self._lock:
                    self._inflight.pop(key, None)

    def _materialize(self, key: Key, raw: Raw, *, used: bool) -> Weights:
        """Cache freshly read bytes; ``used`` counts it as an access (a real
        routing request), unlike an absorbed prefetch."""
        with self._lock:
            self._inflight.pop(key, None)
            w = self._cache.get(key)
        if w is not None:
            return w
        w = _to_mx(raw, self.index.layers[key[0]])
        if self.quant is not None and "gate" in w:
            w = _quantize(w, *self.quant)
        # Insert BEFORE pushing: a heap rebuild inside _push enumerates
        # _cache, and a key missing from it would never be evictable.
        self._cache[key] = w
        self._nbytes += sum(a.nbytes for a in w.values())
        if used:
            self._touch(key)
        else:
            self._push(key)
        # Evicting only drops the cache's reference: an expert still used by
        # an unevaluated graph stays alive until that graph runs (so the
        # newcomer itself may be the LFU victim — it is still returned).
        while self._nbytes > self.budget_bytes and self._cache:
            if not self._evict_one():
                self._rebuild_heap()
                if not self._evict_one():
                    break
        return w

    def _touch(self, key: Key) -> None:
        self._freq[key] = self._freq.get(key, 0) + 1
        self._push(key)

    def _push(self, key: Key) -> None:
        self._tick += 1
        self._last[key] = self._tick
        heapq.heappush(self._heap, (self._freq.get(key, 0), self._tick, key))
        if len(self._heap) > 4 * len(self._cache) + _HEAP_SLACK:
            self._rebuild_heap()

    def _rebuild_heap(self) -> None:
        self._heap = [(self._freq.get(k, 0), self._last.get(k, 0), k) for k in self._cache]
        heapq.heapify(self._heap)

    def _evict_one(self) -> bool:
        while self._heap:
            _, tick, key = heapq.heappop(self._heap)
            if key in self._cache and tick == self._last.get(key, 0):
                old = self._cache.pop(key)
                self._nbytes -= sum(a.nbytes for a in old.values())
                self.evictions += 1
                return True
        return False

    def _fd(self, path: str) -> int:  # caller holds _lock
        fd = self._fds.get(path)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
            _set_nocache(fd)
            self._fds[path] = fd
        return fd

    def _read(self, layer: int, eid: int) -> Raw:
        """Worker thread: pure I/O — whole-slice preads into NumPy buffers."""
        li = self.index.layers[layer]
        raw: Raw = {}
        for sl in li.slices[eid]:
            with self._lock:
                fd = self._fd(sl.file or str(self.shard_dir / li.bank_file))
            buf = np.empty(sl.nbytes, dtype=np.uint8)
            _pread_into(fd, memoryview(buf), sl.offset)
            raw[sl.slot] = (buf, sl.dtype_str, sl.rows, sl.cols)
        return raw


def _to_mx(raw: Raw, li: ExpertLayerIndex) -> Weights:
    """Compute thread: NumPy bytes → MLX arrays (see ``Weights``)."""
    parts: dict[str, mx.array] = {}
    for slot, (buf, dtype_str, rows, cols) in raw.items():
        carrier, mx_dtype = _DTYPES[dtype_str]
        parts[slot] = mx.array(buf.view(carrier)).view(mx_dtype).reshape(rows, cols)
    if "gate" not in parts:  # quantized MLX layout: kept packed as stored
        return parts
    h, i = li.hidden, li.intermediate
    if parts["gate"].shape == (h, i) and h != i:  # mixtral-stacked [H, I] orientation
        parts["gate"], parts["up"] = parts["gate"].T, parts["up"].T
    if parts["down"].shape == (i, h) and h != i:
        parts["down"] = parts["down"].T
    return parts


def _quantize(w: Weights, group_size: int, bits: int) -> Weights:
    """Full-precision expert → packed MLX layout (``"<proj>.weight|scales|biases"``),
    the same ``mx.quantize`` affine math ``mlx_lm.convert`` applies."""
    out: Weights = {}
    for proj, mat in w.items():
        if mat.shape[-1] % group_size:
            return w  # width not divisible by the group: keep this expert exact
        out[f"{proj}.weight"], out[f"{proj}.scales"], out[f"{proj}.biases"] = mx.quantize(
            mat, group_size=group_size, bits=bits)
    return out


def _pread_into(fd: int, view: memoryview, offset: int) -> None:
    got = 0
    while got < len(view):
        n = os.preadv(fd, [view[got:]], offset + got)
        if not n:
            raise OSError(f"short expert read at offset {offset + got}")
        got += n
