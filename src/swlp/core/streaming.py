"""Disk-streaming scheduler for SWLP.

When ``runtime.shard_dir`` points at a directory of per-layer shards produced
by ``swlp.model.shard.shard_model_by_layer``, ``StreamingScheduler`` loads
layer weights from disk into block modules on demand, keeping at most
``window_size`` layers materialized in RAM at any moment.

This is the SSD-bound path used on Apple Silicon (M5 unified memory). The
in-memory ``ThreadedScheduler`` swaps already-loaded blocks between CPU and
device — irrelevant on unified memory, where the goal is to
avoid holding the full model in RAM at all.

Copy-elimination hot path (Phase 20)
------------------------------------
Per token the old path copied every shard three times on the compute thread
(chunk-join → safetensors deserialize → host-to-device cast) plus a wasted
``to_empty(device)`` allocation that ``assign=True`` immediately replaced.
Now ``_prepare_device_state()`` runs on a persistent worker pool: it reads the
shard once into a reusable per-worker buffer (``shard_io.read_shard_payload``,
which also decompresses ``.swz`` shards), reinterprets zero-copy tensor views
inside the payload, and performs the single host→device copy — so ``ensure()``
on the compute thread is reduced to a pointer-assigning
``load_state_dict(assign=True)``.

Direct I/O policy: ``direct_io=True`` (the safe default for models larger
than RAM) bypasses the macOS page cache via ``F_NOCACHE``; passing ``False``
lets the OS cache shard reads — free residency for models that fit in RAM.

Adaptive Residency (Phase 4)
-----------------------------
``resident_count`` layers (indices 0..resident_count-1) are pre-loaded from
SSD into **CPU RAM** once at startup via ``load_resident_layers()``.  Each
token, ``ensure()`` applies them CPU-RAM → device (fast unified-memory copy,
no SSD read), then ``evict()`` returns them to the meta device as normal.

This is *CPU-RAM residency*, not *MPS-residency*.  Keeping large tensors
permanently in MPS fragments the Metal allocator and causes severe slowdowns
when streaming layers allocate additional Metal buffers.  Holding the state
dicts in CPU RAM instead eliminates MPS pressure while still avoiding the
SSD read for the resident portion of the model each token.
"""
from __future__ import annotations

import io
import logging
import threading
import time
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from queue import Empty, SimpleQueue

import torch

from .. import codec
from ..model.quant import dequantize_layer_state
from ..model.sparse import decode_sparse
from .profiler import LayerProfiler
from .scheduler import PrefetchError, SchedulerConfig
from .shard_io import _read_file_nocache as _read_file_nocache  # noqa: PLC0414 — test surface
from .shard_io import _safetensors_metadata as _safetensors_metadata  # noqa: PLC0414
from .shard_io import (
    load_safetensors_shard,
    nest_fp8_state,
    parse_safetensors_views,
    read_shard_mmap,
    read_shard_payload,
)

LOGGER = logging.getLogger(__name__)


def has_shards(shard_dir: str | Path) -> bool:
    p = Path(shard_dir)
    return p.exists() and (p / "shard_manifest.json").exists()


# "auto" direct I/O allows page-cache reads only while the model occupies at
# most this fraction of currently-available RAM.
_DIRECT_IO_AUTO_FRACTION = 0.6

# At most this many worker threads may run a .swz decompress concurrently.
# zipnn fans out over all cores internally, so a second concurrent decode
# does not add throughput — it triples per-layer latency through CPU and
# memory-bandwidth thrash (measured 95 ms vs 30 ms per layer, Phase 22).
_MAX_CONCURRENT_DECOMPRESS = 1


def resolve_direct_io(mode: str, total_model_bytes: int, available_ram_bytes: int) -> bool:
    """Decide whether shard reads should bypass the OS page cache.

    - ``"on"``:  always bypass — controlled benchmark mode, and the right call
      for models far larger than RAM.
    - ``"off"``: never bypass — let the OS cache shard reads.
    - ``"auto"``: bypass only when the model cannot plausibly stay cached.
      Cyclic layer access through an LRU cache that cannot hold the full model
      gets ~zero hits and only evicts more useful pages.  When the model fits
      comfortably (≤ 60% of available RAM) the page cache gives warm-read
      residency for free — and unlike heap residency (Phase 4) it is
      reclaimable under memory pressure, so it cannot trigger the macOS
      memory-compressor collapse.
    """
    normalized = mode.strip().lower()
    if normalized == "on":
        return True
    if normalized == "off":
        return False
    if normalized != "auto":
        LOGGER.warning("swlp_direct_io_invalid_mode", extra={"mode": mode})
        return True
    if total_model_bytes <= 0 or available_ram_bytes <= 0:
        return True
    return total_model_bytes > _DIRECT_IO_AUTO_FRACTION * available_ram_bytes


def _load_safetensors_shard(path: Path) -> dict:
    """Standalone shard load (fresh buffer; the result owns its memory)."""
    return load_safetensors_shard(path)


def _to_meta_preserving_cached_experts(block: torch.nn.Module) -> None:
    """``to_empty(meta)`` a block while sparing cached-expert descendants.

    Layer eviction must not recurse into ``SwlpCachedExperts`` slot arrays:
    those hold the cross-token expert cache and are managed exclusively by
    the ExpertScheduler (marker attribute, so core/ needs no runner import).
    The module can sit at any depth (qwen3-moe: ``mlp.experts``), so all
    descendants are scanned, not just direct children.
    """
    preserved = [
        (name, module)
        for name, module in block.named_modules()
        if name and getattr(module, "_swlp_preserve_on_meta", False)
    ]
    if not preserved:
        block.to_empty(device="meta")
        return
    detached: list[tuple[torch.nn.Module, str, torch.nn.Module]] = []
    for name, module in preserved:
        parent_name, _, leaf = name.rpartition(".")
        parent = block.get_submodule(parent_name) if parent_name else block
        setattr(parent, leaf, torch.nn.Module())  # placeholder during to_empty
        detached.append((parent, leaf, module))
    block.to_empty(device="meta")
    for parent, leaf, module in detached:
        setattr(parent, leaf, module)


def _cached_expert_param_prefixes(block: torch.nn.Module) -> tuple[str, ...]:
    """Qualified names of scheduler-managed slot modules inside ``block``."""
    return tuple(
        f"{name}."
        for name, module in block.named_modules()
        if name and getattr(module, "_swlp_preserve_on_meta", False)
    )


class StreamingScheduler:
    """Loads layer weights from disk shards into block modules on demand.

    Same surface as ``ThreadedScheduler``: ``prefetch``, ``ensure``, ``evict``,
    ``window_end_index``, ``cleanup``, plus a ``config`` attribute.

    CPU-RAM residency (Phase 4)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    When ``resident_count > 0``, the first ``resident_count`` layers have
    their shard state-dicts pre-loaded into ``_resident_data`` (CPU RAM) once
    via ``load_resident_layers()``.  During inference:

    - ``prefetch(idx)`` for resident layers returns immediately — no disk read.
    - ``ensure(idx)`` for resident layers applies the cached CPU state-dict to
      the device (a fast CPU→MPS unified-memory copy).
    - ``evict(idx)`` for resident layers moves the block back to the meta
      device to reclaim MPS memory, but leaves ``_resident_data[idx]`` intact
      so the next token can re-apply without another SSD read.

    Non-resident layers stream from disk via the worker pool.
    """

    def __init__(
        self,
        blocks: Iterable[torch.nn.Module],
        device: torch.device,
        config: SchedulerConfig,
        shard_dir: str | Path,
        resident_count: int = 0,
        direct_io: bool = True,
        extra_volumes: list[Path] | None = None,
    ) -> None:
        self.blocks = list(blocks)
        self.device = device
        self.config = config
        self.shard_dir = Path(shard_dir)
        # Phase 26: shard striping across volumes. Layers are assigned
        # round-robin to a volume; the read pool then parallelizes across
        # spindles/controllers (internal NVMe + TB4 external ≈ additive GB/s).
        # ``shard_dir`` stays the manifest root; volumes only widen where
        # ``layer_XXX`` files may live.
        self._volumes: list[Path] = [self.shard_dir]
        for vol in extra_volumes or []:
            v = Path(vol)
            if v not in self._volumes:
                self._volumes.append(v)
        self._direct_io = bool(direct_io)
        self._resident_count = max(0, min(resident_count, len(self.blocks)))
        # CPU-RAM cache for resident layers: shard index → state_dict (CPU tensors).
        self._resident_data: dict[int, dict] = {}
        self._loaded: set[int] = set()
        self._futures: dict[int, Future] = {}
        self._lock = threading.Lock()
        self._prefetch_enabled = config.prefetch
        # Persistent worker pool — a fresh thread per prefetch (the old design)
        # cost a thread spawn per layer per token and capped reads at one in
        # flight; the pool also owns the reusable per-worker read buffers.
        self._pool = ThreadPoolExecutor(
            max_workers=max(2, config.prefetch_depth),
            thread_name_prefix="swlp-stream",
        )
        # Second pool for MPS uploads — overlaps SSD reads of layer N+1
        # with upload of layer N, cutting per-layer wall time from
        # (SSD + upload) to max(SSD, upload).
        self._upload_pool = ThreadPoolExecutor(
            max_workers=max(2, config.prefetch_depth),
            thread_name_prefix="swlp-upload",
        )
        self._tls = threading.local()
        # Read-buffer free-list (Phase 24 fix): a buffer is handed out by
        # ``_read_and_deserialize`` and returns here only after its payload's
        # upload completed — reusing it earlier let a worker overwrite bytes
        # an in-flight upload still referenced, silently corrupting weights.
        self._buf_pool: SimpleQueue[torch.Tensor] = SimpleQueue()
        # Serializes .swz decompression across workers (see module constant).
        self._decomp_gate = threading.BoundedSemaphore(_MAX_CONCURRENT_DECOMPRESS)
        # Unified memory has no host→device DMA to overlap, so there is no
        # pinned-buffer tier here: the "upload" is a dtype/device cast that
        # already shares physical pages with the read buffer.
        self._pin = False
        # Overlap-tracking counters (Phase 11).
        # hit  — ensure() found the prefetch future already done (full overlap).
        # wait — ensure() blocked on a still-running future (partial overlap).
        # miss — ensure() fell back to a synchronous read (no prefetch running).
        self._overlap_hits: int = 0
        self._overlap_waits: int = 0
        self._overlap_misses: int = 0
        # Fine-grained pipeline profiler.
        self.profiler: LayerProfiler | None = None

    # ── internal helpers ──────────────────────────────────────────────────────

    def _volume_order(self, idx: int) -> list[Path]:
        """Volumes to probe for layer ``idx`` — its striped volume first."""
        if len(self._volumes) <= 1:
            return self._volumes
        i = idx % len(self._volumes)
        return [self._volumes[i], *self._volumes[:i], *self._volumes[i + 1:]]

    def _shard_path(self, idx: int) -> Path | None:
        # Auto-detect shard format by extension: .safetensors (Phase 17),
        # compressed .safetensors.swz, then legacy .pt — probing the layer's
        # striped volume first, then the remaining volumes.
        for vol in self._volume_order(idx):
            st_path = vol / f"layer_{idx:03d}.safetensors"
            if st_path.exists():
                return st_path
            swz_path = codec.compressed_path(st_path)
            if swz_path.exists():
                return swz_path
            pt_path = vol / f"layer_{idx:03d}.pt"
            if pt_path.exists():
                return pt_path
        LOGGER.warning(
            "streaming_shard_missing",
            extra={"layer": idx, "path": str(self.shard_dir / f"layer_{idx:03d}.safetensors")},
        )
        return None

    def _read_shard(self, idx: int) -> dict | None:
        """Standalone CPU read (fresh buffer per call; result safe to retain).

        Used for resident layers whose state dicts live across tokens. The
        streaming hot path goes through ``_prepare_device_state`` instead.
        """
        path = self._shard_path(idx)
        if path is None:
            return None
        try:
            if path.suffix != ".pt":
                return load_safetensors_shard(path, nocache=self._direct_io)
            data = _read_file_nocache(path, nocache=self._direct_io)
            return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        except Exception:
            LOGGER.exception("streaming_shard_read_failed", extra={"layer": idx})
            return None

    def _to_device_state(self, state: dict) -> dict[str, torch.Tensor]:
        """Dequantize/decode and copy a shard state dict to the device.

        ``copy=True`` is required: the source tensors may be views into a
        reused read buffer, and on same-device (CPU) targets ``.to`` without
        a copy would alias that buffer and be corrupted by the next read.
        """
        state = dequantize_layer_state(state)
        state = decode_sparse(state)  # Phase 13: no-op on dense shards
        out = {
            k: v.to(device=self.device, copy=True, non_blocking=self._pin)
            for k, v in state.items()
        }
        return out

    def _read_and_deserialize(
        self, idx: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None] | None:
        """Stage 1: Read shard from SSD and deserialize to CPU state dict.
        Runs on the SSD worker pool.  Returns CPU tensors plus the reusable
        read buffer backing them — the buffer comes from a free-list and is
        only recycled once :meth:`_upload_to_device` has copied the payload
        out, so a worker may never overwrite bytes an in-flight upload still
        references (that race silently corrupted weights under prefetch).
        """
        path = self._shard_path(idx)
        if path is None:
            return None
        prof = self.profiler
        buffer = None
        try:
            if prof:
                prof.begin_read(idx)
            if path.suffix != ".pt":
                if self._direct_io:
                    try:
                        buffer = self._buf_pool.get_nowait()
                    except Empty:
                        buffer = None  # first use / pool drained → allocate
                    payload, size, buffer = read_shard_payload(
                        path,
                        buffer=buffer,
                        nocache=True,
                        pin=self._pin,
                        decompress_gate=self._decomp_gate,
                    )
                else:
                    # Page-cache mode: mmap the file. The kernel's cache +
                    # readahead manage residency (llama.cpp PR #26003).
                    payload, size = read_shard_mmap(path)
                if prof:
                    prof.end_read(idx)
                    prof.begin_deserialize(idx)
                state, metadata = parse_safetensors_views(payload, size)
                if metadata.get("__swlp_quant__", "") == "float8":
                    state = nest_fp8_state(state)
            else:
                data = _read_file_nocache(path, nocache=self._direct_io)
                if prof:
                    prof.end_read(idx)
                    prof.begin_deserialize(idx)
                state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
            if prof:
                prof.end_deserialize(idx)
            return state, buffer
        except Exception:
            if buffer is not None:
                self._buf_pool.put(buffer)
            LOGGER.exception("streaming_shard_read_failed", extra={"layer": idx})
            return None

    def _upload_to_device(
        self, idx: int, state: dict[str, torch.Tensor], buffer: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor] | None:
        """Stage 2: Upload CPU state dict to MPS.  Runs on the upload pool
        so it overlaps with SSD reads of subsequent layers.  The read buffer
        returns to the free-list only after the copy has completed.
        """
        prof = self.profiler
        try:
            if prof:
                prof.begin_upload(idx)
            result = self._to_device_state(state)
            if prof:
                prof.end_upload(idx)
            return result
        except Exception:
            LOGGER.exception("streaming_upload_failed", extra={"layer": idx})
            return None
        finally:
            if buffer is not None:
                self._buf_pool.put(buffer)

    def _materialize_meta_buffers(self, block: torch.nn.Module) -> None:
        """Give device storage to buffers still on meta after the assign-load.

        Non-persistent buffers (e.g. GPT-2's causal-mask ``bias``) are absent
        from shard state dicts, so they stay on meta. They get uninitialised
        device storage — matching the old ``to_empty(device)`` behaviour —
        so a forward pass never touches a meta tensor.
        """
        for name, buf in list(block.named_buffers()):
            if buf is None or buf.device.type != "meta":
                continue
            mod_path, _, leaf = name.rpartition(".")
            owner = block.get_submodule(mod_path) if mod_path else block
            owner._buffers[leaf] = torch.empty_like(buf, device=self.device)

    def _apply_shard(self, idx: int, state_dict: dict, *, device_ready: bool = False) -> None:
        """Materialise ``state_dict`` into ``blocks[idx]`` on the target device.

        With ``device_ready=True`` (the streaming path) the dict already holds
        device tensors and the assign-load is a pointer swap — no allocation
        and no copy on the compute thread. The old ``to_empty(device)``
        pre-materialisation is gone: it allocated a full layer on device only
        for ``assign=True`` to immediately replace every tensor.
        """
        block = self.blocks[idx]
        try:
            if not device_ready:
                state_dict = self._to_device_state(state_dict)
            missing, unexpected = block.load_state_dict(state_dict, strict=False, assign=True)
            # Slot-array params of cached-expert modules are deliberately
            # absent from dense shards — they are not "missing weights".
            expert_prefixes = _cached_expert_param_prefixes(block)
            if expert_prefixes:
                missing = [k for k in missing if not k.startswith(expert_prefixes)]
                # v1-hybrid dirs carry expert tensors inside the layer file;
                # the scheduler owns expert loading — not "unexpected weights".
                unexpected = [k for k in unexpected if not k.startswith(expert_prefixes)]
            if missing:
                LOGGER.warning(
                    "streaming_missing_keys",
                    extra={"layer": idx, "count": len(missing), "sample": list(missing)[:3]},
                )
            if unexpected:
                LOGGER.warning(
                    "streaming_unexpected_keys",
                    extra={"layer": idx, "count": len(unexpected), "sample": list(unexpected)[:3]},
                )
            self._materialize_meta_buffers(block)
        finally:
            self._loaded.add(idx)

    # ── resident-layer management ─────────────────────────────────────────────

    def load_resident_layers(self) -> None:
        """Pre-load the first ``resident_count`` shards into CPU RAM.

        Called once at startup.  Stores raw CPU state-dicts in
        ``_resident_data`` — does NOT materialise them to the device yet.
        The device is only used during inference (``ensure`` → ``evict``
        per token), so MPS memory stays near-zero at startup.
        """
        loaded = 0
        for idx in range(self._resident_count):
            if idx in self._resident_data:
                loaded += 1
                continue
            state = self._read_shard(idx)
            if state is None:
                LOGGER.warning("streaming_resident_shard_missing", extra={"layer": idx})
                continue
            self._resident_data[idx] = state
            loaded += 1
            LOGGER.debug("streaming_resident_cached", extra={"layer": idx})
        LOGGER.info(
            "streaming_resident_layers_cached",
            extra={"resident_count": self._resident_count, "cached": loaded},
        )

    # ── background prefetch ───────────────────────────────────────────────────

    def prefetch(self, layer_index: int) -> None:
        if not self._prefetch_enabled:
            return
        if layer_index < 0 or layer_index >= len(self.blocks):
            return
        # Resident layers load from CPU RAM — no background disk read needed.
        if layer_index < self._resident_count:
            return
        with self._lock:
            if layer_index in self._loaded or layer_index in self._futures:
                return
        if self.profiler:
            self.profiler.record_prefetch_submit(layer_index)
            self.profiler.record_queue_snapshot(
                event="prefetch_submit",
                read_queue_depth=self._pool._work_queue.qsize(),
                upload_queue_depth=self._upload_pool._work_queue.qsize(),
                ready_layers=len(self._loaded),
                active_reads=sum(1 for f in self._futures.values() if not f.done()),
                active_uploads=0,  # tracked via upload futures
            )
        # Two-stage pipeline: SSD read (pool) → MPS upload (upload_pool).
        # The upload of layer N overlaps with the SSD read of layer N+1.
        final: Future[dict[str, torch.Tensor] | None] = Future()
        try:
            ssd_future = self._pool.submit(self._read_and_deserialize, layer_index)
        except RuntimeError as exc:
            raise PrefetchError(f"Prefetch pool unavailable for layer {layer_index}") from exc

        def _chain_upload(ssd_fut: Future) -> None:
            """Callback: when SSD read completes, submit upload to the second pool."""
            try:
                result = ssd_fut.result()
                if result is None:
                    final.set_result(None)
                    return
                cpu_state, buffer = result
                upload_fut = self._upload_pool.submit(
                    self._upload_to_device, layer_index, cpu_state, buffer
                )
                # Forward upload result to the final future.
                def _forward_upload(upload_fut: Future) -> None:
                    try:
                        final.set_result(upload_fut.result())
                    except Exception as exc:
                        final.set_exception(exc)
                upload_fut.add_done_callback(_forward_upload)
            except Exception as exc:
                final.set_exception(exc)

        ssd_future.add_done_callback(_chain_upload)
        with self._lock:
            self._futures[layer_index] = final

    def disable_prefetch(self) -> None:
        self._prefetch_enabled = False
        LOGGER.warning("streaming_prefetch_disabled")

    # ── per-layer materialise / evict ─────────────────────────────────────────

    def ensure(self, layer_index: int) -> torch.nn.Module:
        prof = self.profiler
        if prof:
            prof.begin_ensure(layer_index)
            prof.record_queue_snapshot(
                event="ensure_enter",
                read_queue_depth=self._pool._work_queue.qsize(),
                upload_queue_depth=self._upload_pool._work_queue.qsize(),
                ready_layers=len(self._loaded),
                active_reads=sum(1 for f in self._futures.values() if not f.done()),
                active_uploads=0,
            )
        _ensure_start = time.perf_counter() if prof else 0.0
        overlap_status = "unknown"
        # Fast path for resident layers: apply from CPU RAM (no disk I/O).
        if layer_index < self._resident_count and layer_index in self._resident_data:
            with self._lock:
                already = layer_index in self._loaded
            if not already:
                if prof:
                    prof.begin_upload(layer_index)
                try:
                    self._apply_shard(layer_index, self._resident_data[layer_index])
                except Exception as exc:
                    raise PrefetchError(
                        f"Failed to apply resident shard for layer {layer_index}"
                    ) from exc
                if prof:
                    prof.end_upload(layer_index)
                if prof:
                    prof.end_ensure(layer_index, ensure_wait=0.0, status="resident")
                if prof:
                    prof.record_queue_snapshot(
                        event="ensure_exit",
                        read_queue_depth=self._pool._work_queue.qsize(),
                        upload_queue_depth=self._upload_pool._work_queue.qsize(),
                        ready_layers=len(self._loaded),
                        active_reads=sum(1 for f in self._futures.values() if not f.done()),
                        active_uploads=0,
                    )
            return self.blocks[layer_index]

        # Streaming path: consume the prefetch future, or fall back to a
        # synchronous read.  Track which case occurred so overlap efficiency
        # can be reported via overlap_stats().
        with self._lock:
            if layer_index in self._loaded:
                if prof:
                    prof.end_ensure(layer_index, ensure_wait=0.0, status="hit")
                if prof:
                    prof.record_queue_snapshot(
                        event="ensure_exit",
                        read_queue_depth=self._pool._work_queue.qsize(),
                        upload_queue_depth=self._upload_pool._work_queue.qsize(),
                        ready_layers=len(self._loaded),
                        active_reads=sum(1 for f in self._futures.values() if not f.done()),
                        active_uploads=0,
                    )
                return self.blocks[layer_index]
            future = self._futures.pop(layer_index, None)

        state: dict[str, torch.Tensor] | None = None
        if future is not None:
            if future.done():
                self._overlap_hits += 1  # data ready before compute needed it
                overlap_status = "hit"
            else:
                self._overlap_waits += 1  # still reading; partial overlap at best
                overlap_status = "wait"
            try:
                state = future.result()
            except Exception:
                LOGGER.exception(
                    "streaming_prefetch_future_failed", extra={"layer": layer_index}
                )
                state = None
        if state is None:
            if future is None:
                self._overlap_misses += 1  # no prefetch was running — sync fallback
                overlap_status = "miss"
            read_result = self._read_and_deserialize(layer_index)
            if read_result is None:
                raise PrefetchError(f"Failed to read shard for layer {layer_index}")
            cpu_state, buffer = read_result
            state = self._upload_to_device(layer_index, cpu_state, buffer)
            if state is None:
                raise PrefetchError(f"Failed to upload shard for layer {layer_index}")
        if prof:
            _ensure_end = time.perf_counter()
            prof.end_ensure(
                layer_index,
                wait=_ensure_end - _ensure_start,
                status=overlap_status if future is not None else "sync",
            )
        try:
            self._apply_shard(layer_index, state, device_ready=True)
        except Exception as exc:
            raise PrefetchError(f"Failed to apply shard for layer {layer_index}") from exc
        if prof:
            prof.record_queue_snapshot(
                event="ensure_exit",
                read_queue_depth=self._pool._work_queue.qsize(),
                upload_queue_depth=self._upload_pool._work_queue.qsize(),
                ready_layers=len(self._loaded),
                active_reads=sum(1 for f in self._futures.values() if not f.done()),
                active_uploads=0,
            )
        return self.blocks[layer_index]

    def evict(self, layer_index: int) -> None:
        prof = self.profiler
        if layer_index < 0 or layer_index >= len(self.blocks):
            return
        if layer_index not in self._loaded:
            return
        if prof:
            prof.record_queue_snapshot(
                event="evict",
                read_queue_depth=self._pool._work_queue.qsize(),
                upload_queue_depth=self._upload_pool._work_queue.qsize(),
                ready_layers=len(self._loaded),
                active_reads=sum(1 for f in self._futures.values() if not f.done()),
                active_uploads=0,
            )
        # Phase 23 fix: never evict resident layers when the model fully fits
        # in RAM.  Evicting forces a CPU→MPS re-apply + load_state_dict on
        # the next ensure(), which is pure Python overhead on unified memory.
        # When all layers are resident, keeping them on-device eliminates
        # per-token re-loading and dramatically improves throughput.
        if layer_index < self._resident_count and self._resident_count >= len(self.blocks):
            return
        try:
            if prof:
                prof.begin_evict(layer_index)
            # Move block back to meta device — frees device memory — while
            # sparing scheduler-managed expert slots (they ARE the cache).
            # For resident layers this is fine: _resident_data[idx] keeps the
            # CPU state-dict, so the next ensure() re-applies without SSD I/O.
            _to_meta_preserving_cached_experts(self.blocks[layer_index])
            with self._lock:
                self._loaded.discard(layer_index)
                if layer_index >= self._resident_count:
                    self._futures.pop(layer_index, None)
            if prof:
                prof.end_evict(layer_index)
        except Exception:
            LOGGER.exception("streaming_evict_failed", extra={"layer": layer_index})

    def overlap_stats(self) -> dict[str, int | float]:
        """Return prefetch overlap-efficiency metrics (Phase 11).

        - ``hits``:    ensure() found the prefetch future done (full overlap).
        - ``waits``:   ensure() blocked on a running future (partial overlap).
        - ``misses``:  ensure() fell back to a synchronous read (no overlap).
        - ``hit_rate``: hits / total, in ``[0.0, 1.0]``.
        """
        total = self._overlap_hits + self._overlap_waits + self._overlap_misses
        hit_rate = self._overlap_hits / total if total > 0 else 0.0
        return {
            "hits": self._overlap_hits,
            "waits": self._overlap_waits,
            "misses": self._overlap_misses,
            "total": total,
            "hit_rate": hit_rate,
        }

    def window_end_index(self, current_index: int) -> int:
        return current_index + self.config.prefetch_depth

    def cleanup(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)
        self._upload_pool.shutdown(wait=True, cancel_futures=True)
        for idx in list(self._loaded):
            try:
                _to_meta_preserving_cached_experts(self.blocks[idx])
                with self._lock:
                    self._loaded.discard(idx)
            except Exception:
                LOGGER.exception("streaming_cleanup_evict_failed", extra={"layer": idx})
        with self._lock:
            self._futures.clear()
        self._resident_data.clear()
        # Drop any buffer still parked in the free-list (its payload's upload
        # already completed — nothing references the bytes anymore).
        while not self._buf_pool.empty():
            self._buf_pool.get_nowait()
