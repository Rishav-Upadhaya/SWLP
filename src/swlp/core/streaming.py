"""Disk-streaming scheduler for SWLP.

When ``runtime.shard_dir`` points at a directory of per-layer ``.pt`` shards
produced by ``swlp.model.shard.shard_model_by_layer``, ``StreamingScheduler``
loads layer weights from disk into pre-allocated empty block modules on demand,
keeping at most ``window_size`` layers materialized in RAM at any moment.

This is the SSD-bound path used on Apple Silicon (M5 unified memory). The
existing in-memory ``ThreadedScheduler`` swaps already-loaded blocks between
CPU and device — useful for CUDA-VRAM streaming but irrelevant when the goal
is to avoid holding the full model in RAM at all.

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

Per-token disk I/O drops from (all_layers × layer_mb) to
(streaming_layers × layer_mb), at the cost of a fast CPU→device copy for the
resident layers instead of the slow SSD→CPU read.
"""
from __future__ import annotations

import json
import logging
import os
import struct
import sys
import threading
from collections.abc import Iterable
from pathlib import Path

import torch

from ..model.quant import dequantize_layer_state
from ..model.sparse import decode_sparse
from .scheduler import PrefetchError, SchedulerConfig

LOGGER = logging.getLogger(__name__)


def _read_file_nocache(path: Path) -> bytes:
    """Read a file bypassing the OS page cache on macOS (F_NOCACHE).

    Prevents large model-shard reads from evicting other data from the unified
    memory page cache.  Falls back to a normal read on non-macOS platforms.
    Uses 4 MB sequential chunks for optimal SSD throughput.
    """
    fd = os.open(str(path), os.O_RDONLY)
    try:
        if sys.platform == "darwin":
            import fcntl
            # F_NOCACHE disables UBC (Unified Buffer Cache) for this fd.
            fcntl.fcntl(fd, getattr(fcntl, "F_NOCACHE", 48), 1)
        size = os.fstat(fd).st_size
        chunks: list[bytes] = []
        remaining = size
        while remaining > 0:
            chunk = os.read(fd, min(remaining, 4 * 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _safetensors_metadata(data: bytes) -> dict[str, str]:
    """Parse the ``__metadata__`` dict from a safetensors file's binary header."""
    if len(data) < 8:
        return {}
    header_len = struct.unpack_from("<Q", data, 0)[0]
    if len(data) < 8 + header_len:
        return {}
    try:
        header = json.loads(data[8 : 8 + header_len])
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return header.get("__metadata__", {}) or {}


def has_shards(shard_dir: str | Path) -> bool:
    p = Path(shard_dir)
    return p.exists() and (p / "shard_manifest.json").exists()


def _load_safetensors_shard(path: Path) -> dict:
    """Load a .safetensors shard, reconstructing the FP8 nested format if needed.

    Plain FP16 shards return a flat ``{name: tensor}`` dict.
    FP8 shards (metadata ``__swlp_quant__ = float8``) return the nested
    ``{"__swlp_quant__": "float8", "weights": {name: {"data": ..., "scale": ...}}}``
    format expected by ``dequantize_layer_state()``.

    Uses direct I/O (F_NOCACHE on macOS) to prevent shard reads from polluting
    the OS page cache with model weights that are only needed once per token.
    """
    from safetensors.torch import load as _st_load

    data = _read_file_nocache(path)
    metadata = _safetensors_metadata(data)
    flat: dict[str, torch.Tensor] = _st_load(data)

    quant_scheme = metadata.get("__swlp_quant__", "")
    if quant_scheme == "float8":
        # Reconstruct nested FP8 format from flat safetensors keys.
        weights: dict = {}
        _fp8_suffix = "__fp8_data"
        _scale_suffix = "__fp8_scale"
        for k, v in flat.items():
            if k.endswith(_fp8_suffix):
                name = k[: -len(_fp8_suffix)]
                entry = weights.setdefault(name, {})
                entry["data"] = v
            elif k.endswith(_scale_suffix):
                name = k[: -len(_scale_suffix)]
                entry = weights.setdefault(name, {})
                entry["scale"] = v
            else:
                # 1-D tensor stored directly (no scale).
                weights[k] = {"data": v}
        # Use the same key as quant.py's _QUANT_KEY = "_swlp_quant" so that
        # dequantize_layer_state() recognises it without modification.
        return {"_swlp_quant": "float8", "weights": weights}

    return flat


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

    Non-resident layers stream from disk as before.
    """

    def __init__(
        self,
        blocks: Iterable[torch.nn.Module],
        device: torch.device,
        config: SchedulerConfig,
        shard_dir: str | Path,
        resident_count: int = 0,
    ) -> None:
        self.blocks = list(blocks)
        self.device = device
        self.config = config
        self.shard_dir = Path(shard_dir)
        self._resident_count = max(0, min(resident_count, len(self.blocks)))
        # CPU-RAM cache for resident layers: shard index → state_dict (CPU tensors).
        self._resident_data: dict[int, dict] = {}
        self._loaded: set[int] = set()
        self._pending: dict[int, dict] = {}
        self._threads: dict[int, threading.Thread] = {}
        self._lock = threading.Lock()
        self._prefetch_enabled = config.prefetch
        # Overlap-tracking counters (Phase 11).
        # hit  — ensure() found data already ready in _pending (full overlap achieved).
        # wait — ensure() had to join() a still-running prefetch thread (partial overlap).
        # miss — ensure() fell back to a synchronous disk read (no prefetch was running).
        self._overlap_hits: int = 0
        self._overlap_waits: int = 0
        self._overlap_misses: int = 0

    # ── internal helpers ──────────────────────────────────────────────────────

    def _read_shard(self, idx: int) -> dict | None:
        # Phase 17: auto-detect shard format by extension.  Try .safetensors
        # first; fall back to .pt for legacy shard directories.
        st_path = self.shard_dir / f"layer_{idx:03d}.safetensors"
        pt_path = self.shard_dir / f"layer_{idx:03d}.pt"
        if st_path.exists():
            path = st_path
        elif pt_path.exists():
            path = pt_path
        else:
            LOGGER.warning("streaming_shard_missing", extra={"layer": idx,
                                                              "path": str(st_path)})
            return None
        try:
            if path.suffix == ".safetensors":
                state = _load_safetensors_shard(path)
            else:
                import io
                data = _read_file_nocache(path)
                state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
            # Pin CPU memory for faster CPU→CUDA DMA transfers (no-op on unified
            # memory / MPS). Wrapped in try/except — pin_memory() can raise on
            # platforms that do not support it.
            if self.config.pin_memory:
                pinned: dict = {}
                for k, v in state.items():
                    if v.is_floating_point() and v.device.type == "cpu":
                        try:
                            pinned[k] = v.pin_memory()
                        except Exception:
                            pinned[k] = v  # pin unsupported; keep as-is
                    else:
                        pinned[k] = v
                state = pinned
            return state
        except Exception:
            LOGGER.exception("streaming_shard_read_failed", extra={"layer": idx})
            return None

    def _apply_shard(self, idx: int, state_dict: dict) -> None:
        """Materialise ``state_dict`` into ``blocks[idx]`` on the target device.

        Accepts either a plain FP16 state dict or a quantized (FP8) shard dict —
        ``dequantize_layer_state`` is a no-op on the former. Resident layers keep
        their quantized form in ``_resident_data``; the FP8→FP16 expansion
        happens here, per token, so resident RAM stays at the FP8 footprint.
        """
        block = self.blocks[idx]
        block.to_empty(device=self.device)
        try:
            state_dict = dequantize_layer_state(state_dict)
            state_dict = decode_sparse(state_dict)  # Phase 13: no-op on dense shards
            cast_state = {k: v.to(device=self.device, dtype=v.dtype) for k, v in state_dict.items()}
            missing, unexpected = block.load_state_dict(cast_state, strict=False, assign=True)
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

    def _prefetch_worker(self, idx: int) -> None:
        sd = self._read_shard(idx)
        with self._lock:
            self._pending[idx] = sd if sd is not None else {}
            self._threads.pop(idx, None)

    def prefetch(self, layer_index: int) -> None:
        if not self._prefetch_enabled:
            return
        if layer_index < 0 or layer_index >= len(self.blocks):
            return
        # Resident layers load from CPU RAM — no background disk read needed.
        if layer_index < self._resident_count:
            return
        with self._lock:
            if (
                layer_index in self._loaded
                or layer_index in self._threads
                or layer_index in self._pending
            ):
                return
        t = threading.Thread(
            target=self._prefetch_worker,
            args=(layer_index,),
            daemon=True,
            name=f"swlp-stream-{layer_index}",
        )
        with self._lock:
            self._threads[layer_index] = t
        t.start()

    def disable_prefetch(self) -> None:
        self._prefetch_enabled = False
        LOGGER.warning("streaming_prefetch_disabled")

    # ── per-layer materialise / evict ─────────────────────────────────────────

    def ensure(self, layer_index: int) -> torch.nn.Module:
        # Fast path for resident layers: apply from CPU RAM (no disk I/O).
        if layer_index < self._resident_count and layer_index in self._resident_data:
            with self._lock:
                already = layer_index in self._loaded
            if not already:
                try:
                    self._apply_shard(layer_index, self._resident_data[layer_index])
                except Exception as exc:
                    raise PrefetchError(
                        f"Failed to apply resident shard for layer {layer_index}"
                    ) from exc
            return self.blocks[layer_index]

        # Streaming path: wait for background thread, use pending state, or
        # fall back to a synchronous disk read.  Track which case occurred so
        # overlap efficiency can be reported via overlap_stats().
        with self._lock:
            t = self._threads.get(layer_index)
        if t is not None:
            self._overlap_waits += 1  # thread still running; partial overlap at best
            t.join()

        with self._lock:
            if layer_index in self._loaded:
                return self.blocks[layer_index]
            state = self._pending.pop(layer_index, None)

        if state is None:
            self._overlap_misses += 1  # no prefetch was running — sync fallback
            state = self._read_shard(layer_index)
            if state is None:
                raise PrefetchError(f"Failed to load shard for layer {layer_index}")
        elif t is None:
            # Thread had already finished before ensure() was called: full overlap.
            self._overlap_hits += 1
        # else: t was not None → _overlap_waits already incremented above.
        try:
            self._apply_shard(layer_index, state)
        except Exception as exc:
            raise PrefetchError(f"Failed to apply shard for layer {layer_index}") from exc
        return self.blocks[layer_index]

    def evict(self, layer_index: int) -> None:
        if layer_index < 0 or layer_index >= len(self.blocks):
            return
        if layer_index not in self._loaded:
            return
        try:
            # Move block back to meta device — frees MPS memory.
            # For resident layers this is fine: _resident_data[idx] keeps the
            # CPU state-dict, so the next ensure() re-applies without SSD I/O.
            self.blocks[layer_index].to_empty(device="meta")
            with self._lock:
                self._loaded.discard(layer_index)
                if layer_index >= self._resident_count:
                    # Non-resident: discard any pending / thread state too.
                    self._pending.pop(layer_index, None)
                    self._threads.pop(layer_index, None)
        except Exception:
            LOGGER.exception("streaming_evict_failed", extra={"layer": layer_index})

    def overlap_stats(self) -> dict[str, int | float]:
        """Return prefetch overlap-efficiency metrics (Phase 11).

        - ``hits``:    ensure() found data already in ``_pending`` (full overlap).
        - ``waits``:   ensure() joined a still-running thread (partial overlap).
        - ``misses``:  ensure() fell back to a synchronous disk read (no overlap).
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
        for idx in list(self._loaded):
            try:
                self.blocks[idx].to_empty(device="meta")
                with self._lock:
                    self._loaded.discard(idx)
            except Exception:
                LOGGER.exception("streaming_cleanup_evict_failed", extra={"layer": idx})
        with self._lock:
            self._pending.clear()
            self._threads.clear()
        self._resident_data.clear()
