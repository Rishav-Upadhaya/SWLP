"""Low-level shard I/O for SWLP streaming (Phase 20).

Disk → CPU-tensor loading with explicit page-cache control and a zero-copy
safetensors parse. The old path copied every shard three times per token
(chunk-list join → safetensors deserialize → host-to-device cast); this module
reads a shard **once** into a reusable ``uint8`` buffer via ``readinto`` and
reinterprets tensor views directly inside that buffer, so the single
host→device copy in the scheduler is the only remaining per-token copy.

Page-cache policy: ``nocache=True`` applies ``F_NOCACHE`` on macOS so streaming
reads of models that cannot fit in RAM do not thrash the unified-memory page
cache. Callers gate it via the ``SWLP_DIRECT_IO`` policy (see
``core.streaming``): models that *do* fit get page-cache residency for free.
"""
from __future__ import annotations

import json
import logging
import mmap
import os
import struct
import sys
import threading
import warnings
from contextlib import nullcontext
from pathlib import Path

import torch

from .. import codec

LOGGER = logging.getLogger(__name__)

# Numeric fallback for fcntl.F_NOCACHE on macOS, used when the constant is
# missing from the fcntl module build.
_F_NOCACHE_FALLBACK = 48

_READ_CHUNK_BYTES = 4 * 1024 * 1024

_SAFETENSORS_TORCH_DTYPES: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}

_FP8_DATA_SUFFIX = "__fp8_data"
_FP8_SCALE_SUFFIX = "__fp8_scale"


def _set_nocache(fd: int) -> None:
    """Disable the unified buffer cache for this fd on macOS (F_NOCACHE)."""
    if sys.platform == "darwin":
        import fcntl

        fcntl.fcntl(fd, getattr(fcntl, "F_NOCACHE", _F_NOCACHE_FALLBACK), 1)


_MADV_WILLNEED = getattr(mmap, "MADV_WILLNEED", 3)  # Linux value; absent on macOS


def _madvise_willneed(mm: mmap.mmap, size: int) -> None:
    """Best-effort kernel readahead hint. Skipped on macOS: mmap.mmap exposes no
    fd there and default fault-ahead already covers sequential layer reads."""
    try:
        mm.madvise(_MADV_WILLNEED)
    except Exception:
        pass


def read_shard_mmap(path: Path) -> tuple[torch.Tensor, int]:
    """Zero-copy mmap of a shard file: SSD pages aliased into tensors.

    Returns ``(payload, size)`` where ``payload`` is a uint8 tensor whose
    storage is the file's page mapping. Tensor views parsed from it touch
    disk pages directly — no explicit read, and the OS page cache + readahead
    manages residency (llama.cpp PR #26003 found this beats manual pinning).
    """
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        mm = mmap.mmap(f.fileno(), size, access=mmap.ACCESS_READ)
        _madvise_willneed(mm, size)
    # ponytail: attach handle so the mapping outlives parsed views.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # read-only buffer is the point of mmap
        payload = torch.frombuffer(mm, dtype=torch.uint8)
    payload._swlp_mmap = mm  # type: ignore[attr-defined]
    return payload, size


def _advise_willneed(fd: int, nbytes: int) -> None:
    """Best-effort readahead advice (Phase 26).

    Linux: ``posix_fadvise(WILLNEED)`` — the kernel schedules async readahead
    for the range. macOS: ``F_READAHEAD`` — one-shot readahead of the first
    ``nbytes``. Both are hints: silence on any platform is fine.
    """
    try:
        if sys.platform.startswith("linux"):
            os.posix_fadvise(fd, 0, nbytes, os.POSIX_FADV_WILLNEED)
        elif sys.platform == "darwin":
            import fcntl

            fcntl.fcntl(fd, fcntl.F_READAHEAD, nbytes)
    except (AttributeError, OSError):
        pass


def _read_file_nocache(path: Path, nocache: bool = True) -> bytes:
    """Read a file as bytes, optionally bypassing the OS page cache.

    Used for legacy ``.pt`` shards (``torch.load`` needs a bytes object) and
    by tests; the safetensors hot path uses :func:`read_into_tensor` instead.
    """
    fd = os.open(str(path), os.O_RDONLY)
    try:
        if nocache:
            _set_nocache(fd)
        else:
            size = os.fstat(fd).st_size
            _advise_willneed(fd, size)  # page-cache mode: hint async readahead
        size = os.fstat(fd).st_size
        chunks: list[bytes] = []
        remaining = size
        while remaining > 0:
            chunk = os.read(fd, min(remaining, _READ_CHUNK_BYTES))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def read_into_tensor(
    path: Path,
    buffer: torch.Tensor | None = None,
    nocache: bool = True,
    pin: bool = False,
) -> tuple[torch.Tensor, int]:
    """Read a file into a reusable ``uint8`` CPU tensor via ``readinto``.

    Returns ``(buffer, size)``. The buffer is grown (never shrunk) when the
    file is larger than the one passed in, so a per-worker buffer is allocated
    once and reused for every same-sized layer shard — no allocation and no
    intermediate copy per read. Only ``buffer[:size]`` holds valid data.

    ``pin=True`` allocates the buffer in pinned memory so the
    later host→device copy can be an async DMA straight from the read buffer.
    """
    with open(path, "rb", buffering=0) as f:
        if nocache:
            _set_nocache(f.fileno())
        size = os.fstat(f.fileno()).st_size
        if buffer is None or buffer.numel() < size:
            try:
                buffer = torch.empty(size, dtype=torch.uint8, pin_memory=pin)
            except RuntimeError:
                # pin_memory unsupported on this platform — plain allocation.
                buffer = torch.empty(size, dtype=torch.uint8)
        view = memoryview(buffer.numpy())[:size]
        got = 0
        while got < size:
            n = f.readinto(view[got:])
            if not n:
                break
            got += n
    return buffer, got


def read_shard_payload(
    path: Path,
    buffer: torch.Tensor | None = None,
    nocache: bool = True,
    pin: bool = False,
    decompress_gate: threading.Semaphore | None = None,
) -> tuple[torch.Tensor, int, torch.Tensor | None]:
    """Read a shard file and return parse-ready safetensors bytes.

    Returns ``(payload, payload_size, buffer)``:

    - Plain ``.safetensors`` — ``payload`` *is* the (possibly grown) reusable
      read buffer, exactly :func:`read_into_tensor` semantics.
    - Compressed ``.swz`` — the file is read into the reusable buffer (which
      then holds compressed bytes only) and decompressed; ``payload`` wraps
      the decompressed ``bytearray`` zero-copy via ``torch.frombuffer``.
      Tensor views parsed from it keep that storage alive, so the payload is
      safe to retain like any fresh buffer. Note: the payload is not pinned
      even with ``pin=True`` (only the compressed read buffer is) — on the
      MPS/unified-memory target this is moot.

    ``decompress_gate`` caps how many threads decompress at once: zipnn fans
    out over all cores internally, so concurrent decodes thrash each other
    (measured 95 ms vs 30 ms per layer with two in flight). Reads are not
    gated — they overlap a holder's decode.

    Callers must keep ``buffer`` for the next call to preserve buffer reuse.
    """
    buffer, size = read_into_tensor(path, buffer=buffer, nocache=nocache, pin=pin)
    if path.suffix != codec.COMPRESSED_SUFFIX:
        return buffer, size, buffer
    with decompress_gate if decompress_gate is not None else nullcontext():
        raw = codec.decompress_bytes(memoryview(buffer.numpy())[:size])
    payload = torch.frombuffer(raw, dtype=torch.uint8)
    return payload, payload.numel(), buffer


def parse_safetensors_views(
    buffer: torch.Tensor, size: int
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Zero-copy parse of a safetensors blob held in a ``uint8`` tensor.

    Returns ``({name: tensor}, metadata)`` where each tensor is a *view* into
    ``buffer`` — callers must copy tensors out (e.g. ``.to(device, copy=True)``)
    before the buffer is reused for the next read.
    """
    if size < 8:
        return {}, {}
    header_len = struct.unpack("<Q", buffer[:8].numpy().tobytes())[0]
    data_start = 8 + header_len
    if size < data_start:
        return {}, {}
    header = json.loads(buffer[8:data_start].numpy().tobytes())
    metadata = header.pop("__metadata__", None) or {}
    tensors: dict[str, torch.Tensor] = {}
    for name, info in header.items():
        dtype = _SAFETENSORS_TORCH_DTYPES[info["dtype"]]
        start, end = info["data_offsets"]
        raw = buffer[data_start + start : data_start + end]
        try:
            t = raw.view(dtype)
        except RuntimeError:
            # Misaligned slice (possible in mixed-dtype shards) — clone the
            # raw bytes so the reinterpret starts at storage offset 0.
            t = raw.clone().view(dtype)
        tensors[name] = t.reshape(info["shape"])
    return tensors, metadata


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


def nest_fp8_state(flat: dict[str, torch.Tensor]) -> dict:
    """Reconstruct the nested FP8 dict expected by ``dequantize_layer_state()``.

    FP8 shards store ``{name}__fp8_data`` / ``{name}__fp8_scale`` pairs as flat
    safetensors keys; ``_swlp_quant`` matches quant.py's ``_QUANT_KEY``.
    """
    weights: dict = {}
    for k, v in flat.items():
        if k.endswith(_FP8_DATA_SUFFIX):
            weights.setdefault(k[: -len(_FP8_DATA_SUFFIX)], {})["data"] = v
        elif k.endswith(_FP8_SCALE_SUFFIX):
            weights.setdefault(k[: -len(_FP8_SCALE_SUFFIX)], {})["scale"] = v
        else:
            # 1-D tensor stored directly (no scale).
            weights[k] = {"data": v}
    return {"_swlp_quant": "float8", "weights": weights}


def load_safetensors_shard(path: Path, nocache: bool = True) -> dict:
    """Standalone load of a ``.safetensors`` or compressed ``.swz`` shard.

    Plain FP16 shards return a flat ``{name: tensor}`` dict; FP8 shards
    (metadata ``__swlp_quant__ = float8``) return the nested format expected
    by ``dequantize_layer_state()``. The result owns its memory and is safe to
    retain — used for resident layers and one-off loads. The streaming hot
    path uses :func:`read_shard_payload` + :func:`parse_safetensors_views`
    with a reused buffer instead.
    """
    payload, size, _ = read_shard_payload(path, buffer=None, nocache=nocache)
    tensors, metadata = parse_safetensors_views(payload, size)
    if metadata.get("__swlp_quant__", "") == "float8":
        return nest_fp8_state(tensors)
    return tensors
