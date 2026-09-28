"""Lossless shard codec — compressed ``.swz`` layer shards (SWLP-Max step 1).

A ``.swz`` file holds a byte-grouped, entropy-coded (zipnn: zstd + Huffman)
image of the original shard bytes. FP16/BF16 high bytes (sign + exponent) are
highly skewed, so grouping them and entropy-coding the group recovers ~31% of
shard bytes with **bit-exact** reconstruction — measured 0.683 ratio at
15 GB/s standalone decode on a real Mistral-7B layer shard
(``scripts/research/codec_eval.py``).

Throughput trade-off (measured, Phase 22): under end-to-end streaming load
the decoder contends with MPS compute and host→device copies for unified
memory bandwidth and drops to ~7 GB/s effective. Compressed streaming
therefore only beats plain ``.safetensors`` when the SSD is the bottleneck —
crossover ≈ 3.5 GB/s sequential read. On a ~6.9 GB/s SSD (M5) it costs ~25%
tok/s; on PCIe-3-class NVMe (~2–3.5 GB/s) it is a win. The always-on win is
the ~31% disk footprint; ``swlp compress-shards --revert`` restores plain
shards for fast-SSD machines.

Container layout (little-endian):

    magic    4s    b"SWZ1"
    raw      u64   decompressed payload size in bytes
    sha256   32s   SHA-256 of the raw payload (compress-time + offline checks)
    crc      u32   CRC-32 of the compressed blob (checked on every decompress)
    blob     ...   zipnn-compressed payload

The CRC gate is a hard safety requirement, not an optimization: zipnn 0.5.4's
C core can segfault on malformed input streams (measured: 7/51 single-byte
corruptions crash the process). Only CRC-verified bytes ever reach it, so a
corrupt or tampered ``.swz`` file fails with :class:`CodecError` instead of
killing the process. At ~25 GB/s (hardware CRC) the check is invisible next
to the SSD read.

Like ``config``/``metrics``/``logging`` this is a leaf module importable from
both ``model/`` (shard conversion) and ``core/`` (streaming reads) without
violating the inter-package import rules.

zipnn 0.5.4 constraint: the byte-grouping hint is always ``float16``. The
grouping is identical for any 2-byte dtype, and the ``bfloat16`` hint was
measured to produce non-bit-exact roundtrips on real shard data. Safety does
not rest on the hint: :func:`compress_bytes` verifies the roundtrip with a
fresh decoder before returning, and decoding a verified blob is deterministic.
"""
from __future__ import annotations

import hashlib
import struct
import zlib
from pathlib import Path

MAGIC = b"SWZ1"
COMPRESSED_SUFFIX = ".swz"
_HEADER = struct.Struct("<4sQ32sI")
HEADER_SIZE = _HEADER.size
# Byte-grouping hint passed to zipnn (see module docstring).
_DTYPE_HINT = "float16"
# Phase 22 measured crossover: compressed streaming only beats plain shards
# when the SSD delivers less than ~3.5 GB/s sequential read (the decoder's
# ~7 GB/s effective rate contends for unified-memory bandwidth above that).
COMPRESSION_CROSSOVER_GBPS = 3.5


def recommend_compression(ssd_read_gbps: float) -> bool:
    """Whether ``.swz`` compressed shards are expected to *increase* tok/s.

    Uses the measured Phase 22 crossover. Disk footprint always shrinks ~31%;
    this predicate answers only the throughput question, and is what
    ``swlp doctor`` surfaces alongside a machine's measured SSD bandwidth.
    """
    return ssd_read_gbps > 0 and ssd_read_gbps < COMPRESSION_CROSSOVER_GBPS


class CodecError(RuntimeError):
    """A ``.swz`` blob is malformed, truncated, or failed verification."""


def _zipnn():
    """Fresh zipnn codec per call — instances are not shared across threads."""
    try:
        from zipnn import ZipNN
    except ImportError as exc:  # optional extra: `pip install swlp[codec]`
        raise CodecError(
            "the .swz codec needs zipnn — install it with `pip install swlp[codec]`"
        ) from exc

    return ZipNN(input_format="byte", bytearray_dtype=_DTYPE_HINT)


def compressed_path(path: Path) -> Path:
    """Map ``layer_000.safetensors`` -> ``layer_000.safetensors.swz``."""
    return path.with_name(path.name + COMPRESSED_SUFFIX)


def is_swz(data: bytes | bytearray | memoryview) -> bool:
    """True when ``data`` starts with the ``.swz`` container magic."""
    return bytes(data[: len(MAGIC)]) == MAGIC


def _parse_header(blob: bytes | bytearray | memoryview) -> tuple[int, bytes, int]:
    """Validate and unpack a ``.swz`` header -> (raw_size, sha256, crc32)."""
    if len(blob) < _HEADER.size:
        raise CodecError(f"swz blob truncated: {len(blob)} < header {_HEADER.size}")
    magic, raw_size, digest, crc = _HEADER.unpack_from(bytes(blob[: _HEADER.size]))
    if magic != MAGIC:
        raise CodecError(f"bad swz magic: {magic!r}")
    return raw_size, digest, crc


def decompressed_size(blob: bytes | bytearray | memoryview) -> int:
    """Parse the raw payload size from a ``.swz`` header."""
    raw_size, _, _ = _parse_header(blob)
    return raw_size


def compress_bytes(data: bytes | memoryview, verify: bool = True) -> bytes:
    """Compress shard bytes into a ``.swz`` container.

    ``verify=True`` (the default, and mandatory before deleting an original
    shard) decompresses the result with a fresh decoder and compares SHA-256 —
    a blob that verifies once decodes deterministically forever.
    """
    raw = data if isinstance(data, bytes) else bytes(data)
    if not raw:
        raise CodecError("refusing to compress an empty payload")
    digest = hashlib.sha256(raw).digest()
    blob = _zipnn().compress(raw)
    out = _HEADER.pack(MAGIC, len(raw), digest, zlib.crc32(blob)) + blob
    if verify:
        restored = decompress_bytes(out)
        if hashlib.sha256(restored).digest() != digest:
            raise CodecError("compress verification failed: roundtrip is not bit-exact")
    return out


def decompress_bytes(
    blob: bytes | bytearray | memoryview, check_sha: bool = False
) -> bytearray:
    """Decompress a ``.swz`` container back to the original shard bytes.

    Returns a writable ``bytearray`` so callers can wrap it zero-copy (e.g.
    ``torch.frombuffer``). The compressed payload's CRC-32 is checked before
    the bytes reach zipnn's C core (see module docstring); the stored SHA-256
    is only re-checked with ``check_sha=True`` — the right call for offline
    paths (verification, shard revert), too slow for the per-layer streaming
    hot path.
    """
    raw_size, digest, crc = _parse_header(blob)
    payload = blob[_HEADER.size :]
    payload = payload if isinstance(payload, bytes) else bytes(payload)
    if zlib.crc32(payload) != crc:
        raise CodecError("swz payload corrupted (crc mismatch)")
    try:
        out = _zipnn().decompress(payload)
    except Exception as exc:  # zipnn/zstd raise library-specific errors on corrupt data
        raise CodecError(f"swz payload decode failed: {exc}") from exc
    if len(out) != raw_size:
        raise CodecError(f"swz payload decoded to {len(out)} bytes, header says {raw_size}")
    if check_sha and hashlib.sha256(out).digest() != digest:
        raise CodecError("swz payload corrupted (sha256 mismatch)")
    return out


def verify_blob(blob: bytes | bytearray | memoryview) -> bool:
    """Full offline check: decode and compare against the stored SHA-256."""
    try:
        decompress_bytes(blob, check_sha=True)
    except CodecError:
        return False
    return True
