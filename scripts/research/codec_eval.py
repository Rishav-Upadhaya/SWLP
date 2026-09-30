"""Phase 23a measurement gate — lossless shard-codec candidates on a real shard.

Decides the codec for compressed layer shards (docs/swlp_max_throughput_plan.html,
step 23a). Candidates are evaluated on an actual ``.safetensors`` layer shard for:

- compression ratio (compressed / raw bytes),
- one-time compression cost,
- single-thread decompression throughput *including* reconstruction of the
  contiguous interleaved buffer (what the streaming hot path must do).

Gate (from the plan): ratio <= 0.75 and logical decompress throughput
>= 2.6 GB/s per thread (4 prefetch workers x 2.6 ~= 10.4 GB/s >= the 6.93 GB/s
SSD delivering compressed bytes at ratio 0.67).

The byte-plane transform: BF16 little-endian pairs are split into an LSB plane
(mantissa bytes, near-random) and an MSB plane (sign+exponent bytes, highly
skewed -> entropy-codes well). Reconstruction interleaves the planes back.

Usage:
    python scripts/research/codec_eval.py [shard_path]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import zlib
from collections.abc import Callable
from pathlib import Path

DEFAULT_SHARD = Path("shards/mistral-7b/layer_000.safetensors")
GATE_MAX_RATIO = 0.75
GATE_MIN_DECODE_GBPS = 2.6
_REPS = 3


def _plane_split(data: bytes) -> tuple[bytes, bytes]:
    """Split little-endian 2-byte values into (lsb_plane, msb_plane)."""
    mv = memoryview(data)
    return bytes(mv[0::2]), bytes(mv[1::2])


def _interleave_mv(lsb: bytes, msb: bytes, n: int) -> bytes:
    """Reconstruct interleaved bytes via C-level memoryview step assignment."""
    out = bytearray(n)
    out[0::2] = lsb
    out[1::2] = msb
    return bytes(out)


def _interleave_np(lsb: bytes, msb: bytes, n: int) -> bytes:
    """Reconstruct interleaved bytes via numpy strided stores."""
    import numpy as np

    out = np.empty(n, dtype=np.uint8)
    out[0::2] = np.frombuffer(lsb, dtype=np.uint8)
    out[1::2] = np.frombuffer(msb, dtype=np.uint8)
    return out.tobytes()


def _zstd_codecs(level: int, interleave: Callable) -> tuple[Callable, Callable]:
    import zstandard

    comp = zstandard.ZstdCompressor(level=level)
    dec = zstandard.ZstdDecompressor()

    def compress(data: bytes) -> tuple[bytes, bytes]:
        lsb, msb = _plane_split(data)
        return comp.compress(lsb), comp.compress(msb)

    def decompress(blobs: tuple[bytes, bytes], n: int) -> bytes:
        lsb = dec.decompress(blobs[0], max_output_size=(n + 1) // 2)
        msb = dec.decompress(blobs[1], max_output_size=(n + 1) // 2)
        return interleave(lsb, msb, n)

    return compress, decompress


def _zstd_full(level: int) -> tuple[Callable, Callable]:
    import zstandard

    comp = zstandard.ZstdCompressor(level=level)
    dec = zstandard.ZstdDecompressor()
    return (
        lambda data: (comp.compress(data),),
        lambda blobs, n: dec.decompress(blobs[0], max_output_size=n),
    )


def _zlib_planes(level: int, interleave: Callable) -> tuple[Callable, Callable]:
    def compress(data: bytes) -> tuple[bytes, bytes]:
        lsb, msb = _plane_split(data)
        return zlib.compress(lsb, level), zlib.compress(msb, level)

    def decompress(blobs: tuple[bytes, bytes], n: int) -> bytes:
        return interleave(zlib.decompress(blobs[0]), zlib.decompress(blobs[1]), n)

    return compress, decompress


def _zipnn_codec() -> tuple[Callable, Callable]:
    from zipnn import ZipNN

    zn = ZipNN(input_format="byte", bytearray_dtype="bfloat16")
    return (
        lambda data: (zn.compress(data),),
        lambda blobs, n: zn.decompress(blobs[0]),
    )


def _evaluate(name: str, data: bytes, compress: Callable, decompress: Callable) -> dict | None:
    n = len(data)
    digest = hashlib.sha256(data).digest()
    try:
        t0 = time.perf_counter()
        blobs = compress(data)
        compress_s = time.perf_counter() - t0
        decode_s = []
        for _ in range(_REPS):
            t0 = time.perf_counter()
            out = decompress(blobs, n)
            decode_s.append(time.perf_counter() - t0)
        if hashlib.sha256(out).digest() != digest:
            print(f"  {name:34s}  FAILED roundtrip (not bit-exact)")
            return None
    except Exception as exc:  # eval-only script: report and move on
        print(f"  {name:34s}  ERROR: {exc}")
        return None
    comp_bytes = sum(len(b) for b in blobs)
    ratio = comp_bytes / n
    gbps = n / min(decode_s) / 1e9
    return {"name": name, "ratio": ratio, "compress_s": compress_s, "decode_gbps": gbps}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", nargs="?", default=str(DEFAULT_SHARD))
    args = parser.parse_args()
    path = Path(args.shard)
    if not path.is_file():
        print(f"shard not found: {path}")
        return 1
    data = path.read_bytes()
    print(f"shard: {path}  ({len(data) / 1e9:.3f} GB raw)\n")

    candidates: list[tuple[str, Callable, Callable]] = []
    for level in (3, 9):
        c, d = _zstd_full(level)
        candidates.append((f"zstd-{level} (full bytes)", c, d))
        c, d = _zstd_codecs(level, _interleave_np)
        candidates.append((f"zstd-{level} + planes (np interleave)", c, d))
    c, d = _zstd_codecs(3, _interleave_mv)
    candidates.append(("zstd-3 + planes (mv interleave)", c, d))
    c, d = _zlib_planes(1, _interleave_np)
    candidates.append(("zlib-1 + planes (np interleave)", c, d))
    try:
        c, d = _zipnn_codec()
        candidates.append(("zipnn (bfloat16 byte mode)", c, d))
    except Exception as exc:
        print(f"  zipnn unavailable: {exc}")

    print(f"  {'candidate':34s}  {'ratio':>6s}  {'comp s':>7s}  {'decode GB/s':>11s}  gate")
    results = []
    for name, compress, decompress in candidates:
        r = _evaluate(name, data, compress, decompress)
        if r is None:
            continue
        ok = r["ratio"] <= GATE_MAX_RATIO and r["decode_gbps"] >= GATE_MIN_DECODE_GBPS
        print(
            f"  {r['name']:34s}  {r['ratio']:6.3f}  {r['compress_s']:7.2f}"
            f"  {r['decode_gbps']:11.2f}  {'PASS' if ok else 'fail'}"
        )
        results.append(r)
    passing = [
        r
        for r in results
        if r["ratio"] <= GATE_MAX_RATIO and r["decode_gbps"] >= GATE_MIN_DECODE_GBPS
    ]
    if passing:
        best = min(passing, key=lambda r: r["ratio"])
        print(f"\ngate PASSED — best ratio among passing: {best['name']} ({best['ratio']:.3f})")
        return 0
    print("\ngate FAILED — no candidate meets ratio <= 0.75 and decode >= 2.6 GB/s")
    return 2


if __name__ == "__main__":
    sys.exit(main())
