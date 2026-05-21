"""Phase 7: re-quantize an existing FP16 shard directory to FP8.

Reads ``layer_*.pt`` shards in place (local disk, no HF download), quantizes
each to ``float8_e4m3`` with per-output-channel scales, and writes a new shard
directory plus an updated manifest (``weight_dtype = "float8"``).

Usage:
    python scripts/requantize_fp8.py <src_shard_dir> <dst_shard_dir>

Example:
    python scripts/requantize_fp8.py ./shards/mistral-7b ./shards/mistral-7b-fp8
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swlp.logging import configure_logging
from swlp.model.quant import requantize_shards

configure_logging("INFO", json_logs=False)
LOGGER = logging.getLogger(__name__)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 1
    src, dst = Path(argv[1]), Path(argv[2])
    if not (src / "shard_manifest.json").is_file():
        print(f"No shard_manifest.json found in {src}")
        return 1

    LOGGER.info("requantize_fp8_start", extra={"src": str(src), "dst": str(dst)})
    start = time.perf_counter()
    manifest = requantize_shards(src, dst, scheme="float8")
    elapsed = time.perf_counter() - start

    src_total = (src / "shard_manifest.json").read_text(encoding="utf-8")
    print(f"\nRe-quantized {manifest.num_layers} layers to FP8 in {elapsed:.1f}s.")
    print(f"FP8 total layer weight: {manifest.total_weight_mb:.1f} MB "
          f"({manifest.total_weight_mb / 1024:.2f} GB)")
    print(f"FP8 per-layer: {manifest.layer_weight_mb:.1f} MB")
    print(f"Manifest (weight_dtype={manifest.weight_dtype}) written to: "
          f"{dst}/shard_manifest.json")
    LOGGER.info("requantize_fp8_done", extra={"elapsed_seconds": elapsed,
                                              "src_manifest_bytes": len(src_total)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
