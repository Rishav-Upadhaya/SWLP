"""Phase 1 POC: measure ThreadedPipeline overlap on real shards.

Loads each layer .pt shard through ThreadedPipeline with prefetch on vs. off,
then reports the overlap gain. Also does a window-size sweep (W=2,4,6).

Usage:
    python scripts/run_pipeline_forward.py --shard-dir ./shards/mistral-7b
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swlp.core.pipeline import PipelineConfig, ThreadedPipeline
from swlp.logging import configure_logging
from swlp.model.shard import list_layer_paths, load_manifest

LOGGER = logging.getLogger(__name__)


def _simulate_compute(seconds: float) -> None:
    """Stand-in for per-layer compute so prefetch has something to overlap with."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def measure(
    shard_dir: Path,
    num_layers: int,
    window_size: int,
    prefetch: bool,
    compute_seconds_per_layer: float,
) -> dict:
    config = PipelineConfig(shard_dir=str(shard_dir), window_size=window_size, prefetch_depth=2)
    pipe = ThreadedPipeline(config)

    if prefetch:
        pipe.warmup(num_layers=num_layers)

    started = time.perf_counter()
    transfer_total = 0.0
    compute_total = 0.0

    for i in range(num_layers):
        tx_start = time.perf_counter()
        weights = pipe.get_layer(i)
        tx_end = time.perf_counter()
        if weights is None:
            raise RuntimeError(f"layer {i} returned None — shard missing")
        transfer_total += tx_end - tx_start

        next_idx = i + window_size if prefetch else None
        if next_idx is not None and next_idx < num_layers:
            pipe.prefetch(next_idx)

        compute_start = time.perf_counter()
        _simulate_compute(compute_seconds_per_layer)
        compute_total += time.perf_counter() - compute_start

        pipe.evict(i - (window_size - 1))

    total = time.perf_counter() - started
    pipe.cleanup()

    return {
        "window_size": window_size,
        "prefetch": prefetch,
        "total_seconds": total,
        "transfer_blocking_seconds": transfer_total,
        "compute_seconds": compute_total,
        "num_layers": num_layers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, default=Path("./shards/mistral-7b"))
    parser.add_argument("--compute-ms", type=float, default=20.0)
    parser.add_argument("--windows", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--max-layers", type=int, default=0, help="0 = all layers")
    args = parser.parse_args()

    configure_logging("INFO", json_logs=False)

    if not args.shard_dir.exists():
        LOGGER.error("shard_dir_missing", extra={"path": str(args.shard_dir)})
        return 1

    manifest = load_manifest(args.shard_dir)
    layer_files = list_layer_paths(args.shard_dir)
    num_layers = args.max_layers or min(len(layer_files), manifest.num_layers)
    compute_seconds = args.compute_ms / 1000.0

    LOGGER.info(
        "poc_start",
        extra={
            "shard_dir": str(args.shard_dir),
            "model": manifest.model_id,
            "num_layers": num_layers,
            "layer_weight_mb": manifest.layer_weight_mb,
            "compute_ms": args.compute_ms,
        },
    )

    rows: list[dict] = []
    for window in args.windows:
        seq = measure(
            args.shard_dir,
            num_layers,
            window_size=window,
            prefetch=False,
            compute_seconds_per_layer=compute_seconds,
        )
        pipe = measure(
            args.shard_dir,
            num_layers,
            window_size=window,
            prefetch=True,
            compute_seconds_per_layer=compute_seconds,
        )
        gain = (seq["total_seconds"] - pipe["total_seconds"]) / seq["total_seconds"]
        rows.append({**pipe, "sequential_seconds": seq["total_seconds"], "overlap_gain": gain})
        print(
            f"W={window:>2} | seq={seq['total_seconds']:.3f}s | "
            f"pipe={pipe['total_seconds']:.3f}s | gain={gain*100:5.1f}% | "
            f"transfer_blocked={pipe['transfer_blocking_seconds']:.3f}s"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
