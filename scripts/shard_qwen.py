"""Phase 6: shard Qwen/Qwen2.5-14B-Instruct to ./shards/qwen2.5-14b using the stream sharder."""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swlp.logging import configure_logging
from swlp.model.shard import shard_model_by_layer

configure_logging("INFO", json_logs=False)
LOGGER = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
OUTPUT_DIR = Path("./shards/qwen2.5-14b")

if __name__ == "__main__":
    LOGGER.info("starting_shard", extra={"model_id": MODEL_ID, "output_dir": str(OUTPUT_DIR)})
    start_time = time.perf_counter()
    
    manifest = shard_model_by_layer(
        model_id=MODEL_ID,
        output_dir=OUTPUT_DIR,
        dtype_str="float16",
    )
    
    elapsed = time.perf_counter() - start_time
    LOGGER.info(
        "shard_done",
        extra={
            "num_layers": manifest.num_layers,
            "layer_weight_mb": manifest.layer_weight_mb,
            "total_weight_mb": manifest.total_weight_mb,
            "model_type": manifest.model_type,
            "elapsed_seconds": elapsed,
        },
    )
    print(f"\nSuccessfully sharded {manifest.num_layers} layers in {elapsed:.1f}s.")
    total_gb = manifest.total_weight_mb / 1024
    print(f"Total weight: {manifest.total_weight_mb:.1f} MB ({total_gb:.2f} GB)")
    print(f"Per-layer: {manifest.layer_weight_mb:.1f} MB")
    print(f"Manifest written to: {OUTPUT_DIR}/shard_manifest.json")
