"""Phase 1: shard unsloth/mistral-7b-instruct-v0.2 to ./shards/mistral-7b."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swlp.logging import configure_logging
from swlp.model.shard import shard_model_by_layer

configure_logging("INFO", json_logs=False)
LOGGER = logging.getLogger(__name__)

MODEL_ID = "unsloth/mistral-7b-instruct-v0.2"
OUTPUT_DIR = Path("./shards/mistral-7b")

if __name__ == "__main__":
    LOGGER.info("starting_shard", extra={"model_id": MODEL_ID, "output_dir": str(OUTPUT_DIR)})
    manifest = shard_model_by_layer(
        model_id=MODEL_ID,
        output_dir=OUTPUT_DIR,
        dtype_str="float16",
    )
    LOGGER.info(
        "shard_done",
        extra={
            "num_layers": manifest.num_layers,
            "layer_weight_mb": manifest.layer_weight_mb,
            "total_weight_mb": manifest.total_weight_mb,
            "model_type": manifest.model_type,
        },
    )
    print(f"Sharded {manifest.num_layers} layers, {manifest.total_weight_mb:.1f} MB total")
    print(f"Per-layer: {manifest.layer_weight_mb:.1f} MB")
    print(f"Manifest: {OUTPUT_DIR}/shard_manifest.json")
