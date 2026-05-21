"""Phase 2: KV compression quality + ratio measurement.

Two modes:

  default  — run the same prompt with KV compression OFF then ON, confirm the
             completions are identical (lossless), and report the compression
             ratio (analytical uncompressed KV size / measured compressed bytes).

  --sweep  — sweep zlib compression levels 1,3,6,9 with compression ON and
             report compressed bytes + generate time for each.

Usage:
    python scripts/kv_compare.py --config configs/swlp_mistral_mps.toml
    python scripts/kv_compare.py --config configs/swlp_mistral_mps.toml --sweep
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swlp.config import load_config
from swlp.logging import configure_logging
from swlp.runner.base import execute_baseline

LOGGER = logging.getLogger(__name__)

DEFAULT_PROMPT = "Explain why running large language models locally matters."


def _kv_bytes_per_token(model_id: str, cache_dir: str) -> int:
    """Full-model KV bytes for a single token (all layers, FP16)."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
    n_layer = int(getattr(cfg, "num_hidden_layers", 0))
    n_kv = int(getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", 1)))
    head_dim = int(
        getattr(cfg, "head_dim", 0)
        or getattr(cfg, "hidden_size", 0) // max(getattr(cfg, "num_attention_heads", 1), 1)
    )
    return 2 * n_layer * n_kv * head_dim * 2  # 2 = (k,v), 2 = FP16 bytes


def _run(config, prompt: str, compression: bool, level: int) -> dict:
    config.runtime.backend = "swlp"
    config.runtime.kv_compression = compression
    config.runtime.kv_tiering = compression
    config.runtime.kv_compression_level = level
    config.runtime.swlp_fallback_to_baseline = False
    result = execute_baseline(config, prompt)
    m = result.metrics.to_dict()
    return {
        "completion": result.completion,
        "generate_seconds": m.get("generate_seconds"),
        "compressed_bytes": m.get("kv_cache_compressed_bytes") or 0,
        "peak_total_bytes": m.get("kv_cache_peak_total_bytes") or 0,
        "compressions": m.get("kv_cache_compressions") or 0,
        "decompressions": m.get("kv_cache_decompressions") or 0,
        "input_tokens": m.get("input_tokens") or 0,
        "generated_tokens": m.get("generated_tokens") or 0,
    }


def _compare(config, prompt: str) -> dict:
    kv_per_token = _kv_bytes_per_token(
        config.model.model_id, str(config.cache.cache_dir)
    )
    off = _run(load_config_copy(config), prompt, compression=False, level=0)
    on = _run(load_config_copy(config), prompt, compression=True, level=6)

    seq_len = on["input_tokens"] + on["generated_tokens"]
    uncompressed = kv_per_token * seq_len
    ratio = uncompressed / on["compressed_bytes"] if on["compressed_bytes"] else 0.0
    text_match = off["completion"].strip() == on["completion"].strip()

    return {
        "prompt": prompt,
        "lossless_text_match": text_match,
        "completion_off": off["completion"],
        "completion_on": on["completion"],
        "kv_uncompressed_bytes": uncompressed,
        "kv_compressed_bytes": on["compressed_bytes"],
        "compression_ratio": round(ratio, 3),
        "generate_seconds_off": round(off["generate_seconds"] or 0.0, 2),
        "generate_seconds_on": round(on["generate_seconds"] or 0.0, 2),
        "compressions": on["compressions"],
        "decompressions": on["decompressions"],
    }


def _sweep(config, prompt: str) -> dict:
    kv_per_token = _kv_bytes_per_token(
        config.model.model_id, str(config.cache.cache_dir)
    )
    rows = []
    for level in (1, 3, 6, 9):
        run = _run(load_config_copy(config), prompt, compression=True, level=level)
        seq_len = run["input_tokens"] + run["generated_tokens"]
        uncompressed = kv_per_token * seq_len
        ratio = uncompressed / run["compressed_bytes"] if run["compressed_bytes"] else 0.0
        rows.append(
            {
                "level": level,
                "compressed_bytes": run["compressed_bytes"],
                "compression_ratio": round(ratio, 3),
                "generate_seconds": round(run["generate_seconds"] or 0.0, 2),
            }
        )
        print(
            f"level={level} | ratio={ratio:.3f}x | "
            f"compressed={run['compressed_bytes'] / 1024:.1f} KB | "
            f"generate={run['generate_seconds']:.1f}s"
        )
    return {"prompt": prompt, "sweep": rows}


def load_config_copy(config):
    """Fresh config object so per-run mutations do not leak between runs."""
    from copy import deepcopy

    return deepcopy(config)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/swlp_mistral_mps.toml"))
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--sweep", action="store_true", help="Sweep compression levels 1-9")
    args = parser.parse_args()

    configure_logging("WARNING", json_logs=False)
    config = load_config(args.config)

    if args.sweep:
        report = _sweep(config, args.prompt)
    else:
        report = _compare(config, args.prompt)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
