#!/usr/bin/env python3
"""Run baseline (HF) and SWLP runners and compare outputs and timing traces.

Minimal utility for Phase 8 prototype validation.
"""
from __future__ import annotations

import json
import os
import time
import difflib
import argparse
from pathlib import Path

from swlp.config import load_config
from swlp.runtime import build_runner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--out", type=str, default="swlp_compare_output.json")
    args = parser.parse_args()

    cfg = load_config()
    prompt = args.prompt or cfg.generation.prompt

    results: dict = {"prompt": prompt, "runs": {}}

    # Baseline (HF)
    cfg.runtime.backend = "hf"
    hf_runner = build_runner(cfg)
    t0 = time.perf_counter()
    hf_res = hf_runner.run(prompt, profile=args.profile)
    t1 = time.perf_counter()
    results["runs"]["baseline"] = {
        "completion": hf_res.completion,
        "metrics": hf_res.metrics.to_dict() if hasattr(hf_res.metrics, "to_dict") else {},
        "elapsed": t1 - t0,
    }

    # SWLP run
    cfg.runtime.backend = "swlp"
    swlp_runner = build_runner(cfg)
    t0 = time.perf_counter()
    swlp_res = swlp_runner.run(prompt, profile=args.profile)
    t1 = time.perf_counter()
    results["runs"]["swlp"] = {
        "completion": swlp_res.completion,
        "metrics": swlp_res.metrics.to_dict() if hasattr(swlp_res.metrics, "to_dict") else {},
        "elapsed": t1 - t0,
    }

    # similarity
    seq = difflib.SequenceMatcher(None, hf_res.completion, swlp_res.completion)
    results["similarity_ratio"] = seq.ratio()

    # collect trace if available on runner
    trace = None
    if hasattr(swlp_runner, "_trace"):
        trace = getattr(swlp_runner, "_trace")
        results["swlp_trace"] = trace

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"Wrote comparison to {out_path}")
    print(f"Similarity ratio: {results['similarity_ratio']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
