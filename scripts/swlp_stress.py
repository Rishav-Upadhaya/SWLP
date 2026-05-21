#!/usr/bin/env python3
"""Stress test SWLP runtime by running repeated runs and checking resource stability."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import psutil

from swlp.config import load_config
from swlp.runtime import build_runner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--out", type=str, default="swlp_stress_results.json")
    args = parser.parse_args()

    cfg = load_config()
    prompt = args.prompt or cfg.generation.prompt
    results = {"iterations": args.iterations, "runs": []}

    for i in range(args.iterations):
        t0 = time.perf_counter()
        runner = build_runner(cfg)
        try:
            res = runner.run(prompt, profile=False)
            t1 = time.perf_counter()
            proc = psutil.Process()
            rss = proc.memory_info().rss
            results["runs"].append({"index": i, "elapsed": t1 - t0, "rss": rss, "generated_len": len(res.completion)})
            print(f"Iteration {i}: elapsed={t1-t0:.3f}s rss={rss}")
        except Exception as exc:
            print(f"Iteration {i} failed: {exc}")
            results["runs"].append({"index": i, "error": str(exc)})
        finally:
            # try some cleanup hints
            try:
                import gc, torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote stress results to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
