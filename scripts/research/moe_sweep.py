#!/usr/bin/env python3
"""Phase 25/28 research harness: sweep expert-cache budgets for a sharded MoE model.

Runs the SWLP streaming backend across a list of ``SWLP_EXPERT_CACHE_MB``
values and reports decode throughput plus ExpertScheduler hit-rate statistics
per budget, so the FreeToken-style cache-size/miss-rate curve can be measured
on real hardware (predicted guidance lives in ``swlp doctor``).

Usage:
    python scripts/research/moe_sweep.py --shard-dir ./shards/qwen3-30b-a3b \
        --model Qwen/Qwen3-30B-A3B-Instruct-2507 --budgets 0,2048,6144 \
        --runs 3 --max-tokens 32 --output benchmarks/moe-sweep.json

Requires a MoE shard directory produced by the v2 sharder (expert banks).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.bench_common import provenance, summarize_runs  # noqa: E402
from swlp.config import load_config  # noqa: E402
from swlp.runner.swlp import SWLPRunner  # noqa: E402

PROMPT = (
    "Explain how mixture-of-experts routing works, then summarize the "
    "trade-offs between fine-grained and coarse-grained expert pools."
)


class StatsRunner(SWLPRunner):
    """Captures ExpertScheduler stats at cleanup time (base class logs+nulls)."""

    last_expert_stats: dict[str, object] = {}

    def _cleanup_resources(self, scheduler=None) -> None:
        expert_sched = getattr(self, "_expert_sched", None)
        if expert_sched is not None:
            try:
                StatsRunner.last_expert_stats = dict(expert_sched.stats())
            except Exception:
                StatsRunner.last_expert_stats = {}
        super()._cleanup_resources(scheduler)


def run_budget(
    shard_dir: str, model_id: str, budget_mb: int, runs: int, max_tokens: int
) -> dict[str, object]:
    import os

    os.environ["SWLP_EXPERT_CACHE_MB"] = str(budget_mb)
    os.environ["SWLP_BACKEND"] = "swlp"
    os.environ["SWLP_SHARD_DIR"] = shard_dir
    os.environ["SWLP_MODEL_ID"] = model_id
    os.environ["SWLP_MAX_NEW_TOKENS"] = str(max_tokens)
    os.environ["SWLP_LOG_LEVEL"] = "WARNING"

    throughputs: list[float] = []
    expert_stats: dict[str, object] = {}
    # One runner per budget: load() is idempotent, so model loading is paid
    # once and excluded from the decode-throughput measurement below.
    runner = StatsRunner(load_config())
    runner.run(PROMPT)  # warmup: page cache, resident planning, JIT paths
    for _ in range(max(1, runs)):
        StatsRunner.last_expert_stats = {}
        result = runner.run(PROMPT)
        # Decode-only throughput (generate_seconds), NOT wall time — model
        # load would swamp budget differences on large models.
        tps = result.metrics.throughput_tokens_per_second
        if tps and tps > 0:
            throughputs.append(float(tps))
        if StatsRunner.last_expert_stats:
            expert_stats = StatsRunner.last_expert_stats
        else:
            print(f"  warning: budget {budget_mb} MB produced no expert "
                  "stats (model has no expert banks?)")
    del runner
    return {
        "expert_cache_mb": budget_mb,
        "runs": len(throughputs),
        "throughput_tokens_per_second": summarize_runs(throughputs),
        "expert_stats": expert_stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", required=True, help="MoE shard directory (v2 format)")
    parser.add_argument("--model", required=True, help="HuggingFace model id for the tokenizer")
    parser.add_argument("--budgets", default="0,2048,6144",
                        help="comma-separated SWLP_EXPERT_CACHE_MB values")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--output", default="benchmarks/moe-sweep.json")
    args = parser.parse_args()

    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    results = [
        run_budget(args.shard_dir, args.model, b, args.runs, args.max_tokens)
        for b in budgets
    ]

    report = {
        "schema_version": 1,
        "provenance": provenance(),
        "shard_dir": args.shard_dir,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\n{'BUDGET(MB)':>10} {'MEDIAN t/s':>11} {'HIT RATE':>9}  STATS")
    for row in results:
        tp = row["throughput_tokens_per_second"]
        stats = row["expert_stats"]
        hit = stats.get("hit_rate", stats.get("cache_hit_rate", 0.0))
        med = tp.get("median", 0.0) if isinstance(tp, dict) else 0.0
        print(f"{row['expert_cache_mb']:>10} {med:>11.3f} {hit!s:>9}  {stats}")
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
