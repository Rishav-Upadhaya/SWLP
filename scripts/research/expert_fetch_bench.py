#!/usr/bin/env python3
"""Micro-benchmark: expert miss-path fetch strategies (Phase 25 perf round).

Measures, on a synthetic expert bank sized to real per-expert weights:
  1. serial ensure() loop vs prepare_set() concurrent staging (cold page cache)
  2. warm repeat reads (macOS unified buffer cache serving repeats from RAM)

Usage:
    python scripts/research/expert_fetch_bench.py --experts 256 --bytes-per-expert-mb 12.7

Defaults mirror DeepSeek-V4-Flash: 256 experts x ~12.7 MB.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.bench_common import provenance  # noqa: E402


def build_bank(root: Path, num_experts: int, expert_mb: float) -> Path:
    """Synthetic v2 bank. I=H=1024 fp16 gives 6 MiB/expert exactly; the
    expert count scales the pool (defaults mirror DSV4-Flash's ~12.7 MB)."""
    from safetensors.torch import save_file

    root.mkdir(parents=True, exist_ok=True)
    intermediate = 1024
    hidden = 1024
    gate_up = torch.zeros(num_experts, 2 * intermediate, hidden, dtype=torch.float16)
    down = torch.zeros(num_experts, hidden, intermediate, dtype=torch.float16)
    save_file(
        {"experts.gate_up_proj": gate_up, "experts.down_proj": down},
        str(root / "layer_000.experts.safetensors"),
    )
    gu_bytes = 2 * intermediate * hidden * 2
    dn_bytes = hidden * intermediate * 2
    header = {
        "experts.gate_up_proj": {
            "dtype": "F16",
            "shape": [num_experts, 2 * intermediate, hidden],
            "data_offsets": [0, num_experts * gu_bytes],
        },
        "experts.down_proj": {
            "dtype": "F16",
            "shape": [num_experts, hidden, intermediate],
            "data_offsets": [num_experts * gu_bytes, num_experts * (gu_bytes + dn_bytes)],
        },
    }
    data_start = 8 + len(json.dumps(header).encode())
    slices = []
    for j in range(num_experts):
        base = data_start + j * gu_bytes
        half = gu_bytes // 2
        slices.append(
            [
                {
                    "slot": "gate",
                    "offset": base,
                    "nbytes": half,
                    "dtype": "F16",
                    "rows": intermediate,
                    "cols": hidden,
                },
                {
                    "slot": "up",
                    "offset": base + half,
                    "nbytes": half,
                    "dtype": "F16",
                    "rows": intermediate,
                    "cols": hidden,
                },
                {
                    "slot": "down",
                    "offset": data_start + num_experts * gu_bytes + j * dn_bytes,
                    "nbytes": dn_bytes,
                    "dtype": "F16",
                    "rows": hidden,
                    "cols": intermediate,
                },
            ]
        )
    index = {
        "0": {
            "bank_file": "layer_000.experts.safetensors",
            "num_experts": num_experts,
            "dtype": "F16",
            "hidden": hidden,
            "intermediate": intermediate,
            "slices": slices,
        }
    }
    (root / "expert_index.json").write_text(json.dumps(index))
    return root


def drop_page_cache(path: Path) -> None:
    """Best-effort cold-cache: macOS purgeable + POSIX_FADV_DONTNEED on Linux."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            if hasattr(os, "POSIX_FADV_DONTNEED"):
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--bytes-per-expert-mb", type=float, default=12.7)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--routed", type=int, default=6, help="distinct experts routed per token (top-k union)"
    )
    parser.add_argument("--output", default="benchmarks/expert-fetch-bench.json")
    args = parser.parse_args()

    import tempfile

    from swlp.model.expert_bank import EXPERT_INDEX_FILE, ExpertIndex

    tmp = Path(tempfile.mkdtemp())
    build_bank(tmp, args.experts, args.bytes_per_expert_mb)
    index_path = tmp / EXPERT_INDEX_FILE
    bank_file = tmp / "layer_000.experts.safetensors"

    from swlp.runner.expert_scheduler import ExpertScheduler as _ES

    def make_sched(mode: str, workers: int) -> _ES:
        return _ES(
            ExpertIndex.load(index_path),
            tmp,
            torch.device("cpu"),
            torch.float16,
            budget_bytes=int(args.bytes_per_expert_mb * 1024 * 1024) * args.experts,
            mode=mode,
            workers=workers,
        )

    import torch

    eids = list(range(args.routed))
    results: dict[str, object] = {}

    for label, mode, workers in (
        ("serial_cold", "off", 1),
        ("staged_cold_w2", "predictive", 2),
        ("staged_cold_w4", "predictive", 4),
    ):
        sched = make_sched(mode, workers)
        module = _attach(sched, 0, args.experts, workers)
        timings = []
        for _rep in range(args.repeats):
            sched._modules[0].reset_slots()
            drop_page_cache(bank_file)
            t0 = time.perf_counter()
            module(torch.randn(4, 1024, dtype=torch.float16), _idx(eids), _w(_idx(eids).shape[1]))
            timings.append(time.perf_counter() - t0)
        results[label] = {
            "median_s": statistics.median(timings),
            "min_s": min(timings),
            "all": timings,
        }
        sched.cleanup()

    # Warm repeats (page cache hot): same experts, no eviction.
    sched = make_sched("off", 1)
    module = _attach(sched, 0, args.experts, 1)
    module(torch.randn(4, 1024, dtype=torch.float16), _idx(eids), _w(_idx(eids).shape[1]))  # warm
    warm = []
    for _ in range(args.repeats):
        module.reset_slots()
        t0 = time.perf_counter()
        module(torch.randn(4, 1024, dtype=torch.float16), _idx(eids), _w(_idx(eids).shape[1]))
        warm.append(time.perf_counter() - t0)
    results["serial_warm"] = {"median_s": statistics.median(warm), "min_s": min(warm), "all": warm}
    sched.cleanup()

    report = {
        "provenance": provenance(),
        "config": vars(args) | {"tmp_bank": str(tmp)},
        "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"{'strategy':<18}{'median':>10}{'min':>10}")
    for k, v in results.items():
        print(f"{k:<18}{v['median_s']:>9.4f}s{v['min_s']:>9.4f}s")
    print(f"\nreport: {out}")
    return 0


def _idx(eids: list[int]) -> torch.Tensor:
    rows = (list(dict.fromkeys(eids)) * 2)[:8]  # 2 tokens x top-4, distinct-heavy
    return torch.tensor([rows[:4], rows[4:8]])


def _w(k: int = 4) -> torch.Tensor:
    return torch.full((2, k), 0.5)


def _attach(sched, layer: int, num_experts: int, workers: int):
    """Attach one SwlpCachedExperts module for `layer` (TYPE_CHECKING import only)."""
    from swlp.runner.experts import SwlpCachedExperts

    li = sched.index.layers[layer]
    module = SwlpCachedExperts(
        layer,
        li.num_experts,
        li.hidden,
        li.intermediate,
        torch.float16,
        torch.device("cpu"),
        "silu",
        sched,
        slots=li.num_experts,
    )
    sched.register(layer, module)
    return module


if __name__ == "__main__":
    import torch  # noqa: E402

    raise SystemExit(main())
