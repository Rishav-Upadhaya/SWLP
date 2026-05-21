#!/usr/bin/env python3
"""
compare_airllm_swlp.py — Rigorous head-to-head benchmark: AirLLM vs SWLP.

Models tested:
  1. Mistral-7B  (unsloth/mistral-7b-instruct-v0.2)   — full SWLP vs AirLLM
  2. Mistral-Small-24B  (mistralai/Mistral-Small-24B-Instruct-2501) — SWLP vs AirLLM

Metrics per run:
  • TTFT (ms)          — wall-clock from generation start → first token emitted
  • Prefill (ms)       — time for the full prefill sweep (SWLP only; N/A for AirLLM)
  • Throughput (tok/s) — new tokens generated / total generate wall time
  • RAM peak (GB)      — psutil RSS high-water mark during generation
  • Completion         — raw decoded text (for quality comparison)

Test prompts (3 per model, consistent across backends):
  P1  "What is 2+2?"                                    ← minimal, TTFT-focused
  P2  "Explain how transformer attention works in one paragraph."  ← medium
  P3  "Write a detailed description of the solar system."  ← longer generation

Each prompt: 1 warmup run (not counted) + 2 timed runs → median reported.
max_new_tokens = 32, temperature = 0 (greedy), seed = 42.

Usage:
  python scripts/compare_airllm_swlp.py [--models 7b] [--skip-airllm] [--runs N]
"""

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import psutil

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

PROMPTS = [
    ("P1-minimal",  "What is 2+2?"),
    ("P2-medium",   "Explain how transformer attention works in one paragraph."),
    ("P3-long",     "Write a detailed description of the solar system."),
]

MAX_NEW_TOKENS = 32
TEMPERATURE    = 0.0

MODEL_CONFIGS = {
    "7b": {
        "label":      "Mistral-7B",
        "model_id":   "unsloth/mistral-7b-instruct-v0.2",
        "shard_dir":  str(PROJECT_ROOT / "shards" / "mistral-7b"),
        "window":     2,
        "device":     "mps",
        "airllm_id":  "unsloth/mistral-7b-instruct-v0.2",
    },
    "24b": {
        "label":      "Mistral-Small-24B",
        "model_id":   "mistralai/Mistral-Small-24B-Instruct-2501",
        "shard_dir":  str(PROJECT_ROOT / "shards" / "Mistral-Small-24B-Instruct-2501"),
        "window":     2,
        "device":     "mps",
        "airllm_id":  "mistralai/Mistral-Small-24B-Instruct-2501",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _peak_rss_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)


def _median(vals: list[float]) -> float:
    return statistics.median(vals) if vals else float("nan")


def _pct_token_overlap(a: str, b: str) -> float:
    """Rough token-level overlap between two decoded strings (whitespace split)."""
    ta, tb = a.strip().split(), b.strip().split()
    if not ta or not tb:
        return 0.0
    shared = sum(1 for w in ta if w in set(tb))
    return round(100.0 * shared / max(len(ta), len(tb)), 1)


def _gc() -> None:
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# SWLP runner
# ─────────────────────────────────────────────────────────────────────────────

def _build_swlp_runner(cfg: dict) -> Any:
    """Build a SWLP AppConfig + runner (model loaded on first run() call)."""
    from swlp.config import AppConfig, ModelConfig, CacheConfig, GenerationConfig, RuntimeConfig
    from swlp.runner import build_runner

    app_cfg = AppConfig(
        model=ModelConfig(model_id=cfg["model_id"]),
        cache=CacheConfig(),
        generation=GenerationConfig(
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=False,
            seed=42,
        ),
        runtime=RuntimeConfig(
            device=cfg["device"],
            dtype="float16",
            backend="swlp",
            swlp_window_size=cfg["window"],
            swlp_prefetch_depth=2,
            swlp_prefetch=True,
            shard_dir=cfg["shard_dir"],
            swlp_residency="auto",
            profile=True,
            log_level="WARNING",
        ),
    )
    # runner.run(prompt) calls self.load() internally (idempotent after first call)
    runner = build_runner(app_cfg)
    return runner, app_cfg


def _run_swlp(runner: Any, app_cfg: Any, prompt: str, n_runs: int = 2) -> dict:
    """Run SWLP n_runs times on prompt, return median metrics.

    runner.run(prompt, profile=True) is the correct call signature —
    the AppConfig is stored on the runner at construction time;
    run() accepts the prompt string directly.
    """
    ttft_list, prefill_list, tps_list, gen_s_list, ram_list = [], [], [], [], []
    completion = ""

    for _i in range(n_runs):
        rss_before = _peak_rss_gb()

        t0 = time.perf_counter()
        result = runner.run(prompt, profile=True)
        wall = time.perf_counter() - t0

        rss_after = _peak_rss_gb()

        m = result.metrics
        ttft_ms = (m.time_to_first_token_seconds or 0.0) * 1000
        prefill_ms = (m.prefill_seconds or 0.0) * 1000
        ttft_list.append(ttft_ms)
        prefill_list.append(prefill_ms)
        tps_list.append(m.throughput_tokens_per_second or 0.0)
        gen_s_list.append(wall)
        # Use runner-reported RAM if available, otherwise psutil RSS
        rss_peak = max(rss_before, rss_after)
        if m.ram_peak_bytes:
            rss_peak = max(rss_peak, m.ram_peak_bytes / (1024 ** 3))
        ram_list.append(rss_peak)
        completion = result.completion or ""
        _gc()

    return {
        "ttft_ms":    _median(ttft_list),
        "prefill_ms": _median(prefill_list),
        "tps":        _median(tps_list),
        "gen_s":      _median(gen_s_list),
        "ram_gb":     _median(ram_list),
        "completion": completion.strip(),
        "runs":       n_runs,
    }

# ─────────────────────────────────────────────────────────────────────────────
# AirLLM runner
# ─────────────────────────────────────────────────────────────────────────────

def _build_airllm_runner(model_id: str) -> Any:
    """Load AirLLMLlamaMlx; returns (model, tokenizer) or raises."""
    import mlx.core as mx  # noqa: F401  — ensure MLX is ready
    from airllm import AirLLMLlamaMlx

    print(f"    [AirLLM] loading {model_id} (may create layer shards on first run)…")
    model = AirLLMLlamaMlx(model_id, max_seq_len=512, profiling_mode=False)
    return model


def _run_airllm(model: Any, prompt: str, n_runs: int = 2) -> dict:
    """Run AirLLM n_runs times on prompt, return median metrics."""
    import mlx.core as mx

    ttft_list, tps_list, gen_s_list, ram_list = [], [], [], []
    completion = ""

    tokenizer = model.tokenizer
    ids = tokenizer.encode(prompt)
    x = mx.array([ids])

    for i in range(n_runs):
        rss_before = _peak_rss_gb()
        t_start = time.perf_counter()
        ttft: float | None = None
        tokens = []

        for token in model.model_generate(x, temperature=0):
            if ttft is None:
                ttft = time.perf_counter() - t_start
            tokens.append(token)
            if len(tokens) >= MAX_NEW_TOKENS:
                break

        gen_s = time.perf_counter() - t_start
        rss_after = _peak_rss_gb()

        ttft_list.append((ttft or gen_s) * 1000)
        tps_list.append(len(tokens) / gen_s if gen_s > 0 else 0.0)
        gen_s_list.append(gen_s)
        ram_list.append(max(rss_before, rss_after))
        completion = tokenizer.decode([t.item() for t in tokens])
        _gc()

    return {
        "ttft_ms":    _median(ttft_list),
        "prefill_ms": float("nan"),   # not available from AirLLM
        "tps":        _median(tps_list),
        "gen_s":      _median(gen_s_list),
        "ram_gb":     _median(ram_list),
        "completion": completion.strip(),
        "runs":       n_runs,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v: float, decimals: int = 2, suffix: str = "") -> str:
    if v != v:  # nan
        return "N/A"
    return f"{v:.{decimals}f}{suffix}"


def _print_row(label: str, swlp: dict | None, air: dict | None,
               swlp_comp: str = "", air_comp: str = "") -> None:
    def cell(d: dict | None, key: str, dec: int = 2, suf: str = "") -> str:
        if d is None:
            return "—"
        return _fmt(d.get(key, float("nan")), dec, suf)

    print(f"  {label:<22} "
          f"SWLP {cell(swlp,'ttft_ms',0,'ms'):>9}  "
          f"{cell(swlp,'prefill_ms',0,'ms'):>9}  "
          f"{cell(swlp,'tps',3,' tok/s'):>12}  "
          f"{cell(swlp,'ram_gb',2,' GB'):>8}  "
          f"|| "
          f"AirLLM {cell(air,'ttft_ms',0,'ms'):>9}  "
          f"{'N/A':>9}  "
          f"{cell(air,'tps',3,' tok/s'):>12}  "
          f"{cell(air,'ram_gb',2,' GB'):>8}")


def _print_comparison_header() -> None:
    print()
    print(f"  {'Prompt':<22} "
          f"{'SWLP TTFT':>14} {'SWLP Prefill':>14} {'SWLP tok/s':>14} {'SWLP RAM':>10}  "
          f"  "
          f"{'AirLLM TTFT':>16} {'AirLLM Prefill':>14} {'AirLLM tok/s':>14} {'AirLLM RAM':>10}")
    print("  " + "─" * 130)


def _quality_line(swlp_comp: str, air_comp: str) -> str:
    if not air_comp:
        return "  Quality: AirLLM completion unavailable"
    overlap = _pct_token_overlap(swlp_comp, air_comp)
    match = "✅ identical" if swlp_comp.strip() == air_comp.strip() else f"~{overlap}% word overlap"
    return f"  Quality: SWLP vs AirLLM → {match}"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AirLLM vs SWLP benchmark")
    parser.add_argument("--models",       default="7b,24b",
                        help="comma-separated model keys: 7b,24b (default: 7b,24b)")
    parser.add_argument("--skip-airllm",  action="store_true",
                        help="skip AirLLM and run SWLP only")
    parser.add_argument("--skip-swlp",   action="store_true",
                        help="skip SWLP and run AirLLM only")
    parser.add_argument("--runs",         type=int, default=2,
                        help="timed runs per prompt (default: 2)")
    args = parser.parse_args()

    model_keys = [k.strip() for k in args.models.split(",") if k.strip()]
    n_runs = max(1, args.runs)
    skip_swlp = args.skip_swlp

    all_results: dict[str, dict] = {}

    for mkey in model_keys:
        if mkey not in MODEL_CONFIGS:
            print(f"[WARN] unknown model key '{mkey}', skipping")
            continue

        cfg   = MODEL_CONFIGS[mkey]
        label = cfg["label"]
        print(f"\n{'═'*70}")
        print(f"  MODEL: {label}  ({cfg['model_id']})")
        print(f"{'═'*70}")

        model_results: dict[str, dict[str, dict]] = {"swlp": {}, "airllm": {}}

        # ── SWLP ──────────────────────────────────────────────────────────
        if skip_swlp:
            print(f"\n  ── SWLP skipped (--skip-swlp) ──")
        else:
            print(f"\n  ── SWLP (W={cfg['window']}, FP16 streaming) ──")
            try:
                runner, app_cfg = _build_swlp_runner(cfg)
                print(f"  SWLP loaded ✓")

                for pid, prompt in PROMPTS:
                    print(f"  [{pid}] '{prompt[:55]}…'" if len(prompt) > 55 else f"  [{pid}] '{prompt}'", end="", flush=True)
                    # warmup (one run to warm model cache; result discarded)
                    try:
                        runner.run(prompt, profile=False)
                    except Exception:
                        pass
                    _gc()
                    # timed runs
                    res = _run_swlp(runner, app_cfg, prompt, n_runs=n_runs)
                    model_results["swlp"][pid] = res
                    print(f" → {_fmt(res['tps'],3)} tok/s, TTFT {_fmt(res['ttft_ms'],0)}ms")

                del runner, app_cfg
                _gc()

            except Exception as exc:
                print(f"\n  [SWLP ERROR] {exc}")
                import traceback; traceback.print_exc()

        # ── AirLLM ────────────────────────────────────────────────────────
        if not args.skip_airllm:
            print(f"\n  ── AirLLM (FP16 MLX, layer-by-layer) ──")
            try:
                air_model = _build_airllm_runner(cfg["airllm_id"])
                print(f"  AirLLM loaded ✓")

                for pid, prompt in PROMPTS:
                    print(f"  [{pid}] '{prompt[:55]}…'" if len(prompt) > 55 else f"  [{pid}] '{prompt}'", end="", flush=True)
                    # warmup (one short run)
                    try:
                        ids = air_model.tokenizer.encode(prompt)
                        import mlx.core as mx
                        x = mx.array([ids])
                        for tok in air_model.model_generate(x, temperature=0):
                            break  # just prefill
                    except Exception:
                        pass
                    _gc()
                    # timed runs
                    res = _run_airllm(air_model, prompt, n_runs=n_runs)
                    model_results["airllm"][pid] = res
                    print(f" → {_fmt(res['tps'],3)} tok/s, TTFT {_fmt(res['ttft_ms'],0)}ms")

                del air_model
                _gc()

            except Exception as exc:
                print(f"\n  [AirLLM ERROR] {exc}")
                import traceback; traceback.print_exc()

        all_results[mkey] = model_results

        # ── Print comparison table ─────────────────────────────────────────
        print(f"\n{'─'*70}")
        print(f"  RESULTS: {label}")
        print(f"{'─'*70}")
        print(f"  {'Prompt':<22}  {'TTFT':>10}  {'Prefill':>10}  {'tok/s':>10}  {'RAM':>8}  {'Backend'}")
        print(f"  {'─'*80}")

        for pid, prompt in PROMPTS:
            for backend in ("swlp", "airllm"):
                res = model_results.get(backend, {}).get(pid)
                if res is None:
                    continue
                pf_str = _fmt(res['prefill_ms'], 0, 'ms') if res['prefill_ms'] == res['prefill_ms'] else "N/A"
                print(f"  {pid:<22}  "
                      f"{_fmt(res['ttft_ms'],0,'ms'):>10}  "
                      f"{pf_str:>10}  "
                      f"{_fmt(res['tps'],3):>10}  "
                      f"{_fmt(res['ram_gb'],2,'GB'):>8}  "
                      f"{'SWLP' if backend=='swlp' else 'AirLLM'}")

            # Quality comparison
            sw = model_results.get("swlp", {}).get(pid, {}).get("completion", "")
            ai = model_results.get("airllm", {}).get(pid, {}).get("completion", "")
            if sw or ai:
                overlap = _pct_token_overlap(sw, ai) if sw and ai else 0.0
                match_str = "identical ✅" if sw.strip() == ai.strip() else f"~{overlap:.0f}% word overlap"
                print(f"  {'  quality':>22}  {'SWLP vs AirLLM: ' + match_str:>50}")
                print(f"    SWLP:   {sw[:90]!r}")
                if ai:
                    print(f"    AirLLM: {ai[:90]!r}")
            print()

    # ── Save JSON ────────────────────────────────────────────────────────────
    out_dir = PROJECT_ROOT / "benchmarks"
    out_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = out_dir / f"airllm_vs_swlp_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({"timestamp": ts, "max_new_tokens": MAX_NEW_TOKENS,
                   "runs_per_prompt": n_runs, "results": all_results}, f, indent=2)
    print(f"\n✓ Results saved → {out_path}")

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("  SUMMARY — Speedup: SWLP vs AirLLM  (median across prompts)")
    print("═" * 80)
    print(f"  {'Model':<20}  {'Metric':<20}  {'SWLP':>12}  {'AirLLM':>12}  {'SWLP/AirLLM':>14}")
    print(f"  {'─'*78}")

    for mkey in model_keys:
        mr = all_results.get(mkey, {})
        sw_all = [mr.get("swlp", {}).get(pid, {}) for _, pid, *_ in [(p, p) for p, _ in PROMPTS]]
        # fix: use pid correctly
        sw_all = [mr.get("swlp", {}).get(pid, {}) for pid, _ in PROMPTS]
        ai_all = [mr.get("airllm", {}).get(pid, {}) for pid, _ in PROMPTS]

        label = MODEL_CONFIGS[mkey]["label"]

        for metric, key, dec, suf in [
            ("TTFT (ms)",       "ttft_ms", 0, "ms"),
            ("Prefill (ms)",    "prefill_ms", 0, "ms"),
            ("Throughput",      "tps",     3, " tok/s"),
            ("RAM peak (GB)",   "ram_gb",  2, " GB"),
        ]:
            sw_vals = [d.get(key) for d in sw_all if d.get(key) is not None]
            ai_vals = [d.get(key) for d in ai_all if d.get(key) is not None]

            sw_med = _median(sw_vals) if sw_vals else float("nan")
            ai_med = _median(ai_vals) if ai_vals else float("nan")

            if key in ("ttft_ms", "prefill_ms"):
                ratio = f"{ai_med/sw_med:.1f}× slower" if sw_med and ai_med else "N/A"
            elif key == "tps":
                ratio = f"{sw_med/ai_med:.1f}× faster" if sw_med and ai_med else "N/A"
            elif key == "ram_gb":
                ratio = f"{ai_med/sw_med:.1f}× more RAM" if sw_med and ai_med else "N/A"
            else:
                ratio = "N/A"

            print(f"  {label:<20}  {metric:<20}  "
                  f"{_fmt(sw_med, dec, suf):>12}  "
                  f"{_fmt(ai_med, dec, suf):>12}  "
                  f"{ratio:>14}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
