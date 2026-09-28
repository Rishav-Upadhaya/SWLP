#!/usr/bin/env python3
"""compare_baselines.py — Honest multi-baseline streaming-inference benchmark.

The paper's original comparison was SWLP-vs-AirLLM only. AirLLM lacks prefetch,
so beating it is a weak headline. This harness measures SWLP against the *real*
prior art under one protocol, so the comparison survives a literature-aware
reviewer.

Baselines and where each can run:

  | backend     | mechanism                                   | runs on M5? |
  |-------------|---------------------------------------------|-------------|
  | swlp        | sliding-window SSD→RAM streaming (ours)      | ✅ (shards)  |
  | airllm      | layer-by-layer MLX streaming, no prefetch    | ✅           |
  | accelerate  | HF `device_map` disk offload (the one-liner) | ✅ (cpu+disk)|
  | ollm        | SSD layer+KV streaming, FlashAttention-2     | ❌ CUDA only |
  | zero        | DeepSpeed ZeRO-Inference NVMe offload        | ❌ CUDA only |

`accelerate` is the most important *new* baseline: it is the trivial,
already-installed, lossless prior-art alternative to SWLP, and it runs here.
`ollm` and `zero` need a CUDA GPU (oLLM also needs flash-attn, which does not
build on macOS) — their adapters self-report as unavailable on this machine and
are ready for the Pop!_OS/MX230 box.

SWLP and AirLLM logic is **imported** from ``compare_airllm_swlp`` (no
duplication); page-cache control and provenance come from ``bench_common``.

Usage:
  python scripts/research/compare_baselines.py --backends swlp,accelerate --models 7b
  python scripts/research/compare_baselines.py --backends swlp,accelerate,airllm --cold   # needs sudo
  python scripts/research/compare_baselines.py --list                                     # show availability
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
SCRIPTS_DIR = Path(__file__).parent.resolve()
for _p in (str(PROJECT_ROOT / "src"), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bench_common  # noqa: E402  — shared cold-cache + provenance helpers

# Reuse the existing SWLP + AirLLM adapters and shared constants (no duplication).
from compare_airllm_swlp import (  # noqa: E402
    MAX_NEW_TOKENS,
    MODEL_CONFIGS,
    PROMPTS,
    _build_airllm_runner,
    _build_swlp_runner,
    _fmt,
    _gc,
    _median,
    _peak_rss_gb,
    _run_airllm,
    _run_swlp,
)

# Default forced-offload budget for the Accelerate baseline. Tight enough that
# most layers spill to disk (matching SWLP's "few layers resident" regime), large
# enough that the biggest single layer + embeddings still fit.
ACCEL_DEFAULTS = {
    "accel_cpu_gb": "3GiB",
    "offload_dir": str(PROJECT_ROOT / ".cache" / "accel_offload"),
}


# ─────────────────────────────────────────────────────────────────────────────
# New baseline: HuggingFace Accelerate disk offload (the lossless one-liner)
# ─────────────────────────────────────────────────────────────────────────────

def _accelerate_available(cfg: dict) -> tuple[bool, str]:
    try:
        import accelerate  # noqa: F401
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, f"transformers/accelerate/torch not importable: {exc}"
    return True, "cpu+disk offload (works on any platform)"


def _build_accelerate_runner(cfg: dict) -> Any:
    """Load the model with HF Accelerate forced disk offload.

    Uses ``device_map="auto"`` with a tight ``max_memory`` so most transformer
    blocks live on disk and are streamed CPU↔disk per forward — the stock,
    lossless FP16 prior-art alternative to SWLP. Computes on CPU (the reliable
    macOS path); ``compute_device`` is recorded so the comparison is not
    misread as GPU-vs-GPU.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged = {**ACCEL_DEFAULTS, **cfg}
    model_id = merged["model_id"]
    offload_dir = Path(merged["offload_dir"])
    offload_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Only the real device (cpu) gets a budget; with offload_folder set,
    # device_map="auto" spills the overflow to disk automatically.
    max_memory = {"cpu": merged["accel_cpu_gb"]}
    print(f"    [accelerate] loading {model_id} with disk offload "
          f"(cpu budget {merged['accel_cpu_gb']}, offload → {offload_dir})…")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="auto",
        max_memory=max_memory,
        offload_folder=str(offload_dir),
        offload_state_dict=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tok.pad_token_id
    return {"model": model, "tok": tok, "compute_device": "cpu"}


def _run_accelerate(handle: Any, prompt: str, n_runs: int = 2, cold: bool = False) -> dict:
    """Time HF Accelerate disk-offload generation. TTFT from a 1-token generate;
    throughput from a full ``MAX_NEW_TOKENS`` generate (separate calls so the
    first-token latency is isolated, mirroring the other adapters)."""
    import torch

    model, tok = handle["model"], handle["tok"]
    ttft_list, tps_list, gen_s_list, ram_list = [], [], [], []
    cache_states: list[str] = []
    completion = ""

    inputs = tok(prompt, return_tensors="pt")  # cpu tensors; accelerate hooks move per layer

    for _i in range(n_runs):
        cache_states.append(bench_common.drop_page_cache() if cold else "warm")
        rss_before = _peak_rss_gb()

        with torch.no_grad():
            t0 = time.perf_counter()
            model.generate(**inputs, max_new_tokens=1, do_sample=False)
            ttft_s = time.perf_counter() - t0

            t1 = time.perf_counter()
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            gen_s = time.perf_counter() - t1

        rss_after = _peak_rss_gb()
        new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
        ttft_list.append(ttft_s * 1000)
        tps_list.append(new_tokens / gen_s if gen_s > 0 else 0.0)
        gen_s_list.append(gen_s)
        ram_list.append(max(rss_before, rss_after))
        completion = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        _gc()

    return {
        "ttft_ms": _median(ttft_list),
        "prefill_ms": float("nan"),  # not separately instrumented
        "tps": _median(tps_list),
        "gen_s": _median(gen_s_list),
        "ram_gb": _median(ram_list),
        "cache_state": bench_common.summarize_cache(cache_states),
        "completion": completion.strip(),
        "compute_device": handle.get("compute_device", "cpu"),
        "runs": n_runs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CUDA-only baselines: oLLM and DeepSpeed ZeRO-Inference (scaffolds)
# ─────────────────────────────────────────────────────────────────────────────
# These adapters are NOT validated on this M5 (no CUDA; oLLM needs flash-attn,
# which does not build on macOS). available() gates them off here; the build/run
# bodies are documented starting points to fill in on the CUDA machine.

def _cuda_present() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _ollm_available(cfg: dict) -> tuple[bool, str]:
    try:
        import ollm  # noqa: F401
    except Exception:  # noqa: BLE001
        return False, "oLLM not installed (needs flash-attn; no macOS build) — run on CUDA box"
    if not _cuda_present():
        return False, "oLLM requires a CUDA GPU"
    return True, "ok"


def _build_ollm_runner(cfg: dict) -> Any:  # pragma: no cover - CUDA only
    # Scaffold — verify against the installed oLLM API on the CUDA machine:
    #   from ollm import Inference
    #   o = Inference(cfg["model_id"], device="cuda:0")
    #   o.ini_model(models_dir=cfg.get("ollm_models_dir"), force_download=False)
    #   o.offload_layers_to_cpu()  # or to_ssd, per oLLM version
    #   return {"o": o, "tok": o.tokenizer}
    raise NotImplementedError("oLLM adapter is a CUDA-only scaffold; implement on the MX230 box.")


def _cuda_only_run(handle: Any, prompt: str, n_runs: int, cold: bool) -> dict:  # pragma: no cover
    """Placeholder run for CUDA-only baselines; never reached on this machine
    (their ``available()`` returns False and ``build()`` raises first)."""
    raise NotImplementedError("CUDA-only baseline; implement build()/run() on the MX230 box.")


def _zero_available(cfg: dict) -> tuple[bool, str]:
    try:
        import deepspeed  # noqa: F401
    except Exception:  # noqa: BLE001
        return False, "deepspeed not installed — run on CUDA box"
    if not _cuda_present():
        return False, "ZeRO-Inference requires a CUDA GPU"
    return True, "ok"


def _build_zero_runner(cfg: dict) -> Any:  # pragma: no cover - CUDA only
    # Scaffold — DeepSpeed ZeRO-Inference with NVMe weight offload:
    #   ds_config = {"zero_optimization": {"stage": 3,
    #       "offload_param": {"device": "nvme", "nvme_path": cfg["nvme_path"]}}}
    #   from transformers.integrations import HfDeepSpeedConfig
    #   _dschf = HfDeepSpeedConfig(ds_config)  # keep alive before from_pretrained
    #   model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], dtype=torch.float16)
    #   ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    #   return {"engine": ds_engine.module, "tok": AutoTokenizer.from_pretrained(cfg["model_id"])}
    raise NotImplementedError("ZeRO-Inference adapter is a CUDA-only scaffold; implement on MX230.")


# ─────────────────────────────────────────────────────────────────────────────
# Baseline registry
# ─────────────────────────────────────────────────────────────────────────────

BuildFn = Callable[[dict], Any]
RunFn = Callable[[Any, str, int, bool], dict]
AvailFn = Callable[[dict], "tuple[bool, str]"]


def _swlp_available(cfg: dict) -> tuple[bool, str]:
    shard_dir = Path(cfg.get("shard_dir", ""))
    if shard_dir.is_dir() and (shard_dir / "shard_manifest.json").exists():
        return True, "shards present"
    return False, f"shards missing at {shard_dir} — run `swlp download`"


def _airllm_available(cfg: dict) -> tuple[bool, str]:
    import platform
    if platform.system() != "Darwin":
        return False, "AirLLM MLX path is Apple-Silicon only here"
    try:
        import airllm  # noqa: F401
        import mlx.core  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, f"airllm/mlx not installed: {exc} (pip install airllm)"
    return True, "ok"


# Each entry: build(cfg)->handle, run(handle, prompt, n, cold)->metrics, available(cfg).
BASELINES: dict[str, dict[str, Any]] = {
    "swlp": {
        "label": "SWLP (W=2 FP16 streaming)",
        "available": _swlp_available,
        "build": lambda cfg: _build_swlp_runner(cfg),                 # (runner, app_cfg)
        "run": lambda h, p, n, cold: _run_swlp(h[0], h[1], p, n_runs=n, cold=cold),
    },
    "airllm": {
        "label": "AirLLM (MLX layer-by-layer)",
        "available": _airllm_available,
        "build": lambda cfg: _build_airllm_runner(cfg["airllm_id"]),
        "run": lambda h, p, n, cold: _run_airllm(h, p, n_runs=n, cold=cold),
    },
    "accelerate": {
        "label": "HF Accelerate (disk offload)",
        "available": _accelerate_available,
        "build": _build_accelerate_runner,
        "run": _run_accelerate,
    },
    "ollm": {
        "label": "oLLM (SSD streaming, CUDA)",
        "available": _ollm_available,
        "build": _build_ollm_runner,
        "run": _cuda_only_run,
    },
    "zero": {
        "label": "DeepSpeed ZeRO-Inference (NVMe, CUDA)",
        "available": _zero_available,
        "build": _build_zero_runner,
        "run": _cuda_only_run,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Run loop + reporting
# ─────────────────────────────────────────────────────────────────────────────

def _warmup(backend: str, handle: Any, prompt: str) -> None:
    """One discarded run to warm model/page caches, matching the other harness."""
    try:
        if backend == "swlp":
            handle[0].run(prompt, profile=False)
        elif backend == "airllm":
            import mlx.core as mx
            x = mx.array([handle.tokenizer.encode(prompt)])
            for _tok in handle.model_generate(x, temperature=0):
                break
        elif backend == "accelerate":
            import torch
            inp = handle["tok"](prompt, return_tensors="pt")
            with torch.no_grad():
                handle["model"].generate(**inp, max_new_tokens=1, do_sample=False)
    except Exception:  # noqa: BLE001
        pass
    _gc()


def _list_availability(model_keys: list[str]) -> None:
    print("\nBaseline availability on this machine:")
    print("  " + "─" * 72)
    for mkey in model_keys:
        cfg = {**ACCEL_DEFAULTS, **MODEL_CONFIGS[mkey]}
        print(f"  model: {cfg['label']}")
        for name, spec in BASELINES.items():
            ok, reason = spec["available"](cfg)
            mark = "✅" if ok else "⏭️ "
            print(f"    {mark} {name:<11} — {reason}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Honest multi-baseline streaming benchmark")
    parser.add_argument("--backends", default="swlp,accelerate",
                        help="comma list from: swlp,airllm,accelerate,ollm,zero")
    parser.add_argument("--models", default="7b", help="comma list of model keys (default: 7b)")
    parser.add_argument("--runs", type=int, default=2, help="timed runs per prompt (default: 2)")
    parser.add_argument("--cold", action="store_true",
                        help="drop OS page cache before each timed run (needs sudo)")
    parser.add_argument("--offload-dir", default=None, help="Accelerate disk-offload folder")
    parser.add_argument("--list", action="store_true",
                        help="print which baselines can run here, then exit")
    args = parser.parse_args()

    model_keys = [k.strip() for k in args.models.split(",") if k.strip() in MODEL_CONFIGS]
    if not model_keys:
        print("No valid model keys. Available:", ", ".join(MODEL_CONFIGS))
        return

    if args.list:
        _list_availability(model_keys)
        return

    backends = [b.strip() for b in args.backends.split(",") if b.strip() in BASELINES]
    n_runs = max(1, args.runs)
    cold = args.cold
    print(f"  backends: {backends}")
    print(f"  cache mode: {'COLD (page cache dropped between runs)' if cold else 'warm (default)'}")

    out_dir = PROJECT_ROOT / "benchmarks"
    out_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = out_dir / f"baselines_{ts}.json"
    all_results: dict[str, Any] = {}

    def _persist() -> None:
        with open(out_path, "w") as f:
            json.dump({
                "timestamp": ts,
                "max_new_tokens": MAX_NEW_TOKENS,
                "runs_per_prompt": n_runs,
                "cache_mode": "cold" if cold else "warm",
                "backends": backends,
                "provenance": bench_common.provenance(),
                "results": all_results,
            }, f, indent=2)

    for mkey in model_keys:
        cfg = {**ACCEL_DEFAULTS, **MODEL_CONFIGS[mkey]}
        if args.offload_dir:
            cfg["offload_dir"] = args.offload_dir
        label = cfg["label"]
        print(f"\n{'═' * 70}\n  MODEL: {label}  ({cfg['model_id']})\n{'═' * 70}")
        model_results: dict[str, dict] = {b: {} for b in backends}
        all_results[mkey] = model_results

        for backend in backends:
            spec = BASELINES[backend]
            ok, reason = spec["available"](cfg)
            if not ok:
                print(f"\n  ── {spec['label']} — ⏭️  skipped: {reason} ──")
                model_results[backend] = {"_skipped": reason}
                _persist()
                continue

            print(f"\n  ── {spec['label']} ──")
            try:
                handle = spec["build"](cfg)
                print("  loaded ✓")
                for pid, prompt in PROMPTS:
                    shown = prompt if len(prompt) <= 55 else prompt[:55] + "…"
                    print(f"  [{pid}] '{shown}'", end="", flush=True)
                    _warmup(backend, handle, prompt)
                    res = spec["run"](handle, prompt, n_runs, cold)
                    model_results[backend][pid] = res
                    _persist()
                    print(f" → {_fmt(res['tps'], 3)} tok/s, TTFT {_fmt(res['ttft_ms'], 0)}ms "
                          f"[{res.get('cache_state', 'warm')}]")
                del handle
                _gc()
            except Exception as exc:  # noqa: BLE001
                print(f"\n  [{backend} ERROR] {exc}")
                import traceback
                traceback.print_exc()
                model_results[backend]["_error"] = str(exc)
                _persist()

        _report_model(label, backends, model_results)

    _persist()
    print(f"\n✓ Results saved → {out_path}")


def _report_model(label: str, backends: list[str], results: dict[str, dict]) -> None:
    """Per-prompt table across all backends, then a tok/s summary."""
    print(f"\n{'─' * 70}\n  RESULTS: {label}\n{'─' * 70}")
    print(f"  {'Prompt':<14} {'Backend':<14} {'TTFT':>10} {'tok/s':>10} {'RAM':>9}  cache")
    print(f"  {'─' * 74}")
    for pid, _prompt in PROMPTS:
        for backend in backends:
            res = results.get(backend, {}).get(pid)
            if not res or "_error" in results.get(backend, {}):
                continue
            print(f"  {pid:<14} {backend:<14} "
                  f"{_fmt(res['ttft_ms'], 0, 'ms'):>10} {_fmt(res['tps'], 3):>10} "
                  f"{_fmt(res['ram_gb'], 2, 'GB'):>9}  {res.get('cache_state', 'warm')}")
        print()

    # tok/s summary (median across prompts), with SWLP as the reference if present.
    print(f"  {'Backend':<32} {'median tok/s':>14} {'vs SWLP':>12}")
    print(f"  {'─' * 60}")
    medians: dict[str, float] = {}
    for backend in backends:
        vals = [results.get(backend, {}).get(pid, {}).get("tps")
                for pid, _ in PROMPTS]
        vals = [v for v in vals if isinstance(v, (int, float)) and v == v]
        medians[backend] = _median(vals) if vals else float("nan")
    ref = medians.get("swlp")
    for backend in backends:
        m = medians[backend]
        if ref and m == m and backend != "swlp":
            rel = f"{m / ref:.2f}×" if ref else "—"
        else:
            rel = "ref" if backend == "swlp" else "—"
        print(f"  {BASELINES[backend]['label']:<32} {_fmt(m, 3):>14} {rel:>12}")
    print()


if __name__ == "__main__":
    main()
