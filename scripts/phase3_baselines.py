"""Phase 3: baseline comparison harness for the SWLP paper table.

Runs Mistral-7B through four local-inference systems on the same prompt and
records throughput, time-to-first-token, peak RAM and the completion text:

  swlp    — SWLP sliding-window streaming (FP16, best window W=2)
  airllm  — AirLLM layer-by-layer streaming (FP16, the paper's competitor)
  mlx     — MLX-lm naive full-model load (FP16, no streaming)
  ollama  — Ollama (quantized Q4_K_M by default — a labelled reference point,
            NOT part of the FP16 equal-quality tier)

Each baseline is isolated in try/except so one failure does not abort the run.
Results are appended to a JSON report consumed by docs/results.md.

Usage:
    # start the ollama server first (separate process):
    ollama serve &
    python scripts/phase3_baselines.py --baseline all
    python scripts/phase3_baselines.py --baseline mlx --out benchmarks/phase3.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swlp.config import load_config  # noqa: E402
from swlp.hardware.detect import detect_hardware, fits_in_memory  # noqa: E402
from swlp.logging import configure_logging  # noqa: E402
from swlp.runner.base import execute_baseline  # noqa: E402

# Rough FP16 model size: 2 bytes × num_params. Mistral-7B ≈ 7e9 × 2 = 14 GB.
_MISTRAL_FP16_BYTES = int(14.0 * 1024 ** 3)

LOGGER = logging.getLogger(__name__)

MODEL_ID = "unsloth/mistral-7b-instruct-v0.2"
PROMPT = "Explain in one sentence what makes a Macbook Air good for development."
MAX_NEW_TOKENS = 32
MLX_REPO = "mlx-community/Mistral-7B-Instruct-v0.2-4bit"  # see bench_mlx note
OLLAMA_MODEL = "mistral"  # Ollama default tag → Q4_K_M
OLLAMA_URL = "http://localhost:11434"
SWLP_CONFIG = Path("configs/swlp_mistral_mps.toml")
SWLP_WINDOW = 2  # best end-to-end window on M5 (Phase 1 finding)


def _record(
    system: str,
    backend: str,
    quant: str,
    tier: str,
    tok_per_s: float | None,
    ttft_s: float | None,
    generate_s: float | None,
    peak_ram_gb: float | None,
    completion: str | None,
    error: str | None = None,
) -> dict:
    return {
        "system": system,
        "backend": backend,
        "quant": quant,
        "tier": tier,  # "fp16" (equal-quality) or "quantized" (reference)
        "model": MODEL_ID,
        "tokens_per_second": round(tok_per_s, 3) if tok_per_s else None,
        "ttft_seconds": round(ttft_s, 4) if ttft_s else None,
        "generate_seconds": round(generate_s, 2) if generate_s else None,
        "peak_ram_gb": round(peak_ram_gb, 3) if peak_ram_gb else None,
        "max_new_tokens": MAX_NEW_TOKENS,
        "completion": completion,
        "error": error,
    }


def bench_ollama() -> dict:
    """Quantized reference point — hits the local ollama HTTP API."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            tags = json.loads(resp.read())
        have = any(OLLAMA_MODEL in m.get("name", "") for m in tags.get("models", []))
        if not have:
            LOGGER.warning("ollama model %s not pulled — run `ollama pull %s`",
                            OLLAMA_MODEL, OLLAMA_MODEL)
            return _record("Ollama", "ollama", "Q4_K_M", "quantized",
                           None, None, None, None, None,
                           error=f"model {OLLAMA_MODEL} not pulled")

        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "prompt": PROMPT,
            "stream": True,
            "options": {"num_predict": MAX_NEW_TOKENS, "temperature": 0.0, "seed": 42},
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate", data=payload,
            headers={"Content-Type": "application/json"},
        )
        chunks: list[str] = []
        ttft: float | None = None
        eval_count = 0
        eval_duration_ns = 0
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                obj = json.loads(line)
                piece = obj.get("response", "")
                if piece and ttft is None:
                    ttft = time.perf_counter() - start
                chunks.append(piece)
                if obj.get("done"):
                    eval_count = int(obj.get("eval_count", 0))
                    eval_duration_ns = int(obj.get("eval_duration", 0))
        tok_s = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns else None
        return _record("Ollama", "ollama", "Q4_K_M", "quantized",
                        tok_s, ttft, time.perf_counter() - start, None,
                        "".join(chunks).strip())
    except (urllib.error.URLError, OSError) as exc:
        return _record("Ollama", "ollama", "Q4_K_M", "quantized",
                        None, None, None, None, None, error=str(exc))


def bench_mlx() -> dict:
    """Naive full-model MLX-lm load.

    NOTE: MLX-lm has no layer streaming — it holds the whole model resident.
    A 7B FP16 model is ~14 GB and does NOT fit on a 16 GB M5 (the local FP16
    conversion aborts with a Metal OOM). That OOM is itself a Phase 3 finding:
    it is the exact reason streaming runtimes (SWLP, AirLLM) exist. To still
    obtain an MLX-lm speed number we measure the 4-bit quantized model, which
    is reported in the *quantized* tier alongside Ollama (NOT equal-quality).
    """
    hw = detect_hardware()
    if not fits_in_memory(_MISTRAL_FP16_BYTES, hw):
        LOGGER.warning(
            "bench_mlx: FP16 model (~14 GB) does not fit available RAM (%.1f GB) "
            "— skipping FP16 load; measuring 4-bit quantized instead",
            hw.memory_gb,
        )
    try:
        from mlx_lm import load, stream_generate

        LOGGER.info("loading %s (4-bit — FP16 naive load OOMs on 16 GB)", MLX_REPO)
        model, tokenizer = load(MLX_REPO)
        start = time.perf_counter()
        ttft: float | None = None
        last = None
        text: list[str] = []
        for resp in stream_generate(model, tokenizer, PROMPT, max_tokens=MAX_NEW_TOKENS):
            if ttft is None:
                ttft = time.perf_counter() - start
            text.append(resp.text)
            last = resp
        generate_s = time.perf_counter() - start
        tok_s = last.generation_tps if last else None
        peak = last.peak_memory if last else None
        return _record("MLX-lm", "mlx", "4bit", "quantized",
                        tok_s, ttft, generate_s, peak, "".join(text).strip())
    except Exception as exc:  # noqa: BLE001 — isolate baseline failure
        return _record("MLX-lm", "mlx", "4bit", "quantized",
                        None, None, None, None, None, error=repr(exc))


def bench_airllm() -> dict:
    """AirLLM layer-by-layer streaming — FP16, the paper's competitor."""
    try:
        import mlx.core as mx
        from airllm import AirLLMLlamaMlx

        model = AirLLMLlamaMlx(MODEL_ID, max_seq_len=512, profiling_mode=False)
        tokenizer = model.tokenizer
        ids = tokenizer.encode(PROMPT)
        x = mx.array([ids])

        start = time.perf_counter()
        ttft: float | None = None
        tokens = []
        for token in model.model_generate(x, temperature=0):
            if ttft is None:
                ttft = time.perf_counter() - start
            tokens.append(token)
            if len(tokens) >= MAX_NEW_TOKENS:
                break
        generate_s = time.perf_counter() - start
        completion = tokenizer.decode([t.item() for t in tokens])
        tok_s = len(tokens) / generate_s if generate_s else None
        return _record("AirLLM", "airllm", "fp16", "fp16",
                        tok_s, ttft, generate_s, None, completion.strip())
    except Exception as exc:  # noqa: BLE001 — isolate baseline failure
        return _record("AirLLM", "airllm", "fp16", "fp16",
                        None, None, None, None, None, error=repr(exc))


def _stats(values: list[float]) -> dict | None:
    """Mean/median/std for a list of floats. Returns None for empty lists."""
    if not values:
        return None
    import statistics as _st
    ordered = sorted(values)
    return {
        "mean": sum(ordered) / len(ordered),
        "median": ordered[len(ordered) // 2],
        "std": _st.pstdev(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "count": len(ordered),
    }


def bench_swlp(runs: int = 1) -> dict:
    """SWLP sliding-window streaming — FP16, best window W=2 (Phase 1).

    When ``runs > 1`` the first run is a warmup (not counted) and subsequent
    runs are aggregated into mean/median/std in the returned record.
    """
    try:
        config = load_config(SWLP_CONFIG)
        config.runtime.swlp_window_size = SWLP_WINDOW
        config.generation.max_new_tokens = MAX_NEW_TOKENS

        tps_list: list[float] = []
        ttft_list: list[float] = []
        gen_list: list[float] = []
        peak_list: list[float] = []
        completion: str | None = None

        total_runs = runs + (1 if runs > 1 else 0)  # +1 warmup when N>1
        for i in range(total_runs):
            is_warmup = runs > 1 and i == 0
            result = execute_baseline(config, PROMPT)
            if is_warmup:
                LOGGER.info("swlp warmup run complete")
                continue
            m = result.metrics
            if m.throughput_tokens_per_second is not None:
                tps_list.append(m.throughput_tokens_per_second)
            if m.time_to_first_token_seconds is not None:
                ttft_list.append(m.time_to_first_token_seconds)
            if m.generate_seconds is not None:
                gen_list.append(m.generate_seconds)
            if m.ram_peak_bytes is not None:
                peak_list.append(m.ram_peak_bytes / 1e9)
            completion = result.completion.strip()

        record = _record(
            "SWLP", "swlp", "fp16", "fp16",
            tps_list[0] if len(tps_list) == 1 else (sum(tps_list) / len(tps_list) if tps_list else None),
            ttft_list[0] if len(ttft_list) == 1 else (sum(ttft_list) / len(ttft_list) if ttft_list else None),
            gen_list[0] if len(gen_list) == 1 else (sum(gen_list) / len(gen_list) if gen_list else None),
            peak_list[0] if len(peak_list) == 1 else (sum(peak_list) / len(peak_list) if peak_list else None),
            completion,
        )
        if runs > 1:
            record["stats"] = {
                "tokens_per_second": _stats(tps_list),
                "ttft_seconds": _stats(ttft_list),
                "generate_seconds": _stats(gen_list),
                "peak_ram_gb": _stats(peak_list),
                "measured_runs": runs,
            }
        return record
    except Exception as exc:  # noqa: BLE001 — isolate baseline failure
        return _record("SWLP", "swlp", "fp16", "fp16",
                        None, None, None, None, None, error=repr(exc))


_NON_SWLP_BASELINES = {
    "ollama": bench_ollama,
    "mlx": bench_mlx,
    "airllm": bench_airllm,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        choices=[*_NON_SWLP_BASELINES, "swlp", "all"],
        default="all",
    )
    parser.add_argument("--out", type=Path, default=Path("benchmarks/phase3.json"))
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of timed SWLP runs (first is a warmup when >1). Default: 1.",
    )
    args = parser.parse_args()

    configure_logging("INFO", json_logs=False)
    selected = (
        [*_NON_SWLP_BASELINES, "swlp"]
        if args.baseline == "all"
        else [args.baseline]
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    if args.out.exists():
        report = json.loads(args.out.read_text(encoding="utf-8"))
    report.setdefault("prompt", PROMPT)
    report.setdefault("model", MODEL_ID)
    report.setdefault("results", {})

    for name in selected:
        LOGGER.info("running baseline: %s", name)
        if name == "swlp":
            record = bench_swlp(runs=args.runs)
        else:
            record = _NON_SWLP_BASELINES[name]()
        report["results"][name] = record
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        status = record.get("error") or f"{record.get('tokens_per_second')} tok/s"
        LOGGER.info("baseline %s done: %s", name, status)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
