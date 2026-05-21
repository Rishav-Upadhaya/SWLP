"""Formatted command reference for the SWLP CLI.

``print_swlp_help()`` prints a concise, categorised reference — a friendlier
alternative to argparse's default ``--help`` output.  Called by both
``swlp help`` and bare ``swlp`` (no arguments).
"""
from __future__ import annotations

import sys


def _color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def print_swlp_help() -> None:
    """Print the SWLP command reference to stdout."""
    b, d, r = ("", "", "")
    if _color():
        b, d, r = "\033[1m", "\033[2m", "\033[0m"

    W = 78  # total rule width

    def rule() -> None:
        print(d + "─" * W + r)

    def section(title: str) -> None:
        print(f"\n{b}{title}{r}")

    def cmd(command: str, description: str, pad: int = 54) -> None:
        print(f"  {command:<{pad}}  {d}{description}{r}")

    def row(name: str, value: str, pad: int = 14) -> None:
        print(f"  {b}{name:<{pad}}{r}  {value}")

    # ── header ────────────────────────────────────────────────────────────
    print(f"\n{b}SWLP{r} — run large language models on consumer hardware, without quantization.\n")
    rule()

    # ── INFERENCE ─────────────────────────────────────────────────────────
    section("INFERENCE")
    cmd('swlp --model mistral-7b --prompt "..."',
        "HuggingFace backend  (auto device)")
    cmd('swlp --model mistral-7b --backend mlx --quant int8 --prompt "..."',
        "MLX fast inference   (Apple Silicon, recommended)")
    cmd('swlp --model mistral-7b --backend mlx --quant int4 --prompt "..."',
        "MLX fastest          (minor quality tradeoff)")
    cmd('swlp --shard-dir ./shards/mistral-7b --window 2 --prompt "..."',
        "SWLP streaming       (huge models, low RAM)")
    cmd('swlp --backend mock --prompt "..."',
        "Offline / no model needed")

    # ── CHAT ──────────────────────────────────────────────────────────────
    section("CHAT")
    cmd("swlp chat --model mistral-7b",
        "Chat with HuggingFace backend")
    cmd("swlp chat --model mistral-7b --backend mlx --quant int8",
        "Chat with MLX  (recommended on Apple Silicon)")
    cmd("swlp chat --shard-dir ./shards/mistral-7b --window 2",
        "Chat with SWLP streaming")
    cmd("swlp chat --max-chat-tokens 1024",
        "Set max reply length  (default: 512)")

    # ── DOWNLOAD ──────────────────────────────────────────────────────────
    section("DOWNLOAD MODELS  (one-time; saves shards for --shard-dir)")
    cmd("swlp download --model mistral-7b",   "Mistral 7B Instruct    13.9 GB FP16")
    cmd("swlp download --model mistral-24b",  "Mistral 24B Instruct   48 GB  FP16")
    cmd("swlp download --model qwen-0.5b",    "Qwen2.5 0.5B Instruct  1 GB   FP16")
    cmd("swlp download --model qwen-1.5b",    "Qwen2.5 1.5B Instruct  3 GB   FP16")
    cmd("swlp download --model qwen-3b",      "Qwen2.5 3B Instruct    6 GB   FP16")
    cmd("swlp download --model qwen-7b",      "Qwen2.5 7B Instruct    14 GB  FP16")
    cmd("swlp download --model qwen-14b",     "Qwen2.5 14B Instruct   26 GB  FP16")
    cmd("swlp download --model smollm-1.7b",  "SmolLM2 1.7B Instruct  3.4 GB FP16")
    cmd("swlp download --model smollm-360m",  "SmolLM2 360M Instruct  720 MB FP16")
    cmd("swlp download --model phi-3.5",      "Phi-3.5 mini Instruct  7 GB   FP16")
    cmd("swlp download --model <hf-id>",      "Any HuggingFace model id")

    # ── BENCHMARK ─────────────────────────────────────────────────────────
    section("BENCHMARK & PROFILING")
    cmd("swlp benchmark --model mistral-7b --runs 5 --report",
        "Benchmark 5 runs, print report")
    cmd("swlp benchmark --batch-size 8 --report",
        "Batched throughput benchmark (Phase 10)")
    cmd("swlp report benchmarks/baseline-<ts>.json",
        "Print a saved benchmark report")
    cmd("swlp suite --suite configs/bench_suite.toml --report",
        "Run full benchmark suite")

    # ── SIMULATION ────────────────────────────────────────────────────────
    section("SIMULATION  (no model needed)")
    cmd("swlp simulate --scenario configs/sim_baseline.toml --report",
        "Simulate streaming throughput math")
    cmd("swlp simulate --scenario configs/sim_m5.toml --report",
        "Simulate M5 Apple Silicon scenario")

    # ── MODEL ALIASES ─────────────────────────────────────────────────────
    section("MODEL ALIASES")
    aliases = [
        ("tiny-gpt2",    "sshleifer/tiny-gpt2"),
        ("mistral-7b",   "unsloth/mistral-7b-instruct-v0.2"),
        ("mistral-24b",  "mistralai/Mistral-Small-24B-Instruct-2501"),
        ("qwen-0.5b",    "Qwen/Qwen2.5-0.5B-Instruct"),
        ("qwen-1.5b",    "Qwen/Qwen2.5-1.5B-Instruct"),
        ("qwen-3b",      "Qwen/Qwen2.5-3B-Instruct"),
        ("qwen-7b",      "Qwen/Qwen2.5-7B-Instruct"),
        ("qwen-14b",     "Qwen/Qwen2.5-14B-Instruct"),
        ("phi-3.5",      "microsoft/Phi-3.5-mini-instruct"),
        ("smollm-1.7b",  "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
        ("smollm-360m",  "HuggingFaceTB/SmolLM2-360M-Instruct"),
    ]
    for alias, hf_id in aliases:
        row(alias, hf_id)

    # ── BACKENDS ──────────────────────────────────────────────────────────
    section("BACKENDS")
    backends = [
        ("hf",          "Standard HuggingFace — full model load, any hardware"),
        ("mlx",         "Apple Silicon native — fast quantized compute  (bf16 | int8 | int4)"),
        ("swlp",        "Layer streaming — huge models in low RAM  (requires --shard-dir)"),
        ("speculative", "SWLP + prompt-lookup decoding  (1–3× speedup on repetitive text)"),
        ("mock",        "Offline testing — no model download needed"),
    ]
    for name, desc in backends:
        row(name, desc, pad=12)

    # ── GLOBAL FLAGS ──────────────────────────────────────────────────────
    section("GLOBAL FLAGS")
    flags = [
        ("--config <path>",  "Load settings from a TOML config file"),
        ("--device <dev>",   "Force device: auto | cuda | mps | cpu"),
        ("--max-tokens <n>", "Maximum new tokens to generate"),
        ("--window <n>",     "SWLP sliding-window depth — layers resident at once  (default: 2)"),
        ("--json",           "Print full metrics as JSON"),
        ("--profile",        "Collect detailed timing metrics"),
    ]
    for flag, desc in flags:
        print(f"  {flag:<22}  {desc}")

    # ── footer ────────────────────────────────────────────────────────────
    print()
    rule()
    print(f"{d}Full reference: swlp --help  |  swlp <subcommand> --help{r}\n")
