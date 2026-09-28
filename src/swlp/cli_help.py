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

    W = 76

    def rule() -> None:
        print(d + "─" * W + r)

    def blank() -> None:
        print()

    def section(title: str) -> None:
        print(f"\n{b}{title}{r}\n")

    def cmd(command: str, description: str, pad: int = 50) -> None:
        if len(command) <= pad:
            print(f"  {command:<{pad}}  {d}{description}{r}")
        else:
            print(f"  {command}")
            print(f"    {d}{description}{r}")

    def flag(name: str, description: str, pad: int = 18) -> None:
        print(f"  {b}{name:<{pad}}{r}  {description}")

    def ex(command: str, comment: str = "") -> None:
        suffix = f"  {d}# {comment}{r}" if comment else ""
        print(f"  {command}{suffix}")

    # ── header ────────────────────────────────────────────────────────────
    print(f"\n{b}SWLP{r}  ─  LLMs on Apple Silicon: fast when they fit, "
          f"lossless when they don't.\n")
    rule()

    # ── QUICK START ───────────────────────────────────────────────────────
    section("QUICK START")
    cmd("swlp doctor",                          "① what this Mac can run, and the tuning to apply")
    cmd("swlp chat qwen-7b --quant int4",       "② fast chat — model fits in RAM")
    cmd("swlp run mistral-7b --prompt \"…\"",    "③ download + shard + run, one command")
    cmd("swlp serve mistral-7b",                "④ OpenAI-compatible server on :8080")

    # ── DOWNLOAD ──────────────────────────────────────────────────────────
    section("DOWNLOAD")
    cmd("swlp download --model qwen-0.5b",   "0.5 B  ·   720 MB  ·  fast on any machine")
    cmd("swlp download --model qwen-7b",     "7 B   ·  14 GB   ·  strong quality")
    cmd("swlp download --model mistral-7b",  "7 B   ·  14 GB   ·  Mistral Instruct")
    cmd("swlp download --model <hf-id>",     "any HuggingFace model ID")

    # ── INFERENCE ─────────────────────────────────────────────────────────
    section("TWO WAYS TO RUN")
    print(f"  {b}fits in RAM{r} — native MLX, interactive speed")
    cmd('swlp --model <model> --quant int4 --prompt "…"', "~28 tok/s, near-lossless")
    cmd('swlp --model <model> --quant int8 --prompt "…"', "~16 tok/s, byte-identical to fp16")
    blank()
    print(f"  {b}bigger than RAM{r} — FP16 layer streaming, exact, slow")
    cmd('swlp --shard-dir ./shards/<model> --prompt "…"', "peak RAM ≈ 2 × one layer")
    cmd('swlp --backend mock --prompt "…"',  "offline smoke-test, no model needed")

    # ── CHAT ──────────────────────────────────────────────────────────────
    section("CHAT")
    cmd("swlp chat --shard-dir ./shards/<model>",               "streaming  (works in any RAM)")
    cmd("swlp chat --model <model> --backend mlx --quant int8", "MLX native  (Apple Silicon)")
    cmd("swlp chat --model <model>",                            "HuggingFace  (full model)")
    cmd("swlp chat --backend mock",                             "offline, no model needed")

    # ── FLAGS ─────────────────────────────────────────────────────────────
    section("FLAGS")
    flag("--model <name>",    "alias or HuggingFace ID  (e.g. mistral-7b, qwen-14b)")
    flag("--shard-dir <dir>", "pre-sharded model path  (auto-selects streaming backend)")
    flag("--window <n>",      "sliding window depth  (default: 2)")
    flag("--quant <tier>",    "MLX quantization: bf16 | int8 | int4  (implies --backend mlx)")
    flag("--max-tokens <n>",  "max new tokens to generate")
    flag("--device <dev>",    "auto | mps | cpu")
    flag("--json",            "print full metrics as JSON")
    flag("--profile",         "collect per-layer timing breakdown")

    # ── APPLE SILICON TUNING ──────────────────────────────────────────────
    section("APPLE SILICON TUNING")
    flag("--kv-bits <4|8>",   "quantize KV cache — 4-bit is FASTER than fp16 here")
    flag("--draft-model <m>", "speculative decoding: 1.9-2.1x, same family only")
    flag("--draft-tokens <n>", "drafts per step (default 4; 4-6 is the sweet spot)")
    flag("--max-kv-size <n>", "cap KV length; bounds long-context RAM (lossy)")
    flag("--wired-limit <v>", "Metal wired ceiling: auto | off | <MB>  (default auto)")
    blank()
    print(f"  {d}Run `swlp doctor` — it prints the exact sysctl for this machine.{r}")

    # ── MISTRAL-7B END-TO-END ─────────────────────────────────────────────
    section("MISTRAL-7B  END-TO-END")
    ex("swlp download --model mistral-7b", "download & shard  (~14 GB, one-time)")
    blank()
    ex('swlp --shard-dir ./shards/mistral-7b --prompt "Explain transformers."',
       "single inference")
    ex("swlp chat --shard-dir ./shards/mistral-7b",          "streaming chat")
    ex("swlp chat --shard-dir ./shards/mistral-7b --window 4", "wider context window")
    blank()
    ex("swlp chat --model mistral-7b --backend mlx --quant int8",
       "MLX native  (Apple Silicon, no shards needed)")
    ex("swlp chat --shard-dir ./shards/mistral-7b --backend speculative",
       "speculative decoding  (faster on MPS)")

    # ── BENCHMARKING ──────────────────────────────────────────────────────
    section("BENCHMARKING")
    ex("swlp benchmark --shard-dir ./shards/mistral-7b --runs 5 --report")
    ex("swlp benchmark --model mistral-7b --prompt-set short --warmup-runs 1 --report")
    ex("swlp suite     --suite configs/bench_suite.toml --report")
    ex("swlp simulate  --scenario configs/sim_m5.toml --report")
    ex("swlp policy-report experiments/policy_matrix.csv")
    ex("swlp report    benchmarks/baseline-<timestamp>.json")
    ex("swlp suite-report  benchmarks/suite-<timestamp>.json")

    # ── KV CACHE ──────────────────────────────────────────────────────────
    section("KV CACHE OPTIONS")
    flag("--kv-window <n>",  "keep last N KV positions only  (0 = unbounded, saves RAM)")
    flag("--kv-quant int4",  "INT4 KV quantization  (lossy · ~4× smaller)")
    flag("--kv-compression", "zlib KV compression  (lossless)")

    # ── MODEL TOOLS ───────────────────────────────────────────────────────
    section("MODEL TOOLS")
    cmd("swlp doctor",                               "hardware check + per-model advice")
    cmd("swlp models",                               "list aliases, sizes, HF ids")
    cmd("swlp validate-package ./shards/mistral-7b", "verify shard integrity")
    cmd("swlp compress-shards ./shards/mistral-7b",  "lossless ~31% shrink  (--revert undoes)")
    cmd("swlp layer ./shards/mistral-7b 0",          "inspect layer 0 tensor shapes")
    cmd("swlp package <checkpoint> <output-dir>",    "package a raw checkpoint")

    # ── footer ────────────────────────────────────────────────────────────
    blank()
    rule()
    aliases = "qwen-0.5b  qwen-7b  qwen-14b  mistral-7b  smollm-360m  phi-3.5  …"
    print(f"  {d}Aliases  {aliases}{r}")
    print(f"  {d}Help     swlp --help  ·  swlp <command> --help{r}\n")
