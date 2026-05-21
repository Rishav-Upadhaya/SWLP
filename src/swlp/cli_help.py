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

    W = 72

    def rule() -> None:
        print(d + "─" * W + r)

    def section(title: str) -> None:
        print(f"\n{b}{title}{r}")

    def cmd(command: str, description: str, pad: int = 48) -> None:
        print(f"  {command:<{pad}}  {d}{description}{r}")

    def flag(name: str, description: str, pad: int = 20) -> None:
        print(f"  {b}{name:<{pad}}{r}  {description}")

    # ── header ────────────────────────────────────────────────────────────
    print(f"\n{b}SWLP{r} — run large LLMs on consumer hardware, without quantization.\n")
    rule()

    # ── QUICK START ───────────────────────────────────────────────────────
    section("QUICK START")
    cmd("swlp download --model qwen-0.5b",
        "① download a model once")
    cmd("swlp chat --shard-dir ./shards/qwen-0.5b",
        "② chat  (streaming, any RAM)")
    cmd('swlp --shard-dir ./shards/qwen-0.5b --prompt "hello"',
        "③ single-turn inference")

    # ── DOWNLOAD ──────────────────────────────────────────────────────────
    section("DOWNLOAD MODELS  (one-time setup)")
    cmd("swlp download --model qwen-0.5b",   "0.5 B  ·  720 MB  ·  fast on any machine")
    cmd("swlp download --model qwen-7b",     "7 B   ·  14 GB   ·  strong quality")
    cmd("swlp download --model mistral-7b",  "7 B   ·  14 GB   ·  Mistral Instruct")
    cmd("swlp download --model <hf-id>",     "any HuggingFace model ID")

    # ── CHAT & INFERENCE ──────────────────────────────────────────────────
    section("CHAT & INFERENCE")
    cmd("swlp chat --shard-dir ./shards/<model>",
        "streaming  (huge models, low RAM)")
    cmd("swlp chat --model <model> --backend mlx --quant int8",
        "MLX native  (fast on Apple Silicon)")
    cmd("swlp chat --model <model>",
        "HuggingFace  (full-model load)")
    cmd("swlp chat --backend mock",
        "offline testing, no download needed")

    # ── KEY FLAGS ─────────────────────────────────────────────────────────
    section("KEY FLAGS")
    flag("--model <name>",     "alias or HuggingFace ID  (e.g. qwen-7b)")
    flag("--shard-dir <dir>",  "path to downloaded shards  (streaming)")
    flag("--window <n>",       "streaming window size  (default: 2)")
    flag("--quant <tier>",     "MLX quant: bf16 | int8 | int4")
    flag("--max-tokens <n>",   "tokens to generate")
    flag("--device <dev>",     "auto | cuda | mps | cpu")

    # ── footer ────────────────────────────────────────────────────────────
    print()
    rule()
    aliases = "qwen-0.5b  qwen-7b  qwen-14b  mistral-7b  smollm-360m  phi-3.5  …"
    print(f"  {d}Model aliases: {aliases}{r}")
    print(f"  {d}Full reference: swlp --help  |  swlp <subcommand> --help{r}\n")
