"""The ``swlp`` / ``swlp -h`` screen: what SWLP is, the seven commands, examples."""
from __future__ import annotations

from . import ui


def print_swlp_help() -> None:
    ui.header("swlp", [
        ("what", "run LLMs bigger than your Mac's RAM"),
        ("how", "fast when a model fits · lossless streaming when it doesn't"),
    ])
    ui.section("Commands")
    ui.commands([
        ("swlp chat MODEL", "talk to a model"),
        ("swlp run MODEL \"prompt\"", "answer one prompt and exit"),
        ("swlp serve MODEL", "OpenAI-compatible API on http://127.0.0.1:8080"),
        ("swlp pull MODEL", "download + prepare a model for streaming"),
        ("swlp models", "installed models, and what this Mac can run"),
        ("swlp rm MODEL", "delete a model (shards, downloads, converted copies)"),
        ("swlp doctor", "check chip, memory, SSD and tuning"),
        ("swlp bench MODEL", "measure tok/s"),
    ])
    ui.section("Examples")
    ui.commands([
        ("swlp chat gemma4-26b", "26B MoE on 16 GB · ~14 tok/s (4-bit experts streamed)"),
        ("swlp chat qwen-7b -q int4", "fits in RAM · fastest"),
        ("swlp pull qwen3.6-35b && swlp chat qwen3.6-35b", "35B MoE · lossless bf16"),
        ("swlp run mistral-7b \"Explain RoPE\" --json", "answer + full metrics"),
        ("swlp chat qwen3.6-35b -d", "which backends fit this model, and their settings"),
        ("swlp models -d", "details of every installed model"),
    ])
    ui.section("Backends  (chosen for you; override with --backend NAME)")
    from .cli_resolve import BACKENDS_INFO

    ui.commands([(name, info.what) for name, info in BACKENDS_INFO.items()])
    ui.section("Options")
    ui.commands([
        ("-q, --quant int4|int8|bf16", "run resident on MLX at this precision"),
        ("-n, --max-tokens N", "max tokens per answer"),
        ("-d, --details", "show what would run + which backends fit + their settings"),
        ("--backend NAME", "override the automatic choice (bare --backend lists them)"),
        ("--resident N|auto|off", "layer streaming: keep the first N layers in RAM"),
        ("--window W", "layer streaming: layers on the GPU at once (default 2)"),
        ("-v, --verbose", "show runtime logs"),
    ])
    ui.console.print()
    ui.note("  The backend is chosen for you from the model and this Mac.")
    ui.note("  Advanced tuning: SWLP_* environment variables or --config TOML.")
    ui.note("  In chat: /help /clear /stats /exit  ·  swlp COMMAND -h for options\n")
