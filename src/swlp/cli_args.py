"""Argument parser for the SWLP CLI — seven commands, a handful of flags.

Users name a model; the backend is chosen for them (``cli_resolve.py``).
Advanced tuning (window, KV cache, draft model, expert cache, …) lives in
``SWLP_*`` environment variables or a ``--config`` TOML, not in flags.
"""
from __future__ import annotations

import argparse
from pathlib import Path

BACKENDS = ["mlx", "mlx-moe", "swlp", "speculative", "hf", "mock"]
QUANTS = ["bf16", "int8", "int4"]

# Short names for models measured with SWLP. Unknown names pass through, so
# any HuggingFace id (or local directory) works too.
MODEL_ALIASES = {
    "tiny-gpt2": "sshleifer/tiny-gpt2",
    "smollm-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "smollm-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-14b": "Qwen/Qwen2.5-14B-Instruct",
    "phi-3.5": "microsoft/Phi-3.5-mini-instruct",
    "mistral-7b": "unsloth/mistral-7b-instruct-v0.2",
    "mistral-24b": "mistralai/Mistral-Small-24B-Instruct-2501",
    # MoE — expert-streamed (swlp doctor explains the trade-offs)
    "gemma4-26b": "mlx-community/gemma-4-26b-a4b-it-4bit",  # measured: 14.5 tok/s, 16 GB
    "qwen3.6-35b": "Qwen/Qwen3.6-35B-A3B",
    "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "olmoe-7b": "allenai/OLMoE-1B-7B-0125-Instruct",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "deepseek-v4-flash": "deepseek-ai/DeepSeek-V4-Flash-0731",
}


def resolve_model(name: str) -> str:
    """Map a friendly alias to its HuggingFace id; pass through unknown names."""
    return MODEL_ALIASES.get(name.lower(), name)


def _model_flags(p: argparse.ArgumentParser, model_required: bool = True) -> None:
    p.add_argument("model", nargs=None if model_required else "?", metavar="MODEL",
                   help="alias (swlp models), HuggingFace id, or local directory")
    p.add_argument("-q", "--quant", choices=QUANTS, default=None,
                   help="run resident on MLX at this precision (int4 fastest, bf16 exact)")
    p.add_argument("-d", "--details", action="store_true",
                   help="show what would run: chosen backend, alternatives, settings")
    p.add_argument("-n", "--max-tokens", "--max_tokens", "--max-new-tokens", type=int,
                   default=None, metavar="N", dest="max_tokens",
                   help="stop answers after N tokens (default: when the model finishes)")
    adv = p.add_argument_group("advanced")
    adv.add_argument("--backend", nargs="?", const="?", default=None, metavar="NAME",
                     help="override the automatic choice; bare --backend lists them "
                          f"({', '.join(BACKENDS)})")
    adv.add_argument("--resident", default=None, metavar="N|auto|off",
                     help="layer streaming: keep the first N layers in RAM (default auto)")
    adv.add_argument("--window", type=int, default=None, metavar="W",
                     help="layer streaming: layers on the GPU at once (default 2)")
    adv.add_argument("--config", type=Path, default=None, metavar="TOML",
                     help="config profile (see configs/); SWLP_* env vars also apply")
    adv.add_argument("-v", "--verbose", action="store_true", help="show runtime logs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swlp", add_help=False,
        description="Run LLMs bigger than your Mac's RAM — fast when they fit, "
                    "lossless when they don't.",
    )
    parser.add_argument("-h", "--help", action="store_true")
    parser.add_argument("--version", action="store_true")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    chat = sub.add_parser("chat", help="talk to a model")
    _model_flags(chat)

    run = sub.add_parser("run", help="answer one prompt and exit")
    _model_flags(run)
    run.add_argument("prompt", nargs="?", default=None, help="the prompt ('-' reads stdin)")
    run.add_argument("--json", action="store_true", help="print the answer + full metrics as JSON")

    serve = sub.add_parser("serve", help="OpenAI-compatible HTTP API")
    _model_flags(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)

    pull = sub.add_parser("pull", help="download a model and prepare it for streaming")
    pull.add_argument("model", metavar="MODEL", help="alias or HuggingFace id")
    pull.add_argument("--output-dir", type=Path, default=None, metavar="DIR",
                      help="where shards go (default: shards/<name>)")

    models = sub.add_parser("models", help="installed models, and which ones this Mac can run")
    models.add_argument("-d", "--details", action="store_true",
                        help="per model: path, precision, layers, experts, how it runs")

    rm = sub.add_parser("rm", help="delete a model's shards, downloads and converted copies")
    rm.add_argument("model", metavar="MODEL", help="alias, HuggingFace id, or shard directory")
    rm.add_argument("-y", "--yes", action="store_true", help="don't ask for confirmation")

    doctor = sub.add_parser("doctor", help="check this Mac: chip, memory, SSD, tuning")
    doctor.add_argument("model", nargs="?", default=None, metavar="MODEL",
                        help="also measure how this model would stream")

    bench = sub.add_parser("bench", help="measure tok/s for a model")
    _model_flags(bench)
    bench.add_argument("--runs", type=int, default=3, help="timed runs (default 3)")
    bench.add_argument("--json", action="store_true", help="print the summary as JSON")
    return parser
