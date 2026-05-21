"""Argument-parser construction for the SWLP CLI.

Kept separate from ``cli.py`` (dispatch logic) so each file stays focused and
under the line budget. ``build_parser()`` is the only export the CLI needs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark.run import PROMPT_SETS

BACKENDS = ["hf", "mock", "swlp", "speculative", "mlx"]
QUANTS = ["bf16", "int8", "int4"]

# Short, friendly names for models used across the SWLP phases. A name that is
# not an alias is passed through unchanged, so any HuggingFace id still works.
MODEL_ALIASES = {
    # tiny / test
    "tiny-gpt2": "sshleifer/tiny-gpt2",
    # Mistral family
    "mistral-7b": "unsloth/mistral-7b-instruct-v0.2",
    "mistral-24b": "mistralai/Mistral-Small-24B-Instruct-2501",
    # Qwen 2.5 family
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    # Phi-3.5 (Microsoft, Apache 2.0)
    "phi-3.5": "microsoft/Phi-3.5-mini-instruct",
    # SmolLM2 (HuggingFace, Apache 2.0 — great for testing on low RAM)
    "smollm-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "smollm-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
}


def resolve_model(name: str) -> str:
    """Map a friendly alias to its HuggingFace id; pass through unknown names."""
    return MODEL_ALIASES.get(name.lower(), name)


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    """Attach the inference/runtime arguments shared by the run path and benchmark."""
    parser.add_argument("--config", type=Path, default=None, help="Optional TOML config file")
    parser.add_argument(
        "--model", "--model-id", dest="model", type=str, default=None,
        help="Model: an alias (mistral-7b, qwen-14b, tiny-gpt2) or any HuggingFace id",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Prompt to send to the model")
    parser.add_argument(
        "--backend", "--runner", dest="backend", choices=BACKENDS, default=None,
        help="Runner backend (default: auto from --shard-dir / --quant, else hf)",
    )
    parser.add_argument(
        "--quant", choices=QUANTS, default=None,
        help="MLX quantization tier; implies --backend mlx",
    )
    parser.add_argument(
        "--window", "--window-size", "--swlp-window-size", dest="window",
        type=int, default=None, help="SWLP sliding-window size (streaming backend)",
    )
    parser.add_argument(
        "--max-tokens", dest="max_tokens", type=int, default=None,
        help="Maximum new tokens to generate",
    )
    parser.add_argument("--device", type=str, default=None, help="Device: auto | cuda | mps | cpu")
    parser.add_argument(
        "--shard-dir", dest="shard_dir", type=Path, default=None,
        help="Directory of per-layer shards; implies --backend swlp",
    )
    parser.add_argument("--cache-dir", type=Path, default=None, help="HF cache directory")
    parser.add_argument(
        "--json", "--json-output", dest="json_output", action="store_true",
        help="Print the full result as JSON instead of a summary",
    )
    parser.add_argument("--profile", action="store_true", help="Collect detailed timing metrics")
    # Advanced SWLP / KV tuning — rarely needed, hidden from --help.
    parser.add_argument("--swlp-prefetch-depth", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--swlp-no-prefetch", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--swlp-no-double-buffer", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--swlp-no-pin-memory", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--kv-budget-mb", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--kv-compression", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--kv-compression-level", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--kv-tiering", action="store_true", help=argparse.SUPPRESS)
    # Phase 16: sliding-window KV budget (0 = unbounded)
    parser.add_argument(
        "--kv-window",
        type=int,
        default=None,
        help="Keep only the most recent N KV token positions (0 = unbounded).",
    )
    # Phase 18: INT4 KV quantization (lossy, off by default)
    parser.add_argument(
        "--kv-quant",
        choices=["none", "int4"],
        default=None,
        help="KV cache quantization: none (lossless, default) | int4 (lossy ~4× smaller).",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the full SWLP argument parser: run-path args + tool subcommands."""
    parser = argparse.ArgumentParser(
        prog="swlp",
        description="SWLP — run large language models on consumer hardware.",
        epilog='Run inference:  swlp --model mistral-7b --prompt "Hello"',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_run_args(parser)
    subparsers = parser.add_subparsers(dest="command")

    package_parser = subparsers.add_parser(
        "package", help="Convert a checkpoint into the SWLP layer package layout"
    )
    package_parser.add_argument("checkpoint", type=Path, help="Checkpoint file or directory")
    package_parser.add_argument("output_dir", type=Path, help="Directory to write the package")
    package_parser.add_argument(
        "--model-name", type=str, default=None, help="Override the manifest model name"
    )
    package_parser.add_argument("--json-output", action="store_true", help="Print manifest as JSON")

    validate_parser = subparsers.add_parser(
        "validate-package", help="Validate a packaged SWLP model directory"
    )
    validate_parser.add_argument("path", type=Path, help="Path to a package directory")
    validate_parser.add_argument(
        "--json-output", action="store_true", help="Print validation as JSON"
    )

    layer_parser = subparsers.add_parser(
        "layer", help="Load a single layer from a packaged SWLP model"
    )
    layer_parser.add_argument("path", type=Path, help="Path to a package directory")
    layer_parser.add_argument("layer", type=str, help="Layer name, file name, or numeric index")
    layer_parser.add_argument("--json-output", action="store_true", help="Print payload as JSON")

    benchmark_parser = subparsers.add_parser("benchmark", help="Run the benchmark and save metrics")
    _add_run_args(benchmark_parser)
    benchmark_parser.add_argument("--runs", type=int, default=1, help="Number of benchmark runs")
    benchmark_parser.add_argument(
        "--batch-size", dest="batch_size", type=int, default=1,
        help="Sequences per disk sweep (Phase 10 batched streaming; swlp backend only)",
    )
    benchmark_parser.add_argument(
        "--warmup-runs", type=int, default=1, help="Warm-up runs before timing each prompt"
    )
    benchmark_parser.add_argument(
        "--prompt-set", choices=[*PROMPT_SETS.keys(), "all"], default=None,
        help="Use a predefined prompt set (short, medium, long, or all)",
    )
    benchmark_parser.add_argument("--format", choices=["json", "csv"], default="json")
    benchmark_parser.add_argument("--output", type=Path, default=None, help="Output path")
    benchmark_parser.add_argument("--report", action="store_true", help="Print a summary report")

    report_parser = subparsers.add_parser("report", help="Print a report from a benchmark file")
    report_parser.add_argument("path", type=Path, help="Path to a benchmark JSON or CSV file")

    simulate_parser = subparsers.add_parser("simulate", help="Run SWLP streaming simulations")
    simulate_parser.add_argument(
        "--scenario", type=Path, default=Path("configs/sim_baseline.toml"),
        help="Path to a simulation scenario TOML",
    )
    simulate_parser.add_argument("--format", choices=["json", "csv"], default="json")
    simulate_parser.add_argument("--output", type=Path, default=None, help="Output path")
    simulate_parser.add_argument("--report", action="store_true", help="Print a summary report")

    suite_parser = subparsers.add_parser("suite", help="Run the benchmark suite (baseline vs SWLP)")
    suite_parser.add_argument("--config", type=Path, default=None, help="Path to a TOML config")
    suite_parser.add_argument(
        "--suite", type=Path, default=Path("configs/bench_suite.toml"), help="Suite TOML"
    )
    suite_parser.add_argument("--format", choices=["json", "csv"], default="json")
    suite_parser.add_argument("--output", type=Path, default=None, help="Output path")
    suite_parser.add_argument("--report", action="store_true", help="Print a summary report")

    suite_report_parser = subparsers.add_parser(
        "suite-report", help="Print a summary from a suite output file"
    )
    suite_report_parser.add_argument("path", type=Path, help="Path to a suite JSON or CSV file")

    chat_parser = subparsers.add_parser(
        "chat", help="Interactive chat session with token streaming"
    )
    _add_run_args(chat_parser)
    chat_parser.add_argument(
        "--max-chat-tokens", dest="max_chat_tokens", type=int, default=512,
        help="Max new tokens per reply in chat mode (default: 512)",
    )

    subparsers.add_parser("help", help="Show a categorised command reference")

    download_parser = subparsers.add_parser(
        "download",
        help="Download and shard a model for SWLP layer streaming",
        description=(
            "Downloads a HuggingFace model and splits it into per-layer shards so it\n"
            "can be streamed with --backend swlp, even if it is larger than available RAM.\n\n"
            "Examples:\n"
            "  swlp download --model mistral-7b\n"
            "  swlp download --model mistral-24b\n"
            "  swlp download --model qwen-7b\n"
            "  swlp download --model mistralai/Mistral-Small-24B-Instruct-2501"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    download_parser.add_argument(
        "--model", "--model-id", dest="model", type=str, required=True,
        help=(
            "Model alias (mistral-7b, mistral-24b, qwen-7b, smollm-1.7b, phi-3.5, …) "
            "or any HuggingFace id"
        ),
    )
    download_parser.add_argument(
        "--output-dir", dest="output_dir", type=Path, default=None,
        help="Where to write the shards (default: ./shards/<model-name>)",
    )
    download_parser.add_argument(
        "--cache-dir", type=Path, default=None, help="HF cache directory"
    )

    return parser
