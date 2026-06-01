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
    """Attach inference/runtime arguments, grouped for readable --help output."""

    model_grp = parser.add_argument_group("model")
    model_grp.add_argument(
        "--model", "--model-id", dest="model", type=str, default=None,
        metavar="MODEL",
        help="alias (mistral-7b, qwen-14b, …) or any HuggingFace model ID",
    )
    model_grp.add_argument(
        "--shard-dir", dest="shard_dir", type=Path, default=None,
        metavar="DIR",
        help="pre-sharded model directory; auto-selects the streaming backend",
    )
    model_grp.add_argument(
        "--backend", "--runner", dest="backend", choices=BACKENDS, default=None,
        metavar="BACKEND",
        help="hf | swlp | mlx | speculative | mock  (default: auto-detected)",
    )
    model_grp.add_argument(
        "--quant", choices=QUANTS, default=None,
        help="MLX quantization tier: bf16 | int8 | int4  (implies --backend mlx)",
    )
    model_grp.add_argument(
        "--device", type=str, default=None,
        help="target device: auto | cuda | mps | cpu",
    )
    model_grp.add_argument(
        "--cache-dir", type=Path, default=None,
        metavar="DIR",
        help="HuggingFace cache directory",
    )
    model_grp.add_argument(
        "--draft-model", dest="draft_model", type=str, default=None,
        metavar="MODEL",
        help=(
            "draft model for MLX speculative decoding  (alias or HF ID; "
            "must share the same tokenizer as --model, e.g. smollm-360m)"
        ),
    )
    model_grp.add_argument(
        "--config", type=Path, default=None,
        help="optional TOML config file (flags override any file values)",
    )

    gen_grp = parser.add_argument_group("generation")
    gen_grp.add_argument(
        "--prompt", type=str, default=None,
        help="prompt text to send to the model",
    )
    gen_grp.add_argument(
        "--max-tokens", dest="max_tokens", type=int, default=None,
        metavar="N",
        help="maximum new tokens to generate",
    )

    out_grp = parser.add_argument_group("output")
    out_grp.add_argument(
        "--json", "--json-output", dest="json_output", action="store_true",
        help="print full result and metrics as JSON instead of a summary",
    )
    out_grp.add_argument(
        "--profile", action="store_true",
        help="collect and print per-layer timing breakdown",
    )

    adv_grp = parser.add_argument_group("advanced")
    adv_grp.add_argument(
        "--window", "--window-size", "--swlp-window-size", dest="window",
        type=int, default=None, metavar="N",
        help="sliding window depth — layers kept in memory at once (default: 2)",
    )
    adv_grp.add_argument(
        "--kv-window", type=int, default=None, metavar="N",
        help="keep only the N most recent KV token positions (0 = unbounded)",
    )
    adv_grp.add_argument(
        "--kv-quant", choices=["none", "int4"], default=None,
        help="KV cache quantization: none (lossless, default) | int4 (lossy, ~4× smaller)",
    )
    adv_grp.add_argument(
        "--kv-compression", action="store_true",
        help="enable lossless zlib KV compression (reduces RAM at minor CPU cost)",
    )
    adv_grp.add_argument(
        "--kv-budget-mb", type=int, default=None, metavar="MB",
        help="hard RAM budget for the KV cache in megabytes",
    )

    # Internal SWLP tuning knobs — suppressed from help output.
    parser.add_argument("--swlp-prefetch-depth", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--swlp-no-prefetch", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--swlp-no-double-buffer", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--swlp-no-pin-memory", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--kv-tiering", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--kv-compression-level", type=int, default=None,
                        help=argparse.SUPPRESS)


def build_parser() -> argparse.ArgumentParser:
    """Build the full SWLP argument parser: run-path args + tool subcommands."""
    parser = argparse.ArgumentParser(
        prog="swlp",
        description=(
            "SWLP — stream large language models on consumer hardware, without quantization.\n"
            "Each transformer block loads on demand; only a sliding window of W layers\n"
            "ever resides in memory at once."
        ),
        epilog=(
            "Examples:\n"
            "  swlp download --model mistral-7b\n"
            "  swlp chat --shard-dir ./shards/mistral-7b\n"
            '  swlp --shard-dir ./shards/mistral-7b --prompt "Explain transformers."\n'
            "  swlp chat --model mistral-7b --backend mlx --quant int8\n\n"
            "Run  swlp help  for a full categorised reference."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_run_args(parser)
    subparsers = parser.add_subparsers(dest="command", title="commands", metavar="<command>")

    # ── download ──────────────────────────────────────────────────────────
    download_parser = subparsers.add_parser(
        "download",
        help="download and shard a model for streaming inference",
        description=(
            "Download a HuggingFace model and split it into per-layer shards so it\n"
            "can be streamed with --backend swlp, even if it is larger than available RAM.\n\n"
            "Examples:\n"
            "  swlp download --model mistral-7b\n"
            "  swlp download --model qwen-7b\n"
            "  swlp download --model mistral-24b\n"
            "  swlp download --model mistralai/Mistral-Small-24B-Instruct-2501"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    download_parser.add_argument(
        "--model", "--model-id", dest="model", type=str, required=True,
        metavar="MODEL",
        help=(
            "alias (mistral-7b, mistral-24b, qwen-7b, smollm-1.7b, phi-3.5, …) "
            "or any HuggingFace model ID"
        ),
    )
    download_parser.add_argument(
        "--output-dir", dest="output_dir", type=Path, default=None,
        metavar="DIR",
        help="destination directory for shards  (default: ./shards/<model-name>)",
    )
    download_parser.add_argument(
        "--cache-dir", type=Path, default=None,
        metavar="DIR",
        help="HuggingFace cache directory",
    )

    # ── chat ──────────────────────────────────────────────────────────────
    chat_parser = subparsers.add_parser(
        "chat",
        help="interactive multi-turn chat with token streaming",
        description=(
            "Start an interactive chat session that keeps conversation history across\n"
            "turns and streams tokens as they are generated.\n"
            "Press Ctrl-C or Ctrl-D to exit.\n\n"
            "Examples:\n"
            "  swlp chat --shard-dir ./shards/mistral-7b\n"
            "  swlp chat --model mistral-7b --backend mlx --quant int8\n"
            "  swlp chat --shard-dir ./shards/mistral-7b --backend speculative\n"
            "  swlp chat --backend mock"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_run_args(chat_parser)
    chat_parser.add_argument(
        "--max-chat-tokens", dest="max_chat_tokens", type=int, default=512,
        metavar="N",
        help="max new tokens per reply  (default: 512)",
    )

    # ── benchmark ─────────────────────────────────────────────────────────
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="time inference across N runs and save metrics to disk",
        description=(
            "Run inference one or more times, record timing and memory metrics,\n"
            "and write results to a JSON or CSV file.\n\n"
            "Examples:\n"
            "  swlp benchmark --shard-dir ./shards/mistral-7b --runs 5 --report\n"
            "  swlp benchmark --model mistral-7b --prompt-set short --warmup-runs 1 --report\n"
            "  swlp benchmark --backend mock --runs 3 --prompt-set short --report"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_run_args(benchmark_parser)
    benchmark_parser.add_argument(
        "--runs", type=int, default=1, metavar="N",
        help="number of timed benchmark runs  (default: 1)",
    )
    benchmark_parser.add_argument(
        "--warmup-runs", type=int, default=1, metavar="N",
        help="warm-up runs before timing starts  (default: 1)",
    )
    benchmark_parser.add_argument(
        "--prompt-set", choices=[*PROMPT_SETS.keys(), "all"], default=None,
        help="predefined prompt set to sweep: short | medium | long | all",
    )
    benchmark_parser.add_argument(
        "--batch-size", dest="batch_size", type=int, default=1, metavar="N",
        help="sequences per forward pass for batched streaming  (swlp backend only)",
    )
    benchmark_parser.add_argument(
        "--format", choices=["json", "csv"], default="json",
        help="output file format: json | csv  (default: json)",
    )
    benchmark_parser.add_argument(
        "--output", type=Path, default=None,
        help="output file path  (default: benchmarks/baseline-<timestamp>.<fmt>)",
    )
    benchmark_parser.add_argument(
        "--report", action="store_true",
        help="print a formatted summary table after saving",
    )

    # ── suite ─────────────────────────────────────────────────────────────
    suite_parser = subparsers.add_parser(
        "suite",
        help="sweep multiple configs and compare baseline vs SWLP",
        description=(
            "Run a benchmark suite defined in a TOML file, sweeping multiple prompt sets\n"
            "and window configurations, then compare baseline vs SWLP throughput.\n\n"
            "Examples:\n"
            "  swlp suite --suite configs/bench_suite.toml --report\n"
            "  swlp suite --suite configs/suite_phase3.toml --report"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    suite_parser.add_argument(
        "--config", type=Path, default=None,
        help="optional TOML runtime config",
    )
    suite_parser.add_argument(
        "--suite", type=Path, default=Path("configs/bench_suite.toml"),
        help="suite definition TOML  (default: configs/bench_suite.toml)",
    )
    suite_parser.add_argument(
        "--format", choices=["json", "csv"], default="json",
        help="output file format: json | csv  (default: json)",
    )
    suite_parser.add_argument(
        "--output", type=Path, default=None,
        help="output file path  (default: benchmarks/suite-<timestamp>.<fmt>)",
    )
    suite_parser.add_argument(
        "--report", action="store_true",
        help="print a formatted summary table after saving",
    )

    # ── simulate ──────────────────────────────────────────────────────────
    simulate_parser = subparsers.add_parser(
        "simulate",
        help="estimate throughput from hardware specs — no model needed",
        description=(
            "Run a pure-math bottleneck simulation to estimate streaming throughput\n"
            "and memory usage for a given hardware configuration.  No model or GPU\n"
            "is required — results are based on bandwidth and compute bounds.\n\n"
            "Examples:\n"
            "  swlp simulate --scenario configs/sim_m5.toml --report\n"
            "  swlp simulate --scenario configs/sim_baseline.toml --output sim.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    simulate_parser.add_argument(
        "--scenario", type=Path, default=Path("configs/sim_baseline.toml"),
        help="simulation scenario TOML  (default: configs/sim_baseline.toml)",
    )
    simulate_parser.add_argument(
        "--format", choices=["json", "csv"], default="json",
        help="output file format: json | csv  (default: json)",
    )
    simulate_parser.add_argument(
        "--output", type=Path, default=None,
        help="output file path  (default: benchmarks/sim-<timestamp>.<fmt>)",
    )
    simulate_parser.add_argument(
        "--report", action="store_true",
        help="print a formatted summary table after saving",
    )

    # ── report ────────────────────────────────────────────────────────────
    report_parser = subparsers.add_parser(
        "report",
        help="print a formatted report from a saved benchmark file",
        description=(
            "Load a benchmark JSON or CSV file and print a formatted summary table.\n\n"
            "Example:\n"
            "  swlp report benchmarks/baseline-2025-06-01T12-00-00.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    report_parser.add_argument("path", type=Path, help="path to a benchmark JSON or CSV file")

    # ── suite-report ──────────────────────────────────────────────────────
    suite_report_parser = subparsers.add_parser(
        "suite-report",
        help="print a formatted report from a saved suite file",
        description=(
            "Load a suite output JSON or CSV file and print a formatted summary.\n\n"
            "Example:\n"
            "  swlp suite-report benchmarks/suite-2025-06-01T12-00-00.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    suite_report_parser.add_argument("path", type=Path, help="path to a suite JSON or CSV file")

    # ── package ───────────────────────────────────────────────────────────
    package_parser = subparsers.add_parser(
        "package",
        help="convert a raw checkpoint into the SWLP layer-package format",
        description=(
            "Convert a HuggingFace checkpoint directory into the SWLP layer package\n"
            "layout (one .safetensors file per transformer block).\n\n"
            "Example:\n"
            "  swlp package /path/to/checkpoint ./shards/my-model --model-name my-model"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    package_parser.add_argument("checkpoint", type=Path,
                                help="checkpoint file or directory to convert")
    package_parser.add_argument("output_dir", type=Path,
                                help="directory to write the packaged shards")
    package_parser.add_argument("--model-name", type=str, default=None,
                                help="override the model name written to the manifest")
    package_parser.add_argument("--json-output", action="store_true",
                                help="print the resulting manifest as JSON")

    # ── validate-package ──────────────────────────────────────────────────
    validate_parser = subparsers.add_parser(
        "validate-package",
        help="verify the integrity of a packaged SWLP model directory",
        description=(
            "Check that a SWLP package directory is complete and internally consistent:\n"
            "manifest present, all shard files accounted for, checksums valid.\n\n"
            "Example:\n"
            "  swlp validate-package ./shards/mistral-7b"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    validate_parser.add_argument("path", type=Path,
                                 help="path to the SWLP package directory")
    validate_parser.add_argument("--json-output", action="store_true",
                                 help="print the validation report as JSON")

    # ── layer ─────────────────────────────────────────────────────────────
    layer_parser = subparsers.add_parser(
        "layer",
        help="inspect tensor shapes and dtypes in a single packaged layer",
        description=(
            "Load one transformer block from a SWLP package and print its tensor\n"
            "names, shapes, dtypes, and sizes.\n\n"
            "Examples:\n"
            "  swlp layer ./shards/mistral-7b 0\n"
            "  swlp layer ./shards/mistral-7b model.layers.3"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    layer_parser.add_argument("path", type=Path,
                              help="path to the SWLP package directory")
    layer_parser.add_argument("layer", type=str,
                              help="layer to inspect: numeric index (0, 1, …) or full name")
    layer_parser.add_argument("--json-output", action="store_true",
                              help="print the tensor payload as JSON")

    # ── help ──────────────────────────────────────────────────────────────
    subparsers.add_parser("help", help="show a categorised command reference")

    return parser
