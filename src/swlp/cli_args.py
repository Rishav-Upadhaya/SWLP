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
    # MoE (Phase 25 — expert-streamed; see swlp doctor for guidance)
    "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "deepseek-v4-flash": "deepseek-ai/DeepSeek-V4-Flash-0731",
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
        help="target device: auto | mps | cpu",
    )
    model_grp.add_argument(
        "--kv-bits", type=int, choices=(4, 8), default=None, dest="mlx_kv_bits",
        help="quantize the KV cache (MLX). 4-bit is typically FASTER than fp16 "
             "on unified memory — decode is bandwidth-bound, not compute-bound",
    )
    model_grp.add_argument(
        "--max-kv-size", type=int, default=None, dest="max_kv_size",
        help="cap KV cache length; bounds long-context RAM (lossy: drops oldest)",
    )
    model_grp.add_argument(
        "--draft-tokens", type=int, default=None, dest="mlx_num_draft_tokens",
        help="draft tokens per speculative step (default 4; 4-6 is the sweet spot)",
    )
    model_grp.add_argument(
        "--wired-limit", type=str, default=None, dest="mlx_wired_limit",
        help="Metal wired-memory ceiling: auto | off | <MB>  (default auto)",
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
            "draft model for speculative decoding  (alias or HF ID; must share "
            "the target's tokenizer, e.g. qwen-0.5b for qwen-14b; with "
            "--shard-dir selects the speculative backend, with --quant the MLX one)"
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

    # Phase 23: quality-neutral speedups (always-on, no quality tradeoff).
    perf_grp = parser.add_argument_group(
        "performance (quality-neutral)",
        "Speedups that do NOT affect output quality. Safe to enable always.",
    )
    perf_grp.add_argument(
        "--no-activation-cache", action="store_true",
        help="disable activation caching for prompt prefix reuse",
    )
    perf_grp.add_argument(
        "--no-prealloc-buffer", action="store_true",
        help="disable pre-allocated generate buffer (uses torch.cat instead)",
    )

    # Phase 23: opt-in quality tradeoffs (opt-in via flags).
    tradeoff_grp = parser.add_argument_group(
        "quality tradeoffs (opt-in)",
        "Speedups that may reduce output quality. Each has a configurable level.",
    )
    tradeoff_grp.add_argument(
        "--early-exit", default=None, metavar="THRESHOLD",
        help=(
            "skip remaining layers when next-token entropy < THRESHOLD (0.0-1.0). "
            "Lower = more aggressive = faster but more quality risk. "
            "Typical: 0.5 = moderate, 0.3 = aggressive. 'off' to disable."
        ),
    )
    tradeoff_grp.add_argument(
        "--layer-pruning", default=None, metavar="MODE",
        choices=["off", "light", "aggressive"],
        help=(
            "remove least-important layers: light (~10%% removed, <0.5%% quality loss) "
            "or aggressive (~25%% removed, ~1-3%% quality loss). Requires calibration."
        ),
    )
    tradeoff_grp.add_argument(
        "--adaptive-precision", default=None, metavar="MODE",
        choices=["off", "fp8_late", "int8_late"],
        help=(
            "use FP16 for early layers, lower precision for later layers. "
            "fp8_late: FP8 for last 50%% of layers. "
            "int8_late: INT8 for last 50%% of layers."
        ),
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

    # ── run (one-command: auto-download + auto-shard + generate) ─────────
    run_parser = subparsers.add_parser(
        "run",
        help="one command: download, shard, and run a model  (skips work already done)",
        description=(
            "The quick-start path: give an alias or HuggingFace id and SWLP handles\n"
            "the rest — downloads weights, shards them for streaming if needed, and\n"
            "generates. Already-sharded models are detected via the shard manifest\n"
            "and reused without re-downloading.\n\n"
            "Examples:\n"
            "  swlp run mistral-7b --prompt \"Explain transformers.\"\n"
            '  swlp run mistral-7b --chat\n'
            "  swlp run google/gemma-2-2b-it --backend mlx --quant int4 --prompt \"Hi\""
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_run_args(run_parser)
    run_parser.add_argument(
        "model_pos", nargs="?", default=None, metavar="MODEL",
        help="model alias or HuggingFace id  (same as --model)",
    )
    run_parser.add_argument(
        "--max-chat-tokens", dest="max_chat_tokens", type=int, default=512,
        metavar="N",
        help="max new tokens per reply with --chat  (default: 512)",
    )
    run_parser.add_argument(
        "--chat", action="store_true",
        help="start interactive chat after setup instead of one-shot generation",
    )

    # ── serve ────────────────────────────────────────────────────────────
    serve_parser = subparsers.add_parser(
        "serve",
        help="OpenAI-compatible HTTP server with SSE token streaming",
        description=(
            "Serve a model behind /v1/chat/completions and /v1/completions using\n"
            "the OpenAI API shape (stream=true yields SSE chunks). Works with any\n"
            "backend including the FP16 streaming path.\n\n"
            "Examples:\n"
            "  swlp serve ./shards/mistral-7b\n"
            "  swlp serve mistral-7b --backend mlx --quant int8\n"
            "  swlp serve --shard-dir ./shards/qwen-14b --port 9000"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_run_args(serve_parser)
    serve_parser.add_argument(
        "model_pos", nargs="?", default=None, metavar="MODEL",
        help="alias or HF id or shard dir  (same as --model/--shard-dir)",
    )
    serve_parser.add_argument("--host", default="127.0.0.1",
                              help="bind address  (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=8080,
                              help="port  (default: 8080)")

    # ── download / pull ───────────────────────────────────────────────────
    for cmd in ("download", "pull"):
        is_pull = cmd == "pull"
        download_parser = subparsers.add_parser(
            cmd,
            help=(
                "download and shard a model for streaming inference"
                + (" (with disk-space preflight)" if is_pull else "")
            ),
            description=(
                "Download a HuggingFace model and split it into per-layer shards so it\n"
                "can be streamed with --backend swlp, even if it is larger than available RAM.\n\n"
                + (
                    "pull additionally checks free disk space against the model's\n"
                    "known FP16 size before downloading.\n\n"
                    if is_pull else ""
                )
                + "Examples:\n"
                "  swlp download --model mistral-7b\n"
                "  swlp pull --model qwen-7b\n"
                "  swlp pull --model mistral-24b\n"
                "  swlp pull --model mistralai/Mistral-Small-24B-Instruct-2501"
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

    # ── compress-shards ───────────────────────────────────────────────────
    compress_parser = subparsers.add_parser(
        "compress-shards",
        help="losslessly compress a shard directory in place (~31%% smaller, bit-exact)",
        description=(
            "Convert layer shards to compressed .swz files (zipnn: zstd + Huffman over\n"
            "byte-grouped FP16/BF16 weights). Bit-exact: every layer is roundtrip-verified\n"
            "before its original is deleted. Streaming reads compressed shards directly.\n\n"
            "Throughput depends on your SSD: below ~3.5 GB/s sequential read the smaller\n"
            "reads win; above it (fast Apple Silicon SSDs) decompression costs ~25% tok/s.\n"
            "The ~31% disk saving applies either way. Use --revert to restore plain\n"
            "shards when streaming speed matters more than disk space.\n\n"
            "Examples:\n"
            "  swlp compress-shards ./shards/mistral-7b\n"
            "  swlp compress-shards --revert ./shards/mistral-7b"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    compress_parser.add_argument(
        "shard_dir", type=Path,
        metavar="SHARD_DIR",
        help="shard directory produced by `swlp download`",
    )
    compress_parser.add_argument(
        "--revert", action="store_true",
        help="restore plain .safetensors layers from .swz (SHA-verified, resumable)",
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
        "model_pos", nargs="?", default=None, metavar="MODEL",
        help="model alias or HuggingFace id  (same as --model)",
    )
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

    # ── policy-report ─────────────────────────────────────────────────────
    policy_report_parser = subparsers.add_parser(
        "policy-report",
        help="evaluate scheduler policy accuracy from cross-machine experiment data",
        description=(
            "Load a policy validation matrix (CSV/JSON) and report predictive accuracy:\n"
            "resident-layer MAE, within ±2-layer rate, and throughput regret.\n\n"
            "Required columns: machine, model, pipeline_ratio, free_ram_gb,\n"
            "resident_count, throughput_tokens_per_second\n\n"
            "Optional columns: workload, quant\n\n"
            "Example:\n"
            "  swlp policy-report experiments/policy_matrix.csv"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    policy_report_parser.add_argument(
        "path",
        type=Path,
        help="path to a policy validation matrix CSV or JSON file",
    )

    # ── profile ───────────────────────────────────────────────────────────
    profile_parser = subparsers.add_parser(
        "profile",
        help="run inference with full pipeline profiling and export JSON traces",
        description=(
            "Run inference with the fine-grained profiler enabled.  Collects per-layer\n"
            "timestamps for SSD read, deserialization, upload, compute, and eviction.\n"
            "Exports a JSON trace file with hardware metadata, pipeline metrics, and\n"
            "raw timestamps for timeline visualization.\n\n"
            "Examples:\n"
            "  swlp profile --shard-dir ./shards/mistral-7b --prompt \"Hello\" "
            "--output traces.json\n"
            "  swlp profile --model mistral-7b --max-tokens 16\n"
            "  swlp profile --backend mock --max-tokens 8 --output mock_traces.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_run_args(profile_parser)
    profile_parser.add_argument(
        "--output", type=Path, default=None,
        help="output JSON trace file  (default: layer_traces.json)",
    )
    profile_parser.add_argument(
        "--timeline", action="store_true",
        help="print the per-layer timeline table to stdout",
    )
    profile_parser.add_argument(
        "--detail", action="store_true",
        help="print per-layer block detail to stdout",
    )
    profile_parser.add_argument(
        "--summary", action="store_true",
        help="print pipeline metrics summary to stdout",
    )

    # ── sim ───────────────────────────────────────────────────────────────
    sim_parser = subparsers.add_parser(
        "sim",
        help="run the scheduler simulator — no model or GPU required",
        description=(
            "Simulate the SWLP streaming scheduler with configurable parameters.\n"
            "Produces the same timeline and metrics as the real profiler but without\n"
            "loading any model.  Useful for rapid parameter exploration.\n\n"
            "Examples:\n"
            "  swlp sim --layers 32 --layer-size-mb 512 --window 2 --prefetch 4\n"
            "  swlp sim --layers 80 --layer-size-mb 700 --window 4 --tokens 20 --report\n"
            "  swlp sim --layers 32 --resident 8 --output sim.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sim_parser.add_argument("--layers", type=int, default=32, metavar="N",
                            help="number of transformer layers (default: 32)")
    sim_parser.add_argument("--layer-size-mb", type=float, default=512, metavar="MB",
                            help="size of each layer in MB (default: 512)")
    sim_parser.add_argument("--ram-gb", type=float, default=16.0, metavar="GB",
                            help="total RAM in GB (default: 16)")
    sim_parser.add_argument("--window", type=int, default=2, metavar="N",
                            help="sliding window size (default: 2)")
    sim_parser.add_argument("--prefetch", type=int, default=4, metavar="N",
                            help="prefetch depth (default: 4)")
    sim_parser.add_argument("--workers", type=int, default=2, metavar="N",
                            help="worker thread count (default: 2)")
    sim_parser.add_argument("--compute-ms", type=float, default=50.0, metavar="MS",
                            help="simulated compute time per layer in ms (default: 50)")
    sim_parser.add_argument("--ssd-ms", type=float, default=30.0, metavar="MS",
                            help="simulated SSD read latency in ms (default: 30)")
    sim_parser.add_argument("--upload-ms", type=float, default=10.0, metavar="MS",
                            help="simulated upload latency in ms (default: 10)")
    sim_parser.add_argument("--evict-ms", type=float, default=5.0, metavar="MS",
                            help="simulated eviction latency in ms (default: 5)")
    sim_parser.add_argument("--tokens", type=int, default=10, metavar="N",
                            help="number of tokens to simulate (default: 10)")
    sim_parser.add_argument("--resident", type=int, default=0, metavar="N",
                            help="number of resident (never-evicted) layers (default: 0)")
    sim_parser.add_argument("--output", type=Path, default=None,
                            help="output JSON file  (default: sim_traces.json)")
    sim_parser.add_argument("--report", action="store_true",
                            help="print a formatted summary to stdout")

    # ── analyze ──────────────────────────────────────────────────────────
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="analyze profiler traces and produce actionable recommendations",
        description=(
            "Load a JSON trace file produced by `swlp profile` and analyze it.\n"
            "Identifies the pipeline bottleneck, GPU idle reasons, prefetch efficiency,\n"
            "memory headroom, and generates specific recommendations.\n\n"
            "Examples:\n"
            "  swlp analyze layer_traces.json\n"
            "  swlp analyze layer_traces.json --json\n"
            "  swlp analyze traces.json --output analysis.txt"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    analyze_parser.add_argument(
        "trace_file", type=Path, help="path to JSON trace file from swlp profile"
    )
    analyze_parser.add_argument("--json", dest="analyze_json", action="store_true",
                                help="output analysis as JSON instead of text")
    analyze_parser.add_argument("--output", type=Path, default=None,
                                help="write report to file instead of stdout")

    # ── sweep ────────────────────────────────────────────────────────────
    sweep_parser = subparsers.add_parser(
        "sweep",
        help="sweep simulator parameters and produce a structured dataset",
        description=(
            "Run the scheduler simulator across multiple parameter combinations.\n"
            "Produces a structured JSON or CSV dataset for analysis.\n\n"
            "Examples:\n"
            "  swlp sweep --layers 32,80 --ram 16,24,32 --window 2,4 --resident 0,4,8\n"
            "  swlp sweep --layers 32 --layer-size-mb 512 --ram 16,24,32 --output sweep.json\n"
            "  swlp sweep --layers 80 --ram 16,24,32,48,64 --resident 0,2,4,8,16 --csv"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sweep_parser.add_argument("--layers", type=str, default="32",
                              help="comma-separated layer counts (default: 32)")
    sweep_parser.add_argument("--layer-size-mb", type=str, default="512",
                              help="comma-separated layer sizes in MB (default: 512)")
    sweep_parser.add_argument("--ram", type=str, default="16",
                              help="comma-separated RAM sizes in GB (default: 16)")
    sweep_parser.add_argument("--window", type=str, default="2",
                              help="comma-separated window sizes (default: 2)")
    sweep_parser.add_argument("--prefetch", type=str, default="4",
                              help="comma-separated prefetch depths (default: 4)")
    sweep_parser.add_argument("--resident", type=str, default="0",
                              help="comma-separated resident layer counts (default: 0)")
    sweep_parser.add_argument("--tokens", type=int, default=10, metavar="N",
                              help="number of tokens to simulate (default: 10)")
    sweep_parser.add_argument("--compute-ms", type=float, default=50.0, metavar="MS",
                              help="compute time per layer in ms (default: 50)")
    sweep_parser.add_argument("--ssd-ms", type=float, default=30.0, metavar="MS",
                              help="SSD read latency in ms (default: 30)")
    sweep_parser.add_argument("--upload-ms", type=float, default=10.0, metavar="MS",
                              help="upload latency in ms (default: 10)")
    sweep_parser.add_argument("--workers", type=int, default=2, metavar="N",
                              help="worker thread count (default: 2)")
    sweep_parser.add_argument("--output", type=Path, default=None,
                              help="output file path  (default: sweep_results.json)")
    sweep_parser.add_argument("--csv", action="store_true",
                              help="output as CSV instead of JSON")
    sweep_parser.add_argument("--report", action="store_true",
                              help="print a formatted table to stdout")

    # ── evaluate ─────────────────────────────────────────────────────────
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="compare scheduling policies side-by-side",
        description=(
            "Run multiple scheduling policies on the same workload and compare.\n"
            "Policies: baseline, resident_N, window_N, prefetch_N\n\n"
            "Examples:\n"
            "  swlp evaluate --policies baseline,resident_4,resident_8\n"
            "  swlp evaluate --layers 80 --ram 24 "
            "--policies baseline,resident_4,resident_8,window_4\n"
            "  swlp evaluate --policies baseline,resident_2,resident_4,resident_8,resident_16 "
            "--output eval.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    evaluate_parser.add_argument(
        "--policies", type=str, default="baseline,resident_4,resident_8",
        help="comma-separated policy names (default: baseline,resident_4,resident_8)",
    )
    evaluate_parser.add_argument("--layers", type=int, default=32, metavar="N",
                                 help="number of transformer layers (default: 32)")
    evaluate_parser.add_argument("--layer-size-mb", type=float, default=512, metavar="MB",
                                 help="size of each layer in MB (default: 512)")
    evaluate_parser.add_argument("--ram", type=float, default=16, metavar="GB",
                                 help="total RAM in GB (default: 16)")
    evaluate_parser.add_argument("--window", type=int, default=2, metavar="N",
                                 help="base window size (default: 2)")
    evaluate_parser.add_argument("--prefetch", type=int, default=4, metavar="N",
                                 help="base prefetch depth (default: 4)")
    evaluate_parser.add_argument("--tokens", type=int, default=10, metavar="N",
                                 help="number of tokens to simulate (default: 10)")
    evaluate_parser.add_argument("--compute-ms", type=float, default=50.0, metavar="MS",
                                 help="compute time per layer in ms (default: 50)")
    evaluate_parser.add_argument("--ssd-ms", type=float, default=30.0, metavar="MS",
                                 help="SSD read latency in ms (default: 30)")
    evaluate_parser.add_argument("--upload-ms", type=float, default=10.0, metavar="MS",
                                 help="upload latency in ms (default: 10)")
    evaluate_parser.add_argument("--workers", type=int, default=2, metavar="N",
                                 help="worker thread count (default: 2)")
    evaluate_parser.add_argument("--output", type=Path, default=None,
                                 help="output file path  (default: eval_results.json)")
    evaluate_parser.add_argument("--csv", action="store_true",
                                 help="output as CSV instead of JSON")
    evaluate_parser.add_argument("--report", action="store_true",
                                 help="print a formatted comparison table to stdout")

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

    # ── doctor ────────────────────────────────────────────────────────────
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="diagnose hardware and predict best-observed scheduling configuration",
        description=(
            "Probe hardware, measure pipeline characteristics, predict best-observed\n"
            "resident cache configuration, and explain every decision.\n\n"
            "Examples:\n"
            "  swlp doctor                    # diagnose all known models\n"
            "  swlp doctor mistral-7b         # diagnose specific model"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    doctor_parser.add_argument(
        "model", nargs="?", default=None,
        help="optional model alias to diagnose (default: all known models)",
    )

    # ── models ────────────────────────────────────────────────────────────
    subparsers.add_parser(
        "models",
        help="list model aliases with sizes and HuggingFace ids",
    )

    # ── help ──────────────────────────────────────────────────────────────
    subparsers.add_parser("help", help="show a categorised command reference")

    return parser
