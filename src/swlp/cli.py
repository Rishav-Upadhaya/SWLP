"""SWLP command-line interface.

Primary use — run inference directly, no config file required:

    swlp --model mistral-7b --prompt "Hey, what can you do?"
    swlp --model mistral-7b --backend mlx --quant int8 --prompt "..."
    swlp --shard-dir ./shards/mistral-7b --window 2 --prompt "..."

Interactive chat with streaming:

    swlp chat --model mistral-7b --backend mlx --quant int8

Tool subcommands handle everything else: ``benchmark``, ``simulate``, ``suite``,
``package``, ``validate-package``, ``layer``, ``report``, ``suite-report``.

Parser construction lives in ``cli_args.py``; this module is dispatch only.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import asdict

from .benchmark.run import default_benchmark_path, run_benchmark, save_benchmark
from .benchmark.simulator import (
    default_simulation_path,
    load_scenario,
    save_simulation,
    simulate_scenario,
)
from .benchmark.suite import default_suite_path, load_suite, run_suite, save_suite
from .cli_args import build_parser, resolve_model
from .config import AppConfig, load_config
from .logging import configure_logging
from .model.package import load_layer, package_checkpoint, validate_package
from .reporting.run_report import print_report
from .reporting.sim_report import print_simulation_report
from .reporting.suite_report import print_suite_report
from .runner.base import check_hf_oom, execute_baseline

LOGGER = logging.getLogger(__name__)


def _resolve_backend(args: argparse.Namespace, config: AppConfig) -> str:
    """Pick the backend: explicit flag wins, else infer from --quant / --shard-dir."""
    if args.backend is not None:
        return args.backend
    if args.quant is not None:
        return "mlx"
    if args.shard_dir is not None or config.runtime.shard_dir is not None:
        return "swlp"
    return config.runtime.backend


def _apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    if args.model is not None:
        config.model.model_id = resolve_model(args.model)
    if args.device is not None:
        config.runtime.device = args.device
    if args.cache_dir is not None:
        config.cache.cache_dir = args.cache_dir
    if args.prompt is not None:
        config.generation.prompt = args.prompt
    if args.max_tokens is not None:
        config.generation.max_new_tokens = args.max_tokens
    if args.shard_dir is not None:
        config.runtime.shard_dir = args.shard_dir
        # If the user didn't specify --model, pull the model_id from the shard
        # manifest so the right architecture is built (not the default tiny-gpt2).
        if args.model is None:
            try:
                from .model.shard import load_manifest
                manifest = load_manifest(args.shard_dir)
                config.model.model_id = manifest.model_id
                LOGGER.info(
                    "model_id_from_manifest",
                    extra={"model_id": manifest.model_id, "shard_dir": str(args.shard_dir)},
                )
            except Exception as exc:
                LOGGER.warning(
                    "manifest_read_failed",
                    extra={"shard_dir": str(args.shard_dir), "error": str(exc)},
                )
    if args.quant is not None:
        config.runtime.mlx_quant = args.quant
    if getattr(args, "draft_model", None):
        config.runtime.mlx_draft_model = resolve_model(args.draft_model)
    if args.window is not None:
        config.runtime.swlp_window_size = args.window
    if args.profile:
        config.runtime.profile = True
    if args.swlp_prefetch_depth is not None:
        config.runtime.swlp_prefetch_depth = args.swlp_prefetch_depth
    if args.swlp_no_prefetch:
        config.runtime.swlp_prefetch = False
    if args.swlp_no_double_buffer:
        config.runtime.swlp_double_buffer = False
    if args.swlp_no_pin_memory:
        config.runtime.swlp_pin_memory = False
    if args.kv_budget_mb is not None:
        config.runtime.kv_memory_budget_mb = args.kv_budget_mb
    if args.kv_compression:
        config.runtime.kv_compression = True
    if args.kv_compression_level is not None:
        config.runtime.kv_compression_level = args.kv_compression_level
    if args.kv_tiering:
        config.runtime.kv_tiering = True
    if getattr(args, "kv_window", None) is not None:
        config.runtime.kv_window = args.kv_window
    if getattr(args, "kv_quant", None) is not None:
        config.runtime.kv_quant = args.kv_quant
    config.runtime.backend = _resolve_backend(args, config)
    return config


def _summary_line(metrics) -> str:
    bits = [f"backend={metrics.backend}", f"device={metrics.device}"]
    if metrics.throughput_tokens_per_second:
        bits.append(f"{metrics.throughput_tokens_per_second:.1f} tok/s")
    if metrics.time_to_first_token_seconds:
        bits.append(f"first token {metrics.time_to_first_token_seconds:.2f}s")
    if metrics.generate_seconds:
        bits.append(f"generate {metrics.generate_seconds:.1f}s")
    if metrics.ram_peak_bytes:
        bits.append(f"RAM {metrics.ram_peak_bytes / 1e9:.2f} GB")
    return "  ·  ".join(bits)


def _print_result(result, json_output: bool) -> None:
    if json_output:
        payload = {
            "prompt": result.prompt,
            "completion": result.completion,
            "metrics": result.metrics.to_dict(),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(f"\nPrompt:\n{result.prompt}\n")
    print(f"Completion:\n{result.completion}\n")
    print(_summary_line(result.metrics))
    print("(use --json for full metrics)")


def _print_payload(payload, json_output: bool) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _layer_identifier(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def _run_inference(args: argparse.Namespace) -> int:
    config = _apply_overrides(load_config(args.config), args)
    configure_logging(config.runtime.log_level, config.runtime.json_logs)
    check_hf_oom(config)
    LOGGER.info(
        "baseline_started",
        extra={
            "backend": config.runtime.backend,
            "model_id": config.model.model_id,
            "device": config.runtime.device,
        },
    )
    if args.command == "benchmark":
        config.runtime.profile = True
        run = run_benchmark(
            config, args.prompt, args.runs, args.prompt_set, args.warmup_runs,
            getattr(args, "batch_size", 1),
        )
        output_path = args.output or default_benchmark_path(args.format)
        save_benchmark(run.records, output_path, args.format, run.metadata, run.summary)
        print(f"Saved benchmark metrics to {output_path}")
        if args.report:
            print_report(output_path)
        return 0

    result = execute_baseline(config, args.prompt)
    LOGGER.info("baseline_finished", extra=result.metrics.to_dict())
    _print_result(result, args.json_output)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not raw:
        from .cli_help import print_swlp_help
        print_swlp_help()
        return 0
    args = parser.parse_args(raw)

    if args.command == "help":
        from .cli_help import print_swlp_help
        print_swlp_help()
        return 0

    if args.command == "download":
        from pathlib import Path as _Path

        from .model.shard import shard_model_by_layer

        model_id = resolve_model(args.model)
        slug = args.model.split("/")[-1]  # friendly dir name (alias or last HF id segment)
        output_dir: _Path = args.output_dir or _Path("shards") / slug
        cache_dir = str(args.cache_dir) if args.cache_dir else None

        print(f"\nDownloading and sharding: {model_id}")
        print(f"Output directory:         {output_dir}")
        print("─" * 56)
        print("This is a one-time operation.  Large models (7B+) may")
        print("take 20–60 min depending on your internet connection.\n")

        manifest = shard_model_by_layer(model_id, output_dir, cache_dir=cache_dir)
        total_gb = manifest.total_weight_mb / 1024

        print(f"\n✓  Sharded to {output_dir}")
        print(f"   {manifest.num_layers} layers  ·  {total_gb:.1f} GB total\n")
        print("─" * 56)
        print("Run inference:")
        print(f'  swlp --shard-dir {output_dir} --window 2 --prompt "..."')
        print("\nChat:")
        print(f"  swlp chat --shard-dir {output_dir} --window 2")
        print()
        return 0

    if args.command == "package":
        manifest = package_checkpoint(args.checkpoint, args.output_dir, args.model_name)
        _print_payload(asdict(manifest), args.json_output)
        return 0

    if args.command == "validate-package":
        _print_payload(asdict(validate_package(args.path)), args.json_output)
        return 0

    if args.command == "layer":
        record, tensors = load_layer(args.path, _layer_identifier(args.layer))
        payload = {
            "layer": asdict(record),
            "tensors": [
                {
                    "name": name,
                    "shape": [int(d) for d in tensor.shape],
                    "dtype": str(tensor.dtype).replace("torch.", ""),
                    "size_bytes": int(tensor.element_size() * tensor.numel()),
                }
                for name, tensor in tensors.items()
            ],
        }
        _print_payload(payload, args.json_output)
        return 0

    if args.command == "report":
        print_report(args.path)
        return 0

    if args.command == "suite-report":
        print_suite_report(args.path)
        return 0

    if args.command == "simulate":
        scenario = load_scenario(args.scenario)
        results = simulate_scenario(scenario)
        output_path = args.output or default_simulation_path(args.format)
        save_simulation(results, output_path, args.format)
        print(f"Saved simulation results to {output_path}")
        if args.report:
            print_simulation_report(output_path)
        return 0

    if args.command == "suite":
        config = load_config(args.config)
        configure_logging(config.runtime.log_level, config.runtime.json_logs)
        suite = load_suite(args.suite)
        results = run_suite(config, suite)
        output_path = args.output or default_suite_path(args.format)
        save_suite(results, output_path, args.format)
        print(f"Saved suite results to {output_path}")
        if args.report:
            print_suite_report(output_path)
        return 0

    if args.command == "chat":
        from .chat import run_chat

        config = _apply_overrides(load_config(args.config), args)
        configure_logging(config.runtime.log_level, config.runtime.json_logs)
        max_chat_tokens = getattr(args, "max_chat_tokens", 512)
        run_chat(config, max_tokens=max_chat_tokens)
        return 0

    return _run_inference(args)


if __name__ == "__main__":
    raise SystemExit(main())
