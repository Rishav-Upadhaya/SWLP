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
from pathlib import Path

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
from .core.streaming import has_shards
from .logging import configure_logging
from .model.package import load_layer, package_checkpoint, validate_package
from .reporting.policy_report import print_policy_report
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
        # A draft model on the streaming path means speculative decoding.
        if getattr(args, "draft_model", None) or config.runtime.swlp_draft_model:
            return "speculative"
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
        draft_id = resolve_model(args.draft_model)
        config.runtime.mlx_draft_model = draft_id
        config.runtime.swlp_draft_model = draft_id
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
    # Apple Silicon tuning flags (see runner/mlx_tune.py).
    for flag in ("mlx_kv_bits", "mlx_num_draft_tokens", "mlx_wired_limit"):
        value = getattr(args, flag, None)
        if value is not None:
            setattr(config.runtime, flag, value)
    if getattr(args, "max_kv_size", None) is not None:
        config.runtime.kv_window = args.max_kv_size
    # Phase 23: quality-neutral speedups
    if getattr(args, "no_activation_cache", False):
        config.runtime.swlp_activation_cache = False
    if getattr(args, "no_prealloc_buffer", False):
        config.runtime.swlp_prealloc_buffer = False
    # Phase 23: opt-in quality tradeoffs
    if getattr(args, "early_exit", None) is not None:
        config.runtime.swlp_early_exit = args.early_exit
    if getattr(args, "layer_pruning", None) is not None:
        config.runtime.swlp_layer_pruning = args.layer_pruning
    if getattr(args, "adaptive_precision", None) is not None:
        config.runtime.swlp_adaptive_precision = args.adaptive_precision
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
    if metrics.degradation_count:
        bits.append(f"⚠ {metrics.degradation_count} degradation(s)")
    return "  ·  ".join(bits)


def _run_profile(args: argparse.Namespace) -> int:
    """Run inference with full profiling and export JSON traces."""
    from .runner.base import build_runner

    config = _apply_overrides(load_config(args.config), args)
    configure_logging(config.runtime.log_level, config.runtime.json_logs)
    config.runtime.profile = True
    LOGGER.info(
        "profile_started",
        extra={
            "backend": config.runtime.backend,
            "model_id": config.model.model_id,
            "device": config.runtime.device,
        },
    )
    runner = build_runner(config)
    # Route the profiler dump through --output before the run starts.
    if args.output:
        runner._trace_output = str(Path(args.output))
    if args.timeline or args.detail or args.summary:
        runner._profile_prints = {
            "timeline": bool(args.timeline),
            "detail": bool(args.detail),
            "summary": bool(args.summary),
        }
    result = runner.run(args.prompt or config.generation.prompt, profile=True)

    # Print results
    if not args.json_output:
        print(f"\nPrompt:\n{result.prompt}\n")
        print(f"Completion:\n{result.completion}\n")
        print(_summary_line(result.metrics))
    else:
        print(json.dumps(result.metrics.to_dict(), indent=2, sort_keys=True))

    trace_path = getattr(runner, "_last_trace_path", None)
    if trace_path and Path(trace_path).is_file():
        print(f"Trace file: {trace_path}")
        if args.timeline or args.detail or args.summary:
            print("Timeline and per-stage detail print above; full JSON in the trace file.")
    else:
        print("No trace recorded (backend produced no scheduler profiler data).")

    return 0


def _run_sim(args: argparse.Namespace) -> int:
    """Run the scheduler simulator."""
    from .core.simulator import SimulatorConfig, simulate

    config = SimulatorConfig(
        num_layers=args.layers,
        layer_size_mb=args.layer_size_mb,
        ram_capacity_gb=args.ram_gb,
        window_size=args.window,
        prefetch_depth=args.prefetch,
        worker_count=args.workers,
        compute_time_ms=args.compute_ms,
        ssd_read_latency_ms=args.ssd_ms,
        upload_latency_ms=args.upload_ms,
        eviction_latency_ms=args.evict_ms,
        num_tokens=args.tokens,
        resident_count=args.resident,
        use_resident_cache=args.resident > 0,
    )
    result = simulate(config)

    output_path = args.output or Path("sim_traces.json")
    result.dump(output_path)
    print(f"Simulation saved to {output_path}")

    if args.report:
        result.print_timeline()
        result.print_summary()

    return 0


def _run_analyze(args: argparse.Namespace) -> int:
    """Analyze profiler traces and produce a report."""
    from .core.analyzer import analyze_traces

    if not args.trace_file.exists():
        print(f"Error: trace file not found: {args.trace_file}")
        return 1

    report = analyze_traces(path=args.trace_file)

    if args.analyze_json:
        import json as _json
        payload = {
            "summary": report.summary,
            "diagnoses": [
                {"observation": d.observation, "diagnosis": d.diagnosis,
                 "severity": d.severity, "confidence": d.confidence,
                 "evidence": [{"label": e.label, "value": e.value} for e in d.evidence]}
                for d in report.diagnoses
            ],
            "recommendations": [
                {"priority": r.priority, "action": r.action, "reason": r.reason,
                 "expected_impact": r.expected_impact, "category": r.category,
                 "confidence": r.confidence, "is_prediction": r.is_prediction}
                for r in report.recommendations
            ],
        }
        text = _json.dumps(payload, indent=2)
    else:
        text = report.format()

    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"Analysis written to {args.output}")
    else:
        print(text)

    return 0


def _parse_comma_list(value: str, cast):
    """Parse a comma-separated list of values."""
    return [cast(v.strip()) for v in value.split(",") if v.strip()]


def _run_sweep(args: argparse.Namespace) -> int:
    """Run automated simulator sweeps."""
    from .core.sweep import SweepConfig, run_sweep

    config = SweepConfig(
        num_layers=_parse_comma_list(args.layers, int),
        layer_size_mb=_parse_comma_list(args.layer_size_mb, float),
        ram_gb=_parse_comma_list(args.ram, float),
        window_size=_parse_comma_list(args.window, int),
        prefetch_depth=_parse_comma_list(args.prefetch, int),
        resident_count=_parse_comma_list(args.resident, int),
        num_tokens=args.tokens,
        compute_time_ms=args.compute_ms,
        ssd_read_latency_ms=args.ssd_ms,
        upload_latency_ms=args.upload_ms,
        worker_count=args.workers,
    )

    result = run_sweep(config)

    if args.csv:
        output_path = args.output or Path("sweep_results.csv")
        result.to_csv(output_path)
    else:
        output_path = args.output or Path("sweep_results.json")
        result.to_json(output_path)

    print(f"Sweep complete: {len(result.points)} configurations")
    print(f"Results saved to {output_path}")

    if args.report:
        result.print_table()

    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    """Evaluate scheduling policies side-by-side."""
    from .core.evaluator import EvaluationConfig, evaluate_policies

    policies = _parse_comma_list(args.policies, str)

    config = EvaluationConfig(
        num_layers=args.layers,
        layer_size_mb=args.layer_size_mb,
        ram_gb=args.ram,
        num_tokens=args.tokens,
        compute_time_ms=args.compute_ms,
        ssd_read_latency_ms=args.ssd_ms,
        upload_latency_ms=args.upload_ms,
        worker_count=args.workers,
        base_window_size=args.window,
        base_prefetch_depth=args.prefetch,
        policies=policies,
    )

    result = evaluate_policies(config)

    if args.csv:
        output_path = args.output or Path("eval_results.csv")
        result.to_csv(output_path)
    else:
        output_path = args.output or Path("eval_results.json")
        result.to_json(output_path)

    print(f"Evaluated {len(result.results)} policies")
    print(f"Results saved to {output_path}")

    if args.report:
        result.print_table()

    return 0


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


def _shard_progress(layer: int, total: int, layer_mb: float) -> None:
    """Single-line progress display for ``swlp download`` sharding."""
    bar_width = 24
    filled = int(bar_width * layer / total)
    bar = "█" * filled + "░" * (bar_width - filled)
    end = "\n" if layer == total else ""
    print(f"\r  [{bar}] layer {layer:>3}/{total}  ·  {layer_mb:.0f} MB", end=end, flush=True)


# Manifests and safetensors headers add a little on top of the raw weights;
# 15% headroom keeps the preflight from passing by a rounding hair.
_PREFLIGHT_HEADROOM = 1.15


def _preflight_disk_space(model_arg: str, output_dir: Path) -> bool:
    """Refuse a download that cannot finish: known model size vs free disk.

    Only aliases with a known FP16 size are checked — for arbitrary HF ids the
    size is discovered only after the config downloads, so we stay silent.
    """
    import shutil

    from .cli_doctor import KNOWN_FP16_GB

    known_gb = KNOWN_FP16_GB.get(model_arg)
    if known_gb is None:
        return True
    needed_gb = known_gb * _PREFLIGHT_HEADROOM
    target = output_dir if output_dir.exists() else output_dir.parent
    try:
        free_gb = shutil.disk_usage(target).free / 1024**3
    except OSError:
        return True
    if free_gb >= needed_gb:
        return True
    LOGGER.error(
        "preflight_disk_space_failed",
        extra={"model": model_arg, "needed_gb": round(needed_gb, 1), "free_gb": round(free_gb, 1)},
    )
    print(f"\n✗  Not enough disk space for {model_arg}.")
    print(
        f"   Needs ~{needed_gb:.0f} GB (FP16 + shard overhead), "
        f"free: {free_gb:.0f} GB at {target}."
    )
    print("   Point --output-dir at a volume with more space, or free some disk first.")
    return False


def _fold_positional_model(args: argparse.Namespace) -> argparse.Namespace:
    """`swlp run <model>` / `swlp serve <model>` positional → args.model."""
    pos = getattr(args, "model_pos", None)
    if pos is not None:
        # A path that looks like a shard dir wins over --model.
        p = Path(pos)
        if (p / "shard_manifest.json").exists():
            if args.shard_dir is None:
                args.shard_dir = p
            if args.model is None:
                try:
                    from .model.shard import load_manifest
                    args.model = load_manifest(p).model_id
                except Exception:
                    pass
        elif args.model is None:
            args.model = pos
    return args


def _ensure_shards(config: AppConfig, model_arg: str) -> AppConfig:
    """One-command setup: shard the model for streaming unless already sharded.

    No-op when the backend doesn't need shards (mock/mlx/hf) or --shard-dir
    already resolves. Cache-aware: an existing manifest skips straight through.
    """
    from .runner.base import build_runner  # noqa: F401 — import check only

    if config.runtime.backend in ("mock", "mlx") or config.runtime.backend == "hf":
        return config
    if config.runtime.shard_dir is not None and has_shards(config.runtime.shard_dir):
        return config

    slug = model_arg.split("/")[-1]
    output_dir = Path("shards") / slug
    if has_shards(output_dir):
        print(f"  ✓  reusing existing shards: {output_dir}")
    else:
        from .model.shard import shard_model_by_layer
        print(f"\nDownloading and sharding: {config.model.model_id}")
        print(f"Output directory:         {output_dir}\n")
        shard_model_by_layer(
            config.model.model_id, output_dir,
            cache_dir=str(config.cache.cache_dir) if config.cache.cache_dir else None,
            progress=_shard_progress,
        )
        from .model.shard import load_manifest
        total_gb = load_manifest(output_dir).total_weight_mb / 1024
        print(f"\n✓  Sharded to {output_dir}  ·  {total_gb:.1f} GB\n")
    config.runtime.shard_dir = output_dir
    if config.runtime.backend not in ("speculative",):
        config.runtime.backend = "swlp"
    return config


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

    if args.command in ("run", "serve"):
        args = _fold_positional_model(args)
        if args.model is None and args.shard_dir is None:
            print("Error: `swlp run` needs a model:  swlp run mistral-7b", file=sys.stderr)
            return 1
        config = _apply_overrides(load_config(args.config), args)
        configure_logging(config.runtime.log_level, config.runtime.json_logs)
        model_arg = args.model or str(args.shard_dir)
        config = _ensure_shards(config, model_arg)

        if args.command == "serve":
            from .serve import serve
            serve(config, args.host, args.port)
            return 0

        check_hf_oom(config)
        if getattr(args, "chat", False):
            from .chat import run_chat
            run_chat(config, max_tokens=getattr(args, "max_chat_tokens", 512))
            return 0
        result = execute_baseline(config, args.prompt)
        LOGGER.info("baseline_finished", extra=result.metrics.to_dict())
        _print_result(result, args.json_output)
        return 0

    if args.command == "doctor":
        from .cli_doctor import print_doctor
        print_doctor(model=args.model)
        return 0

    if args.command == "models":
        from .cli_doctor import print_models
        print_models()
        return 0

    if args.command in ("download", "pull"):
        from pathlib import Path as _Path

        from .model.shard import shard_model_by_layer

        model_id = resolve_model(args.model)
        slug = args.model.split("/")[-1]  # friendly dir name (alias or last HF id segment)
        output_dir: _Path = args.output_dir or _Path("shards") / slug
        cache_dir = str(args.cache_dir) if args.cache_dir else None

        if not _preflight_disk_space(args.model, output_dir):
            return 1

        print(f"\nDownloading and sharding: {model_id}")
        print(f"Output directory:         {output_dir}")
        print("─" * 56)
        print("This is a one-time operation.  Large models (7B+) may")
        print("take 20–60 min depending on your internet connection.\n")

        manifest = shard_model_by_layer(
            model_id, output_dir, cache_dir=cache_dir, progress=_shard_progress,
        )
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

    if args.command == "compress-shards":
        from .model.shard import compress_shards, decompress_shards, list_layer_paths

        before = sum(p.stat().st_size for p in list_layer_paths(args.shard_dir))

        def _compress_progress(done: int, total: int, ratio: float) -> None:
            print(f"  layer {done}/{total}  ratio {ratio:.3f}", flush=True)

        if args.revert:
            print(f"\nRestoring plain shards in place: {args.shard_dir}")
            print("Each layer is SHA-256-verified before its .swz source is deleted.\n")
            manifest = decompress_shards(args.shard_dir, progress=_compress_progress)
            after = sum(p.stat().st_size for p in list_layer_paths(args.shard_dir))
            print(f"\n✓  {manifest.num_layers} layers restored to .safetensors")
            print(f"   {before / 1e9:.2f} GB → {after / 1e9:.2f} GB\n")
            return 0

        print(f"\nCompressing shards in place: {args.shard_dir}")
        print("Each layer is roundtrip-verified before its original is deleted.\n")

        manifest = compress_shards(args.shard_dir, progress=_compress_progress)
        after = sum(p.stat().st_size for p in list_layer_paths(args.shard_dir))
        print(f"\n✓  {manifest.num_layers} layers compressed")
        print(
            f"   {before / 1e9:.2f} GB → {after / 1e9:.2f} GB"
            f"  (saved {(before - after) / 1e9:.2f} GB, ratio {after / before:.3f})\n"
        )
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

    if args.command == "policy-report":
        print_policy_report(args.path)
        return 0

    if args.command == "profile":
        return _run_profile(args)

    if args.command == "sim":
        return _run_sim(args)

    if args.command == "analyze":
        return _run_analyze(args)

    if args.command == "sweep":
        return _run_sweep(args)

    if args.command == "evaluate":
        return _run_evaluate(args)

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

        args = _fold_positional_model(args)
        config = _apply_overrides(load_config(args.config), args)
        configure_logging(config.runtime.log_level, config.runtime.json_logs)
        max_chat_tokens = getattr(args, "max_chat_tokens", 512)
        run_chat(config, max_tokens=max_chat_tokens)
        return 0

    return _run_inference(args)


if __name__ == "__main__":
    raise SystemExit(main())
