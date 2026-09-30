"""Research CLI for the scheduling simulators: ``python -m scripts.research.simtools <cmd>``.

Moved out of the ``swlp`` package: these tools model schedules, they don't run models.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.research.simtools.policy_report import print_policy_report


def _run_sim(args: argparse.Namespace) -> int:
    """Run the scheduler simulator."""
    from scripts.research.simtools.simulator import SimulatorConfig, simulate

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
    from scripts.research.simtools.analyzer import analyze_traces

    if not args.trace_file.exists():
        print(f"Error: trace file not found: {args.trace_file}")
        return 1

    report = analyze_traces(path=args.trace_file)

    if args.analyze_json:
        import json as _json

        payload = {
            "summary": report.summary,
            "diagnoses": [
                {
                    "observation": d.observation,
                    "diagnosis": d.diagnosis,
                    "severity": d.severity,
                    "confidence": d.confidence,
                    "evidence": [{"label": e.label, "value": e.value} for e in d.evidence],
                }
                for d in report.diagnoses
            ],
            "recommendations": [
                {
                    "priority": r.priority,
                    "action": r.action,
                    "reason": r.reason,
                    "expected_impact": r.expected_impact,
                    "category": r.category,
                    "confidence": r.confidence,
                    "is_prediction": r.is_prediction,
                }
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
    from scripts.research.simtools.sweep import SweepConfig, run_sweep

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
    from scripts.research.simtools.evaluator import EvaluationConfig, evaluate_policies

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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="simtools")
    subparsers = parser.add_subparsers(dest="command", required=True)

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
            "  simtools policy-report experiments/policy_matrix.csv"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    policy_report_parser.add_argument(
        "path",
        type=Path,
        help="path to a policy validation matrix CSV or JSON file",
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
            "  simtools sim --layers 32 --layer-size-mb 512 --window 2 --prefetch 4\n"
            "  simtools sim --layers 80 --layer-size-mb 700 --window 4 --tokens 20 --report\n"
            "  simtools sim --layers 32 --resident 8 --output sim.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sim_parser.add_argument(
        "--layers",
        type=int,
        default=32,
        metavar="N",
        help="number of transformer layers (default: 32)",
    )
    sim_parser.add_argument(
        "--layer-size-mb",
        type=float,
        default=512,
        metavar="MB",
        help="size of each layer in MB (default: 512)",
    )
    sim_parser.add_argument(
        "--ram-gb", type=float, default=16.0, metavar="GB", help="total RAM in GB (default: 16)"
    )
    sim_parser.add_argument(
        "--window", type=int, default=2, metavar="N", help="sliding window size (default: 2)"
    )
    sim_parser.add_argument(
        "--prefetch", type=int, default=4, metavar="N", help="prefetch depth (default: 4)"
    )
    sim_parser.add_argument(
        "--workers", type=int, default=2, metavar="N", help="worker thread count (default: 2)"
    )
    sim_parser.add_argument(
        "--compute-ms",
        type=float,
        default=50.0,
        metavar="MS",
        help="simulated compute time per layer in ms (default: 50)",
    )
    sim_parser.add_argument(
        "--ssd-ms",
        type=float,
        default=30.0,
        metavar="MS",
        help="simulated SSD read latency in ms (default: 30)",
    )
    sim_parser.add_argument(
        "--upload-ms",
        type=float,
        default=10.0,
        metavar="MS",
        help="simulated upload latency in ms (default: 10)",
    )
    sim_parser.add_argument(
        "--evict-ms",
        type=float,
        default=5.0,
        metavar="MS",
        help="simulated eviction latency in ms (default: 5)",
    )
    sim_parser.add_argument(
        "--tokens",
        type=int,
        default=10,
        metavar="N",
        help="number of tokens to simulate (default: 10)",
    )
    sim_parser.add_argument(
        "--resident",
        type=int,
        default=0,
        metavar="N",
        help="number of resident (never-evicted) layers (default: 0)",
    )
    sim_parser.add_argument(
        "--output", type=Path, default=None, help="output JSON file  (default: sim_traces.json)"
    )
    sim_parser.add_argument(
        "--report", action="store_true", help="print a formatted summary to stdout"
    )

    # ── analyze ──────────────────────────────────────────────────────────
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="analyze profiler traces and produce actionable recommendations",
        description=(
            "Load a JSON trace file produced by `swlp profile` and analyze it.\n"
            "Identifies the pipeline bottleneck, GPU idle reasons, prefetch efficiency,\n"
            "memory headroom, and generates specific recommendations.\n\n"
            "Examples:\n"
            "  simtools analyze layer_traces.json\n"
            "  simtools analyze layer_traces.json --json\n"
            "  simtools analyze traces.json --output analysis.txt"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    analyze_parser.add_argument(
        "trace_file", type=Path, help="path to JSON trace file from swlp profile"
    )
    analyze_parser.add_argument(
        "--json",
        dest="analyze_json",
        action="store_true",
        help="output analysis as JSON instead of text",
    )
    analyze_parser.add_argument(
        "--output", type=Path, default=None, help="write report to file instead of stdout"
    )

    # ── sweep ────────────────────────────────────────────────────────────
    sweep_parser = subparsers.add_parser(
        "sweep",
        help="sweep simulator parameters and produce a structured dataset",
        description=(
            "Run the scheduler simulator across multiple parameter combinations.\n"
            "Produces a structured JSON or CSV dataset for analysis.\n\n"
            "Examples:\n"
            "  simtools sweep --layers 32,80 --ram 16,24,32 --window 2,4 --resident 0,4,8\n"
            "  simtools sweep --layers 32 --layer-size-mb 512 --ram 16,24,32 --output sweep.json\n"
            "  simtools sweep --layers 80 --ram 16,24,32,48,64 --resident 0,2,4,8,16 --csv"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sweep_parser.add_argument(
        "--layers", type=str, default="32", help="comma-separated layer counts (default: 32)"
    )
    sweep_parser.add_argument(
        "--layer-size-mb",
        type=str,
        default="512",
        help="comma-separated layer sizes in MB (default: 512)",
    )
    sweep_parser.add_argument(
        "--ram", type=str, default="16", help="comma-separated RAM sizes in GB (default: 16)"
    )
    sweep_parser.add_argument(
        "--window", type=str, default="2", help="comma-separated window sizes (default: 2)"
    )
    sweep_parser.add_argument(
        "--prefetch", type=str, default="4", help="comma-separated prefetch depths (default: 4)"
    )
    sweep_parser.add_argument(
        "--resident",
        type=str,
        default="0",
        help="comma-separated resident layer counts (default: 0)",
    )
    sweep_parser.add_argument(
        "--tokens",
        type=int,
        default=10,
        metavar="N",
        help="number of tokens to simulate (default: 10)",
    )
    sweep_parser.add_argument(
        "--compute-ms",
        type=float,
        default=50.0,
        metavar="MS",
        help="compute time per layer in ms (default: 50)",
    )
    sweep_parser.add_argument(
        "--ssd-ms",
        type=float,
        default=30.0,
        metavar="MS",
        help="SSD read latency in ms (default: 30)",
    )
    sweep_parser.add_argument(
        "--upload-ms",
        type=float,
        default=10.0,
        metavar="MS",
        help="upload latency in ms (default: 10)",
    )
    sweep_parser.add_argument(
        "--workers", type=int, default=2, metavar="N", help="worker thread count (default: 2)"
    )
    sweep_parser.add_argument(
        "--output", type=Path, default=None, help="output file path  (default: sweep_results.json)"
    )
    sweep_parser.add_argument("--csv", action="store_true", help="output as CSV instead of JSON")
    sweep_parser.add_argument(
        "--report", action="store_true", help="print a formatted table to stdout"
    )

    # ── evaluate ─────────────────────────────────────────────────────────
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="compare scheduling policies side-by-side",
        description=(
            "Run multiple scheduling policies on the same workload and compare.\n"
            "Policies: baseline, resident_N, window_N, prefetch_N\n\n"
            "Examples:\n"
            "  simtools evaluate --policies baseline,resident_4,resident_8\n"
            "  simtools evaluate --layers 80 --ram 24 "
            "--policies baseline,resident_4,resident_8,window_4\n"
            "  simtools evaluate --policies baseline,resident_2,resident_4,resident_8,resident_16 "
            "--output eval.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    evaluate_parser.add_argument(
        "--policies",
        type=str,
        default="baseline,resident_4,resident_8",
        help="comma-separated policy names (default: baseline,resident_4,resident_8)",
    )
    evaluate_parser.add_argument(
        "--layers",
        type=int,
        default=32,
        metavar="N",
        help="number of transformer layers (default: 32)",
    )
    evaluate_parser.add_argument(
        "--layer-size-mb",
        type=float,
        default=512,
        metavar="MB",
        help="size of each layer in MB (default: 512)",
    )
    evaluate_parser.add_argument(
        "--ram", type=float, default=16, metavar="GB", help="total RAM in GB (default: 16)"
    )
    evaluate_parser.add_argument(
        "--window", type=int, default=2, metavar="N", help="base window size (default: 2)"
    )
    evaluate_parser.add_argument(
        "--prefetch", type=int, default=4, metavar="N", help="base prefetch depth (default: 4)"
    )
    evaluate_parser.add_argument(
        "--tokens",
        type=int,
        default=10,
        metavar="N",
        help="number of tokens to simulate (default: 10)",
    )
    evaluate_parser.add_argument(
        "--compute-ms",
        type=float,
        default=50.0,
        metavar="MS",
        help="compute time per layer in ms (default: 50)",
    )
    evaluate_parser.add_argument(
        "--ssd-ms",
        type=float,
        default=30.0,
        metavar="MS",
        help="SSD read latency in ms (default: 30)",
    )
    evaluate_parser.add_argument(
        "--upload-ms",
        type=float,
        default=10.0,
        metavar="MS",
        help="upload latency in ms (default: 10)",
    )
    evaluate_parser.add_argument(
        "--workers", type=int, default=2, metavar="N", help="worker thread count (default: 2)"
    )
    evaluate_parser.add_argument(
        "--output", type=Path, default=None, help="output file path  (default: eval_results.json)"
    )
    evaluate_parser.add_argument("--csv", action="store_true", help="output as CSV instead of JSON")
    evaluate_parser.add_argument(
        "--report", action="store_true", help="print a formatted comparison table to stdout"
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "policy-report":
        print_policy_report(args.path)
        return 0
    return {
        "sim": _run_sim,
        "analyze": _run_analyze,
        "sweep": _run_sweep,
        "evaluate": _run_evaluate,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
