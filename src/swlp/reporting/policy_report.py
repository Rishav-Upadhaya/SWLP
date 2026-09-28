from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ..core.resident_policy import estimate_optimal_resident_count

_REQUIRED_FIELDS = {
    "machine",
    "model",
    "pipeline_ratio",
    "free_ram_gb",
    "resident_count",
    "throughput_tokens_per_second",
}


def _to_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        return float(value)
    raise ValueError(f"Expected numeric value, got {value!r}")


def _to_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip():
        return int(float(value))
    raise ValueError(f"Expected integer value, got {value!r}")


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _load_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return [dict(item) for item in payload["rows"]]
    raise ValueError("JSON must be either a list of rows or an object with a 'rows' list")


def load_policy_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        return _load_csv(path)
    if path.suffix.lower() == ".json":
        return _load_json(path)
    raise ValueError(f"Unsupported file format: {path.suffix}")


def _scenario_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    machine = str(row.get("machine", "")).strip()
    model = str(row.get("model", "")).strip()
    workload = str(row.get("workload", "default")).strip() or "default"
    quant = str(row.get("quant", "fp16")).strip() or "fp16"
    return machine, model, workload, quant


def _validate_fields(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows found in policy validation input")
    missing = _REQUIRED_FIELDS.difference(rows[0].keys())
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_list}")


def evaluate_policy_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    _validate_fields(rows)

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_scenario_key(row), []).append(row)

    scenarios: list[dict[str, Any]] = []

    for key, scenario_rows in grouped.items():
        machine, model, workload, quant = key
        first = scenario_rows[0]
        pipeline_ratio = _to_float(first["pipeline_ratio"])
        free_ram_gb = _to_float(first["free_ram_gb"])

        by_resident: dict[int, list[float]] = {}
        for row in scenario_rows:
            resident = _to_int(row["resident_count"])
            tp = _to_float(row["throughput_tokens_per_second"])
            by_resident.setdefault(resident, []).append(tp)

        mean_tp_by_resident = {
            resident: sum(values) / len(values)
            for resident, values in by_resident.items()
        }

        best_resident, best_tp = max(
            mean_tp_by_resident.items(),
            key=lambda item: item[1],
        )

        predicted_resident = estimate_optimal_resident_count(
            pipeline_ratio, free_ram_gb
        ).resident_count
        predicted_eval_resident, predicted_eval_tp = min(
            mean_tp_by_resident.items(),
            key=lambda item: (abs(item[0] - predicted_resident), -item[1]),
        )

        regret = 0.0
        if best_tp > 0:
            regret = max(0.0, (best_tp - predicted_eval_tp) / best_tp)

        error_layers = predicted_resident - best_resident

        scenarios.append(
            {
                "machine": machine,
                "model": model,
                "workload": workload,
                "quant": quant,
                "pipeline_ratio": pipeline_ratio,
                "free_ram_gb": free_ram_gb,
                "predicted_resident": predicted_resident,
                "evaluated_predicted_resident": predicted_eval_resident,
                "actual_best_resident": best_resident,
                "resident_error_layers": error_layers,
                "within_2_layers": abs(error_layers) <= 2,
                "best_throughput_tokens_per_second": best_tp,
                "predicted_throughput_tokens_per_second": predicted_eval_tp,
                "throughput_regret_pct": regret * 100.0,
            }
        )

    if not scenarios:
        raise ValueError("No valid scenarios in policy validation input")

    abs_errors = [abs(item["resident_error_layers"]) for item in scenarios]
    regrets = [item["throughput_regret_pct"] for item in scenarios]
    within2 = [item["within_2_layers"] for item in scenarios]

    mae = sum(abs_errors) / len(abs_errors)
    within2_pct = 100.0 * (sum(1 for flag in within2 if flag) / len(within2))
    avg_regret_pct = sum(regrets) / len(regrets)

    return {
        "summary": {
            "scenario_count": len(scenarios),
            "resident_mae_layers": mae,
            "within_2_layers_pct": within2_pct,
            "avg_throughput_regret_pct": avg_regret_pct,
        },
        "scenarios": sorted(
            scenarios,
            key=lambda row: (
                row["machine"],
                row["model"],
                row["workload"],
                row["quant"],
            ),
        ),
    }


def print_policy_report(path: Path) -> None:
    rows = load_policy_rows(path)
    result = evaluate_policy_rows(rows)
    summary = result["summary"]

    print(f"Policy validation report for {path}")
    print(f"Scenarios: {summary['scenario_count']}")
    print(f"Resident MAE: {summary['resident_mae_layers']:.2f} layers")
    print(f"Within ±2 layers: {summary['within_2_layers_pct']:.1f}%")
    print(f"Average throughput regret: {summary['avg_throughput_regret_pct']:.2f}%")
    print()
    print(
        "{:<12} {:<16} {:<12} {:>6} {:>6} {:>7} {:>10}".format(
            "machine",
            "model",
            "workload",
            "pred",
            "best",
            "error",
            "regret%",
        )
    )
    print("-" * 78)
    for row in result["scenarios"]:
        print(
            "{:<12} {:<16} {:<12} {:>6} {:>6} {:>7} {:>9.2f}".format(
                row["machine"][:12],
                row["model"][:16],
                row["workload"][:12],
                row["predicted_resident"],
                row["actual_best_resident"],
                row["resident_error_layers"],
                row["throughput_regret_pct"],
            )
        )
