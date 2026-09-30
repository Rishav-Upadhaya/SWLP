from __future__ import annotations

from scripts.research.simtools.policy_report import evaluate_policy_rows


def test_policy_report_summary_metrics() -> None:
    rows = [
        {
            "machine": "m2-air",
            "model": "mistral-7b",
            "workload": "short",
            "quant": "fp16",
            "pipeline_ratio": 5.0,
            "free_ram_gb": 8.0,
            "resident_count": 8,
            "throughput_tokens_per_second": 1.0,
        },
        {
            "machine": "m2-air",
            "model": "mistral-7b",
            "workload": "short",
            "quant": "fp16",
            "pipeline_ratio": 5.0,
            "free_ram_gb": 8.0,
            "resident_count": 12,
            "throughput_tokens_per_second": 1.4,
        },
        {
            "machine": "m2-air",
            "model": "mistral-7b",
            "workload": "short",
            "quant": "fp16",
            "pipeline_ratio": 5.0,
            "free_ram_gb": 8.0,
            "resident_count": 16,
            "throughput_tokens_per_second": 1.3,
        },
        {
            "machine": "m2-pro",
            "model": "mistral-7b",
            "workload": "short",
            "quant": "fp16",
            "pipeline_ratio": 7.0,
            "free_ram_gb": 12.0,
            "resident_count": 12,
            "throughput_tokens_per_second": 1.2,
        },
        {
            "machine": "m2-pro",
            "model": "mistral-7b",
            "workload": "short",
            "quant": "fp16",
            "pipeline_ratio": 7.0,
            "free_ram_gb": 12.0,
            "resident_count": 20,
            "throughput_tokens_per_second": 1.45,
        },
        {
            "machine": "m2-pro",
            "model": "mistral-7b",
            "workload": "short",
            "quant": "fp16",
            "pipeline_ratio": 7.0,
            "free_ram_gb": 12.0,
            "resident_count": 22,
            "throughput_tokens_per_second": 1.5,
        },
    ]

    result = evaluate_policy_rows(rows)

    summary = result["summary"]
    assert summary["scenario_count"] == 2
    assert summary["resident_mae_layers"] >= 0.0
    assert 0.0 <= summary["within_2_layers_pct"] <= 100.0
    assert summary["avg_throughput_regret_pct"] >= 0.0


def test_policy_report_uses_nearest_tested_resident() -> None:
    rows = [
        {
            "machine": "m2-air",
            "model": "mistral-7b",
            "pipeline_ratio": 5.0,
            "free_ram_gb": 8.0,
            "resident_count": 8,
            "throughput_tokens_per_second": 1.0,
        },
        {
            "machine": "m2-air",
            "model": "mistral-7b",
            "pipeline_ratio": 5.0,
            "free_ram_gb": 8.0,
            "resident_count": 16,
            "throughput_tokens_per_second": 1.2,
        },
    ]

    result = evaluate_policy_rows(rows)
    scenario = result["scenarios"][0]

    # Predicted resident at (ratio=5, ram=8) is 12, nearest tested point is 16
    # due to tie-break toward higher throughput.
    assert scenario["predicted_resident"] == 12
    assert scenario["evaluated_predicted_resident"] == 16
