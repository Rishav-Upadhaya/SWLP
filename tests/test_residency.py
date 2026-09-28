"""Tests for swlp.core.residency — ResidencyPlan and plan_residency()."""
from __future__ import annotations

from swlp.core.residency import (
    _HEADROOM_FACTOR,
    _OS_RESERVE_BYTES,
    _WORKING_RESERVE_BYTES,
    ResidencyPlan,
    build_residency_decision,
    estimate_pipeline_ratio,
    plan_residency,
    run_startup_calibration,
)
from swlp.core.resident_policy import estimate_optimal_resident_count

# ── helpers ──────────────────────────────────────────────────────────────────

def _gb(n: float) -> int:
    return int(n * 1024 ** 3)


# ── basic contract ────────────────────────────────────────────────────────────

def test_plan_residency_returns_plan():
    plan = plan_residency(
        total_memory_bytes=_gb(16),
        layer_weight_bytes=_gb(0.436),
        num_layers=32,
    )
    assert isinstance(plan, ResidencyPlan)


def test_resident_plus_streaming_equals_num_layers():
    for num_layers in [32, 40, 80]:
        plan = plan_residency(
            total_memory_bytes=_gb(16),
            layer_weight_bytes=_gb(0.4),
            num_layers=num_layers,
        )
        assert plan.resident_count + plan.streaming_count == num_layers


def test_resident_bytes_consistent():
    layer_bytes = _gb(0.5)
    plan = plan_residency(
        total_memory_bytes=_gb(24),
        layer_weight_bytes=layer_bytes,
        num_layers=40,
    )
    assert plan.resident_bytes == plan.resident_count * layer_bytes
    assert plan.streaming_bytes == plan.streaming_count * layer_bytes


# ── memory budget maths ───────────────────────────────────────────────────────

def test_model_almost_fits_uses_partial_resident_policy():
    """When full model does not fit, planner chooses a bounded partial resident set."""
    plan = plan_residency(
        total_memory_bytes=_gb(16),
        layer_weight_bytes=_gb(0.436),   # Mistral-7B layer
        num_layers=32,
        pipeline_ratio=5.0,
    )
    assert 0 < plan.resident_count <= 22
    assert plan.streaming_count == 32 - plan.resident_count


def test_huge_model_resident_is_memory_bounded():
    """For very large models, resident count is capped by available memory."""
    plan = plan_residency(
        total_memory_bytes=_gb(16),
        layer_weight_bytes=_gb(0.7),   # ~30B rough layer size
        num_layers=60,
        pipeline_ratio=10.0,
    )
    assert 0 <= plan.resident_count < 60
    assert plan.streaming_count == 60 - plan.resident_count


def test_partial_residency_guard_exact_boundary():
    """Model exactly equal to usable budget → full residency allowed."""
    usable = int((_gb(16) - _gb(4) - _gb(2)) * 0.75)   # = 7,516,192,768
    layer_bytes = usable // 10   # 10 layers fit exactly
    plan = plan_residency(
        total_memory_bytes=_gb(16),
        layer_weight_bytes=layer_bytes,
        num_layers=10,
    )
    # total_model = 10 * layer_bytes = usable → residency allowed.
    assert plan.resident_count == 10
    assert plan.streaming_count == 0


def test_empirical_policy_matches_anchor_points():
    # Ratio 5 table anchors: 8GB->12, 12GB->20, 16GB+->22
    result = estimate_optimal_resident_count(5.0, 8.0)
    assert result.resident_count == 12

    result = estimate_optimal_resident_count(5.0, 12.0)
    assert result.resident_count == 20

    result = estimate_optimal_resident_count(5.0, 16.0)
    assert result.resident_count == 22


def test_empirical_policy_interpolates_smoothly():
    # Midpoint in RAM between 8 and 12 GB for ratio 5.
    result = estimate_optimal_resident_count(5.0, 10.0)
    assert result.resident_count == 16


def test_empirical_policy_clamps_ratio_and_ram_bounds():
    result = estimate_optimal_resident_count(1.0, 6.0)
    assert result.resident_count == 2

    result = estimate_optimal_resident_count(20.0, 64.0)
    assert result.resident_count == 22


def test_estimate_pipeline_ratio_increases_with_layer_size():
    small = estimate_pipeline_ratio(
        layer_weight_bytes=_gb(0.1),
        ssd_bandwidth_gbps=6.5,
    )
    large = estimate_pipeline_ratio(
        layer_weight_bytes=_gb(0.5),
        ssd_bandwidth_gbps=6.5,
    )
    assert large > small


def test_plan_uses_free_memory_when_provided():
    total = _gb(16)
    constrained = _gb(10)
    layer = _gb(0.4)
    with_free = plan_residency(
        total_memory_bytes=total,
        free_memory_bytes=constrained,
        layer_weight_bytes=layer,
        num_layers=32,
        pipeline_ratio=7.0,
    )
    without_free = plan_residency(
        total_memory_bytes=total,
        layer_weight_bytes=layer,
        num_layers=32,
        pipeline_ratio=7.0,
    )
    assert with_free.available_bytes <= without_free.available_bytes
    assert with_free.resident_count <= without_free.resident_count


def test_full_fit_when_model_tiny():
    """tiny-gpt2 (few MB) on 16 GB: all layers resident."""
    plan = plan_residency(
        total_memory_bytes=_gb(16),
        layer_weight_bytes=1 * 1024 * 1024,   # 1 MB per layer
        num_layers=12,
    )
    assert plan.resident_count == 12
    assert plan.streaming_count == 0


def test_resident_count_never_exceeds_num_layers():
    plan = plan_residency(
        total_memory_bytes=_gb(128),    # huge memory
        layer_weight_bytes=_gb(0.1),
        num_layers=10,
    )
    assert plan.resident_count == 10
    assert plan.streaming_count == 0


# ── edge cases ────────────────────────────────────────────────────────────────

def test_zero_layers_returns_zero_resident():
    plan = plan_residency(
        total_memory_bytes=_gb(16),
        layer_weight_bytes=_gb(0.5),
        num_layers=0,
    )
    assert plan.resident_count == 0
    assert plan.streaming_count == 0


def test_zero_layer_weight_returns_zero_resident():
    plan = plan_residency(
        total_memory_bytes=_gb(16),
        layer_weight_bytes=0,
        num_layers=32,
    )
    assert plan.resident_count == 0
    assert plan.streaming_count == 32


def test_insufficient_memory_returns_zero_resident():
    """If after reserves there is no headroom, no layers are resident."""
    tiny_ram = _OS_RESERVE_BYTES + _WORKING_RESERVE_BYTES  # exactly at the boundary
    plan = plan_residency(
        total_memory_bytes=tiny_ram,
        layer_weight_bytes=_gb(0.5),
        num_layers=32,
    )
    assert plan.resident_count == 0
    assert plan.available_bytes == 0


def test_available_bytes_reflects_headroom_factor():
    total = _gb(16)
    headroom = total - _OS_RESERVE_BYTES - _WORKING_RESERVE_BYTES
    expected_usable = int(headroom * _HEADROOM_FACTOR)
    plan = plan_residency(
        total_memory_bytes=total,
        layer_weight_bytes=_gb(10),   # too big to fit — resident_count = 0
        num_layers=5,
    )
    assert plan.available_bytes == expected_usable


# ── startup calibration ──────────────────────────────────────────────────────

def test_startup_calibration_estimates_ratio():
    """Startup calibration should estimate pipeline ratio from hardware."""
    result = run_startup_calibration(
        layer_weight_bytes=_gb(0.416),
        num_layers=32,
        total_memory_bytes=_gb(16),
        free_memory_bytes=_gb(12),
        ssd_bandwidth_gbps=6.5,
    )
    assert result.estimated_pipeline_ratio > 0
    assert result.is_valid is True


def test_startup_calibration_with_measurements():
    """Startup calibration should compute measured ratio when measurements available."""
    result = run_startup_calibration(
        layer_weight_bytes=_gb(0.416),
        num_layers=32,
        total_memory_bytes=_gb(16),
        free_memory_bytes=_gb(12),
        ssd_bandwidth_gbps=6.5,
        measured_read_ms=100.0,
        measured_deserialize_ms=10.0,
        measured_upload_ms=20.0,
        measured_compute_ms=30.0,
    )
    assert result.measured_pipeline_ratio > 0
    assert len(result.reasoning) > 0


# ── build_residency_decision ─────────────────────────────────────────────────

def test_build_residency_decision_has_all_fields():
    """Decision should have all required fields."""
    decision = build_residency_decision(
        total_memory_bytes=_gb(16),
        free_memory_bytes=_gb(12),
        layer_weight_bytes=_gb(0.416),
        num_layers=32,
        pipeline_ratio=5.0,
        ratio_source="measured",
    )
    assert decision.resident_count >= 0
    assert decision.streaming_count >= 0
    assert decision.confidence_score >= 0
    assert decision.confidence_level in ("high", "medium", "low")
    assert len(decision.reasoning) > 0
