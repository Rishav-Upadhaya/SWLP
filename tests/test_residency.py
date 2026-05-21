"""Tests for swlp.core.residency — ResidencyPlan and plan_residency()."""
from __future__ import annotations

from swlp.core.residency import (
    _HEADROOM_FACTOR,
    _OS_RESERVE_BYTES,
    _WORKING_RESERVE_BYTES,
    ResidencyPlan,
    plan_residency,
)

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

def test_model_almost_fits_returns_zero_resident():
    """7B (13.96 GB) on 16 GB: usable budget is only 7.5 GB, so the full model
    does NOT fit → planner returns 0 resident layers (all streaming).

    Partial residency would evict OS page cache for streaming layers, causing
    macOS memory compression and net slowdown.  The full-model-fit guard
    correctly returns 0 here.
    """
    plan = plan_residency(
        total_memory_bytes=_gb(16),
        layer_weight_bytes=_gb(0.436),   # Mistral-7B layer
        num_layers=32,
    )
    # total_model = 13.96 GB > usable = 7.5 GB → no residency.
    assert plan.resident_count == 0, f"expected 0 resident, got {plan.resident_count}"
    assert plan.streaming_count == 32


def test_huge_model_all_streaming():
    """30B on 16 GB: model cannot fit at all → all streaming, zero resident."""
    plan = plan_residency(
        total_memory_bytes=_gb(16),
        layer_weight_bytes=_gb(0.7),   # ~30B rough layer size
        num_layers=60,
    )
    assert plan.resident_count == 0
    assert plan.streaming_count == 60


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
