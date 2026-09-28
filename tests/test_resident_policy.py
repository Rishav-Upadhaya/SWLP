"""Tests for swlp.core.resident_policy."""

from __future__ import annotations

from swlp.core.resident_policy import (
    RAM_ANCHORS_GB,
    RATIO_ANCHORS,
    RESIDENT_GRID,
    ResidentPolicyResult,
    estimate_optimal_resident_count,
)


def test_policy_matches_anchor_points() -> None:
    """Verify interpolation matches calibration grid at anchor points."""
    # Ratio=5.0, RAM=8GB → 12 resident layers
    result = estimate_optimal_resident_count(5.0, 8.0)
    assert result.resident_count == 12
    assert result.in_grid_region is True
    assert result.memory_clamped is False

    # Ratio=5.0, RAM=12GB → 20 resident layers
    result = estimate_optimal_resident_count(5.0, 12.0)
    assert result.resident_count == 20

    # Ratio=5.0, RAM=16GB → 22 resident layers
    result = estimate_optimal_resident_count(5.0, 16.0)
    assert result.resident_count == 22


def test_policy_interpolates_between_ram_anchors() -> None:
    """Verify interpolation works between RAM anchors."""
    result = estimate_optimal_resident_count(5.0, 10.0)
    assert result.resident_count == 16


def test_policy_clamps_out_of_range_inputs() -> None:
    """Verify clamping for inputs outside the grid."""
    # Ratio=0.5 is below the grid minimum (1.0)
    result_low = estimate_optimal_resident_count(0.5, 6.0)
    assert result_low.resident_count == 2
    assert result_low.extrapolated is True

    # Ratio=20.0 is above the grid maximum (15.0)
    result_high = estimate_optimal_resident_count(20.0, 64.0)
    assert result_high.resident_count == 22
    assert result_high.extrapolated is True


def test_policy_monotonicity_in_ram() -> None:
    """More RAM should never produce worse decisions."""
    for ratio in [3.0, 5.0, 7.0, 10.0]:
        prev = 0
        for ram in [4.0, 8.0, 12.0, 16.0, 24.0]:
            result = estimate_optimal_resident_count(ratio, ram)
            assert result.resident_count >= prev, (
                f"Non-monotonic at ratio={ratio}, ram={ram}: "
                f"{result.resident_count} < {prev}"
            )
            prev = result.resident_count


def test_policy_monotonicity_in_ratio() -> None:
    """Higher pipeline ratio should never produce fewer resident layers."""
    for ram in [8.0, 12.0, 16.0]:
        prev = 0
        for ratio in [1.0, 3.0, 5.0, 7.0, 10.0, 15.0]:
            result = estimate_optimal_resident_count(ratio, ram)
            assert result.resident_count >= prev, (
                f"Non-monotonic at ram={ram}, ratio={ratio}: "
                f"{result.resident_count} < {prev}"
            )
            prev = result.resident_count


def test_policy_memory_safety() -> None:
    """Resident count should never exceed safe limits."""
    result = estimate_optimal_resident_count(15.0, 4.0, layer_size_mb=416.0)
    # With only 4GB free, memory limit is very low
    assert result.resident_count <= result.memory_limit
    assert result.memory_clamped is True


def test_policy_returns_metadata() -> None:
    """Verify all metadata fields are populated."""
    result = estimate_optimal_resident_count(5.0, 12.0)
    assert isinstance(result, ResidentPolicyResult)
    assert isinstance(result.raw_float, float)
    assert isinstance(result.memory_limit, int)
    assert isinstance(result.in_grid_region, bool)
    assert isinstance(result.extrapolated, bool)


def test_grid_dimensions_match() -> None:
    """Verify grid dimensions are consistent."""
    assert len(RESIDENT_GRID) == len(RATIO_ANCHORS)
    for ratio in RATIO_ANCHORS:
        assert ratio in RESIDENT_GRID
        assert len(RESIDENT_GRID[ratio]) == len(RAM_ANCHORS_GB)
