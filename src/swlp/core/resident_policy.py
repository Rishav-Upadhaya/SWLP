"""Resident-layer policy model.

Maps (pipeline_ratio, free_ram_gb) to a target resident layer count via
bilinear interpolation over calibrated anchors.

Design invariants:
    1. More RAM never produces worse decisions (monotonicity in RAM).
    2. Impossible configurations are rejected (memory safety).
    3. Resident count never exceeds safe limits.
    4. Policy remains deterministic (no random state).
"""

from __future__ import annotations

from dataclasses import dataclass

# ── Calibration grid ────────────────────────────────────────────────────────
# Anchors from Experiment A (294-config sweep) and real Mistral-7B validation.
#
# Pipeline ratio = (SSD read + deserialize + upload) / compute time.
# Higher ratio → more benefit from resident caching.
#
# Grid dimensions:
#   - 6 ratio anchors: 1.0, 3.0, 5.0, 7.0, 10.0, 15.0
#   - 5 RAM anchors: 4.0, 8.0, 12.0, 16.0, 24.0 GB
#
# The grid is intentionally conservative: values represent the best-observed
# resident count for a 32-layer, 416MB/layer model. Different architectures
# will need different grids (this is a v1.0 limitation).

RATIO_ANCHORS: tuple[float, ...] = (1.0, 3.0, 5.0, 7.0, 10.0, 15.0)
RAM_ANCHORS_GB: tuple[float, ...] = (4.0, 8.0, 12.0, 16.0, 24.0)

# Grid[ratio][ram_index] = best-observed resident layers
# Rows correspond to RATIO_ANCHORS, columns to RAM_ANCHORS_GB.
RESIDENT_GRID: dict[float, tuple[float, ...]] = {
    1.0: (2.0, 2.0, 2.0, 2.0, 2.0),    # ratio≈1: GPU saturated, no benefit
    3.0: (2.0, 2.0, 2.0, 2.0, 2.0),    # ratio=3: minimal benefit
    5.0: (4.0, 12.0, 20.0, 22.0, 22.0), # ratio=5: moderate benefit
    7.0: (8.0, 16.0, 22.0, 22.0, 22.0), # ratio=7: good benefit
    10.0: (10.0, 16.0, 22.0, 22.0, 22.0), # ratio=10: strong benefit
    15.0: (12.0, 18.0, 22.0, 22.0, 22.0), # ratio=15: very strong
}

# Maximum resident layers allowed (conservative: leave room for streaming)
MAX_RESIDENT_LAYERS: int = 28
# Minimum resident layers (always beneficial to cache at least 2)
MIN_RESIDENT_LAYERS: int = 2
# Total layers in the reference model (used for safety clamping)
TOTAL_LAYERS: int = 32


@dataclass(slots=True)
class ResidentPolicyResult:
    """Result of the resident policy computation."""
    resident_count: int
    raw_float: float
    memory_clamped: bool
    memory_limit: int
    in_grid_region: bool
    extrapolated: bool


def _lerp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    """Linear interpolation between two points."""
    if x1 <= x0:
        return y0
    t = max(0.0, min(1.0, (x - x0) / (x1 - x0)))
    return y0 + t * (y1 - y0)


def _interp_1d(x: float, anchors: tuple[float, ...], values: tuple[float, ...]) -> float:
    """1D linear interpolation with clamping at boundaries."""
    if x <= anchors[0]:
        return values[0]
    if x >= anchors[-1]:
        return values[-1]
    for idx in range(len(anchors) - 1):
        left = anchors[idx]
        right = anchors[idx + 1]
        if left <= x <= right:
            return _lerp(x, left, right, values[idx], values[idx + 1])
    return values[-1]


def _is_in_grid(pipeline_ratio: float, free_ram_gb: float) -> bool:
    """Check if the input is within the calibrated grid region."""
    return (
        RATIO_ANCHORS[0] <= pipeline_ratio <= RATIO_ANCHORS[-1]
        and RAM_ANCHORS_GB[0] <= free_ram_gb <= RAM_ANCHORS_GB[-1]
    )


def _compute_memory_limit(free_ram_gb: float, layer_size_mb: float = 416.0) -> int:
    """Compute maximum resident layers given available RAM.

    This is a safety check, not the primary decision maker. The calibration
    grid already accounts for memory constraints. This function only prevents
    configurations that would cause OOM (using 80% of available RAM).

    The grid's highest RAM anchor is 24GB. For RAM above that, the grid
    already saturates at 22 layers. For RAM below 4GB, the grid gives
    minimal resident counts. So this limit rarely triggers.
    """
    # Use 80% of available RAM as the safety limit
    usable_gb = free_ram_gb * 0.80
    layer_gb = layer_size_mb / 1024.0
    if layer_gb <= 0:
        return 0
    return max(0, int(usable_gb / layer_gb))


def estimate_optimal_resident_count(
    pipeline_ratio: float,
    free_ram_gb: float,
    layer_size_mb: float = 416.0,
) -> ResidentPolicyResult:
    """Estimate best-observed resident layers via bilinear interpolation.

    Args:
        pipeline_ratio: Measured or estimated pipeline ratio.
        free_ram_gb: Available RAM in GB.
        layer_size_mb: Size of each transformer layer in MB.

    Returns:
        ResidentPolicyResult with resident count and metadata.
    """
    # Check if we're within the calibrated grid
    in_grid = _is_in_grid(pipeline_ratio, free_ram_gb)

    # Clamp to grid boundaries for interpolation
    ratio = max(RATIO_ANCHORS[0], min(RATIO_ANCHORS[-1], pipeline_ratio))
    ram = max(RAM_ANCHORS_GB[0], min(RAM_ANCHORS_GB[-1], free_ram_gb))

    # Bilinear interpolation: first interpolate across RAM for each ratio anchor,
    # then interpolate across ratio.
    per_ratio_values = []
    for ratio_anchor in RATIO_ANCHORS:
        grid_row = RESIDENT_GRID[ratio_anchor]
        per_ratio_values.append(_interp_1d(ram, RAM_ANCHORS_GB, grid_row))

    resident_float = _interp_1d(ratio, RATIO_ANCHORS, tuple(per_ratio_values))

    # Apply memory safety clamp
    memory_limit = _compute_memory_limit(free_ram_gb, layer_size_mb)
    memory_limit = min(memory_limit, MAX_RESIDENT_LAYERS)
    memory_limit = max(MIN_RESIDENT_LAYERS, memory_limit)

    resident = int(round(resident_float))
    memory_clamped = resident > memory_limit
    if memory_clamped:
        resident = memory_limit

    # Ensure within absolute bounds
    resident = max(MIN_RESIDENT_LAYERS, min(MAX_RESIDENT_LAYERS, resident))

    return ResidentPolicyResult(
        resident_count=resident,
        raw_float=resident_float,
        memory_clamped=memory_clamped,
        memory_limit=memory_limit,
        in_grid_region=in_grid,
        extrapolated=not in_grid,
    )
