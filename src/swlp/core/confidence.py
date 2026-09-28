"""Confidence estimation for scheduler policy decisions.

Confidence is based on:
1. Distance from validated calibration anchors
2. Whether ratio was measured vs estimated
3. Layer size availability
4. Hardware region coverage
5. Calibration quality (if available)

Design: intentionally simple and explainable. Every factor is deterministic
and traceable. No black-box models.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .resident_policy import RAM_ANCHORS_GB, RATIO_ANCHORS


@dataclass(slots=True)
class ConfidenceEstimate:
    """Confidence estimate with full reasoning chain."""
    score: float
    level: str
    reasons: list[str]
    factors: dict[str, float] = field(default_factory=dict)


def _normalize_distance(value: float, lo: float, hi: float) -> float:
    """How far outside [lo, hi] is value? Returns 0 if inside, >0 if outside."""
    if hi <= lo:
        return 0.0
    if lo <= value <= hi:
        return 0.0
    span = hi - lo
    if value < lo:
        return (lo - value) / span
    return (value - hi) / span


def estimate_policy_confidence(
    pipeline_ratio: float,
    free_ram_gb: float,
    *,
    measured_ratio: bool,
    has_layer_size: bool,
    calibration_error_pct: float | None = None,
    hardware_generation: str | None = None,
) -> ConfidenceEstimate:
    """Estimate decision confidence for policy interpolation.

    The score is in [0, 1] and decreases outside calibrated regions.

    Factors:
        - Ratio distance from grid: -35% max penalty
        - RAM distance from grid: -35% max penalty
        - Measured vs estimated ratio: +10% / -10%
        - Layer size available: +0% / -20%
        - Calibration quality: -20% max penalty (if available)
        - Hardware coverage: +5% bonus (if known generation)
    """
    reasons: list[str] = []
    factors: dict[str, float] = {}
    score = 1.0

    # Factor 1: Pipeline ratio distance from calibrated region
    ratio_dist = _normalize_distance(pipeline_ratio, RATIO_ANCHORS[0], RATIO_ANCHORS[-1])
    ratio_penalty = min(0.35, 0.35 * ratio_dist)
    score -= ratio_penalty
    factors["ratio_distance"] = -ratio_penalty
    if ratio_dist > 0:
        reasons.append(
            f"pipeline ratio {pipeline_ratio:.1f} is outside validated range "
            f"[{RATIO_ANCHORS[0]:.1f}, {RATIO_ANCHORS[-1]:.1f}]"
        )
    else:
        reasons.append("pipeline ratio lies within calibrated range")

    # Factor 2: RAM distance from calibrated region
    ram_dist = _normalize_distance(free_ram_gb, RAM_ANCHORS_GB[0], RAM_ANCHORS_GB[-1])
    ram_penalty = min(0.35, 0.35 * ram_dist)
    score -= ram_penalty
    factors["ram_distance"] = -ram_penalty
    if ram_dist > 0:
        reasons.append(
            f"free RAM {free_ram_gb:.1f}GB is outside validated range "
            f"[{RAM_ANCHORS_GB[0]:.1f}, {RAM_ANCHORS_GB[-1]:.1f}]GB"
        )
    else:
        reasons.append("free RAM lies within calibrated range")

    # Factor 3: Measured vs estimated ratio
    if measured_ratio:
        reasons.append("using measured pipeline ratio from profiler")
        score += 0.1
        factors["measured_ratio"] = 0.1
    else:
        reasons.append("using estimated pipeline ratio from hardware model")
        score -= 0.1
        factors["estimated_ratio"] = -0.1

    # Factor 4: Layer size availability
    if not has_layer_size:
        reasons.append("layer size metadata unavailable; estimation fallback applied")
        score -= 0.2
        factors["missing_layer_size"] = -0.2
    else:
        factors["has_layer_size"] = 0.0

    # Factor 5: Calibration quality (if available)
    if calibration_error_pct is not None:
        if calibration_error_pct > 50:
            penalty = min(0.20, 0.20 * (calibration_error_pct / 100))
            score -= penalty
            factors["calibration_error"] = -penalty
            reasons.append(f"calibration error high: {calibration_error_pct:.0f}%")
        elif calibration_error_pct > 20:
            penalty = min(0.10, 0.10 * (calibration_error_pct / 50))
            score -= penalty
            factors["calibration_error"] = -penalty
            reasons.append(f"calibration error moderate: {calibration_error_pct:.0f}%")
        else:
            reasons.append(f"calibration error acceptable: {calibration_error_pct:.0f}%")
            factors["calibration_error"] = 0.0

    # Factor 6: Hardware generation coverage
    known_generations = {"M1", "M2", "M3", "M4", "M5", "A100", "H100", "RTX3090", "RTX4090"}
    if hardware_generation and hardware_generation in known_generations:
        score += 0.05
        factors["hardware_coverage"] = 0.05
        reasons.append(f"hardware generation {hardware_generation} is in validated set")
    elif hardware_generation:
        factors["hardware_unknown"] = -0.05
        score -= 0.05
        reasons.append(f"hardware generation {hardware_generation} not in validated set")

    # Clamp and classify
    clamped = max(0.0, min(1.0, score))
    if clamped >= 0.8:
        level = "high"
    elif clamped >= 0.55:
        level = "medium"
    else:
        level = "low"

    return ConfidenceEstimate(
        score=clamped,
        level=level,
        reasons=reasons,
        factors=factors,
    )
