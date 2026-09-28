"""Adaptive residency planning for SWLP.

Given available memory and per-layer weight size, computes how many transformer
blocks can be kept permanently resident (loaded once, never evicted) vs. streamed
from disk on every token step.

Architecture:
    Hardware Probe → Pipeline Model → Resident Policy → Residency Planner → Streaming Scheduler

This module implements the Residency Planner layer, which consumes the pipeline
model and resident policy to produce an explainable residency decision.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .confidence import ConfidenceEstimate, estimate_policy_confidence
from .pipeline_model import estimate_pipeline_ratio_from_hardware
from .resident_policy import (
    ResidentPolicyResult,
    estimate_optimal_resident_count,
)

# Bytes reserved for the OS, background processes, and Metal driver overhead.
# On Apple Silicon unified memory, MPS tensors are locked in GPU memory and
# cannot be reclaimed by macOS under pressure — so the true OS reserve needs
# to be generous (4 GB covers macOS kernel + services + driver overhead).
_OS_RESERVE_BYTES: int = 4 * 1024 * 1024 * 1024  # 4 GB
# Bytes reserved for embeddings, final norm, lm_head, KV cache, activations,
# streaming layer slots (1-2 × layer_bytes), and PyTorch MPS internal buffers.
_WORKING_RESERVE_BYTES: int = 2 * 1024 * 1024 * 1024  # 2 GB
# Use 75 % of remaining headroom for resident layers to avoid Metal memory
# pressure (macOS compressor activates when free pages drop near zero).
_HEADROOM_FACTOR: float = 0.75


@dataclass(slots=True)
class ResidencyPlan:
    """Pure computation result — no I/O, no torch."""
    resident_count: int    # layers permanently loaded at startup
    streaming_count: int   # layers loaded/evicted per token
    resident_bytes: int    # total bytes locked in RAM
    streaming_bytes: int   # bytes streamed from disk per token
    available_bytes: int   # usable bytes after reserves


@dataclass(slots=True)
class ResidencyDecision:
    """Explainable scheduling decision with full provenance."""
    resident_count: int
    streaming_count: int
    resident_bytes: int
    streaming_bytes: int
    available_bytes: int
    memory_budget_bytes: int
    pipeline_ratio: float
    ratio_source: str
    estimated_resident_count: int
    memory_clamp_applied: bool
    confidence_score: float
    confidence_level: str
    reasoning: list[str]
    # New fields for v1.0
    policy_result: ResidentPolicyResult | None = None
    calibration_timestamp: float | None = None
    hardware_fingerprint: str | None = None


@dataclass(slots=True)
class CalibrationResult:
    """Result of a startup calibration probe."""
    measured_pipeline_ratio: float
    measured_read_ms: float
    measured_deserialize_ms: float
    measured_upload_ms: float
    measured_compute_ms: float
    estimated_pipeline_ratio: float
    ratio_error_pct: float
    calibration_timestamp: float
    hardware_fingerprint: str
    is_valid: bool
    reasoning: list[str]


def estimate_pipeline_ratio(
    layer_weight_bytes: int,
    ssd_bandwidth_gbps: float,
    compute_ms: float = 100.0,
    deserialize_upload_ms: float = 52.0,
) -> float:
    """Backward-compatible wrapper for startup pipeline-ratio estimation."""
    return estimate_pipeline_ratio_from_hardware(
        layer_weight_bytes=layer_weight_bytes,
        ssd_bandwidth_gbps=ssd_bandwidth_gbps,
        compute_ms=compute_ms,
        deserialize_upload_ms=deserialize_upload_ms,
    )


def plan_residency(
    total_memory_bytes: int,
    layer_weight_bytes: int,
    num_layers: int,
    *,
    free_memory_bytes: int | None = None,
    pipeline_ratio: float | None = None,
    layer_size_mb: float = 416.0,
) -> ResidencyPlan:
    """Compute how many layers stay resident vs. stream.

    Args:
        total_memory_bytes: Total system/unified RAM in bytes.
        layer_weight_bytes: Size of one transformer block on disk (bytes).
        num_layers: Total number of transformer blocks in the model.
        free_memory_bytes: Available RAM in bytes (if known).
        pipeline_ratio: Measured or estimated pipeline ratio.
        layer_size_mb: Size of each layer in MB for policy lookup.

    Returns:
        ResidencyPlan with resident_count and streaming_count.
        If the model fits entirely, resident_count == num_layers and
        streaming_count == 0.
    """
    memory_budget = (
        min(total_memory_bytes, free_memory_bytes)
        if free_memory_bytes is not None
        else total_memory_bytes
    )
    headroom = memory_budget - _OS_RESERVE_BYTES - _WORKING_RESERVE_BYTES
    usable = max(0, int(headroom * _HEADROOM_FACTOR))

    if layer_weight_bytes <= 0 or num_layers <= 0:
        return ResidencyPlan(
            resident_count=0,
            streaming_count=num_layers,
            resident_bytes=0,
            streaming_bytes=0,
            available_bytes=usable,
        )

    total_model_bytes = layer_weight_bytes * num_layers

    max_resident_by_memory = min(num_layers, usable // layer_weight_bytes)
    if max_resident_by_memory <= 0:
        return ResidencyPlan(
            resident_count=0,
            streaming_count=num_layers,
            resident_bytes=0,
            streaming_bytes=total_model_bytes,
            available_bytes=usable,
        )

    if total_model_bytes <= usable:
        resident_count = num_layers
    else:
        ratio = 5.0 if pipeline_ratio is None else max(0.1, float(pipeline_ratio))
        free_ram_gb = memory_budget / (1024.0 ** 3)
        policy_result = estimate_optimal_resident_count(ratio, free_ram_gb, layer_size_mb)
        resident_count = max(0, min(max_resident_by_memory, policy_result.resident_count))

    streaming_count = num_layers - resident_count
    return ResidencyPlan(
        resident_count=resident_count,
        streaming_count=streaming_count,
        resident_bytes=resident_count * layer_weight_bytes,
        streaming_bytes=streaming_count * layer_weight_bytes,
        available_bytes=usable,
    )


def run_startup_calibration(
    *,
    layer_weight_bytes: int,
    num_layers: int,
    total_memory_bytes: int,
    free_memory_bytes: int,
    ssd_bandwidth_gbps: float | None = None,
    compute_ms: float | None = None,
    measured_read_ms: float | None = None,
    measured_deserialize_ms: float | None = None,
    measured_upload_ms: float | None = None,
    measured_compute_ms: float | None = None,
    hardware_fingerprint: str = "unknown",
) -> CalibrationResult:
    """Run startup calibration: estimate pipeline ratio, compare to measurements.

    This is the "Hardware Probe → Pipeline Model" step of the architecture.
    It estimates the pipeline ratio from hardware specs, then optionally compares
    to measured values from a calibration run.

    Args:
        layer_weight_bytes: Size of one transformer layer on disk.
        num_layers: Total number of layers.
        total_memory_bytes: Total system RAM.
        free_memory_bytes: Available RAM.
        ssd_bandwidth_gbps: Estimated SSD bandwidth (if known).
        compute_ms: Estimated compute time per layer (if known).
        measured_*: Measured timings from a calibration run (if available).

    Returns:
        CalibrationResult with estimated and measured pipeline ratios.
    """
    reasoning: list[str] = []
    timestamp = time.time()

    # Step 1: Estimate pipeline ratio from hardware specs
    estimated_ratio = 5.0  # default fallback
    if ssd_bandwidth_gbps is not None and layer_weight_bytes > 0:
        estimated_ratio = estimate_pipeline_ratio(
            layer_weight_bytes=layer_weight_bytes,
            ssd_bandwidth_gbps=ssd_bandwidth_gbps,
            compute_ms=compute_ms or 100.0,
            deserialize_upload_ms=52.0,  # typical deserialization + upload
        )
        reasoning.append(f"estimated pipeline ratio from hardware: {estimated_ratio:.2f}")
    else:
        reasoning.append("using default pipeline ratio (no hardware specs available)")

    # Step 2: Compute measured pipeline ratio if measurements are available
    measured_ratio = estimated_ratio
    ratio_error = 0.0
    is_valid = True

    if all(v is not None for v in [measured_read_ms, measured_deserialize_ms,
                                     measured_upload_ms, measured_compute_ms]):
        if measured_compute_ms and measured_compute_ms > 0:
            transfer_ms = measured_read_ms + measured_deserialize_ms + measured_upload_ms
            measured_ratio = transfer_ms / measured_compute_ms
            ratio_error = abs(measured_ratio - estimated_ratio) / max(0.01, estimated_ratio) * 100
            reasoning.append(f"measured pipeline ratio: {measured_ratio:.2f}")
            reasoning.append(f"estimation error: {ratio_error:.1f}%")

            # If error is large, the estimated ratio was wrong
            if ratio_error > 50:
                reasoning.append("large estimation error — using measured ratio")
            else:
                reasoning.append("estimation within acceptable range")
        else:
            reasoning.append("compute_ms is zero — cannot compute measured ratio")
            is_valid = False
    else:
        reasoning.append("no measured timings available — using estimated ratio")

    return CalibrationResult(
        measured_pipeline_ratio=measured_ratio,
        measured_read_ms=measured_read_ms or 0.0,
        measured_deserialize_ms=measured_deserialize_ms or 0.0,
        measured_upload_ms=measured_upload_ms or 0.0,
        measured_compute_ms=measured_compute_ms or 0.0,
        estimated_pipeline_ratio=estimated_ratio,
        ratio_error_pct=ratio_error,
        calibration_timestamp=timestamp,
        hardware_fingerprint=hardware_fingerprint,
        is_valid=is_valid,
        reasoning=reasoning,
    )


def build_residency_decision(
    *,
    total_memory_bytes: int,
    free_memory_bytes: int,
    layer_weight_bytes: int,
    num_layers: int,
    pipeline_ratio: float,
    ratio_source: str,
    layer_size_mb: float = 416.0,
    calibration_result: CalibrationResult | None = None,
) -> ResidencyDecision:
    """Build an explainable scheduling decision around ``plan_residency``.

    This is the "Resident Policy → Residency Planner" step of the architecture.
    It combines the policy model output with memory safety constraints and
    confidence estimation to produce a fully explainable decision.
    """
    memory_budget = min(total_memory_bytes, free_memory_bytes)
    headroom = memory_budget - _OS_RESERVE_BYTES - _WORKING_RESERVE_BYTES
    usable = max(0, int(headroom * _HEADROOM_FACTOR))
    max_resident_by_memory = (
        min(num_layers, usable // layer_weight_bytes)
        if layer_weight_bytes > 0 and num_layers > 0
        else 0
    )

    # Get policy result
    policy_result = None
    estimated_resident = 0
    if layer_weight_bytes > 0 and num_layers > 0:
        free_ram_gb = memory_budget / (1024.0 ** 3)
        policy_result = estimate_optimal_resident_count(
            pipeline_ratio, free_ram_gb, layer_size_mb
        )
        estimated_resident = policy_result.resident_count

    plan = plan_residency(
        total_memory_bytes=total_memory_bytes,
        free_memory_bytes=free_memory_bytes,
        layer_weight_bytes=layer_weight_bytes,
        num_layers=num_layers,
        pipeline_ratio=pipeline_ratio,
        layer_size_mb=layer_size_mb,
    )

    confidence: ConfidenceEstimate = estimate_policy_confidence(
        pipeline_ratio=pipeline_ratio,
        free_ram_gb=memory_budget / (1024.0 ** 3),
        measured_ratio=(ratio_source == "measured"),
        has_layer_size=(layer_weight_bytes > 0),
    )

    # Build reasoning chain
    reasoning = []
    if calibration_result:
        reasoning.extend(calibration_result.reasoning)
    reasoning.append(f"policy interpolated resident target: {estimated_resident}")
    reasoning.append(f"memory-safe resident limit: {max_resident_by_memory}")
    if policy_result and policy_result.memory_clamped:
        reasoning.append(
            f"memory clamp applied: {estimated_resident} → {policy_result.resident_count}"
        )
    if policy_result and policy_result.extrapolated:
        reasoning.append("input outside calibrated grid — extrapolation applied")
    reasoning.extend(confidence.reasons)

    # Determine ratio source with calibration context
    effective_source = ratio_source
    if calibration_result and calibration_result.ratio_error_pct > 50:
        effective_source = "measured (calibration override)"

    return ResidencyDecision(
        resident_count=plan.resident_count,
        streaming_count=plan.streaming_count,
        resident_bytes=plan.resident_bytes,
        streaming_bytes=plan.streaming_bytes,
        available_bytes=plan.available_bytes,
        memory_budget_bytes=memory_budget,
        pipeline_ratio=pipeline_ratio,
        ratio_source=effective_source,
        estimated_resident_count=estimated_resident,
        memory_clamp_applied=(plan.resident_count != estimated_resident),
        confidence_score=confidence.score,
        confidence_level=confidence.level,
        reasoning=reasoning,
        policy_result=policy_result,
        calibration_timestamp=(
            calibration_result.calibration_timestamp if calibration_result else None
        ),
        hardware_fingerprint=(
            calibration_result.hardware_fingerprint if calibration_result else None
        ),
    )
