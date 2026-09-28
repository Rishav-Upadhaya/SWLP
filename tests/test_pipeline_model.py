"""Tests for swlp.core.pipeline_model."""

from __future__ import annotations

from swlp.core.pipeline_model import (
    estimate_pipeline_ratio_from_hardware,
    pipeline_ratio_from_metrics,
)


def _gb(n: float) -> int:
    return int(n * 1024 ** 3)


def test_estimated_ratio_increases_with_larger_layer() -> None:
    small = estimate_pipeline_ratio_from_hardware(
        layer_weight_bytes=_gb(0.1),
        ssd_bandwidth_gbps=6.5,
    )
    large = estimate_pipeline_ratio_from_hardware(
        layer_weight_bytes=_gb(0.5),
        ssd_bandwidth_gbps=6.5,
    )
    assert large > small


def test_estimated_ratio_handles_invalid_inputs() -> None:
    assert estimate_pipeline_ratio_from_hardware(0, 6.5) == 1.0
    assert estimate_pipeline_ratio_from_hardware(_gb(0.4), 0.0) == 1.0


def test_ratio_from_metrics_matches_formula() -> None:
    ratio = pipeline_ratio_from_metrics(
        avg_read_ms=10.0,
        avg_deserialize_ms=20.0,
        avg_upload_ms=5.0,
        avg_compute_ms=7.0,
    )
    assert ratio == 5.0


def test_ratio_from_metrics_clamps_compute_zero() -> None:
    ratio = pipeline_ratio_from_metrics(
        avg_read_ms=10.0,
        avg_deserialize_ms=20.0,
        avg_upload_ms=5.0,
        avg_compute_ms=0.0,
    )
    assert ratio == 1.0
