"""Pipeline-ratio modeling for startup calibration.

This module isolates how SWLP estimates and measures pipeline ratio:

    pipeline_ratio = (read + deserialize + upload) / compute

The resident policy consumes this ratio but does not compute it.
"""

from __future__ import annotations


def estimate_pipeline_ratio_from_hardware(
    layer_weight_bytes: int,
    ssd_bandwidth_gbps: float,
    compute_ms: float = 100.0,
    deserialize_upload_ms: float = 52.0,
) -> float:
    """Estimate pipeline ratio from layer size and SSD bandwidth.

    Used at startup when no measured traces are available yet.
    """
    if layer_weight_bytes <= 0 or ssd_bandwidth_gbps <= 0 or compute_ms <= 0:
        return 1.0
    bandwidth_mb_per_s = ssd_bandwidth_gbps * 1024.0
    if bandwidth_mb_per_s <= 0:
        return 1.0
    layer_mb = layer_weight_bytes / (1024.0 * 1024.0)
    read_ms = (layer_mb / bandwidth_mb_per_s) * 1000.0
    return max(0.1, (read_ms + deserialize_upload_ms) / compute_ms)


def pipeline_ratio_from_metrics(
    avg_read_ms: float,
    avg_deserialize_ms: float,
    avg_upload_ms: float,
    avg_compute_ms: float,
) -> float:
    """Compute pipeline ratio directly from measured profiler metrics."""
    if avg_compute_ms <= 0:
        return 1.0
    transfer_ms = max(0.0, avg_read_ms) + max(0.0, avg_deserialize_ms) + max(0.0, avg_upload_ms)
    return max(0.1, transfer_ms / avg_compute_ms)
