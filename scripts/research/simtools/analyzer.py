"""Post-run trace analyzer for SWLP profiler data.

Reads the JSON trace output from LayerProfiler.dump() and produces a
human-readable analysis report with observation→diagnosis→recommendation
chains backed by measurable evidence.

Usage:
    from scripts.research.simtools.analyzer import analyze_traces
    report = analyze_traces("layer_traces.json")
    print(report.format())
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Data model ─────────────────────────────────────────────────────────────


@dataclass
class Evidence:
    """A single measured fact supporting a diagnosis."""

    label: str  # e.g. "Upload avg"
    value: str  # e.g. "37.2 ms"
    unit: str = ""  # e.g. "ms", "%", "GB"


@dataclass
class Diagnosis:
    """Observation → reasoning → conclusion chain."""

    observation: str  # what was measured
    evidence: list[Evidence] = field(default_factory=list)
    diagnosis: str = ""  # why it happened
    severity: str = "info"  # "info" | "warning" | "critical"
    confidence: float = 0.0  # 0.0 - 1.0, how sure we are


@dataclass
class Recommendation:
    """An actionable recommendation derived from a diagnosis."""

    priority: int
    action: str
    reason: str
    expected_impact: str
    category: str  # "config" | "code" | "hardware" | "experiment"
    is_prediction: bool = False  # True = extrapolated, not measured
    confidence: float = 0.0


@dataclass
class AnalysisReport:
    """Complete analysis output."""

    diagnoses: list[Diagnosis] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
    summary: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    pipeline_occupancy: dict[str, float] = field(default_factory=dict)

    def format(self) -> str:
        lines: list[str] = []
        lines.append("=" * 72)
        lines.append("SWLP ANALYSIS REPORT")
        lines.append("=" * 72)

        # Hardware context
        hw = self.metadata.get("hardware", {})
        if hw:
            lines.append("")
            lines.append("  HARDWARE")
            lines.append(f"    Device:     {hw.get('device_type', '?')} {hw.get('chip_name', '')}")
            lines.append(f"    RAM:        {hw.get('memory_gb', 0):.0f} GB")
            lines.append(f"    Model:      {hw.get('model_id', '?')}")
            lines.append(f"    Layers:     {hw.get('num_layers', '?')}")
            lines.append(f"    Window:     {hw.get('window_size', '?')}")
            lines.append(f"    Prefetch:   {hw.get('prefetch_depth', '?')}")

        # Pipeline summary
        if self.summary:
            lines.append("")
            lines.append("-" * 72)
            lines.append("PIPELINE SUMMARY")
            lines.append("-" * 72)
            for key, val in self.summary.items():
                lines.append(f"  {key:.<34} {val}")

        # Pipeline occupancy
        if self.pipeline_occupancy:
            lines.append("")
            lines.append("-" * 72)
            lines.append("PIPELINE OCCUPANCY")
            lines.append("-" * 72)
            for component, pct in self.pipeline_occupancy.items():
                bar_len = int(pct / 5)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                lines.append(f"  {component:.<20} {bar} {pct:.1f}%")

        # Diagnoses
        if self.diagnoses:
            lines.append("")
            lines.append("-" * 72)
            lines.append("DIAGNOSES")
            lines.append("-" * 72)
            for i, d in enumerate(self.diagnoses, 1):
                sev_icon = {"critical": "!!!", "warning": " ! ", "info": " i "}.get(
                    d.severity, "   "
                )
                lines.append("")
                lines.append(f"  [{sev_icon}] Diagnosis #{i}: {d.observation}")
                if d.evidence:
                    lines.append("       Evidence:")
                    for e in d.evidence:
                        lines.append(
                            f"         - {e.label}: {e.value}{(' ' + e.unit) if e.unit else ''}"
                        )
                if d.diagnosis:
                    lines.append(f"       Diagnosis:  {d.diagnosis}")
                lines.append(f"       Confidence: {d.confidence:.0%}")

        # Recommendations
        if self.recommendations:
            lines.append("")
            lines.append("-" * 72)
            lines.append("RECOMMENDATIONS")
            lines.append("-" * 72)
            for r in sorted(self.recommendations, key=lambda x: x.priority):
                pred_tag = " [prediction]" if r.is_prediction else ""
                lines.append("")
                lines.append(f"  #{r.priority} [{r.category}]{pred_tag} {r.action}")
                lines.append(f"       Why:             {r.reason}")
                lines.append(f"       Expected impact: {r.expected_impact}")
                lines.append(f"       Confidence:      {r.confidence:.0%}")

        lines.append("")
        lines.append("=" * 72)
        return "\n".join(lines)


# ── Helpers ────────────────────────────────────────────────────────────────


def _f(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _i(val: Any, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _sorted_traces(traces: list[dict]) -> list[dict]:
    return sorted(
        [t for t in traces if t.get("compute_start", 0) > 0],
        key=lambda t: t["compute_start"],
    )


# ── GPU analysis ──────────────────────────────────────────────────────────


def _analyze_gpu_idle(traces: list[dict], metrics: dict) -> tuple[Diagnosis, float]:
    gpu_busy = _f(metrics.get("gpu_busy_time_ms"))
    gpu_idle = _f(metrics.get("gpu_idle_time_ms"))
    total = gpu_busy + gpu_idle
    gpu_eff = _f(metrics.get("gpu_efficiency"))
    idle_pct = (1.0 - gpu_eff) * 100 if total > 0 else 0.0

    evidence = [
        Evidence("GPU busy", f"{gpu_busy:.0f}", "ms"),
        Evidence("GPU idle", f"{gpu_idle:.0f}", "ms"),
        Evidence("GPU efficiency", f"{gpu_eff * 100:.1f}", "%"),
    ]

    if total <= 0:
        return Diagnosis(
            observation="GPU utilization",
            evidence=evidence,
            diagnosis="No compute data recorded.",
            severity="info",
            confidence=0.5,
        ), 0.0

    # Classify idle gaps
    sorted_t = _sorted_traces(traces)
    idle_waiting_data = 0.0
    idle_scheduling = 0.0

    for i in range(1, len(sorted_t)):
        prev_end = _f(sorted_t[i - 1].get("compute_end"))
        curr_start = _f(sorted_t[i].get("compute_start"))
        gap_ms = (curr_start - prev_end) * 1000
        if gap_ms <= 0:
            continue

        overlap = sorted_t[i].get("overlap_status", "")
        ensure_wait = _f(sorted_t[i].get("ensure_wait"))

        if overlap in ("miss", "wait") or ensure_wait > 0.001:
            idle_waiting_data += gap_ms
        else:
            idle_scheduling += gap_ms

    evidence.append(Evidence("Idle: waiting for data", f"{idle_waiting_data:.0f}", "ms"))
    evidence.append(Evidence("Idle: scheduling gaps", f"{idle_scheduling:.0f}", "ms"))

    # Determine dominant reason
    if idle_waiting_data > idle_scheduling and idle_waiting_data > 5:
        dominant = "waiting for data (SSD read or upload not finished)"
        diag_confidence = 0.85
    elif idle_scheduling > 5:
        dominant = "scheduling gaps between layers"
        diag_confidence = 0.7
    else:
        dominant = "negligible idle time"
        diag_confidence = 0.9

    severity = "critical" if idle_pct > 30 else "warning" if idle_pct > 15 else "info"

    return Diagnosis(
        observation=f"GPU is idle {idle_pct:.1f}% of pipeline time",
        evidence=evidence,
        diagnosis=f"GPU idle primarily caused by {dominant}.",
        severity=severity,
        confidence=diag_confidence,
    ), idle_pct


def _analyze_prefetch(traces: list[dict], metrics: dict) -> Diagnosis:
    hits = _i(metrics.get("prefetch_hits"))
    waits = _i(metrics.get("prefetch_waits"))
    misses = _i(metrics.get("prefetch_misses"))
    total = hits + waits + misses
    hit_rate = hits / total if total > 0 else 0.0

    evidence = [
        Evidence("Prefetch hits", str(hits)),
        Evidence("Prefetch waits", str(waits)),
        Evidence("Prefetch misses", str(misses)),
        Evidence("Hit rate", f"{hit_rate * 100:.1f}", "%"),
    ]

    avg_read = _f(metrics.get("avg_read_ms"))
    avg_upload = _f(metrics.get("avg_upload_ms"))

    if misses > 0:
        evidence.append(
            Evidence("Sync fallback cost", f"~{avg_read + avg_upload:.0f}", "ms per miss")
        )

    if hit_rate < 0.5:
        severity = "critical"
        diag = f"Prefetch is largely ineffective — {misses} of {total} layers required sync reads."
        conf = 0.9
    elif hit_rate < 0.8:
        severity = "warning"
        diag = (
            f"Prefetch works for most layers but {misses} misses "
            f"and {waits} partial overlaps remain."
        )
        conf = 0.85
    else:
        severity = "info"
        diag = (
            f"Prefetch is effective — {hits}/{total} layers were ready before compute needed them."
        )
        conf = 0.95

    return Diagnosis(
        observation=f"Prefetch hit rate is {hit_rate * 100:.1f}%",
        evidence=evidence,
        diagnosis=diag,
        severity=severity,
        confidence=conf,
    )


def _analyze_bottleneck(traces: list[dict], metrics: dict) -> Diagnosis:
    stages = {
        "SSD Read": _f(metrics.get("avg_read_ms")),
        "Deserialize": _f(metrics.get("avg_deserialize_ms")),
        "Upload to MPS": _f(metrics.get("avg_upload_ms")),
        "Compute": _f(metrics.get("avg_compute_ms")),
        "ensure() Wait": _f(metrics.get("avg_ensure_wait_ms")),
    }

    evidence = [Evidence(k, f"{v:.1f}", "ms") for k, v in stages.items()]

    non_compute = {k: v for k, v in stages.items() if k != "Compute"}
    total_non_compute = sum(non_compute.values())
    compute_ms = stages["Compute"]

    if total_non_compute <= 0:
        return Diagnosis(
            observation="Pipeline bottleneck",
            evidence=evidence,
            diagnosis="No non-compute time recorded.",
            severity="info",
            confidence=0.5,
        )

    dominant = max(non_compute, key=non_compute.get)
    dominant_ms = non_compute[dominant]
    ratio = dominant_ms / compute_ms if compute_ms > 0 else 0

    evidence.append(Evidence("Dominant non-compute", f"{dominant}={dominant_ms:.1f}", "ms"))
    evidence.append(Evidence("Ratio to compute", f"{ratio:.2f}", "x"))

    # Confidence based on how clearly dominant the stage is
    sorted_non_compute = sorted(non_compute.values(), reverse=True)
    if len(sorted_non_compute) >= 2 and sorted_non_compute[1] > 0:
        margin = (sorted_non_compute[0] - sorted_non_compute[1]) / sorted_non_compute[0]
        conf = min(0.95, 0.6 + margin * 0.5)
    else:
        conf = 0.7

    if ratio > 0.5:
        severity = "warning"
        diag = (
            f"{dominant} ({dominant_ms:.1f}ms) is {ratio:.1f}x compute time "
            f"— primary throughput limiter."
        )
    else:
        severity = "info"
        diag = (
            f"Compute ({compute_ms:.1f}ms) dominates; "
            f"{dominant} ({dominant_ms:.1f}ms) is secondary."
        )

    return Diagnosis(
        observation=f"Primary bottleneck: {dominant} at {dominant_ms:.1f}ms avg",
        evidence=evidence,
        diagnosis=diag,
        severity=severity,
        confidence=conf,
    )


def _analyze_memory(traces: list[dict], metrics: dict, hw: dict) -> list[Diagnosis]:
    diags: list[Diagnosis] = []

    peak_rss = _i(metrics.get("peak_rss_bytes"))
    total_ram = _f(hw.get("memory_gb", 0)) * 1e9
    if total_ram <= 0:
        return diags

    peak_pct = peak_rss / total_ram * 100
    headroom = total_ram - peak_rss

    evidence = [
        Evidence("Peak RSS", f"{peak_rss / 1e9:.2f}", "GB"),
        Evidence("Total RAM", f"{total_ram / 1e9:.0f}", "GB"),
        Evidence("Headroom", f"{headroom / 1e9:.2f}", "GB"),
    ]

    diags.append(
        Diagnosis(
            observation=f"Peak memory usage is {peak_pct:.0f}% of available RAM",
            evidence=evidence,
            diagnosis=(
                f"Peak RSS is {peak_rss / 1e9:.2f} GB with {headroom / 1e9:.2f} GB headroom. "
                f"macOS typically reserves 3-4 GB for kernel and services."
            ),
            severity="warning" if peak_pct > 85 else "info",
            confidence=0.9,
        )
    )

    # Resident cache estimate
    num_layers = _i(hw.get("num_layers"))
    if num_layers > 0 and headroom > 0:
        # Use shard manifest layer size if available, else estimate from peak
        layer_size_bytes = _f(hw.get("layer_size_bytes"))
        if layer_size_bytes <= 0:
            # Rough: assume streaming window of 2-4 layers in peak RSS
            window = _i(hw.get("window_size")) or 2
            layer_size_bytes = peak_rss / max(window + 1, 2)

        if layer_size_bytes > 0:
            available_for_resident = max(0, headroom - 2e9)  # reserve 2 GB
            max_resident = int(available_for_resident / layer_size_bytes)

            if max_resident > 0:
                est = [
                    Evidence("Est. layer size", f"{layer_size_bytes / 1e6:.0f}", "MB"),
                    Evidence("Available for resident", f"{available_for_resident / 1e9:.1f}", "GB"),
                    Evidence("Max resident layers", str(max_resident)),
                ]
                diags.append(
                    Diagnosis(
                        observation=f"~{max_resident} layers could fit in resident cache",
                        evidence=est,
                        diagnosis=(
                            f"With {available_for_resident / 1e9:.1f} GB available and "
                            f"~{layer_size_bytes / 1e6:.0f} MB per layer, up to {max_resident} "
                            f"layers could stay resident without memory pressure."
                        ),
                        severity="info",
                        confidence=0.65,  # estimate, not measured
                    )
                )

    return diags


def _analyze_ensure_stalls(traces: list[dict], metrics: dict) -> Diagnosis:
    max_wait = _f(metrics.get("max_ensure_wait_ms"))
    avg_wait = _f(metrics.get("avg_ensure_wait_ms"))
    total_wait = _f(metrics.get("total_ensure_wait_ms"))
    stall_count = _i(metrics.get("pipeline_stall_count"))
    num_layers = _i(metrics.get("num_layers")) or len(traces)

    evidence = [
        Evidence("Max ensure() wait", f"{max_wait:.1f}", "ms"),
        Evidence("Avg ensure() wait", f"{avg_wait:.1f}", "ms"),
        Evidence("Total ensure() wait", f"{total_wait:.1f}", "ms"),
        Evidence("Sync fallbacks", str(stall_count)),
    ]

    if num_layers > 0:
        stall_pct = stall_count / num_layers * 100
        evidence.append(Evidence("Stall rate", f"{stall_pct:.0f}", "%"))

    if max_wait > 20:
        severity = "warning"
        diag = f"Significant ensure() stalls — longest wait was {max_wait:.1f}ms."
    elif max_wait > 5:
        severity = "info"
        diag = f"Mild ensure() stalls — max {max_wait:.1f}ms, avg {avg_wait:.1f}ms."
    else:
        severity = "info"
        diag = "ensure() stalls are negligible."

    return Diagnosis(
        observation=f"ensure() stalls: max={max_wait:.1f}ms, total={total_wait:.1f}ms",
        evidence=evidence,
        diagnosis=diag,
        severity=severity,
        confidence=0.9,
    )


def _analyze_queue_saturation(
    metrics: dict, queue_snapshots: list[dict] | None
) -> Diagnosis | None:
    """Analyze queue saturation from queue snapshots if available."""
    if not queue_snapshots:
        return None

    read_depths = [_i(s.get("read_queue_depth")) for s in queue_snapshots]
    upload_depths = [_i(s.get("upload_queue_depth")) for s in queue_snapshots]

    if not read_depths:
        return None

    avg_read = sum(read_depths) / len(read_depths)
    max_read = max(read_depths)
    avg_upload = sum(upload_depths) / len(upload_depths)
    max_upload = max(upload_depths)

    evidence = [
        Evidence("SSD queue avg depth", f"{avg_read:.1f}"),
        Evidence("SSD queue max depth", str(max_read)),
        Evidence("Upload queue avg depth", f"{avg_upload:.1f}"),
        Evidence("Upload queue max depth", str(max_upload)),
    ]

    read_saturated = max_read >= 4  # typical worker count
    upload_saturated = max_upload >= 4

    if read_saturated and upload_saturated:
        diag = "Both SSD and upload queues saturated — workers are the bottleneck."
        severity = "warning"
        conf = 0.8
    elif read_saturated:
        diag = "SSD queue saturated — reads are the bottleneck."
        severity = "warning"
        conf = 0.75
    elif upload_saturated:
        diag = "Upload queue saturated — CPU→MPS copies are the bottleneck."
        severity = "warning"
        conf = 0.75
    elif avg_read < 0.5 and avg_upload < 0.5:
        diag = "Queues mostly empty — prefetch depth may be excessive."
        severity = "info"
        conf = 0.7
    else:
        diag = "Queue depths moderate — no saturation detected."
        severity = "info"
        conf = 0.65

    return Diagnosis(
        observation=(
            f"SSD queue avg={avg_read:.1f} max={max_read}, "
            f"Upload queue avg={avg_upload:.1f} max={max_upload}"
        ),
        evidence=evidence,
        diagnosis=diag,
        severity=severity,
        confidence=conf,
    )


# ── Pipeline occupancy ────────────────────────────────────────────────────


def _compute_occupancy(traces: list[dict], metrics: dict) -> dict[str, float]:
    """Estimate per-component utilization as percentages."""
    gpu_eff = _f(metrics.get("gpu_efficiency")) * 100

    # SSD utilization: time SSD was doing reads / total wall time
    read_times = [_f(t.get("read_end")) - _f(t.get("read_start")) for t in traces]
    read_times = [r for r in read_times if r > 0]
    total_read = sum(read_times)
    wall = _f(metrics.get("wall_time_ms")) / 1000 if _f(metrics.get("wall_time_ms")) > 0 else 1.0
    ssd_util = min(100, (total_read / wall) * 100) if wall > 0 else 0

    # Upload utilization
    upload_times = [_f(t.get("upload_end")) - _f(t.get("upload_start")) for t in traces]
    upload_times = [u for u in upload_times if u > 0]
    total_upload = sum(upload_times)
    upload_util = min(100, (total_upload / wall) * 100) if wall > 0 else 0

    # CPU utilization (deserialize as proxy)
    deser_times = [_f(t.get("deserialize_end")) - _f(t.get("deserialize_start")) for t in traces]
    deser_times = [d for d in deser_times if d > 0]
    total_deser = sum(deser_times)
    cpu_util = min(100, (total_deser / wall) * 100) if wall > 0 else 0

    return {
        "GPU": gpu_eff,
        "SSD": ssd_util,
        "Upload": upload_util,
        "CPU (deserialize)": cpu_util,
    }


# ── Recommendation generation ─────────────────────────────────────────────


def _generate_recommendations(
    diagnoses: list[Diagnosis],
    metrics: dict,
    hw: dict,
) -> list[Recommendation]:
    recs: list[Recommendation] = []
    priority = 1

    for d in diagnoses:
        # GPU idle recommendations
        if "GPU is idle" in d.observation:
            try:
                idle_pct = float(d.observation.split("idle ")[1].split("%")[0])
            except (ValueError, IndexError):
                idle_pct = 0

            if idle_pct > 30:
                recs.append(
                    Recommendation(
                        priority=priority,
                        action="Increase prefetch depth or enable resident cache",
                        reason=f"GPU idle {idle_pct:.0f}% — compute thread is starved for data",
                        expected_impact="Reduce GPU idle by 10-20%, improve throughput",
                        category="config",
                        confidence=0.8,
                    )
                )
                priority += 1

        # Prefetch recommendations
        if "Prefetch hit rate" in d.observation:
            try:
                rate = float(d.observation.split("is ")[1].split("%")[0])
            except (ValueError, IndexError):
                rate = 100

            if rate < 70:
                recs.append(
                    Recommendation(
                        priority=priority,
                        action="Increase prefetch depth",
                        reason=f"Hit rate only {rate:.0f}% — many layers require sync reads",
                        expected_impact="Reduce sync fallbacks by 20-40%",
                        category="config",
                        confidence=0.85,
                    )
                )
                priority += 1
            elif rate > 95:
                recs.append(
                    Recommendation(
                        priority=priority,
                        action="Consider reducing prefetch depth to save RAM",
                        reason=f"Hit rate {rate:.0f}% — depth may be higher than needed",
                        expected_impact="Free RAM for resident cache or larger window",
                        category="config",
                        confidence=0.7,
                    )
                )
                priority += 1

        # Bottleneck recommendations
        if "Primary bottleneck" in d.observation:
            for e in d.evidence:
                if "Dominant non-compute" in e.label:
                    if "Upload" in e.value:
                        recs.append(
                            Recommendation(
                                priority=priority,
                                action="Increase upload worker count",
                                reason="CPU→MPS upload is the dominant non-compute bottleneck",
                                expected_impact=(
                                    "Overlap uploads more aggressively, reduce per-layer latency"
                                ),
                                category="config",
                                confidence=d.confidence,
                            )
                        )
                        priority += 1
                    elif "SSD Read" in e.value:
                        recs.append(
                            Recommendation(
                                priority=priority,
                                action="Increase prefetch depth to overlap more SSD reads",
                                reason="SSD read is the dominant non-compute bottleneck",
                                expected_impact="More read/compute overlap",
                                category="config",
                                confidence=d.confidence,
                            )
                        )
                        priority += 1

        # Memory / resident cache recommendations
        if "could fit in resident cache" in d.observation:
            max_resident = 0
            for e in d.evidence:
                if "Max resident" in e.label:
                    try:
                        max_resident = int(e.value)
                    except ValueError:
                        pass

            if max_resident >= 4:
                target = min(max_resident, 8)
                recs.append(
                    Recommendation(
                        priority=priority,
                        action=f"Experiment with {target}-layer resident cache in simulator",
                        reason=(
                            f"~{max_resident} layers of headroom — resident cache "
                            f"can skip SSD reads for early layers"
                        ),
                        expected_impact=(
                            "Predicted 10-20% throughput improvement "
                            "[prediction — validate with simulator]"
                        ),
                        category="experiment",
                        is_prediction=True,
                        confidence=0.6,
                    )
                )
                priority += 1

        # Queue saturation recommendations
        if "queues saturated" in (d.diagnosis or ""):
            recs.append(
                Recommendation(
                    priority=priority,
                    action="Increase worker thread count",
                    reason="Worker pools are saturated — more workers would increase parallelism",
                    expected_impact="Better pipeline overlap",
                    category="config",
                    confidence=d.confidence,
                )
            )
            priority += 1

    # Always recommend simulator sweeps
    recs.append(
        Recommendation(
            priority=priority,
            action="Run simulator sweeps: swlp sim --resident 0,2,4,8,16 --window 2,4,6",
            reason="Empirical sweep finds the Pareto-optimal config for your hardware",
            expected_impact="Evidence-based configuration instead of guesswork",
            category="experiment",
            is_prediction=True,
            confidence=0.5,
        )
    )

    return recs


# ── Main entry point ──────────────────────────────────────────────────────


def analyze_traces(
    path: str | Path | None = None,
    data: dict | None = None,
) -> AnalysisReport:
    """Analyze profiler traces and produce a report.

    Args:
        path: Path to the JSON trace file from LayerProfiler.dump()
        data: Already-loaded trace dict (overrides path)
    """
    if data is None:
        if path is None:
            raise ValueError("Either path or data must be provided")
        with open(path) as f:
            data = json.load(f)

    traces = data.get("traces", [])
    metrics = data.get("pipeline_metrics", {})
    hw = data.get("hardware", {})
    queue_snapshots = data.get("queue_snapshots")

    if not traces:
        return AnalysisReport(
            summary={"status": "no data"},
            metadata=data,
        )

    # Run all analyses
    diagnoses: list[Diagnosis] = []

    gpu_diag, gpu_idle_pct = _analyze_gpu_idle(traces, metrics)
    diagnoses.append(gpu_diag)

    diagnoses.append(_analyze_prefetch(traces, metrics))
    diagnoses.append(_analyze_bottleneck(traces, metrics))
    diagnoses.extend(_analyze_memory(traces, metrics, hw))
    diagnoses.append(_analyze_ensure_stalls(traces, metrics))

    q_diag = _analyze_queue_saturation(metrics, queue_snapshots)
    if q_diag is not None:
        diagnoses.append(q_diag)

    # Pipeline occupancy
    occupancy = _compute_occupancy(traces, metrics)

    # Recommendations
    recommendations = _generate_recommendations(diagnoses, metrics, hw)

    # Summary
    summary = {
        "GPU utilization": f"{_f(metrics.get('gpu_efficiency')) * 100:.1f}%",
        "GPU idle": f"{gpu_idle_pct:.1f}%",
        "Prefetch hit rate": f"{_f(metrics.get('prefetch_hit_rate')) * 100:.1f}%",
        "Peak RSS": f"{_i(metrics.get('peak_rss_bytes')) / 1e9:.2f} GB",
        "Pipeline stalls": str(_i(metrics.get("pipeline_stall_count"))),
        "Avg ensure() wait": f"{_f(metrics.get('avg_ensure_wait_ms')):.1f} ms",
        "Avg compute": f"{_f(metrics.get('avg_compute_ms')):.1f} ms",
        "Avg SSD read": f"{_f(metrics.get('avg_read_ms')):.1f} ms",
        "Tokens analyzed": str(_i(metrics.get("num_tokens"))),
        "Layers analyzed": str(_i(metrics.get("num_layers"))),
    }

    # Determine primary bottleneck
    stages = {
        "SSD Read": _f(metrics.get("avg_read_ms")),
        "Upload": _f(metrics.get("avg_upload_ms")),
        "ensure() Wait": _f(metrics.get("avg_ensure_wait_ms")),
    }
    if stages:
        bottleneck = max(stages, key=stages.get)
        summary["Primary bottleneck"] = bottleneck

    if gpu_idle_pct > 30:
        summary["Overall health"] = "NEEDS ATTENTION"
    elif gpu_idle_pct > 15:
        summary["Overall health"] = "FAIR"
    else:
        summary["Overall health"] = "GOOD"

    return AnalysisReport(
        diagnoses=diagnoses,
        recommendations=recommendations,
        summary=summary,
        metadata=data,
        pipeline_occupancy=occupancy,
    )
