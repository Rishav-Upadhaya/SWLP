"""SWLP algorithm engine: scheduler, streaming, KV cache."""
from .analyzer import AnalysisReport, analyze_traces
from .compressed_cache import CompressedDynamicCache, CompressedDynamicLayer
from .evaluator import EvaluationConfig, EvaluationResult, evaluate_policies
from .kv_cache import KVCacheManager
from .pipeline_model import estimate_pipeline_ratio_from_hardware, pipeline_ratio_from_metrics
from .profiler import (
    LayerProfiler,
    LayerTrace,
    PipelineMetrics,
    QueueSnapshot,
    compute_pipeline_metrics,
)
from .residency import (
    CalibrationResult,
    ResidencyDecision,
    ResidencyPlan,
    build_residency_decision,
    plan_residency,
    run_startup_calibration,
)
from .resident_policy import ResidentPolicyResult, estimate_optimal_resident_count
from .scheduler import PrefetchError, SchedulerConfig, ThreadedScheduler
from .simulator import SimResult, SimulatorConfig, simulate
from .speculative import NgramDrafter, SpeculativeConfig, verify_greedy
from .streaming import StreamingScheduler, has_shards
from .sweep import SweepConfig, SweepResult, run_sweep

__all__ = [
    "AnalysisReport",
    "CalibrationResult",
    "CompressedDynamicCache",
    "CompressedDynamicLayer",
    "EvaluationConfig",
    "EvaluationResult",
    "KVCacheManager",
    "LayerProfiler",
    "LayerTrace",
    "PipelineMetrics",
    "pipeline_ratio_from_metrics",
    "estimate_pipeline_ratio_from_hardware",
    "QueueSnapshot",
    "NgramDrafter",
    "ResidentPolicyResult",
    "ResidencyDecision",
    "ResidencyPlan",
    "SimResult",
    "SimulatorConfig",
    "SpeculativeConfig",
    "StreamingScheduler",
    "SweepConfig",
    "SweepResult",
    "ThreadedScheduler",
    "PrefetchError",
    "SchedulerConfig",
    "analyze_traces",
    "build_residency_decision",
    "compute_pipeline_metrics",
    "estimate_optimal_resident_count",
    "evaluate_policies",
    "has_shards",
    "plan_residency",
    "run_sweep",
    "run_startup_calibration",
    "simulate",
    "verify_greedy",
]
