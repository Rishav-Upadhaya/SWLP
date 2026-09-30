"""SWLP algorithm engine: scheduler, streaming, KV cache."""
from .compressed_cache import CompressedDynamicCache, CompressedDynamicLayer
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
from .speculative import NgramDrafter, SpeculativeConfig, verify_greedy
from .streaming import StreamingScheduler, has_shards

__all__ = [
    "CalibrationResult",
    "CompressedDynamicCache",
    "CompressedDynamicLayer",
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
    "SpeculativeConfig",
    "StreamingScheduler",
    "ThreadedScheduler",
    "PrefetchError",
    "SchedulerConfig",
    "build_residency_decision",
    "compute_pipeline_metrics",
    "estimate_optimal_resident_count",
    "has_shards",
    "plan_residency",
    "run_startup_calibration",
    "verify_greedy",
]
