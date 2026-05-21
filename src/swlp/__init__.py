"""swlp — Sliding Window Layer Pipeline."""

from .core.pipeline import PipelineConfig, ThreadedPipeline
from .hardware.detect import HardwareInfo, detect_hardware
from .metrics import RunMetrics, RunResult
from .model.shard import ShardManifest, load_manifest, shard_model_by_layer
from .runner.base import build_runner, execute_baseline

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # hardware
    "HardwareInfo",
    "detect_hardware",
    # model sharding
    "ShardManifest",
    "shard_model_by_layer",
    "load_manifest",
    # pipeline
    "PipelineConfig",
    "ThreadedPipeline",
    # runners
    "build_runner",
    "execute_baseline",
    # metrics
    "RunMetrics",
    "RunResult",
]
