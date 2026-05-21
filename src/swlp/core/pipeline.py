"""Cross-platform sliding-window layer pipeline.

Works on CUDA, MPS (Apple Silicon), and CPU via Python threading.
The CUDA path (SWLPScheduler) uses CUDA streams for overlap.
This threading path hides SSD→RAM transfer latency behind compute
using a background prefetch thread per layer.

Usage (standalone, model already sharded):
    pipeline = ThreadedPipeline(PipelineConfig(shard_dir="./shards", window_size=4))
    pipeline.warmup(num_layers=32)
    for i in range(32):
        weights = pipeline.get_layer(i)
        pipeline.prefetch(i + pipeline.config.window_size)
        # ... apply weights to hidden_states ...
        pipeline.evict(i - 1)
    pipeline.cleanup()
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    shard_dir: str
    window_size: int = 4
    prefetch_depth: int = 2
    device: str = "cpu"     # "cpu" | "mps" | "cuda"


class ThreadedPipeline:
    """
    Prefetches layer weights from disk into RAM in a background thread
    while the current layer is computing. Works on any device.

    Memory model:
      - At most window_size layers resident in RAM at a time.
      - Each layer loaded via a daemon thread; join() only at compute time.
      - Evict is explicit — caller controls the window.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._window: dict[int, Any] = {}
        self._threads: dict[int, threading.Thread] = {}
        self._lock = threading.Lock()

    def _load_layer(self, idx: int) -> None:
        import torch
        path = Path(self.config.shard_dir) / f"layer_{idx:03d}.pt"
        if not path.exists():
            LOGGER.warning("shard_missing", extra={"layer": idx, "path": str(path)})
            return
        weights = torch.load(str(path), map_location="cpu", weights_only=True)
        with self._lock:
            self._window[idx] = weights
        LOGGER.debug("pipeline_loaded", extra={"layer": idx})

    def prefetch(self, idx: int) -> threading.Thread | None:
        """Schedule background load of layer idx. No-op if already loaded or in-flight."""
        if idx < 0:
            return None
        with self._lock:
            if idx in self._window or idx in self._threads:
                return self._threads.get(idx)
        t = threading.Thread(
            target=self._load_layer, args=(idx,), daemon=True, name=f"swlp-prefetch-{idx}"
        )
        with self._lock:
            self._threads[idx] = t
        t.start()
        return t

    def get_layer(self, idx: int) -> Any:
        """Return layer weights, blocking until prefetch completes if needed."""
        with self._lock:
            t = self._threads.get(idx)
        if t is not None:
            t.join()
        with self._lock:
            weights = self._window.get(idx)
        if weights is None:
            # synchronous fallback (prefetch was never called)
            LOGGER.debug("pipeline_sync_load", extra={"layer": idx})
            self._load_layer(idx)
            with self._lock:
                weights = self._window.get(idx)
        return weights

    def evict(self, idx: int) -> None:
        """Free layer from RAM. Call after compute is done on that layer."""
        with self._lock:
            self._window.pop(idx, None)
            self._threads.pop(idx, None)
        LOGGER.debug("pipeline_evicted", extra={"layer": idx})

    def warmup(self, num_layers: int) -> None:
        """Pre-fill the first window of layers before the forward pass starts."""
        for i in range(min(self.config.window_size, num_layers)):
            self.prefetch(i)

    def cleanup(self) -> None:
        with self._lock:
            self._window.clear()
            self._threads.clear()

    def resident_count(self) -> int:
        with self._lock:
            return len(self._window)
