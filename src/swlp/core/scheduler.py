"""Layer schedulers for Apple Silicon.

``ThreadedScheduler`` swaps transformer blocks between CPU RAM and the MPS
device using a background thread pool. The disk-streaming path lives in
``core/streaming.py``; this one is for a model already resident in CPU RAM.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from dataclasses import dataclass

import torch

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SchedulerConfig:
    window_size: int
    prefetch_depth: int
    prefetch: bool
    double_buffer: bool
    pin_memory: bool


class PrefetchError(RuntimeError):
    pass


def _capture_cpu_state(block: torch.nn.Module) -> dict[str, torch.Tensor] | None:
    """Snapshot a block's CPU-resident parameter and buffer tensors by name.

    Returns ``None`` when any tensor is not on CPU — callers fall back to a
    plain ``.to("cpu")`` eviction for such blocks.
    """
    state: dict[str, torch.Tensor] = {}
    for name, param in block.named_parameters():
        if param.device.type != "cpu":
            return None
        state[name] = param.data
    for name, buf in block.named_buffers():
        if buf is None:
            continue
        if buf.device.type != "cpu":
            return None
        state[name] = buf
    return state


def _restore_cpu_state(block: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Point a block's parameters/buffers back at their CPU master tensors.

    Weights are read-only during inference, so the masters are still current —
    re-pointing releases the device copies without the device→host copy that
    ``block.to("cpu")`` performs — on unified memory that copy is pure
    wasted bandwidth, and bandwidth is the binding constraint.
    """
    modules = dict(block.named_modules())
    for name, param in block.named_parameters():
        master = state.get(name)
        if master is not None:
            param.data = master
    for name, _ in list(block.named_buffers()):
        master = state.get(name)
        if master is None:
            continue
        mod_path, _, leaf = name.rpartition(".")
        owner = modules.get(mod_path, block)
        owner._buffers[leaf] = master


class ThreadedScheduler:
    """
    Threading-based layer scheduler for MPS (Apple Silicon) and CPU targets.
    Swaps blocks CPU<->MPS on a background thread pool.
    Used when the model is already resident in CPU RAM (no shard directory).
    """

    def __init__(
        self,
        blocks: Iterable[torch.nn.Module],
        device: torch.device,
        config: SchedulerConfig,
    ) -> None:
        self.blocks = list(blocks)
        self.device = device
        self.config = config
        self._loaded: set[int] = set()
        self._threads: dict[int, threading.Thread] = {}
        self._lock = threading.Lock()
        self._prefetch_enabled = config.prefetch
        self._cpu_masters: dict[int, dict[str, torch.Tensor] | None] = {
            i: _capture_cpu_state(b) for i, b in enumerate(self.blocks)
        }

    def _move_block(self, layer_index: int) -> None:
        self.blocks[layer_index].to(self.device)
        with self._lock:
            self._loaded.add(layer_index)
            self._threads.pop(layer_index, None)

    def prefetch(self, layer_index: int) -> None:
        if not self._prefetch_enabled:
            return
        if layer_index < 0 or layer_index >= len(self.blocks):
            return
        with self._lock:
            if layer_index in self._loaded or layer_index in self._threads:
                return
        t = threading.Thread(
            target=self._move_block, args=(layer_index,), daemon=True,
            name=f"swlp-prefetch-{layer_index}",
        )
        with self._lock:
            self._threads[layer_index] = t
        t.start()
        LOGGER.debug("threaded_prefetch_enqueued", extra={"layer": layer_index})

    def disable_prefetch(self) -> None:
        self._prefetch_enabled = False
        LOGGER.warning("threaded_prefetch_disabled")

    def ensure(self, layer_index: int) -> torch.nn.Module:
        with self._lock:
            t = self._threads.get(layer_index)
        if t is not None:
            t.join()
        if layer_index not in self._loaded:
            LOGGER.debug("threaded_sync_load", extra={"layer": layer_index})
            try:
                self._move_block(layer_index)
            except Exception:
                LOGGER.exception("threaded_sync_load_failed", extra={"layer": layer_index})
                raise PrefetchError(f"Failed to load layer {layer_index}") from None
        return self.blocks[layer_index]

    def evict(self, layer_index: int) -> None:
        if layer_index < 0 or layer_index >= len(self.blocks):
            return
        if layer_index not in self._loaded:
            return
        try:
            masters = self._cpu_masters.get(layer_index)
            if masters is not None:
                _restore_cpu_state(self.blocks[layer_index], masters)
            else:
                self.blocks[layer_index].to("cpu")
            with self._lock:
                self._loaded.discard(layer_index)
                self._threads.pop(layer_index, None)
            LOGGER.debug("threaded_evict", extra={"layer": layer_index})
        except Exception:
            LOGGER.exception("threaded_evict_failed", extra={"layer": layer_index})

    def window_end_index(self, current_index: int) -> int:
        return current_index + self.config.prefetch_depth

    def cleanup(self) -> None:
        for idx in list(self._loaded):
            try:
                self.evict(idx)
            except Exception:
                LOGGER.exception("threaded_cleanup_evict_failed", extra={"layer": idx})
        with self._lock:
            self._threads.clear()
