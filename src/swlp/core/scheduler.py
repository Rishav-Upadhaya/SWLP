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


class SWLPScheduler:
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
        self._prefetch_events: dict[int, torch.cuda.Event] = {}
        self._prefetch_enabled = (
            config.prefetch and device.type == "cuda" and torch.cuda.is_available()
        )
        self._streams: list[torch.cuda.Stream] = []
        self._max_prefetch_retries: int = 2
        self._prefetch_backoff: float = 0.01
        if self._prefetch_enabled:
            stream_count = 2 if config.double_buffer else 1
            self._streams = [torch.cuda.Stream(device=device) for _ in range(stream_count)]
        if config.pin_memory and self._prefetch_enabled:
            self._pin_blocks()

    def _pin_blocks(self) -> None:
        for block in self.blocks:
            for param in block.parameters(recurse=False):
                if param.device.type == "cpu":
                    param.data = param.data.pin_memory()
            for name, buffer in block.named_buffers(recurse=False):
                if buffer is not None and buffer.device.type == "cpu":
                    block._buffers[name] = buffer.pin_memory()

    def _select_stream(self, layer_index: int) -> torch.cuda.Stream:
        return self._streams[layer_index % len(self._streams)]

    def _move_block(self, layer_index: int, non_blocking: bool) -> None:
        block = self.blocks[layer_index]
        block.to(self.device, non_blocking=non_blocking)
        self._loaded.add(layer_index)

    def prefetch(self, layer_index: int) -> None:
        if not self._prefetch_enabled:
            return
        if layer_index < 0 or layer_index >= len(self.blocks):
            return
        if layer_index in self._loaded or layer_index in self._prefetch_events:
            return
        stream = self._select_stream(layer_index)
        last_exc: Exception | None = None
        for attempt in range(self._max_prefetch_retries + 1):
            try:
                with torch.cuda.stream(stream):
                    LOGGER.debug(
                        "swlp_prefetch_start",
                        extra={"layer": layer_index, "attempt": attempt},
                    )
                    self._move_block(layer_index, non_blocking=True)
                event = torch.cuda.Event()
                stream.record_event(event)
                self._prefetch_events[layer_index] = event
                LOGGER.debug(
                    "swlp_prefetch_enqueued",
                    extra={"layer": layer_index, "attempt": attempt},
                )
                return
            except Exception as exc:
                last_exc = exc
                LOGGER.warning(
                    "swlp_prefetch_attempt_failed",
                    extra={"layer": layer_index, "attempt": attempt, "error": str(exc)},
                )
                # simple backoff
                import time

                time.sleep(self._prefetch_backoff * (attempt + 1))
                continue
        # if we reach here, all attempts failed
        raise PrefetchError(f"Failed to prefetch layer {layer_index} after retries") from last_exc

    def disable_prefetch(self) -> None:
        if self._prefetch_enabled:
            LOGGER.warning("swlp_prefetch_disabled")
        self._prefetch_enabled = False
        self._prefetch_events.clear()

    def ensure(self, layer_index: int) -> torch.nn.Module:
        if layer_index in self._prefetch_events:
            event = self._prefetch_events.pop(layer_index)
            try:
                torch.cuda.current_stream().wait_event(event)
                return self.blocks[layer_index]
            except Exception:
                LOGGER.exception("swlp_prefetch_wait_failed", extra={"layer": layer_index})
                # fall back to synchronous move
        if layer_index not in self._loaded:
            try:
                LOGGER.debug("swlp_sync_load", extra={"layer": layer_index})
                self._move_block(layer_index, non_blocking=False)
            except Exception:
                LOGGER.exception("swlp_sync_load_failed", extra={"layer": layer_index})
                raise PrefetchError(f"Failed to ensure layer {layer_index}") from None
        return self.blocks[layer_index]

    def evict(self, layer_index: int) -> None:
        if layer_index < 0 or layer_index >= len(self.blocks):
            return
        if layer_index not in self._loaded:
            return
        try:
            LOGGER.debug("swlp_evict", extra={"layer": layer_index})
            self.blocks[layer_index].to("cpu")
            self._loaded.remove(layer_index)
        except Exception:
            LOGGER.exception("swlp_evict_failed", extra={"layer": layer_index})

    def window_end_index(self, current_index: int) -> int:
        return current_index + self.config.prefetch_depth

    def cleanup(self) -> None:
        """Evict loaded layers and release stream/event resources."""
        try:
            for idx in list(self._loaded):
                try:
                    self.evict(idx)
                except Exception:
                    LOGGER.exception("swlp_cleanup_evict_failed", extra={"layer": idx})
            self._prefetch_events.clear()
            self._streams.clear()
        except Exception:
            LOGGER.exception("swlp_cleanup_failed")


class ThreadedScheduler:
    """
    Threading-based layer scheduler for MPS (Apple Silicon) and CPU targets.
    Same interface as SWLPScheduler but uses Python threads instead of CUDA streams.
    Allows SWLP to run on any device — no CUDA required.
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
