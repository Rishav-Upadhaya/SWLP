"""Tests for swlp.core.scheduler — CPU-master capture/restore and the
no-copy eviction path of ThreadedScheduler."""
from __future__ import annotations

import torch
import torch.nn as nn

from swlp.core.scheduler import (
    SchedulerConfig,
    ThreadedScheduler,
    _capture_cpu_state,
    _restore_cpu_state,
)


def _make_config(prefetch: bool = False) -> SchedulerConfig:
    return SchedulerConfig(
        window_size=2,
        prefetch_depth=1,
        prefetch=prefetch,
        double_buffer=False,
        pin_memory=False,
    )


def _accelerator_device() -> torch.device | None:
    """Return a non-CPU device when one is available (MPS on Apple, else CUDA)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return None


class _BlockWithBuffer(nn.Module):
    def __init__(self, hidden: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)
        self.register_buffer("scale", torch.ones(hidden))


# ── _capture_cpu_state / _restore_cpu_state ──────────────────────────────────


def test_capture_cpu_state_includes_params_and_buffers() -> None:
    block = _BlockWithBuffer()
    state = _capture_cpu_state(block)
    assert state is not None
    assert "linear.weight" in state
    assert "linear.bias" in state
    assert "scale" in state
    # Captured tensors share storage with the live tensors (no copies).
    assert state["linear.weight"].data_ptr() == block.linear.weight.data.data_ptr()


def test_capture_cpu_state_returns_none_for_meta_blocks() -> None:
    with torch.device("meta"):
        block = nn.Linear(4, 4)
    assert _capture_cpu_state(block) is None


def test_restore_cpu_state_points_back_at_masters() -> None:
    block = _BlockWithBuffer()
    state = _capture_cpu_state(block)
    assert state is not None
    # Simulate a device move by detaching params from their masters.
    block.linear.weight.data = block.linear.weight.data.clone()
    block.scale = block.scale.clone()
    assert block.linear.weight.data.data_ptr() != state["linear.weight"].data_ptr()
    _restore_cpu_state(block, state)
    assert block.linear.weight.data.data_ptr() == state["linear.weight"].data_ptr()
    assert block.scale.data_ptr() == state["scale"].data_ptr()


# ── ThreadedScheduler eviction ───────────────────────────────────────────────


def test_threaded_scheduler_captures_masters_at_init() -> None:
    blocks = [_BlockWithBuffer() for _ in range(3)]
    scheduler = ThreadedScheduler(blocks, torch.device("cpu"), _make_config())
    assert set(scheduler._cpu_masters) == {0, 1, 2}
    assert all(m is not None for m in scheduler._cpu_masters.values())


def test_threaded_scheduler_evict_restores_master_identity() -> None:
    """After ensure() + evict(), params must point at the original CPU masters
    (no device→host copy-back creating fresh tensors)."""
    device = _accelerator_device()
    if device is None:
        # CPU-only environment: ensure/evict must still round-trip cleanly.
        device = torch.device("cpu")
    blocks = [_BlockWithBuffer() for _ in range(2)]
    scheduler = ThreadedScheduler(blocks, device, _make_config())
    masters = scheduler._cpu_masters[0]
    assert masters is not None
    original_ptr = masters["linear.weight"].data_ptr()

    scheduler.ensure(0)
    scheduler.evict(0)

    assert 0 not in scheduler._loaded
    assert blocks[0].linear.weight.data.data_ptr() == original_ptr
    assert blocks[0].linear.weight.device.type == "cpu"


def test_threaded_scheduler_reload_after_evict_matches_master() -> None:
    """A second ensure() after evict() must produce the same weight values."""
    device = _accelerator_device() or torch.device("cpu")
    blocks = [_BlockWithBuffer() for _ in range(2)]
    expected = blocks[0].linear.weight.data.clone()
    scheduler = ThreadedScheduler(blocks, device, _make_config())

    scheduler.ensure(0)
    scheduler.evict(0)
    block = scheduler.ensure(0)

    got = block.linear.weight.data.to("cpu")
    assert torch.equal(got, expected)
    scheduler.cleanup()
