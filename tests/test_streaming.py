"""Tests for swlp.core.streaming — StreamingScheduler overlap tracking and pin_memory."""
from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn

from swlp.core.scheduler import SchedulerConfig
from swlp.core.streaming import StreamingScheduler


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_config(
    pin_memory: bool = False,
    prefetch: bool = True,
    window_size: int = 2,
) -> SchedulerConfig:
    return SchedulerConfig(
        window_size=window_size,
        prefetch_depth=1,
        prefetch=prefetch,
        double_buffer=True,
        pin_memory=pin_memory,
    )


def _write_shards(shard_dir: Path, num_layers: int = 4, hidden: int = 4) -> list[nn.Module]:
    """Write tiny .pt shard files; return corresponding meta-device modules."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    blocks: list[nn.Module] = []
    for i in range(num_layers):
        state = {
            "weight": torch.zeros(hidden, hidden),
            "bias": torch.zeros(hidden),
        }
        torch.save(state, str(shard_dir / f"layer_{i:03d}.pt"))
        with torch.device("meta"):
            block = nn.Linear(hidden, hidden)
        blocks.append(block)
    return blocks


# ── tests ─────────────────────────────────────────────────────────────────────


def test_overlap_stats_initially_zero(tmp_path: Path) -> None:
    """Fresh scheduler reports all overlap counters as zero."""
    shard_dir = tmp_path / "shards"
    blocks = _write_shards(shard_dir)
    cfg = _make_config()
    scheduler = StreamingScheduler(blocks, torch.device("cpu"), cfg, shard_dir)
    stats = scheduler.overlap_stats()
    assert stats["hits"] == 0
    assert stats["waits"] == 0
    assert stats["misses"] == 0
    assert stats["total"] == 0
    assert stats["hit_rate"] == 0.0


def test_miss_counted_when_no_prefetch(tmp_path: Path) -> None:
    """ensure() with prefetch disabled → miss counter incremented."""
    shard_dir = tmp_path / "shards"
    blocks = _write_shards(shard_dir)
    cfg = _make_config(prefetch=False)
    scheduler = StreamingScheduler(blocks, torch.device("cpu"), cfg, shard_dir)
    scheduler.ensure(0)
    stats = scheduler.overlap_stats()
    assert stats["misses"] == 1
    assert stats["hits"] == 0
    assert stats["waits"] == 0


def test_prefetch_then_ensure_no_miss(tmp_path: Path) -> None:
    """prefetch() then ensure() → no miss; either hit or wait (thread timing)."""
    shard_dir = tmp_path / "shards"
    blocks = _write_shards(shard_dir)
    cfg = _make_config(prefetch=True)
    scheduler = StreamingScheduler(blocks, torch.device("cpu"), cfg, shard_dir)
    scheduler.prefetch(0)
    # Give the background thread time to finish reading the tiny shard.
    time.sleep(0.15)
    scheduler.ensure(0)
    stats = scheduler.overlap_stats()
    assert stats["misses"] == 0
    assert stats["hits"] + stats["waits"] == 1


def test_evict_removes_from_loaded(tmp_path: Path) -> None:
    """ensure() materialises a layer; evict() removes it from _loaded."""
    shard_dir = tmp_path / "shards"
    blocks = _write_shards(shard_dir)
    cfg = _make_config(prefetch=False)
    scheduler = StreamingScheduler(blocks, torch.device("cpu"), cfg, shard_dir)
    scheduler.ensure(0)
    assert 0 in scheduler._loaded
    scheduler.evict(0)
    assert 0 not in scheduler._loaded


def test_pin_memory_no_crash(tmp_path: Path) -> None:
    """pin_memory=True must not crash even when CUDA is unavailable."""
    shard_dir = tmp_path / "shards"
    blocks = _write_shards(shard_dir)
    cfg = _make_config(pin_memory=True, prefetch=False)
    scheduler = StreamingScheduler(blocks, torch.device("cpu"), cfg, shard_dir)
    # Should complete without error regardless of CUDA availability.
    scheduler.ensure(0)
    assert 0 in scheduler._loaded


def test_multiple_ensures_accumulate_stats(tmp_path: Path) -> None:
    """ensure() called N times with no prefetch → N misses total."""
    shard_dir = tmp_path / "shards"
    blocks = _write_shards(shard_dir, num_layers=4)
    cfg = _make_config(prefetch=False)
    scheduler = StreamingScheduler(blocks, torch.device("cpu"), cfg, shard_dir)
    for i in range(4):
        scheduler.ensure(i)
    stats = scheduler.overlap_stats()
    assert stats["misses"] == 4
    assert stats["total"] == 4
    assert stats["hit_rate"] == 0.0
