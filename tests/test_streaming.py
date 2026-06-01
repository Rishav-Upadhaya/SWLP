"""Tests for swlp.core.streaming — StreamingScheduler overlap tracking, pin_memory,
and the F_NOCACHE direct-I/O helpers."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

from swlp.core.scheduler import SchedulerConfig
from swlp.core.streaming import _read_file_nocache, _safetensors_metadata
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


# ── _read_file_nocache ────────────────────────────────────────────────────────


def test_read_file_nocache_returns_correct_bytes(tmp_path: Path) -> None:
    """_read_file_nocache reads back the exact bytes written to a file."""
    payload = b"swlp-test-" * 1000
    f = tmp_path / "test.bin"
    f.write_bytes(payload)
    result = _read_file_nocache(f)
    assert result == payload


def test_read_file_nocache_large_file(tmp_path: Path) -> None:
    """Reads files larger than the 4 MB chunk size correctly."""
    payload = b"x" * (6 * 1024 * 1024)  # 6 MB > 4 MB chunk
    f = tmp_path / "large.bin"
    f.write_bytes(payload)
    result = _read_file_nocache(f)
    assert len(result) == len(payload)
    assert result == payload


def test_read_file_nocache_empty_file(tmp_path: Path) -> None:
    f = tmp_path / "empty.bin"
    f.write_bytes(b"")
    assert _read_file_nocache(f) == b""


# ── _safetensors_metadata ─────────────────────────────────────────────────────


def test_safetensors_metadata_extracts_custom_key(tmp_path: Path) -> None:
    """Metadata written to a .safetensors file is readable via the binary header."""
    from safetensors.torch import save_file as st_save_file

    tensors = {"w": torch.zeros(4, 4)}
    path = tmp_path / "layer_000.safetensors"
    st_save_file(tensors, str(path), metadata={"__swlp_quant__": "float8", "version": "1"})

    data = _read_file_nocache(path)
    meta = _safetensors_metadata(data)
    assert meta["__swlp_quant__"] == "float8"
    assert meta["version"] == "1"


def test_safetensors_metadata_missing_returns_empty(tmp_path: Path) -> None:
    from safetensors.torch import save_file as st_save_file

    tensors = {"w": torch.zeros(2)}
    path = tmp_path / "layer_000.safetensors"
    st_save_file(tensors, str(path))  # no metadata
    data = _read_file_nocache(path)
    assert _safetensors_metadata(data) == {}


def test_safetensors_metadata_truncated_data() -> None:
    assert _safetensors_metadata(b"") == {}
    assert _safetensors_metadata(b"\x00" * 4) == {}


# ── nocache path used by StreamingScheduler (.safetensors shards) ─────────────


def _write_safetensors_shards(
    shard_dir: Path, num_layers: int = 3, hidden: int = 4
) -> list[nn.Module]:
    from safetensors.torch import save_file as st_save_file

    shard_dir.mkdir(parents=True, exist_ok=True)
    blocks: list[nn.Module] = []
    for i in range(num_layers):
        state = {"weight": torch.zeros(hidden, hidden), "bias": torch.zeros(hidden)}
        st_save_file(state, str(shard_dir / f"layer_{i:03d}.safetensors"))
        with torch.device("meta"):
            block = nn.Linear(hidden, hidden)
        blocks.append(block)
    return blocks


def test_scheduler_loads_safetensors_via_nocache(tmp_path: Path) -> None:
    """StreamingScheduler loads .safetensors shards correctly via the new I/O path."""
    from swlp.core.streaming import StreamingScheduler

    shard_dir = tmp_path / "shards"
    blocks = _write_safetensors_shards(shard_dir, num_layers=3)
    cfg = _make_config(prefetch=False)
    scheduler = StreamingScheduler(blocks, torch.device("cpu"), cfg, shard_dir)

    for i in range(3):
        block = scheduler.ensure(i)
        assert block is not None
        # Verify the block is materialised on CPU (not meta device).
        for p in block.parameters():
            assert p.device.type == "cpu"
