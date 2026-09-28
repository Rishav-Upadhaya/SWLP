"""Tests for swlp.core.streaming — StreamingScheduler overlap tracking, pin_memory,
and the F_NOCACHE direct-I/O helpers."""
from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn

from swlp.core.scheduler import SchedulerConfig
from swlp.core.streaming import (
    StreamingScheduler,
    _read_file_nocache,
    _safetensors_metadata,
)

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


def test_scheduler_prefetch_pool_round_trip(tmp_path: Path) -> None:
    """A full prefetch → ensure → evict → re-ensure cycle through the worker
    pool keeps weights correct (device-ready dicts must not alias buffers)."""
    from safetensors.torch import save_file as st_save_file

    from swlp.core.streaming import StreamingScheduler

    shard_dir = tmp_path / "shards"
    shard_dir.mkdir(parents=True)
    states = []
    blocks = []
    for i in range(3):
        state = {"weight": torch.randn(4, 4), "bias": torch.randn(4)}
        states.append(state)
        st_save_file(state, str(shard_dir / f"layer_{i:03d}.safetensors"))
        with torch.device("meta"):
            blocks.append(nn.Linear(4, 4))

    cfg = _make_config(prefetch=True)
    scheduler = StreamingScheduler(blocks, torch.device("cpu"), cfg, shard_dir)
    for i in range(3):
        scheduler.prefetch(i)
    for i in range(3):
        block = scheduler.ensure(i)
        assert torch.equal(block.weight.data, states[i]["weight"])
        scheduler.evict(i)
    # Re-ensure after evict (sync fallback path).
    block = scheduler.ensure(1)
    assert torch.equal(block.weight.data, states[1]["weight"])
    scheduler.cleanup()


# ── resolve_direct_io ─────────────────────────────────────────────────────────


def test_resolve_direct_io_on_off() -> None:
    from swlp.core.streaming import resolve_direct_io

    assert resolve_direct_io("on", 1, 100) is True
    assert resolve_direct_io("off", 10**12, 1) is False


def test_resolve_direct_io_auto_small_model_uses_page_cache() -> None:
    from swlp.core.streaming import resolve_direct_io

    # 1 GB model, 8 GB available → fits comfortably → cached reads.
    assert resolve_direct_io("auto", 10**9, 8 * 10**9) is False


def test_resolve_direct_io_auto_large_model_bypasses_cache() -> None:
    from swlp.core.streaming import resolve_direct_io

    # 14 GB model, 8 GB available → cannot stay cached → direct I/O.
    assert resolve_direct_io("auto", 14 * 10**9, 8 * 10**9) is True


def test_resolve_direct_io_unknown_sizes_default_safe() -> None:
    from swlp.core.streaming import resolve_direct_io

    assert resolve_direct_io("auto", 0, 8 * 10**9) is True
    assert resolve_direct_io("bogus", 10**9, 8 * 10**9) is True


# ── Phase 22: streaming from compressed .swz shards ──────────────────────────


def _write_swz_shards(
    shard_dir: Path, num_layers: int = 4, hidden: int = 4
) -> tuple[list[nn.Module], list[dict]]:
    """Write compressed .swz shard files; return (meta blocks, original states)."""
    from swlp import codec
    from swlp.model.shard import _save_safetensors

    shard_dir.mkdir(parents=True, exist_ok=True)
    blocks: list[nn.Module] = []
    states: list[dict] = []
    torch.manual_seed(11)
    for i in range(num_layers):
        state = {
            "weight": torch.randn(hidden, hidden, dtype=torch.float16),
            "bias": torch.randn(hidden, dtype=torch.float16),
        }
        st_path = shard_dir / f"layer_{i:03d}.safetensors"
        _save_safetensors(state, st_path)
        codec.compressed_path(st_path).write_bytes(codec.compress_bytes(st_path.read_bytes()))
        st_path.unlink()
        states.append(state)
        with torch.device("meta"):
            block = nn.Linear(hidden, hidden).to(torch.float16)
        blocks.append(block)
    return blocks, states


def test_streaming_swz_shards_bit_exact(tmp_path: Path) -> None:
    """ensure() on compressed shards materialises bit-identical weights."""
    shard_dir = tmp_path / "shards"
    blocks, states = _write_swz_shards(shard_dir)
    sched = StreamingScheduler(blocks, torch.device("cpu"), _make_config(), shard_dir)
    try:
        for i, state in enumerate(states):
            block = sched.ensure(i)
            assert torch.equal(block.weight.detach(), state["weight"])
            assert torch.equal(block.bias.detach(), state["bias"])
            sched.evict(i)
    finally:
        sched.cleanup()


def test_streaming_swz_prefetch_then_ensure(tmp_path: Path) -> None:
    """The prefetch worker path (read + decompress off-thread) serves ensure()."""
    shard_dir = tmp_path / "shards"
    blocks, states = _write_swz_shards(shard_dir)
    sched = StreamingScheduler(blocks, torch.device("cpu"), _make_config(), shard_dir)
    try:
        sched.prefetch(0)
        time.sleep(0.3)
        block = sched.ensure(0)
        assert torch.equal(block.weight.detach(), states[0]["weight"])
        stats = sched.overlap_stats()
        assert stats["misses"] == 0
    finally:
        sched.cleanup()


def test_streaming_swz_resident_layers(tmp_path: Path) -> None:
    """Resident-layer preload (load_safetensors_shard path) handles .swz."""
    shard_dir = tmp_path / "shards"
    blocks, states = _write_swz_shards(shard_dir)
    sched = StreamingScheduler(
        blocks, torch.device("cpu"), _make_config(), shard_dir, resident_count=2
    )
    try:
        sched.load_resident_layers()
        block = sched.ensure(0)
        assert torch.equal(block.weight.detach(), states[0]["weight"])
    finally:
        sched.cleanup()


def test_read_shard_payload_with_decompress_gate(tmp_path: Path) -> None:
    """The gated decompress path returns the same payload and releases the gate."""
    import threading

    from swlp.core.shard_io import read_shard_payload

    shard_dir = tmp_path / "shards"
    _write_swz_shards(shard_dir, num_layers=1)
    path = shard_dir / "layer_000.safetensors.swz"
    gate = threading.BoundedSemaphore(1)
    ungated, size_a, _ = read_shard_payload(path, nocache=False)
    gated, size_b, _ = read_shard_payload(path, nocache=False, decompress_gate=gate)
    assert size_a == size_b
    assert torch.equal(ungated[:size_a], gated[:size_b])
    # BoundedSemaphore raises on over-release, so acquiring proves it was released.
    assert gate.acquire(blocking=False)
    gate.release()


# ── Phase 26: multi-volume striping ─────────────────────────────────────────

def test_scheduler_reads_layers_across_volumes(tmp_path):
    """Layers striped over two volumes load via the round-robin probe order."""
    primary = tmp_path / "shards"
    secondary = tmp_path / "external"
    primary.mkdir()
    secondary.mkdir()
    states = []
    for i in range(4):
        state = {
            "weight": torch.randn(4, 4),
            "bias": torch.randn(4),
        }
        states.append(state)
        target = primary if i % 2 == 0 else secondary
        torch.save(state, str(target / f"layer_{i:03d}.pt"))

    blocks = []
    with torch.device("meta"):
        for _ in range(4):
            blocks.append(nn.Linear(4, 4))
    cfg = _make_config(prefetch=False)
    scheduler = StreamingScheduler(
        blocks, torch.device("cpu"), cfg, primary, extra_volumes=[secondary]
    )
    try:
        for i in range(4):
            block = scheduler.ensure(i)
            assert torch.equal(block.weight.data, states[i]["weight"])
    finally:
        scheduler.cleanup()


def test_volume_order_stripes_round_robin(tmp_path):
    primary = tmp_path / "a"
    other = tmp_path / "b"
    blocks = [nn.Linear(2, 2) for _ in range(3)]
    cfg = _make_config(prefetch=False)
    scheduler = StreamingScheduler(
        blocks, torch.device("cpu"), cfg, primary, extra_volumes=[other]
    )
    try:
        assert scheduler._volume_order(0)[0] == primary
        assert scheduler._volume_order(1)[0] == other
        assert scheduler._volume_order(2)[0] == primary
        for idx in range(4):
            order = scheduler._volume_order(idx)
            assert sorted(order) == sorted([primary, other])
    finally:
        scheduler.cleanup()
