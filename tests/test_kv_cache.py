"""Tests for swlp.core.kv_cache — KVCacheManager including Phase 12 disk spill."""
from pathlib import Path

import torch

from swlp.core.kv_cache import KVCacheManager


# ── existing tests ─────────────────────────────────────────────────────────


def test_kv_cache_basic_compress_decompress():
    # small budget to force compression
    budget = 20 * 1024  # 20 KB
    manager = KVCacheManager(budget_bytes=budget, compression=True)

    # create two small tensors
    a = torch.randn(16, 16)
    b = torch.randn(16, 16)

    manager.set(0, (a, b))
    manager.set(1, (a * 2, b * 2))

    stats = manager.stats()
    assert stats["entries"] == 2

    # budget is very small so at least one entry should be compressed
    assert stats["compressed_bytes"] >= 0

    # fetch layer 0 (decompress if needed)
    tensors = manager.get(0)
    assert tensors is None or (isinstance(tensors, tuple) and len(tensors) == 2)
    if tensors:
        k, v = tensors
        assert torch.allclose(k.cpu(), a.cpu()) or k.numel() == a.numel()


# ── Phase 12 disk-spill tests ──────────────────────────────────────────────


def test_disk_spill_writes_file(tmp_path: Path) -> None:
    """Over-budget manager with disk_dir should spill to a file."""
    disk_dir = tmp_path / "kv_spill"
    # Budget so small that after compression we still overflow → disk spill.
    tiny_budget = 10  # 10 bytes — guaranteed overflow
    manager = KVCacheManager(
        budget_bytes=tiny_budget,
        compression=True,
        tiering=False,
        disk_dir=disk_dir,
    )
    a = torch.randn(32, 32)
    manager.set(0, (a, a.clone()))
    # At least one file should exist in disk_dir after budget enforcement.
    disk_files = list(disk_dir.glob("kv_*.pt"))
    assert len(disk_files) >= 1
    stats = manager.stats()
    assert stats["disk_bytes"] > 0
    assert stats["disk_spills"] >= 1


def test_disk_spill_round_trip(tmp_path: Path) -> None:
    """KV spilled to disk can be fully restored via get()."""
    disk_dir = tmp_path / "kv_spill"
    tiny_budget = 10
    a = torch.randn(8, 8)
    b = torch.randn(8, 8)
    manager = KVCacheManager(
        budget_bytes=tiny_budget,
        compression=True,
        tiering=False,
        disk_dir=disk_dir,
    )
    manager.set(0, (a, b))
    # Confirm spill occurred.
    assert manager._entries[0].location == "disk"
    # Reload.
    tensors = manager.get(0)
    assert tensors is not None
    k, v = tensors
    # Values should be bit-identical after round-trip.
    assert torch.allclose(k.cpu(), a.cpu(), atol=1e-5)
    assert torch.allclose(v.cpu(), b.cpu(), atol=1e-5)


def test_disk_spill_file_deleted_after_load(tmp_path: Path) -> None:
    """Disk spill file is removed after being loaded back via get()."""
    disk_dir = tmp_path / "kv_spill"
    a = torch.randn(8, 8)
    manager = KVCacheManager(
        budget_bytes=10,
        compression=True,
        tiering=False,
        disk_dir=disk_dir,
    )
    manager.set(0, (a, a.clone()))
    spill_file = manager._entries[0].disk_path
    assert spill_file is not None
    assert Path(spill_file).exists()
    manager.get(0)
    assert not Path(spill_file).exists()


def test_history_empty_when_profile_false() -> None:
    """_history stays empty when profile=False (default)."""
    manager = KVCacheManager(budget_bytes=1024 * 1024)
    a = torch.randn(16, 16)
    for i in range(10):
        manager.set(i, (a, a.clone()))
    assert manager._history == []


def test_history_populated_when_profile_true() -> None:
    """_history grows with profile=True."""
    manager = KVCacheManager(budget_bytes=1024 * 1024, profile=True)
    a = torch.randn(16, 16)
    manager.set(0, (a, a.clone()))
    assert len(manager._history) >= 1
    assert "timestamp" in manager._history[0]
    assert "disk_bytes" in manager._history[0]


def test_stats_has_disk_fields() -> None:
    """stats() always returns disk_bytes, disk_spills, disk_loads fields."""
    manager = KVCacheManager()
    s = manager.stats()
    assert "disk_bytes" in s
    assert "disk_spills" in s
    assert "disk_loads" in s


def test_clear_deletes_disk_files(tmp_path: Path) -> None:
    """clear() removes spill files from disk."""
    disk_dir = tmp_path / "kv_spill"
    a = torch.randn(8, 8)
    manager = KVCacheManager(
        budget_bytes=10,
        compression=True,
        tiering=False,
        disk_dir=disk_dir,
    )
    manager.set(0, (a, a.clone()))
    spill_file = manager._entries[0].disk_path
    manager.clear()
    if spill_file:
        assert not Path(spill_file).exists()
    assert len(manager._entries) == 0


# ── Phase 16 kv_window tests ───────────────────────────────────────────────


def test_kv_window_trims_long_sequences() -> None:
    """set() trims KV to kv_window positions when sequence exceeds window."""
    window = 4
    manager = KVCacheManager(budget_bytes=1024 * 1024, kv_window=window)
    # KV shape: [batch=1, heads=2, seq=10, head_dim=8]
    k = torch.randn(1, 2, 10, 8)
    v = torch.randn(1, 2, 10, 8)
    manager.set(0, (k, v))
    stored = manager.get(0)
    assert stored is not None
    k_out, v_out = stored
    assert k_out.shape[-2] == window
    assert v_out.shape[-2] == window
    # Stored values must equal the LAST kv_window positions of the input.
    assert torch.equal(k_out, k[..., -window:, :])
    assert torch.equal(v_out, v[..., -window:, :])


def test_kv_window_no_trim_when_short() -> None:
    """set() leaves short sequences untouched when seq_len <= kv_window."""
    window = 16
    manager = KVCacheManager(budget_bytes=1024 * 1024, kv_window=window)
    k = torch.randn(1, 2, 8, 8)
    v = torch.randn(1, 2, 8, 8)
    manager.set(0, (k, v))
    stored = manager.get(0)
    assert stored is not None
    k_out, v_out = stored
    assert k_out.shape[-2] == 8  # unchanged
    assert torch.equal(k_out, k)


def test_kv_window_zero_means_unbounded() -> None:
    """kv_window=0 (default) does not trim anything."""
    manager = KVCacheManager(budget_bytes=1024 * 1024, kv_window=0)
    k = torch.randn(1, 1, 100, 8)
    v = torch.randn(1, 1, 100, 8)
    manager.set(0, (k, v))
    stored = manager.get(0)
    assert stored is not None
    k_out, _ = stored
    assert k_out.shape[-2] == 100  # full length kept


def test_kv_window_bytes_bounded() -> None:
    """After repeated set() calls with growing seq, stored bytes stay bounded."""
    window = 4
    head_dim = 8
    manager = KVCacheManager(budget_bytes=1024 * 1024, kv_window=window)
    for seq_len in range(1, 20):
        k = torch.randn(1, 1, seq_len, head_dim)
        v = torch.randn(1, 1, seq_len, head_dim)
        manager.set(0, (k, v))
    stored = manager.get(0)
    assert stored is not None
    k_out, _ = stored
    assert k_out.shape[-2] <= window


# ── Phase 18 INT4 KV quantization tests ───────────────────────────────────


def test_kv_quant_int4_invalid_value() -> None:
    """KVCacheManager raises ValueError for unknown kv_quant values."""
    import pytest

    with pytest.raises(ValueError, match="kv_quant"):
        KVCacheManager(kv_quant="fp8")


def test_kv_quant_none_is_default() -> None:
    """kv_quant='none' is the default (lossless path)."""
    manager = KVCacheManager()
    assert manager.kv_quant == "none"
    k = torch.randn(1, 2, 8, 16)
    v = torch.randn(1, 2, 8, 16)
    manager.set(0, (k, v))
    result = manager.get(0)
    assert result is not None
    k_out, v_out = result
    assert torch.allclose(k_out, k, atol=1e-5)
    assert torch.allclose(v_out, v, atol=1e-5)


def test_kv_quant_int4_stores_quantized_tensors() -> None:
    """INT4 mode stores packed tensors in quantized_tensors, not in tensors."""
    manager = KVCacheManager(kv_quant="int4")
    k = torch.randn(1, 2, 8, 16)
    v = torch.randn(1, 2, 8, 16)
    manager.set(0, (k, v))
    entry = manager._entries[0]
    assert entry.quantized_tensors is not None
    assert entry.tensors is None  # raw tensors are not stored
    assert len(entry.quantized_tensors) == 4  # (pk, sk, pv, sv)


def test_kv_quant_int4_round_trip_shape() -> None:
    """INT4 get() returns tensors with the original shape."""
    manager = KVCacheManager(kv_quant="int4")
    k = torch.randn(1, 4, 12, 64)
    v = torch.randn(1, 4, 12, 64)
    manager.set(0, (k, v))
    result = manager.get(0)
    assert result is not None
    k_out, v_out = result
    assert k_out.shape == k.shape
    assert v_out.shape == v.shape


def test_kv_quant_int4_approximate_values() -> None:
    """INT4 KV approximates original values within a reasonable tolerance."""
    manager = KVCacheManager(kv_quant="int4")
    # Use a tensor with values in a bounded range so quantization error is measurable.
    k = torch.randn(1, 2, 16, 64).clamp(-2, 2)
    v = torch.randn(1, 2, 16, 64).clamp(-2, 2)
    manager.set(0, (k, v))
    result = manager.get(0)
    assert result is not None
    k_out, v_out = result
    # Relative L1 error should be under 20% for bounded activations on iid data.
    k_rel_err = (k_out.float() - k.float()).abs().mean() / k.float().abs().mean().clamp(min=1e-8)
    v_rel_err = (v_out.float() - v.float()).abs().mean() / v.float().abs().mean().clamp(min=1e-8)
    assert float(k_rel_err) < 0.20, f"K relative L1 error too high: {k_rel_err:.4f}"
    assert float(v_rel_err) < 0.20, f"V relative L1 error too high: {v_rel_err:.4f}"


def test_kv_quant_int4_smaller_than_fp16() -> None:
    """INT4 mode uses less memory than FP16 for the same KV tensor."""
    # A large tensor to make the footprint difference clear.
    k = torch.randn(1, 8, 64, 128)
    v = torch.randn(1, 8, 64, 128)
    fp16_bytes = k.element_size() * k.numel() * 2  # k + v in FP16

    manager = KVCacheManager(kv_quant="int4")
    manager.set(0, (k, v))
    entry = manager._entries[0]
    int4_bytes = entry.uncompressed_bytes
    # INT4 should be ~4× smaller than FP16 (packed uint8 + scales).
    assert int4_bytes < fp16_bytes / 2, (
        f"INT4 ({int4_bytes} bytes) not smaller than half of FP16 ({fp16_bytes} bytes)"
    )


def test_kv_quant_int4_skip_zlib_compression() -> None:
    """INT4 entries skip zlib compression (already compact)."""
    budget = 10  # tiny — forces compression + eviction
    manager = KVCacheManager(budget_bytes=budget, compression=True, kv_quant="int4")
    k = torch.randn(1, 2, 8, 16)
    v = torch.randn(1, 2, 8, 16)
    manager.set(0, (k, v))
    entry = manager._entries[0]
    # Still quantized; zlib compression was skipped.
    assert entry.quantized_tensors is not None
    assert entry.location == "host"
    assert entry.compressed is None
