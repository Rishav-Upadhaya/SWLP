"""Tests for read_shard_mmap — zero-copy mmap shard reads."""
import json
import struct

import pytest
import torch

from swlp.core.shard_io import parse_safetensors_views, read_shard_mmap


def _write_shard(path, tensors: dict[str, torch.Tensor]) -> None:
    """Minimal safetensors writer (no external deps)."""
    header = {}
    offset = 0
    blobs = []
    for name, t in tensors.items():
        raw = t.contiguous().view(torch.uint8).numpy().tobytes()
        header[name] = {
            "dtype": "F32" if t.dtype == torch.float32 else str(t.dtype).upper(),
            "shape": list(t.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
        blobs.append(raw)
    hjson = json.dumps(header).encode()
    pad = (8 - len(hjson) % 8) % 8  # align data start to 8 bytes like real files
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson) + pad))
        f.write(hjson)
        f.write(b" " * pad)
        for b in blobs:
            f.write(b)


def test_mmap_roundtrip(tmp_path):
    path = tmp_path / "layer_000.safetensors"
    original = {"w": torch.arange(12, dtype=torch.float16).reshape(3, 4)}
    _write_shard(path, {k: v.float() for k, v in original.items()})

    payload, size = read_shard_mmap(path)
    assert size == path.stat().st_size
    views, metadata = parse_safetensors_views(payload, size)
    assert "w" in views
    # Views must be byte-identical to what was written.
    assert torch.equal(views["w"], torch.arange(12, dtype=torch.float32).reshape(3, 4))


def test_mmap_payload_keeps_mapping_alive(tmp_path):
    """Parsed views stay valid after the local payload reference is dropped."""
    path = tmp_path / "layer_001.safetensors"
    expected = torch.randn(8, 8)
    _write_shard(path, {"x": expected})

    def load():
        payload, size = read_shard_mmap(path)
        views, _ = parse_safetensors_views(payload, size)
        return views["x"].clone()

    assert torch.equal(load(), expected)


def test_mmap_missing_file(tmp_path):
    with pytest.raises(OSError):
        read_shard_mmap(tmp_path / "nope.safetensors")
