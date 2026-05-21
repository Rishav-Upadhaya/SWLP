"""Tests for Phase 17 safetensors shard format.

Covers:
- _save_safetensors writes a valid .safetensors file (plain FP16)
- _save_safetensors flattens FP8 quant dicts correctly
- _load_safetensors_shard round-trips plain FP16 state
- _load_safetensors_shard reconstructs FP8 nested format
- _safetensors_file_ok integrity check
- verify_shards detects corrupt safetensors
- get_layer_path uses correct extension per shard_format
- list_layer_paths prefers .safetensors over .pt
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import torch
import pytest

from swlp.model.shard import (
    ShardManifest,
    _safetensors_file_ok,
    _save_safetensors,
    get_layer_path,
    list_layer_paths,
    verify_shards,
    _write_manifest,
)
from swlp.core.streaming import _load_safetensors_shard
from swlp.model.quant import quantize_layer_state, dequantize_layer_state


# ── helpers ─────────────────────────────────────────────────────────────────

def _fp16_state(hidden: int = 4) -> dict[str, torch.Tensor]:
    return {
        "weight": torch.randn(hidden, hidden, dtype=torch.float16),
        "bias": torch.zeros(hidden, dtype=torch.float16),
    }


def _minimal_shard_dir(
    tmp_path: Path,
    num_layers: int = 2,
    shard_format: str = "safetensors",
) -> Path:
    """Write a minimal shard dir with embed.pt, lm_head.pt, and layer_*.* files."""
    d = tmp_path / "shards"
    d.mkdir()
    torch.save({"embed_tokens": {"weight": torch.zeros(4, 4)}}, str(d / "embed.pt"))
    torch.save({"weight": torch.zeros(4, 4)}, str(d / "lm_head.pt"))
    for i in range(num_layers):
        if shard_format == "safetensors":
            _save_safetensors(_fp16_state(), d / f"layer_{i:03d}.safetensors")
        else:
            torch.save(_fp16_state(), str(d / f"layer_{i:03d}.pt"))
    manifest = ShardManifest(
        model_id="test", num_layers=num_layers, layer_weight_mb=0.01,
        total_weight_mb=0.02, embed_file="embed.pt", lm_head_file="lm_head.pt",
        model_type="llama", shard_format=shard_format,
    )
    _write_manifest(d, manifest)
    return d


# ── plain FP16 safetensors round-trip ───────────────────────────────────────

def test_save_and_load_fp16_round_trip(tmp_path: Path) -> None:
    """_save_safetensors + _load_safetensors_shard → bit-identical FP16 state."""
    state = _fp16_state()
    path = tmp_path / "layer_000.safetensors"
    _save_safetensors(state, path)
    loaded = _load_safetensors_shard(path)
    for k in state:
        assert k in loaded
        assert torch.equal(loaded[k], state[k])


def test_save_safetensors_creates_valid_file(tmp_path: Path) -> None:
    """Written .safetensors file passes _safetensors_file_ok."""
    path = tmp_path / "layer.safetensors"
    _save_safetensors(_fp16_state(), path)
    assert _safetensors_file_ok(path)


# ── FP8 safetensors round-trip ───────────────────────────────────────────────

def test_fp8_save_and_load_reconstructs_nested(tmp_path: Path) -> None:
    """_save_safetensors(FP8 quant dict) + _load_safetensors_shard → nested FP8."""
    state = _fp16_state(hidden=8)
    fp8_state = quantize_layer_state(state, scheme="float8")
    path = tmp_path / "layer_fp8.safetensors"
    _save_safetensors(fp8_state, path)
    loaded = _load_safetensors_shard(path)
    assert loaded.get("_swlp_quant") == "float8"
    assert "weights" in loaded


def test_fp8_dequantize_after_safetensors_round_trip(tmp_path: Path) -> None:
    """FP8 → .safetensors → load → dequantize ≈ original FP16."""
    state = {
        "weight": torch.randn(8, 8, dtype=torch.float16),
        "bias": torch.zeros(8, dtype=torch.float16),
    }
    fp8_state = quantize_layer_state(state, scheme="float8")
    path = tmp_path / "layer_fp8.safetensors"
    _save_safetensors(fp8_state, path)
    loaded = _load_safetensors_shard(path)
    deq = dequantize_layer_state(loaded)
    assert "weight" in deq
    # Relative error should be small (FP8 near-lossless).
    rel = (deq["weight"].float() - state["weight"].float()).abs().mean()
    orig_abs = state["weight"].float().abs().mean()
    assert rel / orig_abs < 0.05


# ── _safetensors_file_ok ─────────────────────────────────────────────────────

def test_safetensors_file_ok_rejects_empty(tmp_path: Path) -> None:
    p = tmp_path / "empty.safetensors"
    p.write_bytes(b"")
    assert not _safetensors_file_ok(p)


def test_safetensors_file_ok_rejects_pt_magic(tmp_path: Path) -> None:
    """A .pt file (ZIP magic PK) should fail the safetensors header check."""
    p = tmp_path / "fake.safetensors"
    torch.save({"x": torch.zeros(1)}, str(p))
    assert not _safetensors_file_ok(p)


def test_safetensors_file_ok_missing_file(tmp_path: Path) -> None:
    p = tmp_path / "no_such_file.safetensors"
    assert not _safetensors_file_ok(p)


# ── get_layer_path / list_layer_paths ────────────────────────────────────────

def test_get_layer_path_safetensors_extension(tmp_path: Path) -> None:
    p = get_layer_path(tmp_path, 0, shard_format="safetensors")
    assert p.suffix == ".safetensors"
    assert p.name == "layer_000.safetensors"


def test_get_layer_path_pt_extension(tmp_path: Path) -> None:
    p = get_layer_path(tmp_path, 0, shard_format="pt")
    assert p.suffix == ".pt"


def test_list_layer_paths_prefers_safetensors(tmp_path: Path) -> None:
    """list_layer_paths returns .safetensors files when both formats exist."""
    for i in range(3):
        _save_safetensors(_fp16_state(), tmp_path / f"layer_{i:03d}.safetensors")
        torch.save(_fp16_state(), str(tmp_path / f"layer_{i:03d}.pt"))
    paths = list_layer_paths(tmp_path)
    assert all(p.suffix == ".safetensors" for p in paths)
    assert len(paths) == 3


def test_list_layer_paths_falls_back_to_pt(tmp_path: Path) -> None:
    for i in range(2):
        torch.save(_fp16_state(), str(tmp_path / f"layer_{i:03d}.pt"))
    paths = list_layer_paths(tmp_path)
    assert all(p.suffix == ".pt" for p in paths)
    assert len(paths) == 2


# ── verify_shards with safetensors format ────────────────────────────────────

def test_verify_shards_ok_safetensors(tmp_path: Path) -> None:
    d = _minimal_shard_dir(tmp_path, shard_format="safetensors")
    report = verify_shards(d)
    assert report.ok, report.summary()


def test_verify_shards_ok_legacy_pt(tmp_path: Path) -> None:
    d = _minimal_shard_dir(tmp_path, shard_format="pt")
    report = verify_shards(d)
    assert report.ok, report.summary()


def test_verify_shards_detects_corrupt_safetensors(tmp_path: Path) -> None:
    d = _minimal_shard_dir(tmp_path, shard_format="safetensors")
    # Overwrite a layer shard with garbage.
    corrupt = d / "layer_000.safetensors"
    corrupt.write_bytes(b"NOT_SAFETENSORS")
    report = verify_shards(d)
    assert not report.ok
    assert "layer_000.safetensors" in report.corrupt
