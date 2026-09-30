"""Tests for the safetensors shard format.

Covers:
- _save_safetensors writes a valid .safetensors file (plain FP16)
- _load_safetensors_shard round-trips plain FP16 state
- _safetensors_file_ok integrity check
- verify_shards detects corrupt safetensors
- get_layer_path uses correct extension per shard_format
- list_layer_paths prefers .safetensors over .pt
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from swlp import codec
from swlp.core.streaming import _load_safetensors_shard
from swlp.model.shard import (
    ShardManifest,
    _safetensors_file_ok,
    _save_safetensors,
    _write_manifest,
    compress_shards,
    decompress_shards,
    get_layer_path,
    list_layer_paths,
    load_manifest,
    verify_shards,
)

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


# ── Phase 22: compressed .swz shards ─────────────────────────────────────────

def test_compress_shards_bit_exact_roundtrip(tmp_path: Path) -> None:
    """compress_shards replaces layers with .swz; loads are bit-identical."""
    d = _minimal_shard_dir(tmp_path)
    before = {
        i: _load_safetensors_shard(d / f"layer_{i:03d}.safetensors") for i in range(2)
    }
    manifest = compress_shards(d)
    assert manifest.shard_compression == "swz"
    for i in range(2):
        assert not (d / f"layer_{i:03d}.safetensors").exists()
        swz = d / f"layer_{i:03d}.safetensors.swz"
        assert swz.exists()
        after = _load_safetensors_shard(swz)
        for k in before[i]:
            assert torch.equal(after[k], before[i][k])


def test_compress_shards_updates_manifest_on_disk(tmp_path: Path) -> None:
    d = _minimal_shard_dir(tmp_path)
    compress_shards(d)
    reloaded = json.loads((d / "shard_manifest.json").read_text())
    assert reloaded["shard_compression"] == "swz"


def test_compress_shards_resumes_after_interrupt(tmp_path: Path) -> None:
    """A partially converted dir (layer 0 already .swz) completes cleanly."""
    d = _minimal_shard_dir(tmp_path)
    src = d / "layer_000.safetensors"
    codec.compressed_path(src).write_bytes(codec.compress_bytes(src.read_bytes()))
    src.unlink()
    manifest = compress_shards(d)
    assert manifest.shard_compression == "swz"
    assert sorted(p.name for p in d.glob("layer_*")) == [
        "layer_000.safetensors.swz",
        "layer_001.safetensors.swz",
    ]


def test_compress_shards_rejects_pt_format(tmp_path: Path) -> None:
    d = _minimal_shard_dir(tmp_path, shard_format="pt")
    with pytest.raises(ValueError, match="safetensors"):
        compress_shards(d)


def test_verify_shards_accepts_compressed_dir(tmp_path: Path) -> None:
    d = _minimal_shard_dir(tmp_path)
    compress_shards(d)
    assert verify_shards(d).ok


def test_verify_shards_accepts_mixed_dir(tmp_path: Path) -> None:
    """Mid-conversion dirs (manifest still says "none") verify fine."""
    d = _minimal_shard_dir(tmp_path)
    src = d / "layer_000.safetensors"
    codec.compressed_path(src).write_bytes(codec.compress_bytes(src.read_bytes()))
    src.unlink()
    assert verify_shards(d).ok


def test_verify_shards_detects_corrupt_swz(tmp_path: Path) -> None:
    d = _minimal_shard_dir(tmp_path)
    compress_shards(d)
    (d / "layer_000.safetensors.swz").write_bytes(b"garbage-not-a-container")
    report = verify_shards(d)
    assert not report.ok
    assert "layer_000.safetensors.swz" in report.corrupt


def test_get_layer_path_with_compression(tmp_path: Path) -> None:
    p = get_layer_path("/x", 3, shard_format="safetensors", shard_compression="swz")
    assert p.name == "layer_003.safetensors.swz"
    p = get_layer_path("/x", 3, shard_format="safetensors")
    assert p.name == "layer_003.safetensors"


def test_list_layer_paths_finds_swz(tmp_path: Path) -> None:
    d = _minimal_shard_dir(tmp_path)
    compress_shards(d)
    paths = list_layer_paths(d)
    assert [p.suffix for p in paths] == [".swz", ".swz"]


# ── Phase 22: decompress_shards (revert) ─────────────────────────────────────

def test_decompress_shards_restores_bit_exact_files(tmp_path: Path) -> None:
    """compress → revert restores byte-identical .safetensors files."""
    d = _minimal_shard_dir(tmp_path)
    original = {i: (d / f"layer_{i:03d}.safetensors").read_bytes() for i in range(2)}
    compress_shards(d)
    manifest = decompress_shards(d)
    assert manifest.shard_compression == "none"
    for i in range(2):
        restored = d / f"layer_{i:03d}.safetensors"
        assert restored.read_bytes() == original[i]
        assert not codec.compressed_path(restored).exists()
    reloaded = json.loads((d / "shard_manifest.json").read_text())
    assert reloaded["shard_compression"] == "none"
    assert verify_shards(d).ok


def test_decompress_shards_resumes_after_interrupt(tmp_path: Path) -> None:
    """A partially reverted dir (layer 0 already plain) completes cleanly."""
    d = _minimal_shard_dir(tmp_path)
    compress_shards(d)
    swz = d / "layer_000.safetensors.swz"
    plain = d / "layer_000.safetensors"
    plain.write_bytes(codec.decompress_bytes(swz.read_bytes()))
    swz.unlink()
    manifest = decompress_shards(d)
    assert manifest.shard_compression == "none"
    assert sorted(p.name for p in d.glob("layer_*")) == [
        "layer_000.safetensors",
        "layer_001.safetensors",
    ]


def test_decompress_shards_drops_leftover_swz(tmp_path: Path) -> None:
    """A run killed between rename and unlink leaves both files; revert cleans up."""
    d = _minimal_shard_dir(tmp_path)
    compress_shards(d)
    swz = d / "layer_000.safetensors.swz"
    (d / "layer_000.safetensors").write_bytes(codec.decompress_bytes(swz.read_bytes()))
    decompress_shards(d)  # swz still on disk next to a valid plain file
    assert not swz.exists()
    assert verify_shards(d).ok


def test_decompress_shards_rejects_pt_format(tmp_path: Path) -> None:
    d = _minimal_shard_dir(tmp_path, shard_format="pt")
    with pytest.raises(ValueError, match="safetensors"):
        decompress_shards(d)


# ── Phase 25: MoE expert banks ───────────────────────────────────────────────

def _moe_shard_dir(tmp_path: Path, num_layers: int = 2, with_banks: bool = True) -> Path:
    d = tmp_path / "moe_shards"
    d.mkdir()
    torch.save({"embed_tokens": {"weight": torch.zeros(4, 4)}}, str(d / "embed.pt"))
    torch.save({"weight": torch.zeros(4, 4)}, str(d / "lm_head.pt"))
    for i in range(num_layers):
        _save_safetensors(_fp16_state(), d / f"layer_{i:03d}.safetensors")
        if with_banks:
            _save_safetensors(
                {
                    "mlp.experts.gate_up_proj": torch.randn(4, 8, 4, dtype=torch.float16),
                    "mlp.experts.down_proj": torch.randn(4, 4, 4, dtype=torch.float16),
                },
                d / f"layer_{i:03d}.experts.safetensors",
            )
    manifest = ShardManifest(
        model_id="moe-test", num_layers=num_layers, layer_weight_mb=0.01,
        total_weight_mb=0.02, embed_file="embed.pt", lm_head_file="lm_head.pt",
        model_type="qwen3_moe", shard_format="safetensors",
        num_experts=4, top_k=2, expert_bank=with_banks,
    )
    _write_manifest(d, manifest)
    return d


def test_moe_manifest_roundtrip_new_fields(tmp_path: Path) -> None:
    d = _moe_shard_dir(tmp_path)
    m = load_manifest(d)
    assert m.num_experts == 4
    assert m.top_k == 2
    assert m.expert_bank is True


def test_manifest_back_compat_defaults(tmp_path: Path) -> None:
    """Old manifests without MoE fields load with zeroed defaults."""
    d = _minimal_shard_dir(tmp_path)
    raw = json.loads((d / "shard_manifest.json").read_text())
    for field in ("num_experts", "top_k", "expert_bank"):
        raw.pop(field, None)  # simulate a pre-Phase-25 manifest
    (d / "shard_manifest.json").write_text(json.dumps(raw))
    m = load_manifest(d)
    assert m.num_experts == 0 and m.top_k == 0 and m.expert_bank is False


def test_verify_shards_accepts_expert_banks(tmp_path: Path) -> None:
    d = _moe_shard_dir(tmp_path)
    report = verify_shards(d)
    assert report.ok, report.summary()


def test_verify_shards_detects_corrupt_bank(tmp_path: Path) -> None:
    d = _moe_shard_dir(tmp_path)
    (d / "layer_001.experts.safetensors").write_bytes(b"garbage")
    report = verify_shards(d)
    assert not report.ok
    assert "layer_001.experts.safetensors" in report.corrupt


def test_verify_shards_accepts_v1_moe_without_banks(tmp_path: Path) -> None:
    """expert_bank=True but experts live inside the layer file (v1 dirs)."""
    d = _moe_shard_dir(tmp_path, with_banks=False)
    state = _fp16_state()
    state["mlp.experts.gate_up_proj"] = torch.randn(4, 8, 4, dtype=torch.float16)
    state["mlp.experts.down_proj"] = torch.randn(4, 4, 4, dtype=torch.float16)
    _save_safetensors(state, d / "layer_000.safetensors")
    _save_safetensors(state, d / "layer_001.safetensors")
    report = verify_shards(d)
    assert report.ok, report.summary()
