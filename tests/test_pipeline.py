"""Tests for swlp.core.pipeline — ThreadedPipeline with F_NOCACHE direct I/O."""
from __future__ import annotations

from pathlib import Path

import torch

from swlp.core.pipeline import PipelineConfig, ThreadedPipeline, _read_file_nocache


def _write_pt_shards(shard_dir: Path, num_layers: int = 4, hidden: int = 4) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_layers):
        state = {
            "weight": torch.full((hidden, hidden), float(i)),
            "bias": torch.zeros(hidden),
        }
        torch.save(state, str(shard_dir / f"layer_{i:03d}.pt"))


# ── _read_file_nocache (pipeline module copy) ─────────────────────────────────


def test_pipeline_read_file_nocache(tmp_path: Path) -> None:
    payload = b"pipeline-nocache-" * 500
    f = tmp_path / "data.bin"
    f.write_bytes(payload)
    assert _read_file_nocache(f) == payload


# ── ThreadedPipeline ──────────────────────────────────────────────────────────


def test_pipeline_get_layer_returns_weights(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    _write_pt_shards(shard_dir, num_layers=4)
    cfg = PipelineConfig(shard_dir=str(shard_dir), window_size=2)
    pipeline = ThreadedPipeline(cfg)
    pipeline.warmup(num_layers=4)

    weights = pipeline.get_layer(0)
    assert weights is not None
    # Layer 0 was written with weight filled with 0.0.
    assert torch.allclose(weights["weight"], torch.zeros(4, 4))


def test_pipeline_prefetch_then_get(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    _write_pt_shards(shard_dir, num_layers=4)
    cfg = PipelineConfig(shard_dir=str(shard_dir), window_size=2)
    pipeline = ThreadedPipeline(cfg)

    pipeline.prefetch(1)
    weights = pipeline.get_layer(1)
    assert weights is not None
    # Layer 1 written with weight filled with 1.0.
    assert torch.allclose(weights["weight"], torch.ones(4, 4))


def test_pipeline_evict_frees_layer(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    _write_pt_shards(shard_dir)
    cfg = PipelineConfig(shard_dir=str(shard_dir))
    pipeline = ThreadedPipeline(cfg)

    pipeline.get_layer(0)
    assert pipeline.resident_count() == 1
    pipeline.evict(0)
    assert pipeline.resident_count() == 0


def test_pipeline_missing_shard_returns_none(tmp_path: Path) -> None:
    """Missing shard logs a warning and leaves the slot empty."""
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    cfg = PipelineConfig(shard_dir=str(shard_dir))
    pipeline = ThreadedPipeline(cfg)

    weights = pipeline.get_layer(99)
    assert weights is None


def test_pipeline_cleanup_clears_window(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    _write_pt_shards(shard_dir, num_layers=2)
    cfg = PipelineConfig(shard_dir=str(shard_dir))
    pipeline = ThreadedPipeline(cfg)

    pipeline.get_layer(0)
    pipeline.get_layer(1)
    assert pipeline.resident_count() == 2
    pipeline.cleanup()
    assert pipeline.resident_count() == 0
