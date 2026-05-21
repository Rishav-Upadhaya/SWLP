"""Tests for swlp.model.sparse — COO sparse weight codec."""
from pathlib import Path

import pytest
import torch

from swlp.model.sparse import (
    decode_sparse,
    encode_sparse,
    is_sparse_encoded,
    sparsify_shards,
    sparsity,
)


def test_sparsity_all_zeros() -> None:
    t = torch.zeros(4, 4)
    assert sparsity(t) == 1.0


def test_sparsity_no_zeros() -> None:
    t = torch.ones(4, 4)
    assert sparsity(t) == 0.0


def test_sparsity_half() -> None:
    t = torch.zeros(4, 4)
    t[:2, :] = 1.0
    assert sparsity(t) == pytest.approx(0.5)  # noqa: PT018


def test_encode_sparse_round_trip_bit_exact() -> None:
    """encode then decode → bit-identical tensor."""
    # Create a tensor that is 75% zero.
    t = torch.zeros(8, 8)
    t[0, :2] = 1.0
    t[1, 3] = -2.5
    state = {"weight": t}
    encoded = encode_sparse(state, threshold=0.5)
    assert is_sparse_encoded(encoded)
    decoded = decode_sparse(encoded)
    assert "weight" in decoded
    assert torch.equal(decoded["weight"], t)


def test_decode_sparse_noop_on_dense_dict() -> None:
    """decode_sparse on a plain dense dict returns the same dict unchanged."""
    t = torch.randn(4, 4)
    d = {"weight": t, "bias": torch.zeros(4)}
    result = decode_sparse(d)
    assert result is d  # same object, not a copy


def test_encode_below_threshold_stays_dense() -> None:
    """Tensors below the sparsity threshold are not encoded."""
    t = torch.randn(4, 4)  # near-zero sparsity
    state = {"weight": t}
    encoded = encode_sparse(state, threshold=0.5)
    assert not is_sparse_encoded(encoded)
    assert "weight" in encoded
    assert torch.equal(encoded["weight"], t)


def test_encode_1d_tensors_always_dense() -> None:
    """1-D bias vectors are never COO-encoded (too small to benefit)."""
    t = torch.zeros(16)  # 100% sparse but 1-D
    state = {"bias": t}
    encoded = encode_sparse(state, threshold=0.0)
    assert not is_sparse_encoded(encoded)
    assert "bias" in encoded


def test_is_sparse_encoded_detects_keys() -> None:
    d_dense = {"weight": torch.randn(4, 4)}
    t = torch.zeros(4, 4)
    t[0, 0] = 1.0
    d_sparse = encode_sparse({"weight": t}, threshold=0.0)
    assert not is_sparse_encoded(d_dense)
    assert is_sparse_encoded(d_sparse)


def test_sparsify_shards_round_trip(tmp_path: Path) -> None:
    """sparsify_shards + decode_sparse → bit-identical state dicts."""

    shard_dir = tmp_path / "dense"
    sparse_dir = tmp_path / "sparse"
    shard_dir.mkdir()

    # Write a shard with 75%-zero weight.
    w = torch.zeros(8, 8)
    w[0, :2] = 1.0
    b = torch.zeros(8)
    torch.save({"weight": w, "bias": b}, str(shard_dir / "layer_000.pt"))

    summary = sparsify_shards(shard_dir, sparse_dir, threshold=0.5)
    assert summary["tensors_sparsified"] >= 1

    # Load the sparse shard and decode.
    enc = torch.load(str(sparse_dir / "layer_000.pt"), weights_only=True)
    assert is_sparse_encoded(enc)
    dec = decode_sparse(enc)
    assert torch.equal(dec["weight"], w)
    assert torch.equal(dec["bias"], b)  # bias kept dense
