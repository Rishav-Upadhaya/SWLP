"""Tests for swlp.core.compressed_cache and KV budget recommendation."""
from __future__ import annotations

import torch
from transformers import MistralConfig

from swlp.core.compressed_cache import CompressedDynamicCache, CompressedDynamicLayer
from swlp.core.kv_cache import KVCacheManager
from swlp.hardware.detect import HardwareInfo, kv_budget_recommendation


def _tiny_config(num_layers: int = 3) -> MistralConfig:
    return MistralConfig(
        vocab_size=64,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=num_layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        sliding_window=None,
    )


def _manager() -> KVCacheManager:
    return KVCacheManager(
        budget_bytes=4 * 1024 * 1024,
        compression=True,
        compression_level=6,
        tiering=True,
        device=torch.device("cpu"),
    )


def test_compressed_cache_builds_compressed_layers() -> None:
    cache = CompressedDynamicCache(_tiny_config(num_layers=4), _manager())
    assert len(cache.layers) == 4
    assert all(isinstance(layer, CompressedDynamicLayer) for layer in cache.layers)


def test_compress_decompress_round_trip_is_lossless() -> None:
    cache = CompressedDynamicCache(_tiny_config(), _manager())
    keys = torch.randn(1, 2, 5, 4, dtype=torch.float16)
    values = torch.randn(1, 2, 5, 4, dtype=torch.float16)

    cache.layers[0].update(keys, values, layer_idx=0)
    assert cache.get_seq_length(0) == 5

    cache.compress_layer(0)
    assert cache.layers[0]._cold is True
    # get_seq_length must answer while compressed (mask construction needs it).
    assert cache.get_seq_length(0) == 5

    # Next step appends one token — update auto-decompresses first.
    new_k = torch.randn(1, 2, 1, 4, dtype=torch.float16)
    new_v = torch.randn(1, 2, 1, 4, dtype=torch.float16)
    out_k, out_v = cache.layers[0].update(new_k, new_v, layer_idx=0)

    assert cache.layers[0]._cold is False
    assert out_k.shape[-2] == 6
    # The first 5 positions must be bit-identical to the pre-compression KV.
    assert torch.equal(out_k[:, :, :5, :], keys)
    assert torch.equal(out_v[:, :, :5, :], values)


def test_compress_all_marks_every_layer_cold() -> None:
    cache = CompressedDynamicCache(_tiny_config(num_layers=3), _manager())
    for idx in range(3):
        k = torch.randn(1, 2, 2, 4, dtype=torch.float16)
        cache.layers[idx].update(k, k.clone(), layer_idx=idx)
    cache.compress_all()
    assert all(layer._cold for layer in cache.layers)


def test_compress_noop_on_uninitialized_layer() -> None:
    cache = CompressedDynamicCache(_tiny_config(), _manager())
    # Never updated — compress must be a safe no-op.
    cache.compress_layer(0)
    assert cache.layers[0]._cold is False


def test_kv_budget_recommendation_scales_with_memory() -> None:
    small = HardwareInfo("mps", True, 8.0, 6.9, "torch", "M-test")
    large = HardwareInfo("mps", True, 64.0, 6.9, "torch", "M-test")
    small_budget = kv_budget_recommendation(small, window_size=4, layer_weight_mb=436.0,
                                            num_layers=32)
    large_budget = kv_budget_recommendation(large, window_size=4, layer_weight_mb=436.0,
                                            num_layers=32)
    assert large_budget > small_budget
    assert small_budget >= 256  # always a usable floor


def test_kv_budget_recommendation_floor() -> None:
    tiny = HardwareInfo("cpu", False, 2.0, 3.5, "torch", "cpu-test")
    budget = kv_budget_recommendation(tiny, window_size=6, layer_weight_mb=750.0,
                                      num_layers=60)
    assert budget == 256  # headroom exhausted -> floored
