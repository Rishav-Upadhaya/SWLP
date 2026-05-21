"""KV cache with per-layer compression for Llama/Mistral-family models.

Phase 2: a transformers ``Cache`` that keeps cold layers' KV compressed in host
RAM and decompresses on demand, so a model whose full KV cache would not fit in
16 GB can still run.

Design
------
``SWLPRunner`` computes one transformer block at a time. Within a single token
step, layer N's KV is written once (by its attention's ``cache.update``) and not
read again until the *next* step reaches layer N. That gap — a full sweep across
the other layers — is when layer N's KV can sit compressed.

- ``CompressedDynamicLayer`` extends transformers' ``DynamicLayer``. While *cold*
  it holds no tensors; ``get_seq_length`` still answers from a recorded length so
  attention-mask construction works without decompressing. ``update`` transparently
  decompresses first.
- ``CompressedDynamicCache`` extends ``DynamicCache``, swapping in compressed
  layers and exposing ``compress_layer(idx)`` for the runner to call after each
  block. All compression/offload bookkeeping is delegated to ``KVCacheManager``,
  so its zlib path and stats (ratio, peak bytes) are reused unchanged.

The compression is **lossless** (zlib over exact tensor bytes) — decompressed KV
is bit-identical, so generation output is unchanged.
"""
from __future__ import annotations

import logging

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer

from .kv_cache import KVCacheManager

LOGGER = logging.getLogger(__name__)


class CompressedDynamicLayer(DynamicLayer):
    """A DynamicLayer whose cold KV lives in a KVCacheManager (compressed)."""

    def __init__(self, layer_idx: int, kv_manager: KVCacheManager) -> None:
        super().__init__()
        self._layer_idx = layer_idx
        self._kv_manager = kv_manager
        self._cold = False
        self._cold_seq_len = 0

    def compress(self) -> None:
        """Hand this layer's KV to the manager (offload + zlib) and free locally."""
        if self._cold or not self.is_initialized:
            return
        if self.keys is None or self.keys.numel() == 0:
            return
        seq_len = int(self.keys.shape[-2])
        # Phase 16: if kv_window is active the manager will trim the tensors;
        # record the trimmed length so get_seq_length() stays accurate.
        w = self._kv_manager.kv_window
        self._cold_seq_len = min(seq_len, w) if w > 0 else seq_len
        self._kv_manager.set(self._layer_idx, (self.keys, self.values))
        self._kv_manager.maybe_evict(self._layer_idx)
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self._cold = True

    def decompress(self) -> None:
        """Restore this layer's KV from the manager (decompress + move to device)."""
        if not self._cold:
            return
        restored = self._kv_manager.get(self._layer_idx, self.device)
        if restored is not None:
            self.keys, self.values = restored
        self._cold = False

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cold:
            self.decompress()
        return super().update(key_states, value_states, *args, **kwargs)

    def get_seq_length(self) -> int:
        if self._cold:
            return self._cold_seq_len
        return super().get_seq_length()

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        kv_length = self.get_seq_length() + query_length
        return kv_length, 0


class CompressedDynamicCache(DynamicCache):
    """DynamicCache whose layers compress cold KV through a shared KVCacheManager."""

    def __init__(self, config, kv_manager: KVCacheManager) -> None:
        super().__init__(config=config)
        self._kv_manager = kv_manager
        # Replace the config-derived layers with compression-aware variants.
        # Sliding-window cropping is intentionally not modelled — exact for any
        # context shorter than the model's sliding window.
        self.layers = [
            CompressedDynamicLayer(idx, kv_manager) for idx in range(len(self.layers))
        ]

    def compress_layer(self, layer_idx: int) -> None:
        if 0 <= layer_idx < len(self.layers):
            self.layers[layer_idx].compress()

    def compress_all(self) -> None:
        for layer in self.layers:
            layer.compress()

    @property
    def kv_manager(self) -> KVCacheManager:
        return self._kv_manager
