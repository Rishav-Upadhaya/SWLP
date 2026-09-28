"""Tests for swlp.core.prefix_cache — interval-snapshot prefix KV reuse (Phase 26)."""
from __future__ import annotations

import torch

from swlp.core.prefix_cache import PrefixKVCache


def _kv(layers: int, seq: int, batch: int = 1, heads: int = 2, dim: int = 4):
    g = torch.Generator().manual_seed(7)
    return [
        (torch.randn(batch, heads, seq, dim, generator=g),
         torch.randn(batch, heads, seq, dim, generator=g))
        for _ in range(layers)
    ]


def _ids(n: int, start: int = 0) -> list[int]:
    return list(range(start, start + n))


def test_lookup_exact_longest_prefix():
    c = PrefixKVCache(max_entries=4)
    c.store(_ids(100), _kv(2, 100))
    hit = c.lookup(_ids(150))
    assert hit is not None and hit.length == 100
    # Slices must match the stored prefix.
    reference = _kv(2, 100)
    for (k, _v), (rk, _rv) in zip(hit.layer_kv, reference, strict=False):
        assert torch.equal(k, rk)


def test_lookup_misses_on_divergence():
    c = PrefixKVCache()
    c.store(_ids(100), _kv(2, 100))
    diverging = [0] * 100
    assert c.lookup(diverging) is None


def test_interior_snapshots_at_interval():
    c = PrefixKVCache(max_entries=16, snapshot_every=50)
    c.store(_ids(120), _kv(1, 120))
    # 50 and 100-token interior snapshots + the full 120 exist.
    assert c.lookup(_ids(60)) is not None and c.lookup(_ids(60)).length == 50
    assert c.lookup(_ids(110)) is not None and c.lookup(_ids(110)).length == 100
    assert c.lookup(_ids(120)).length == 120


def test_lru_eviction_keeps_recent():
    c = PrefixKVCache(max_entries=2)
    a = _ids(10, start=100)
    b = _ids(10, start=200)
    d = _ids(10, start=300)
    c.store(a, _kv(1, 10))
    c.store(b, _kv(1, 10))
    c.lookup(a)  # refresh entry A's recency
    c.store(d, _kv(1, 10))
    assert c.lookup(b) is None       # B was evicted (LRU)
    assert c.lookup(a) is not None
    assert c.lookup(d) is not None


def test_same_prefix_replaces_not_duplicates():
    c = PrefixKVCache(max_entries=8)
    c.store(_ids(50), _kv(1, 50))
    c.store(_ids(50), _kv(1, 50))
    assert len(c._entries) == 1


def test_seed_then_update_equals_full_update():
    """Seeded cache + suffix update == one full-length update (bit-exact)."""
    from transformers.cache_utils import DynamicCache, DynamicLayer

    layers_n, prefix_n, suffix_n = 2, 20, 5
    prefix_kv = _kv(layers_n, prefix_n)
    suffix_kv = _kv(layers_n, suffix_n, batch=1)

    seeded = DynamicCache()
    for _ in range(layers_n):
        seeded.layers.append(DynamicLayer())
    c = PrefixKVCache()
    c.seed_dynamic_cache(seeded, type("E", (), {
        "layer_kv": prefix_kv, "length": prefix_n})(), torch.device("cpu"))
    for layer, (k, v) in zip(seeded.layers, suffix_kv, strict=False):
        layer.update(k, v)

    full = DynamicCache()
    for _ in range(layers_n):
        full.layers.append(DynamicLayer())
    for layer, (pk, pv) in zip(full.layers, prefix_kv, strict=False):
        layer.update(pk, pv)
    for layer, (sk, sv) in zip(full.layers, suffix_kv, strict=False):
        layer.update(sk, sv)

    for ls, lf in zip(seeded.layers, full.layers, strict=False):
        assert torch.equal(ls.keys, lf.keys)
        assert torch.equal(ls.values, lf.values)
        assert ls.get_seq_length() == prefix_n + suffix_n


def test_extract_layers_clones_cpu():
    from transformers.cache_utils import DynamicCache, DynamicLayer

    cache = DynamicCache()
    k = torch.randn(1, 2, 10, 4)
    layer = DynamicLayer()
    layer.update(k, k.clone())
    cache.layers.append(layer)
    c = PrefixKVCache()
    out = c.extract_layers(cache, 1)
    assert out[0][0].device.type == "cpu"
    assert torch.equal(out[0][0], k)
    layer.keys = torch.zeros_like(layer.keys)
    assert not torch.equal(out[0][0], layer.keys)  # clone, not a view


def test_stats_and_clear():
    c = PrefixKVCache()
    c.store(_ids(30), _kv(1, 30))
    c.lookup(_ids(40))
    c.lookup([9] * 10)
    s = c.stats()
    assert s["hits"] == 1 and s["misses"] == 1 and s["entries"] >= 1
    c.clear()
    assert c.stats()["entries"] == 0


def test_max_bytes_budget_evicts_and_refuses():
    """Round-3 fix: the byte budget is enforced — oversize snapshots are
    refused, LRU eviction keeps total bytes under budget."""
    import torch

    cache = PrefixKVCache(max_entries=8, max_bytes=4096)
    # One layer-pair of [1,1,16,4] fp32 k+v = 512 B; ids must match KV length.
    small = [(torch.zeros(1, 1, 16, 4), torch.zeros(1, 1, 16, 4))]
    for i in range(12):  # 12 × 512 B > 4096 B budget → evictions must occur
        cache.store(list(range(i, i + 16)), small)
    stats = cache.stats()
    assert stats["bytes"] <= 4096, f"byte budget exceeded: {stats}"
    assert 0 < stats["entries"] <= 8


def test_replace_and_clear_accounting():
    """Round-3 fix: replacing an entry releases its old bytes; clear()
    resets the byte counter (previously leaked → phantom evictions)."""
    import torch

    kv = [(torch.zeros(1, 1, 16, 4), torch.zeros(1, 1, 16, 4))]
    ids = list(range(16))
    cache = PrefixKVCache(max_entries=8)
    cache.store(ids, kv)
    first = cache.stats()["bytes"]
    assert first > 0
    cache.store(ids, kv)  # identical prefix → replace, not accumulate
    assert cache.stats()["bytes"] == first, "replacement double-counted bytes"
    assert len(cache._entries) == 1
    cache.clear()
    assert cache.stats()["bytes"] == 0


def test_every_entry_ids_match_kv_length():
    """Round-3 invariant: a snapshot's id count must equal its KV length.
    A mismatch means ids were stored against misaligned KV (the post-prefix-
    hit store bug), which would silently corrupt later seeded runs."""
    import torch

    kv = [(torch.zeros(1, 2, 20, 4), torch.zeros(1, 2, 20, 4))]
    cache = PrefixKVCache(max_entries=8, snapshot_every=5)
    cache.store(list(range(20)), kv)
    for entry in cache._entries:
        kv_len = entry.layer_kv[0][0].shape[-2]
        assert entry.length == kv_len, (
            f"entry ids ({entry.length}) != KV length ({kv_len})"
        )


def test_oversize_snapshot_refused(caplog):
    """A single snapshot larger than max_bytes is refused loudly."""
    import torch

    cache = PrefixKVCache(max_entries=8, max_bytes=256)
    big_kv = [(torch.zeros(1, 1, 64, 8), torch.zeros(1, 1, 64, 8))]  # 4 KiB
    with caplog.at_level("WARNING"):
        cache.store(list(range(64)), big_kv)
    assert any("prefix_store_too_large" in r.message for r in caplog.records)
    assert not cache._entries


def test_ids_shorter_than_kv_refused(caplog):
    """Ids that don't cover their KV would mis-seed later turns — refused."""
    import torch

    cache = PrefixKVCache(max_entries=8)
    kv = [(torch.zeros(1, 1, 20, 4), torch.zeros(1, 1, 20, 4))]
    with caplog.at_level("WARNING"):
        cache.store([1, 2, 3], kv)
    assert any("prefix_store_ids_shorter_than_kv" in r.message for r in caplog.records)
    assert not cache._entries
