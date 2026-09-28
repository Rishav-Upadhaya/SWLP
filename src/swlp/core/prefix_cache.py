"""Prefix KV caching (Phase 26, FreeToken-inspired).

Every chat/serve turn re-feeds the whole conversation, so each turn's prefill
re-streams the model for the full history. Greedy decoding is deterministic:
identical token prefixes produce bitwise-identical KV. Caching the KV of
previous prefixes therefore skips those prefill sweeps with **zero quality
change** — the lossless analogue of FreeToken's semantic-aware anchor
checkpoints (arXiv:2608.16157). Each stored turn produces one full-length
snapshot plus interior interval snapshots (``snapshot_every``), so later
turns that share only a prefix of a long turn still hit.

v1 keeps at most ``max_entries`` snapshots under a ``max_bytes`` host-RAM
budget (LRU) — simple, bounded, and the right first cut for interactive
sessions where a handful of turn snapshots dominate. Block-hash radix
sharing is a follow-up once multi-tenant serving exists.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import torch

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PrefixEntry:
    """One cached prefix: token ids plus the per-layer KV it produced."""

    ids: list[int]
    layer_kv: list[tuple[torch.Tensor, torch.Tensor]]  # CPU tensors [B, H, S, D]
    bytes: int
    created: float = field(default_factory=time.monotonic)

    @property
    def length(self) -> int:
        return len(self.ids)


class PrefixKVCache:
    """Bounded LRU of exact-match prefix KV snapshots.

    Bounded in BOTH entry count and bytes: ``max_bytes`` (default 2 GiB)
    caps host RAM — a single snapshot larger than the budget is refused.
    Snapshots are taken per stored sequence (one per chat turn in practice)
    plus interior interval snapshots every ``snapshot_every`` tokens, so
    later turns sharing only a prefix of a long turn still hit.
    """

    def __init__(self, max_entries: int = 8, snapshot_every: int = 512,
                 max_bytes: int = 2 * 1024 ** 3) -> None:
        self.max_entries = max(1, int(max_entries))
        # Extra snapshot interval inside one stored sequence (tokens).
        self.snapshot_every = max(1, int(snapshot_every))
        self.max_bytes = max(1, int(max_bytes))
        self._bytes = 0
        self._entries: list[PrefixEntry] = []
        self._hits = 0
        self._misses = 0
        self._stored = 0

    # ── lookup / store ────────────────────────────────────────────────────

    def lookup(
        self,
        ids: list[int],
        min_length: int = 0,
        max_length: int | None = None,
    ) -> PrefixEntry | None:
        """Longest cached prefix of ``ids`` (exact token match), or None.

        ``min_length``/``max_length`` bound the usable match (caller policy,
        e.g. skip trivial or full-length hits); entries rejected by the
        bounds count as misses, not hits.
        """
        best: PrefixEntry | None = None
        for entry in self._entries:
            n = entry.length
            if n > len(ids) or n == 0:
                continue
            if max_length is not None and n >= max_length:
                continue
            if entry.ids == ids[:n] and (best is None or n > best.length):
                best = entry
        if best is not None and best.length < max(1, min_length):
            best = None
        if best is not None:
            best.created = time.monotonic()
            self._hits += 1
        else:
            self._misses += 1
        return best

    def store(
        self,
        ids: list[int],
        layer_kv: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Snapshot ``ids``/KV, plus truncated snapshots at fixed intervals.

        ``layer_kv`` holds CPU tensors covering exactly ``len(ids)`` tokens.
        Interior snapshots every ``snapshot_every`` tokens are slices of the
        same tensors, so later turns with shorter shared history still hit.

        Layers with diverging sequence lengths (e.g. after an early exit)
        would seed a corrupt cache — refuse the whole store loudly instead.
        """
        if not ids or not layer_kv:
            return
        lengths = {int(k.shape[-2]) for k, _v in layer_kv}
        if len(lengths) != 1:
            LOGGER.warning(
                "prefix_store_layer_length_mismatch",
                extra={"lengths": sorted(lengths), "ids": len(ids)},
            )
            return
        seq_len = lengths.pop()
        if seq_len < len(ids):
            # KV shorter than ids (trailing token not yet fed): trim ids so
            # the snapshot stays exactly aligned with the cached KV.
            ids = ids[:seq_len]
        elif seq_len > len(ids):
            # KV longer than ids cannot be labeled truthfully — a snapshot
            # whose ids don't cover its KV would mis-seed later turns.
            LOGGER.warning(
                "prefix_store_ids_shorter_than_kv",
                extra={"ids": len(ids), "kv": seq_len},
            )
            return
        stored = 0
        for n in sorted({seq_len, *range(self.snapshot_every, seq_len, self.snapshot_every)}):
            if n <= 0 or n > seq_len:
                continue
            if self._insert(ids[:n], [
                (k[..., :n, :].clone(), v[..., :n, :].clone()) for k, v in layer_kv
            ]):
                stored += 1
        self._stored += stored

    def _insert(self, ids: list[int], layer_kv: list[tuple[torch.Tensor, torch.Tensor]]) -> bool:
        nbytes = sum(k.numel() * k.element_size() + v.numel() * v.element_size()
                     for k, v in layer_kv)
        if nbytes > self.max_bytes:
            LOGGER.warning(
                "prefix_store_too_large",
                extra={"bytes": nbytes, "budget": self.max_bytes},
            )
            return False
        # Replace an identical-length entry for the same ids instead of
        # growing the list with stale duplicates of one prefix — and release
        # the replaced entry's bytes first or the budget double-counts.
        replaced = [e for e in self._entries if e.length == len(ids) and e.ids == ids]
        if replaced:
            self._entries = [e for e in self._entries if e not in replaced]
            self._bytes -= sum(e.bytes for e in replaced)
        self._entries.append(PrefixEntry(ids=ids, layer_kv=layer_kv, bytes=nbytes))
        self._bytes += nbytes
        self._entries.sort(key=lambda e: e.length)
        # Enforce both bounds, evicting least-recently-created first.
        while len(self._entries) > self.max_entries or self._bytes > self.max_bytes:
            victim = min(self._entries, key=lambda e: e.created)
            self._entries.remove(victim)
            self._bytes -= victim.bytes
            if not self._entries:
                break
        return True

    # ── transformers integration ──────────────────────────────────────────

    def seed_dynamic_cache(self, past_state: object, entry: PrefixEntry,
                           device: torch.device) -> None:
        """Seed a (possibly empty) ``DynamicCache`` with a cached prefix."""
        from transformers.cache_utils import DynamicLayer

        layers = getattr(past_state, "layers", None)
        if layers is None:
            raise TypeError("prefix seeding requires a DynamicCache-style past state")
        while len(layers) < len(entry.layer_kv):
            layers.append(DynamicLayer())
        for layer, (k, v) in zip(layers, entry.layer_kv, strict=True):
            kd, vd = k.to(device), v.to(device)
            layer.keys = kd
            layer.values = vd
            layer.dtype = kd.dtype
            layer.device = kd.device
            layer.is_initialized = True

    def extract_layers(self, past_state: object, num_layers: int
                       ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Pull per-layer KV off a DynamicCache as CPU clones."""
        layers = getattr(past_state, "layers", [])
        out: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in layers[:num_layers]:
            k = layer.keys.detach().to("cpu", copy=True)
            v = layer.values.detach().to("cpu", copy=True)
            out.append((k, v))
        return out

    # ── bookkeeping ───────────────────────────────────────────────────────

    def clear(self) -> None:
        self._entries.clear()
        self._bytes = 0

    def stats(self) -> dict[str, int | float]:
        return {
            "entries": len(self._entries),
            "bytes": sum(e.bytes for e in self._entries),
            "hits": self._hits,
            "misses": self._misses,
            "stored": self._stored,
            "hit_rate": (self._hits / (self._hits + self._misses))
            if (self._hits + self._misses) else 0.0,
        }
