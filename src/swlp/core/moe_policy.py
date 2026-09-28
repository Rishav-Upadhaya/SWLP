"""Pure policy functions for MoE expert streaming (Phase 25).

No I/O, no torch — deterministic math and bookkeeping only, so everything
here is trivially unit-testable:

- :class:`RoutingHistory` — the frequency window that drives predictive
  expert prefetch. (FreeToken's ``q*`` bandwidth-adaptive miss split was
  removed with the CUDA path: it divides work between a host bus and a
  device bus, and unified memory has only one. Hit rate, not placement, *q*
  of them over the transfer bus while the remaining ``m − q`` execute where
  they already live balances finish times when ``q* ≈ m·B_P/(B_P + B_H)``
  with both bandwidths measured on the host machine. On Apple unified
  memory there is no second execution site, so the split degenerates to
  "transfer all" — the function stays the single source of the formula.
- :class:`RoutingHistory` — prior-token expert prediction. Apple's SpecMD
  study and MoE-SpeQ both find expert access does not follow clean LRU
  temporal locality, but the experts a layer routes to repeat heavily across
  adjacent tokens; a per-layer frequency window over recent tokens is the
  cheapest predictor that beats pure LRU.
"""
from __future__ import annotations

from collections import Counter, deque


class RoutingHistory:
    """Per-layer frequency window over recently routed expert ids.

    ``record(layer, ids)`` after each layer compute; ``predict(layer, n)``
    returns the *n* most frequent experts of that layer's last ``window``
    observations, most recent first on ties. Deterministic and O(window).
    """

    def __init__(self, window: int = 32) -> None:
        self._window = max(1, int(window))
        self._layers: dict[int, deque[frozenset[int]]] = {}
        self._last: dict[int, frozenset[int]] = {}

    def record(self, layer: int, expert_ids) -> None:
        ids = frozenset(int(i) for i in expert_ids)
        dq = self._layers.setdefault(layer, deque(maxlen=self._window))
        dq.append(ids)
        self._last[layer] = ids

    def predict(self, layer: int, n: int) -> list[int]:
        """Predicted experts for ``layer``'s next activation, best first.

        The immediately-preceding observation's set leads (token-to-token
        routing repetition is the strongest cheap signal — the
        Mixtral-offloading prefetcher reached 80–90% on it alone); the
        window's frequency ranking, recency tie-broken, fills the rest.
        """
        dq = self._layers.get(layer)
        if not dq or n <= 0:
            return []
        counts: Counter[int] = Counter()
        last_seen: dict[int, int] = {}
        for age, ids in enumerate(reversed(dq)):
            for eid in ids:
                counts[eid] += 1
                last_seen.setdefault(eid, age)
        ranked = sorted(counts, key=lambda e: (-counts[e], last_seen[e], e))
        ordered = list(dict.fromkeys([*sorted(self._last.get(layer, frozenset())), *ranked]))
        return ordered[:n]

    def last(self, layer: int) -> frozenset[int]:
        return self._last.get(layer, frozenset())

    def reset(self) -> None:
        self._layers.clear()
        self._last.clear()
