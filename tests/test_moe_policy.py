"""Tests for swlp.core.moe_policy — q* split and routing history (Phase 25)."""
from __future__ import annotations

from swlp.core.moe_policy import RoutingHistory


def test_history_predicts_frequent_ids():
    h = RoutingHistory(window=8)
    for _ in range(4):
        h.record(0, [1, 2])
    h.record(0, [3])
    # Last observation's set leads (token-to-token repetition signal), then
    # window frequency: 3 is most recent, then frequent 1 and 2.
    assert h.predict(0, 2) == [3, 1]
    assert h.predict(0, 3) == [3, 1, 2]


def test_history_last_set_leads_prediction():
    """Token-to-token repetition is the strongest cheap prefetch signal —
    the last routed set must occupy the top prediction slots."""
    h = RoutingHistory(window=8)
    for _ in range(6):
        h.record(0, [5, 9])  # dominant frequency
    h.record(0, [7])         # one-off, but MOST RECENT
    top = h.predict(0, 2)
    assert top[0] == 7       # recency leads despite low frequency
    assert 5 in top          # frequency fills the remaining slot


def test_history_recency_breaks_ties():
    h = RoutingHistory(window=8)
    h.record(0, [1])
    h.record(0, [2])
    h.record(0, [1])  # 1 seen more recently than 2 (both count 1... 1 counts 2)
    assert h.predict(0, 1) == [1]


def test_history_window_forgets_old_routes():
    h = RoutingHistory(window=2)
    h.record(0, [9])
    h.record(0, [7])
    h.record(0, [5])  # 9 fell out of the 2-observation window
    assert 9 not in h.predict(0, 10)


def test_history_empty_layer_predicts_nothing():
    h = RoutingHistory()
    assert h.predict(3, 4) == []
    assert h.last(3) == frozenset()


def test_history_reset():
    h = RoutingHistory()
    h.record(0, [1])
    h.reset()
    assert h.predict(0, 4) == []
