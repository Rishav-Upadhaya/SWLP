"""Tests for scripts/research/bench_common.py — multi-run statistics."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_BENCH_COMMON_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "research" / "bench_common.py"
)
_spec = importlib.util.spec_from_file_location("bench_common", _BENCH_COMMON_PATH)
assert _spec is not None and _spec.loader is not None
bench_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench_common)

summarize_runs = bench_common.summarize_runs


def test_summarize_runs_empty() -> None:
    s = summarize_runs([])
    assert s == {"n": 0, "median": 0.0, "iqr_low": 0.0, "iqr_high": 0.0}


def test_summarize_runs_single_value() -> None:
    s = summarize_runs([0.42])
    assert s["n"] == 1
    assert s["median"] == 0.42
    assert s["iqr_low"] == 0.42
    assert s["iqr_high"] == 0.42


def test_summarize_runs_median_odd_count() -> None:
    s = summarize_runs([0.3, 0.1, 0.2])
    assert s["n"] == 3
    assert s["median"] == 0.2


def test_summarize_runs_known_quartiles() -> None:
    # 0..4: median 2.0, q1 1.0, q3 3.0 (linear interpolation between ranks).
    s = summarize_runs([4.0, 0.0, 2.0, 3.0, 1.0])
    assert s["median"] == 2.0
    assert s["iqr_low"] == 1.0
    assert s["iqr_high"] == 3.0


def test_summarize_runs_order_independent() -> None:
    a = summarize_runs([5.0, 1.0, 3.0, 2.0, 4.0])
    b = summarize_runs([1.0, 2.0, 3.0, 4.0, 5.0])
    assert a == b
