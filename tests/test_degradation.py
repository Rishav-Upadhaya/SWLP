"""The silent-degradation ledger and SWLP_STRICT.

The streaming hot path catches ~85 failures rather than crashing — a slower
correct answer beats no answer. The risk is that a degraded run looks exactly
like a clean one in the benchmark output. These tests pin the two behaviours
that stop that: the ledger records, and strict mode refuses.
"""
from __future__ import annotations

import pytest

from swlp.config import load_config
from swlp.metrics import RunMetrics
from swlp.runner.swlp import SWLPRunner


def test_degrade_records_and_continues_by_default() -> None:
    runner = SWLPRunner(load_config(None))
    assert runner.degradations == []

    runner.degrade("prefetch_failed(layer=3)", RuntimeError("disk gone"))
    runner.degrade("kv_compress_failed(layer=7)", ValueError("bad shape"))

    assert runner.degradations == [
        "prefetch_failed(layer=3)",
        "kv_compress_failed(layer=7)",
    ]


def test_strict_mode_reraises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SWLP_STRICT", "1")
    runner = SWLPRunner(load_config(None))
    assert runner.config.runtime.swlp_strict is True

    with pytest.raises(RuntimeError, match="disk gone"):
        runner.degrade("prefetch_failed(layer=3)", RuntimeError("disk gone"))

    # It is still recorded before re-raising, so the ledger is complete.
    assert runner.degradations == ["prefetch_failed(layer=3)"]


def test_strict_mode_without_an_exception_cannot_raise() -> None:
    """degrade(reason) with no exception object always continues — there is
    nothing to re-raise, so strict mode must not invent one."""
    import os

    os.environ["SWLP_STRICT"] = "1"
    try:
        runner = SWLPRunner(load_config(None))
        runner.degrade("informational_only")
        assert runner.degradations == ["informational_only"]
    finally:
        del os.environ["SWLP_STRICT"]


def test_metrics_carry_the_ledger() -> None:
    m = RunMetrics(
        "m", "swlp", "cpu", 0.0, 1, 1,
        degradations=["prefetch_failed(layer=3)"],
        degradation_count=1,
    )
    d = m.to_dict()
    assert d["degradation_count"] == 1
    assert d["degradations"] == ["prefetch_failed(layer=3)"]


def test_clean_run_reports_no_degradation_fields() -> None:
    """A clean run must leave both fields None, not 0/[] — the report and the
    JSON consumers treat 'absent' as 'nothing went wrong'."""
    m = RunMetrics("m", "swlp", "cpu", 0.0, 1, 1)
    assert m.degradations is None
    assert m.degradation_count is None
