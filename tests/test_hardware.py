"""Tests for swlp.hardware.detect — fits_in_memory()."""
from __future__ import annotations

from swlp.hardware.detect import _EMBED_RESERVE_MB, _OS_RESERVE_MB, HardwareInfo, fits_in_memory


def _hw(memory_gb: float) -> HardwareInfo:
    return HardwareInfo(
        device_type="mps",
        unified_memory=True,
        memory_gb=memory_gb,
        ssd_bandwidth_gbps=6.5,
        preferred_backend="torch",
        chip_name="Apple M5",
    )


def _gb(n: float) -> int:
    return int(n * 1024 ** 3)


def _reserve_bytes() -> int:
    return (_OS_RESERVE_MB + _EMBED_RESERVE_MB) * 1024 * 1024


def test_tiny_model_fits_on_small_machine():
    assert fits_in_memory(model_bytes=_gb(0.1), hw=_hw(8)) is True


def test_14gb_model_does_not_fit_16gb_machine():
    """Mistral-7B FP16 ~14 GB should not fit 16 GB after reserves."""
    assert fits_in_memory(model_bytes=_gb(14), hw=_hw(16)) is False


def test_model_exactly_at_limit_fits():
    hw = _hw(16)
    available = _gb(16) - _reserve_bytes()
    assert fits_in_memory(model_bytes=available, hw=hw) is True


def test_model_one_byte_over_limit_does_not_fit():
    hw = _hw(16)
    available = _gb(16) - _reserve_bytes()
    assert fits_in_memory(model_bytes=available + 1, hw=hw) is False


def test_model_fits_on_large_machine():
    assert fits_in_memory(model_bytes=_gb(60), hw=_hw(128)) is True


def test_zero_model_bytes_always_fits():
    assert fits_in_memory(model_bytes=0, hw=_hw(8)) is True
