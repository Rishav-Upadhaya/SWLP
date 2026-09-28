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


# ── measured SSD bandwidth cache (Phase 24) ─────────────────────────────────

def test_save_and_load_measured_bandwidth(tmp_path, monkeypatch):
    from swlp.hardware.detect import (
        _measured_ssd_bandwidth,
        bandwidth_cache_path,
        save_measured_bandwidth,
    )

    monkeypatch.setenv("SWLP_HW_CACHE", str(tmp_path / "hardware.json"))
    monkeypatch.delenv("SWLP_SSD_BW_GBPS", raising=False)
    assert _measured_ssd_bandwidth() is None  # no cache yet

    save_measured_bandwidth(5.25)
    assert bandwidth_cache_path().exists()
    assert _measured_ssd_bandwidth() == 5.25


def test_env_override_beats_cache_file(tmp_path, monkeypatch):
    from swlp.hardware.detect import _measured_ssd_bandwidth, save_measured_bandwidth

    monkeypatch.setenv("SWLP_HW_CACHE", str(tmp_path / "hardware.json"))
    save_measured_bandwidth(5.25)
    monkeypatch.setenv("SWLP_SSD_BW_GBPS", "9.9")
    assert _measured_ssd_bandwidth() == 9.9


def test_invalid_env_falls_through_to_cache(tmp_path, monkeypatch):
    from swlp.hardware.detect import _measured_ssd_bandwidth, save_measured_bandwidth

    monkeypatch.setenv("SWLP_HW_CACHE", str(tmp_path / "hardware.json"))
    save_measured_bandwidth(4.0)
    monkeypatch.setenv("SWLP_SSD_BW_GBPS", "not-a-number")
    assert _measured_ssd_bandwidth() == 4.0


def test_missing_cache_file_returns_none(tmp_path, monkeypatch):
    from swlp.hardware.detect import _measured_ssd_bandwidth

    monkeypatch.setenv("SWLP_HW_CACHE", str(tmp_path / "does-not-exist.json"))
    monkeypatch.delenv("SWLP_SSD_BW_GBPS", raising=False)
    assert _measured_ssd_bandwidth() is None
