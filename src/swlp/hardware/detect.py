"""Hardware detection for SWLP cross-platform support."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Platform-default SSD bandwidths (GB/s) used only until a real measurement
# is available — see ``save_measured_bandwidth`` / ``phase0_hardware_check``.
_APPLE_SSD_DEFAULT_GBPS = 6.5
_OTHER_SSD_DEFAULT_GBPS = 3.5


def bandwidth_cache_path() -> Path:
    """Location of the measured-hardware cache (overridable via env)."""
    override = os.getenv("SWLP_HW_CACHE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "swlp" / "hardware.json"


def save_measured_bandwidth(gbps: float) -> None:
    """Persist a measured SSD read bandwidth for future ``detect_hardware`` calls.

    Written by ``scripts/phase0_hardware_check.py``; replaces the hardcoded
    platform defaults in the Hardware Probe so the residency planner reasons
    over real bandwidth.
    """
    path = bandwidth_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ssd_read_gbps": float(gbps)}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _measured_ssd_bandwidth() -> float | None:
    """Best-known SSD read bandwidth: env override → measured cache → None."""
    env = os.getenv("SWLP_SSD_BW_GBPS")
    if env:
        try:
            value = float(env)
            if value > 0:
                return value
        except ValueError:
            pass
    try:
        with open(bandwidth_cache_path(), encoding="utf-8") as fh:
            value = float(json.load(fh).get("ssd_read_gbps", 0.0))
        if value > 0:
            return value
    except Exception:
        pass
    return None


@dataclass(slots=True)
class HardwareInfo:
    device_type: str        # "mps" | "cpu"
    unified_memory: bool    # True on Apple Silicon (no PCIe bus)
    memory_gb: float        # total system RAM (unified on Apple)
    ssd_bandwidth_gbps: float
    preferred_backend: str  # "mlx" | "torch"
    chip_name: str


def _apple_chip_name() -> str:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            if "Chip" in line or "Processor" in line:
                return line.split(":", 1)[-1].strip()
    except Exception:
        pass
    return "Apple Silicon"


def _system_memory_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        return 0.0


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _mlx_available() -> bool:
    try:
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        return False


def detect_hardware() -> HardwareInfo:
    import torch

    if _is_apple_silicon():
        chip = _apple_chip_name()
        mem = _system_memory_gb()
        mps_ok = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        backend = "mlx" if _mlx_available() else "torch"
        return HardwareInfo(
            device_type="mps" if mps_ok else "cpu",
            unified_memory=True,
            memory_gb=mem,
            ssd_bandwidth_gbps=_measured_ssd_bandwidth() or _APPLE_SSD_DEFAULT_GBPS,
            preferred_backend=backend,
            chip_name=chip,
        )

    return HardwareInfo(
        device_type="cpu",
        unified_memory=False,
        memory_gb=_system_memory_gb(),
        ssd_bandwidth_gbps=_measured_ssd_bandwidth() or _OTHER_SSD_DEFAULT_GBPS,
        preferred_backend="torch",
        chip_name=platform.processor() or "Unknown CPU",
    )


def window_size_recommendation(hw: HardwareInfo, layer_weight_mb: float) -> int:
    """
    Heuristic: how many layers to keep resident given hardware bandwidth.
    Transfer time = layer_weight_mb / (bandwidth_mb_per_s).
    Pick window so transfer is mostly hidden by compute.
    """
    bandwidth_mb_per_s = hw.ssd_bandwidth_gbps * 1024
    if bandwidth_mb_per_s <= 0:
        return 2
    transfer_ms = (layer_weight_mb / bandwidth_mb_per_s) * 1000
    # rough compute budget per layer on consumer hardware: ~80-150 ms
    compute_ms = 100.0
    ratio = transfer_ms / compute_ms
    if ratio < 0.5:
        return 2
    if ratio < 1.0:
        return 4
    return 6


# Memory reserved for the OS and other processes; the rest of unified/system
# RAM is available to the runtime.
_OS_RESERVE_MB = 3072
# Conservative allowance for embeddings + final norm + lm_head kept on device.
_EMBED_RESERVE_MB = 1024


def fits_in_memory(model_bytes: int, hw: HardwareInfo) -> bool:
    """Return True if the model can be fully loaded without OOM.

    Leaves room for the OS and the embedding/lm_head that SWLP always keeps
    resident, then checks whether ``model_bytes`` fits in what remains.
    """
    available = (
        hw.memory_gb * 1024 * 1024 * 1024
        - _OS_RESERVE_MB * 1024 * 1024
        - _EMBED_RESERVE_MB * 1024 * 1024
    )
    return model_bytes <= available


def streaming_fits_in_memory(
    resident_bytes: int,
    window_bytes: int,
    hw: HardwareInfo,
) -> bool:
    """Return True if SWLP layer streaming can run at all on this machine.

    Streaming keeps the embeddings, final norm and lm_head permanently resident
    and slides a window of transformer layers through RAM. If even that minimal
    footprint — ``resident_bytes`` (embed + lm_head) plus one ``window_bytes``
    layer window — does not fit after the OS reserve, no amount of streaming
    helps and the run should abort cleanly rather than hard-OOM.
    """
    available = hw.memory_gb * 1024 * 1024 * 1024 - _OS_RESERVE_MB * 1024 * 1024
    return (resident_bytes + window_bytes) <= available


def kv_budget_recommendation(
    hw: HardwareInfo,
    window_size: int,
    layer_weight_mb: float,
    num_layers: int,
) -> int:
    """Recommend a KV-cache RAM budget (MB) given hardware and the layer window.

    Budget math:  total_ram - OS_reserve - window_footprint - embed_reserve.
    The window footprint is ``window_size`` resident layers plus one prefetch
    slot. The result is floored at 256 MB so a budget always exists.
    """
    total_mb = hw.memory_gb * 1024
    window_footprint_mb = (window_size + 1) * max(layer_weight_mb, 0.0)
    headroom_mb = total_mb - _OS_RESERVE_MB - _EMBED_RESERVE_MB - window_footprint_mb
    # Give KV at most half the remaining headroom — the rest cushions activations.
    budget_mb = int(headroom_mb * 0.5)
    return max(budget_mb, 256)
