"""Hardware detection for SWLP cross-platform support."""
from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass


@dataclass(slots=True)
class HardwareInfo:
    device_type: str        # "cuda" | "mps" | "cpu"
    unified_memory: bool    # True on Apple Silicon (no PCIe bus)
    memory_gb: float        # total system RAM (unified on Apple)
    ssd_bandwidth_gbps: float
    preferred_backend: str  # "cuda" | "mlx" | "torch"
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
            ssd_bandwidth_gbps=6.5,  # typical M-series NVMe
            preferred_backend=backend,
            chip_name=chip,
        )

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return HardwareInfo(
            device_type="cuda",
            unified_memory=False,
            memory_gb=_system_memory_gb(),
            ssd_bandwidth_gbps=3.5,
            preferred_backend="cuda",
            chip_name=props.name,
        )

    return HardwareInfo(
        device_type="cpu",
        unified_memory=False,
        memory_gb=_system_memory_gb(),
        ssd_bandwidth_gbps=3.5,
        preferred_backend="torch",
        chip_name=platform.processor() or "Unknown CPU",
    )


def window_size_recommendation(hw: HardwareInfo, layer_weight_mb: float) -> int:
    """
    Heuristic: how many layers to keep resident given hardware bandwidth.
    Transfer time = layer_weight_mb / (bandwidth_mb_per_s).
    Pick window so transfer is mostly hidden by compute.
    """
    bandwidth_mb_per_s = hw.ssd_bandwidth_gbps * 1024 / 8
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
