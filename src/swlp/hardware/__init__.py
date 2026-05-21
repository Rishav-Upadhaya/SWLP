"""Hardware detection for SWLP cross-platform support."""
from .detect import (
    HardwareInfo,
    detect_hardware,
    fits_in_memory,
    streaming_fits_in_memory,
    window_size_recommendation,
)

__all__ = [
    "HardwareInfo",
    "detect_hardware",
    "fits_in_memory",
    "streaming_fits_in_memory",
    "window_size_recommendation",
]
