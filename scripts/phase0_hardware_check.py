"""Phase 0: Measure your actual hardware numbers before writing SWLP.

Run this first on every target machine and record the output.
These numbers drive window_size decisions and the paper's hardware table.

Usage:
    python scripts/phase0_hardware_check.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

# Make sure the src package is importable when run from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def check_hardware() -> None:
    from swlp.hardware import detect_hardware, window_size_recommendation

    hw = detect_hardware()
    print("\n=== Hardware ===")
    print(f"  Chip:            {hw.chip_name}")
    print(f"  Device:          {hw.device_type}")
    print(f"  Unified memory:  {hw.unified_memory}")
    print(f"  RAM:             {hw.memory_gb:.1f} GB")
    print(f"  SSD (estimated): {hw.ssd_bandwidth_gbps:.1f} GB/s")
    print(f"  Backend:         {hw.preferred_backend}")

    # window size recommendations for common model sizes
    for model, mb in [("7B (FP16)", 175), ("30B (FP16)", 750), ("70B (FP16)", 1750)]:
        w = window_size_recommendation(hw, mb)
        print(f"  Suggested window for {model} layer ({mb} MB): W={w}")


def measure_ssd_write(size_mb: int = 512) -> float:
    """Write size_mb to a temp file and return GB/s."""
    data = b"x" * (size_mb * 1024 * 1024)
    with tempfile.NamedTemporaryFile(delete=True) as f:
        t0 = time.perf_counter()
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
        elapsed = time.perf_counter() - t0
    return (size_mb / 1024) / elapsed


def measure_ssd_read(size_mb: int = 512) -> float:
    """Write then read size_mb and return read GB/s."""
    data = b"x" * (size_mb * 1024 * 1024)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        fname = f.name
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

    try:
        t0 = time.perf_counter()
        with open(fname, "rb") as f:
            _ = f.read()
        elapsed = time.perf_counter() - t0
        return (size_mb / 1024) / elapsed
    finally:
        os.unlink(fname)


def check_mlx() -> None:
    print("\n=== MLX (Apple Silicon) ===")
    try:
        import mlx.core as mx
        a = mx.ones((1024, 1024))
        b = mx.ones((1024, 1024))
        t0 = time.perf_counter()
        c = a @ b
        mx.eval(c)
        elapsed = time.perf_counter() - t0
        print(f"  mlx available:  yes")
        print(f"  1024x1024 matmul: {elapsed * 1000:.1f} ms")
    except ImportError:
        print("  mlx available:  no  (pip install mlx mlx-lm)")


def check_torch() -> None:
    import torch

    print("\n=== PyTorch ===")
    print(f"  version:         {torch.__version__}")
    print(f"  CUDA:            {torch.cuda.is_available()}")
    mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    print(f"  MPS:             {mps}")

    device_str = "mps" if mps else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    a = torch.ones(1024, 1024, device=device)
    b = torch.ones(1024, 1024, device=device)
    # warmup
    _ = a @ b
    if device_str == "cuda":
        torch.cuda.synchronize()
    elif device_str == "mps":
        torch.mps.synchronize()

    t0 = time.perf_counter()
    c = a @ b
    if device_str == "cuda":
        torch.cuda.synchronize()
    elif device_str == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"  device:          {device_str}")
    print(f"  1024x1024 matmul ({device_str}): {elapsed * 1000:.1f} ms")


if __name__ == "__main__":
    check_hardware()

    print("\n=== SSD Bandwidth ===")
    w = measure_ssd_write(256)
    print(f"  Write (256 MB): {w:.2f} GB/s")
    r = measure_ssd_read(256)
    print(f"  Read  (256 MB): {r:.2f} GB/s")
    print("  NOTE: record these — they set your window_size target")

    check_torch()
    check_mlx()

    print("\n=== Summary ===")
    print("  Copy these numbers into your research log.")
    print("  Compare against the CTO rethink targets:")
    print("    M5:   SSD ~6.5 GB/s, compute ~20-40ms/layer, window W=4-6")
    print("    MX230: PCIe ~32 GB/s effective, compute ~80-150ms/layer, window W=2-4")
    print()
