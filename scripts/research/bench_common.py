"""Shared benchmark-harness helpers: page-cache control + run provenance.

Imported by both ``scripts/research/compare_airllm_swlp.py`` and
``scripts/research/phase3_baselines.py`` so every benchmark JSON is **self-describing**:
package versions, git commit, hardware, and — most importantly — whether the OS
page cache was *cold* or *warm* for the run.

Why this matters
----------------
SWLP streams weights from SSD. If a model's shards are already resident in the
OS page cache (e.g. left there by a warmup run, or a previous benchmark), then a
"streaming" measurement is really timing a RAM read, not an SSD read — and can
*exceed* the true cold-SSD throughput ceiling (``SSD_bw / model_bytes``). That is
exactly how an earlier run reported 0.505 tok/s for Mistral-7B against a 0.496
tok/s cold ceiling: physically impossible for cold streaming, so it was warm.

These helpers drop the page cache between runs and record the achieved state, so
warm and cold numbers are never silently conflated. ``cache_state`` is stamped
into every result; if the cache could not actually be dropped (no permission),
the JSON says so rather than mislabelling a warm run as cold.
"""

from __future__ import annotations

import platform
import shutil
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

# Packages whose versions pin a benchmark's reproducibility.
_VERSION_PACKAGES = (
    "airllm",
    "mlx",
    "mlx-lm",
    "torch",
    "transformers",
    "safetensors",
    "swlp",
)


def drop_page_cache() -> str:
    """Best-effort drop of the OS page cache; return the *achieved* state.

    Returns one of:
      - ``"cold"``                       — cache was successfully dropped
      - ``"warm (<reason>)"``            — drop unavailable / not permitted

    The caller stamps this string into the result so a warm run is never
    mislabelled as cold. On macOS this needs ``purge`` (often requires sudo);
    on Linux it needs root to write ``/proc/sys/vm/drop_caches``.
    """
    system = platform.system()
    if system == "Darwin":
        purge = shutil.which("purge")
        if purge is None:
            return "warm (purge not found)"
        for cmd in ([purge], ["sudo", "-n", purge]):
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=180)
                if result.returncode == 0:
                    return "cold"
            except Exception:
                continue
        return "warm (purge needs sudo — run harness with sudo, or `sudo purge` first)"
    if system == "Linux":
        try:
            subprocess.run(["sync"], check=False, timeout=60)
            Path("/proc/sys/vm/drop_caches").write_text("3\n")
            return "cold"
        except PermissionError:
            return "warm (drop_caches needs root)"
        except Exception as exc:  # noqa: BLE001 — report any failure verbatim
            return f"warm (drop_caches failed: {exc})"
    return f"warm (no cache-drop mechanism on {system})"


def summarize_cache(states: list[str]) -> str:
    """Collapse per-run cache states into one label for a result.

    Returns ``"cold"`` only if *every* timed run was cold; otherwise surfaces the
    first warm reason so a partially-warm run is never reported as fully cold.
    """
    if not states:
        return "warm"
    if all(s == "cold" for s in states):
        return "cold"
    return next((s for s in states if s != "cold"), "warm")


def summarize_runs(values: list[float]) -> dict[str, float | int]:
    """Median ± IQR summary for a repeated measurement.

    Headline numbers must come from ≥ 5 runs and be reported as
    ``median [iqr_low, iqr_high]`` — the median is robust to the thermal /
    page-cache outliers that single-run figures fall victim to (the Phase 19
    audit traced a 2.4× discrepancy to exactly that). ``n`` travels with the
    summary so a reviewer can see how much evidence backs the number.
    """
    if not values:
        return {"n": 0, "median": 0.0, "iqr_low": 0.0, "iqr_high": 0.0}
    # "inclusive" = linear interpolation between closest ranks (numpy's default).
    q1, med, q3 = (
        statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else values * 3
    )
    return {"n": len(values), "median": med, "iqr_low": q1, "iqr_high": q3}


def package_versions() -> dict[str, str]:
    """Return installed versions of the packages a benchmark depends on."""
    import importlib.metadata as md

    versions: dict[str, str] = {}
    for name in _VERSION_PACKAGES:
        try:
            versions[name] = md.version(name)
        except Exception:
            versions[name] = "not installed"
    return versions


def git_commit() -> str:
    """Short git commit of the working tree, or ``"unknown"``."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def hardware_summary() -> dict[str, object]:
    """A small, JSON-friendly snapshot of the detected hardware."""
    try:
        from swlp.hardware.detect import detect_hardware

        hw = detect_hardware()
        return {
            "chip": hw.chip_name,
            "device_type": hw.device_type,
            "memory_gb": round(hw.memory_gb, 1),
            "unified_memory": hw.unified_memory,
            "ssd_bandwidth_gbps": round(hw.ssd_bandwidth_gbps, 2),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"hardware detection failed: {exc}"}


def provenance() -> dict[str, object]:
    """Assemble the full provenance block for a benchmark report.

    Stamped once per JSON file so the numbers are reproducible and a reviewer
    can see exactly what was measured, on what, and with which library versions.
    """
    return {
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "hardware": hardware_summary(),
        "versions": package_versions(),
    }
