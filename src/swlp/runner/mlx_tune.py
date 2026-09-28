"""Apple Silicon runtime tuning for the MLX backend.

Everything here targets one number: decode tokens/second on a unified-memory
Mac. Three facts drive every choice.

1. **Decode is memory-bandwidth-bound, not compute-bound.** Apple's own M5
   numbers show prefill (compute-bound) gaining up to 3.6x from the M5 Neural
   Accelerators while decode gains only 19-27%, tracking the bandwidth bump
   from 120 GB/s (M4) to 153 GB/s (M5). So on decode, *bytes moved per token*
   is the budget — not FLOPs.

2. **Therefore compression can be free, or better than free.** If a kernel
   costs less time than the bandwidth it saves, quantizing is a pure win. This
   is measured for int4 KV on Apple Silicon (arXiv:2605.05699): ~25 ns/vec of
   kernel overhead against 3x less KV traffic, giving a *speedup* with
   essentially no perplexity change on short contexts. That inverts the usual
   quality-for-latency trade.

3. **macOS caps what the GPU may wire, in two layers.** Metal reports a
   ``max_recommended_working_set_size`` — about 74% of RAM (11.84 GB on a
   16 GB M5). ``mx.set_wired_limit`` can raise this process's wired budget up
   to that cap, but **not past it**: MLX rejects the call outright. Going
   higher needs ``sudo sysctl iogpu.wired_limit_mb=<MB>``, which SWLP will
   not run for you — it prints the exact command instead. Leaving macOS
   under ~4 GB trades a memory win for a swap storm, which is strictly worse.

None of this changes what the model computes, except ``kv_bits``, which is
opt-in and labelled.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

LOGGER = logging.getLogger(__name__)

# Leave this much physical RAM to macOS. Below ~4 GB the window server, the
# shell and the page cache start fighting the model, and a swap storm costs
# far more than the extra wired pages win.
_OS_RESERVE_GB = 4.0

# Never wire more than this fraction of RAM even if asked: past it macOS
# cannot reclaim enough to stay responsive.
_MAX_WIRED_FRACTION = 0.92

# mlx-lm's own default prefill chunk. Larger chunks cut TTFT on long prompts
# but raise transient memory; 2048 is the documented sweet spot.
DEFAULT_PREFILL_STEP = 2048


@dataclass(slots=True)
class TuningReport:
    """What the tuner actually changed — surfaced by `swlp doctor`."""

    wired_limit_mb: int | None = None
    wired_limit_before_mb: int | None = None
    cache_limit_mb: int | None = None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.wired_limit_mb is None:
            return "MLX memory tuning: not applied"
        return (
            f"MLX wired limit: {self.wired_limit_mb / 1024:.1f} GB "
            f"(was {(self.wired_limit_before_mb or 0) / 1024:.1f} GB)"
        )


def recommended_wired_limit_mb(total_ram_gb: float) -> int:
    """How much RAM the GPU *should* be allowed to wire, in MB.

    Total minus a fixed OS reserve, clamped to a safe fraction. Deliberately
    total-RAM-driven: a fraction-only rule gives 8 GB machines far too little
    headroom and 64 GB machines needlessly much.

    This is the target, not necessarily the achievable value — see
    :func:`max_working_set_mb` for the hard cap Metal enforces.
    """
    usable = max(1.0, total_ram_gb - _OS_RESERVE_GB)
    ceiling = total_ram_gb * _MAX_WIRED_FRACTION
    return int(min(usable, ceiling) * 1024)


def max_working_set_mb() -> int | None:
    """Metal's hard ceiling on wired memory for this device, in MB.

    ``mx.set_wired_limit`` raises ``[metal::set_wired_limit] Setting a wired
    limit larger than the maximum working set size is not allowed`` past this,
    so it has to be queried rather than assumed.
    """
    try:
        import mlx.core as mx

        info = mx.device_info()
        size = info.get("max_recommended_working_set_size")
        return int(size) // (1024 * 1024) if size else None
    except Exception:
        return None


def sysctl_advice(total_ram_gb: float) -> str | None:
    """The sudo command that would raise Metal's ceiling, if it would help.

    Returns None when the ceiling is already at or above what we want. SWLP
    never runs this itself: it needs root, it affects the whole machine, and
    it resets on reboot — that is the user's call to make, not a library's.
    """
    cap = max_working_set_mb()
    want = recommended_wired_limit_mb(total_ram_gb)
    if cap is None or cap >= want:
        return None
    return (
        f"sudo sysctl iogpu.wired_limit_mb={want}"
        f"   # raises the GPU ceiling {cap / 1024:.1f} → {want / 1024:.1f} GB "
        f"(resets on reboot)"
    )


def apply_memory_tuning(setting: str, total_ram_gb: float) -> TuningReport:
    """Raise the Metal wired-memory ceiling for this process.

    ``setting`` is ``auto`` (compute it), ``off`` (leave macOS alone), or an
    explicit MB value. Returns what changed; never raises — a machine that
    refuses the call still runs, just with the default ceiling.
    """
    report = TuningReport()
    setting = str(setting).strip().lower()
    if setting in ("off", "", "none", "0"):
        report.notes.append("wired limit untouched (mlx_wired_limit=off)")
        return report

    if setting == "auto":
        target_mb = recommended_wired_limit_mb(total_ram_gb)
    else:
        try:
            target_mb = int(setting)
        except ValueError:
            LOGGER.warning("mlx_wired_limit_invalid", extra={"value": setting})
            report.notes.append(f"ignored invalid mlx_wired_limit={setting!r}")
            return report

    try:
        import mlx.core as mx
    except ImportError:
        report.notes.append("mlx not installed — no memory tuning")
        return report

    # Metal refuses anything above the device working set, so clamp rather
    # than let the call fail and lose the gain we *can* get.
    cap_mb = max_working_set_mb()
    if cap_mb is not None and target_mb > cap_mb:
        report.notes.append(
            f"clamped {target_mb / 1024:.1f} GB → {cap_mb / 1024:.1f} GB "
            "(Metal working-set cap; raise it with sysctl — see `swlp doctor`)"
        )
        target_mb = cap_mb

    try:
        # set_wired_limit returns the PREVIOUS limit, in bytes.
        previous = mx.set_wired_limit(target_mb * 1024 * 1024)
        report.wired_limit_mb = target_mb
        report.wired_limit_before_mb = int(previous) // (1024 * 1024)
        LOGGER.info(
            "mlx_wired_limit_set",
            extra={"limit_mb": target_mb, "previous_mb": report.wired_limit_before_mb},
        )
    except Exception as exc:
        # Raising the limit is an optimisation, never a requirement.
        LOGGER.warning("mlx_wired_limit_failed", extra={"error": str(exc)})
        report.notes.append(f"wired limit not raised: {exc}")

    return report


def generation_kwargs(
    *,
    kv_bits: int,
    kv_group_size: int,
    quantized_kv_start: int,
    max_kv_size: int,
    prefill_step_size: int,
    num_draft_tokens: int,
    has_draft_model: bool,
) -> dict[str, object]:
    """Map SWLP config onto ``mlx_lm.generate_step`` kwargs.

    Only non-default values are emitted, so mlx-lm's own defaults win wherever
    SWLP has no opinion — that keeps this working across mlx-lm releases
    instead of pinning their defaults into our code.
    """
    kwargs: dict[str, object] = {}

    if kv_bits in (4, 8):
        kwargs["kv_bits"] = kv_bits
        kwargs["kv_group_size"] = kv_group_size
        # Quantizing from token 0 costs accuracy on the earliest, most
        # attended-to positions for little gain (the cache is tiny then).
        # Starting later keeps the head of the cache exact.
        kwargs["quantized_kv_start"] = max(0, quantized_kv_start)
    elif kv_bits not in (0,):
        LOGGER.warning("mlx_kv_bits_invalid", extra={"kv_bits": kv_bits})

    if max_kv_size > 0:
        # RotatingKVCache: bounds long-context memory. Lossy — it drops the
        # oldest positions — so it is off unless explicitly asked for.
        kwargs["max_kv_size"] = max_kv_size

    if prefill_step_size > 0:
        kwargs["prefill_step_size"] = prefill_step_size

    if has_draft_model and num_draft_tokens > 0:
        kwargs["num_draft_tokens"] = num_draft_tokens

    return kwargs


