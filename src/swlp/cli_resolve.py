"""Pick the right backend for a model — so users name a model, not a backend.

Rules, first match wins:

1. ``--backend`` given           → honoured (advanced override).
2. a local SWLP shard dir        → experts banked? ``mlx-moe`` (lossless MoE
   (a path, or ``shards/<name>``)   streaming) · MTP head? ``speculative``
                                   (self-drafting, identical output) · else ``swlp``.
3. a local MLX checkpoint dir    → MoE? ``mlx-moe`` (4-bit experts streamed) · else ``mlx``.
4. ``--quant`` given             → ``mlx`` at that tier (resident, fast).
5. an MLX-format Hub repo        → as 3 (read from its config.json only).
6. anything else                 → lossless layer streaming; needs ``swlp pull`` first.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .cli_args import resolve_model
from .model.expert_bank import EXPERT_INDEX_FILE
from .model.shard import MANIFEST_FILE, MTP_FILE

SHARDS_ROOT = Path("shards")


@dataclass(frozen=True)
class Target:
    backend: str
    model_id: str
    label: str                        # one-line human description
    shard_dir: Path | None = None
    local_path: Path | None = None    # MLX checkpoint dir, loaded as-is
    quant: str | None = None
    mtp: bool = False
    needs_pull: bool = False
    prequantized: bool = False        # an MLX checkpoint: runs at its shipped precision
    full_gb: float | None = None      # full-precision size, when known (fit checks)


def shard_dir_for(model: str) -> Path:
    """Where ``swlp pull <model>`` puts shards (last path segment of the name)."""
    return SHARDS_ROOT / model.rstrip("/").split("/")[-1]


def resolve_target(model: str, quant: str | None = None, backend: str | None = None,
                   known_config: dict | None = None) -> Target:
    """``known_config`` is the Hub repo's config.json when known (tests inject it;
    the CLI fetches it lazily only when rules 1–4 did not decide)."""
    if backend == "mock":
        return Target("mock", model, "offline mock (no model)")
    model_id = resolve_model(model)
    path = Path(model).expanduser()
    # A path, or shards pulled under the alias or the HF repo name.
    for local in ([path] if path.is_dir() else [shard_dir_for(model), shard_dir_for(model_id)]):
        found = _from_dir(local, quant) if local.is_dir() and backend is None else None
        if found is not None:
            return found
    if backend is not None:
        return Target(backend, model_id, f"{backend} (forced)", quant=quant)
    cfg = known_config if known_config is not None else hub_config(model_id)
    if cfg and cfg.get("quantization"):
        # Already quantized (MLX repo): -q cannot improve it; re-quantizing
        # would convert a full resident copy (measured: 14 GB for Gemma 4 26B).
        return _mlx_target(model_id, cfg, None, ignored_quant=quant)
    if quant is not None:
        return Target("mlx", model_id, f"MLX {quant} · resident", quant=quant)
    return Target("swlp", model_id, "lossless layer streaming", shard_dir=shard_dir_for(model),
                  needs_pull=True)


def is_moe(cfg: dict) -> bool:
    text = cfg.get("text_config") or {}
    keys = ("num_experts", "num_local_experts", "n_routed_experts")
    return any(int(cfg.get(k) or text.get(k) or 0) > 0 for k in keys)


def _from_dir(path: Path, quant: str | None) -> Target | None:
    if (path / MANIFEST_FILE).is_file():
        manifest = json.loads((path / MANIFEST_FILE).read_text())
        model_id = manifest.get("model_id", str(path))
        dtype = manifest.get("weight_dtype", "float16")
        if manifest.get("source_quant"):  # exact w.r.t. a quantized release only
            dtype = f"{dtype} (from {manifest['source_quant'].upper()} release)"
        if (path / EXPERT_INDEX_FILE).is_file():
            if quant in ("int4", "int8"):
                return Target("mlx-moe", model_id,
                              f"MoE expert streaming · {dtype} → {quant} on load (lossy)",
                              shard_dir=path, quant=quant)
            return Target("mlx-moe", model_id, f"MoE expert streaming · {dtype} · lossless",
                          shard_dir=path)
        if quant is not None:
            # Dense layers are re-read every token, so -q means "resident on
            # MLX" — from the real HF id in the manifest, not the typed name.
            return Target("mlx", model_id, f"MLX {quant} · resident", quant=quant,
                          full_gb=manifest.get("total_weight_mb", 0) / 1024 or None)
        if (path / MTP_FILE).is_file():
            return Target("speculative", model_id,
                          f"layer streaming + MTP self-drafting · {dtype} · lossless",
                          shard_dir=path, mtp=True)
        return Target("swlp", model_id, f"layer streaming · {dtype} · lossless", shard_dir=path)
    cfg_path = path / "config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text())
        if cfg.get("quantization"):
            return _mlx_target(str(path), cfg, path, ignored_quant=quant)
    return None


def _mlx_target(model_id: str, cfg: dict, local: Path | None,
                ignored_quant: str | None = None) -> Target:
    bits = (cfg.get("quantization") or {}).get("bits", "?")
    note = f" (already {bits}-bit; -q {ignored_quant} ignored)" if ignored_quant else ""
    if is_moe(cfg):
        return Target("mlx-moe", model_id, f"MoE expert streaming · {bits}-bit{note}",
                      local_path=local, prequantized=True)
    return Target("mlx", model_id, f"MLX {bits}-bit · resident{note}", local_path=local,
                  quant="bf16", prequantized=True)


def hub_config(model_id: str) -> dict | None:
    """config.json from the Hub (cached after the first call); None if offline."""
    try:
        from huggingface_hub import hf_hub_download

        return json.loads(Path(hf_hub_download(model_id, "config.json")).read_text())
    except Exception:
        return None


@dataclass(frozen=True)
class BackendInfo:
    what: str                          # one line: what it does
    quality: str
    knobs: tuple[tuple[str, str], ...]  # (config field, meaning) — set as SWLP_* env vars


# One catalogue for help, `-d` and the resolver — they cannot disagree.
BACKENDS_INFO: dict[str, BackendInfo] = {
    "mlx-moe": BackendInfo(
        "MoE: dense part in RAM, experts cached from SSD",
        "lossless on bf16 shards · -q int4/int8 on load is lossy",
        (("swlp_expert_cache_mb", "expert cache MB (0 = free RAM, GPU-capped)"),
         ("swlp_moe_quant", "quantize on load: none | int8 | int4 (or -q)"),
         ("swlp_expert_prefetch", "lru | predictive | off (predictive measured slower)"))),
    "swlp": BackendInfo(
        "layers streamed from SSD, one at a time",
        "lossless (shard precision)",
        (("swlp_window_size", "W: layers on the GPU at once (default 2) · --window"),
         ("swlp_prefetch_depth", "layers read ahead while computing (default 2)"),
         ("swlp_residency", "first N layers kept in RAM: auto | off | N · --resident"),
         ("swlp_direct_io", "bypass the page cache: auto | on | off"))),
    "speculative": BackendInfo(
        "layer streaming + drafted tokens per sweep",
        "lossless — output identical to swlp",
        (("swlp_mtp", "draft with the model's own MTP head (auto when present)"),
         ("swlp_draft_model", "or a small same-tokenizer draft model"),
         ("swlp_spec_max_draft", "max drafted tokens per sweep (default 16)"),
         ("swlp_window_size", "W, as for swlp · --window"),
         ("swlp_residency", "resident layers, as for swlp · --resident"))),
    "mlx": BackendInfo(
        "whole model in RAM on MLX (fastest if it fits)",
        "bf16 exact · int8 near-lossless · int4 lossy (-q)",
        (("mlx_quant", "bf16 | int8 | int4 (or -q)"),
         ("mlx_kv_bits", "4-bit KV cache: faster on unified memory"),
         ("mlx_draft_model", "speculative draft model (1.9-2.1x)"))),
    "hf": BackendInfo(
        "plain transformers, whole model in RAM",
        "exact", ()),
    "mock": BackendInfo("offline canned answers, no model", "n/a", ()),
}


def backend_status(model: str, target: Target) -> dict[str, str]:
    """For each backend: can it run *this* model, and how to ask for it."""
    shards = next((d for d in (Path(model), shard_dir_for(model),
                               shard_dir_for(resolve_model(model)))
                   if (d / MANIFEST_FILE).is_file()), None)
    moe_shards = shards is not None and (shards / EXPERT_INDEX_FILE).is_file()
    mlx_repo = target.backend in ("mlx", "mlx-moe") and target.shard_dir is None
    need_pull = f"after  swlp pull {model}"
    has_mtp = shards is not None and (shards / MTP_FILE).is_file()
    status = {
        "mlx-moe": "yes" if moe_shards or target.backend == "mlx-moe"
                   else "no — not a pulled MoE model",
        "swlp": ("yes, but slow for MoE (torch path)" if moe_shards else "yes" if shards
                 else "no — MLX checkpoint" if mlx_repo else need_pull),
        # The MTP drafter handles dense MTP heads only (an MoE model's MTP is MoE).
        "speculative": ("with SWLP_DRAFT_MODEL only (MoE MTP unsupported)" if moe_shards
                        else "yes (MTP head)" if has_mtp
                        else "yes, with SWLP_DRAFT_MODEL" if shards
                        else "no — MLX checkpoint" if mlx_repo else need_pull),
        "mlx": "yes" if target.backend == "mlx" else "with  -q int4|int8  (loads it all in RAM)",
        "hf": "only if the full model fits in RAM",
        "mock": "yes (offline test)",
    }
    status[target.backend] = "✓ chosen automatically"
    return status
