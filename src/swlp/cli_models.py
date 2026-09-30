"""``swlp models``: what is installed here, how each would run, and the aliases."""
from __future__ import annotations

import json
from pathlib import Path

from . import ui
from .cli_args import MODEL_ALIASES
from .cli_resolve import SHARDS_ROOT, resolve_target

_HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"


def installed_models() -> list[tuple[str, str, float, str]]:
    """``(name to type, source, size GB, how it runs)`` for local models."""
    rows: list[tuple[str, str, float, str]] = []
    if SHARDS_ROOT.is_dir():
        for d in sorted(p for p in SHARDS_ROOT.iterdir() if (p / "shard_manifest.json").is_file()):
            size = sum(f.stat().st_size for f in d.iterdir() if f.is_file()) / 1024**3
            model_id = json.loads((d / "shard_manifest.json").read_text()).get("model_id", d.name)
            how = resolve_target(str(d)).label
            rows.append((_alias_for(model_id, d.name), "shards", size, how))
    if _HF_HUB.is_dir():
        for repo in sorted(_HF_HUB.glob("models--*")):
            cfg_paths = sorted(repo.glob("snapshots/*/config.json"))
            if not cfg_paths:
                continue
            cfg = json.loads(cfg_paths[-1].read_text())
            weights = list(cfg_paths[-1].parent.glob("*.safetensors"))
            if not cfg.get("quantization") or not weights:
                continue  # only runnable MLX checkpoints (others need `swlp pull`)
            model_id = repo.name.removeprefix("models--").replace("--", "/")
            size = sum(f.stat().st_size for f in weights) / 1024**3
            how = resolve_target(str(cfg_paths[-1].parent)).label
            rows.append((_alias_for(model_id, model_id), "HF cache", size, how))
    return rows


def print_models() -> None:
    rows = installed_models()
    if rows:
        ui.table(["model", "where", "size", "runs as"],
                 [[n, w, f"{s:.1f} GB", h] for n, w, s, h in rows], title="Installed")
    else:
        ui.note("  nothing installed yet — try:  swlp pull qwen-7b")
    ui.console.print()
    from .cli_doctor import KNOWN_FP16_GB, MOE_MODELS

    known = {**KNOWN_FP16_GB, **{a: float(i["disk_gb"]) for a, i in MOE_MODELS.items()}}
    sizes = {a: f"{known[a]:.0f} GB" if a in known else "" for a in MODEL_ALIASES}
    ui.table(["alias", "full size", "HuggingFace id"],
             [[a, sizes[a], h] for a, h in MODEL_ALIASES.items()],
             title="Aliases (any HuggingFace id works too)")
    ui.note("\n  swlp chat MODEL  ·  swlp pull MODEL  ·  swlp doctor MODEL for a fit check\n")


def _alias_for(model_id: str, fallback: str) -> str:
    """The short alias for a HF id — the name users actually type."""
    return next((a for a, h in MODEL_ALIASES.items() if h.lower() == model_id.lower()), fallback)


def model_artifacts(model: str) -> list[tuple[str, Path, int]]:
    """Everything SWLP keeps on disk for ``model``: ``(kind, path, bytes)``.

    Shard dirs (named after the alias or the HF repo), Hugging Face cache
    repos (the default hub and SWLP's own cache dir), and ``-q`` MLX
    conversions. Nothing outside those locations is ever listed.
    """
    import re

    from huggingface_hub import scan_cache_dir

    from .cli_args import resolve_model
    from .cli_resolve import shard_dir_for
    from .config import load_config

    model_id = resolve_model(model)
    found: dict[Path, tuple[str, int]] = {}
    for d in (Path(model), shard_dir_for(model), shard_dir_for(model_id)):
        if (d / "shard_manifest.json").is_file():
            found[d.resolve()] = ("shards", _dir_bytes(d))
    cache_dir = Path(load_config(None).cache.cache_dir)
    for hub in (None, cache_dir):
        try:
            info = scan_cache_dir(hub) if hub is None or hub.is_dir() else None
        except Exception:
            info = None  # no cache there yet
        for repo in (info.repos if info else ()):
            if repo.repo_id.lower() == model_id.lower():
                found[Path(repo.repo_path).resolve()] = ("HF download", repo.size_on_disk)
    slug = re.sub(r"[^A-Za-z0-9._-]", "_", model_id)  # runner/mlx.py conversion naming
    for d in cache_dir.glob(f"mlx-*-{slug}") if cache_dir.is_dir() else ():
        found[d.resolve()] = (f"MLX {d.name.split('-')[1]} copy", _dir_bytes(d))
    return [(kind, path, size) for path, (kind, size) in sorted(found.items())]


def _dir_bytes(d: Path) -> int:
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file() and not f.is_symlink())
