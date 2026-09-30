"""``swlp chat MODEL -d`` / bare ``--backend`` / ``swlp models -d``: say what
would run and what the options are — before anything is loaded."""
from __future__ import annotations

import json
from pathlib import Path

from . import ui
from .cli_resolve import BACKENDS_INFO, backend_status, resolve_target
from .config import env_name, load_config


def print_plan(model: str, quant: str | None, command: str) -> None:
    """Chosen backend and why, every backend's fit for this model, and the
    chosen backend's settings (current values + the env var that sets them)."""
    target = resolve_target(model, quant, None)
    info = BACKENDS_INFO.get(target.backend)
    ui.header(f"swlp {command} {model} -d", [
        ("model", target.model_id),
        ("runs as", target.label + ("   (needs  swlp pull  first)" if target.needs_pull else "")),
        ("backend", f"{target.backend} — {info.what}" if info else target.backend),
        ("quality", _quality(target, info)),
    ])
    status = backend_status(model, target)
    ui.console.print()
    ui.table(["backend", "for this model", "what it is"],
             [[name, status[name], BACKENDS_INFO[name].what] for name in BACKENDS_INFO],
             title="Backends")
    ui.note(f"  pick one:  swlp {command} {model} --backend NAME")
    if info and info.knobs:
        runtime = load_config(None).runtime
        ui.console.print()
        ui.table(["setting", "now", "meaning"],
                 [[env_name(f), str(getattr(runtime, f)), meaning] for f, meaning in info.knobs],
                 title=f"Settings for {target.backend}")
        f0 = info.knobs[0][0]
        ui.note(f"  set for one run:  {env_name(f0)}=<value> swlp {command} {model}")
    ui.console.print()


def _quality(target, info) -> str:
    """What the user actually gets: a pre-quantized checkpoint runs as shipped."""
    if target.prequantized:
        return "as shipped by the checkpoint (see 'runs as')"
    return info.quality if info else ""


def print_model_details() -> None:
    """``swlp models -d``: one panel per installed model."""
    from .cli_models import installed_models

    rows = installed_models()
    if not rows:
        ui.note("  nothing installed yet — try:  swlp pull qwen-7b")
        return
    for name, where, size, how in rows:
        target = resolve_target(name)
        path = target.shard_dir or target.local_path or Path(where)
        facts = _facts(path)
        ui.header(name, [
            ("model", target.model_id),
            ("where", str(path)),
            ("size", f"{size:.1f} GB"),
            ("runs as", how),
            *facts,
            ("use", f"swlp chat {name}   ·   -d for backends   ·   swlp rm {name}"),
        ])
        ui.console.print()


def _facts(path: Path) -> list[tuple[str, str]]:
    """Architecture facts from a shard manifest or an MLX config.json."""
    manifest = path / "shard_manifest.json"
    if manifest.is_file():
        m = json.loads(manifest.read_text())
        facts = [("precision", str(m.get("weight_dtype", "?"))),
                 ("layers", f"{m.get('num_layers', '?')} · {m.get('layer_weight_mb', 0):.0f} MB"
                            " dense each")]
        if m.get("num_experts"):
            facts.append(("experts", f"{m['num_experts']} per layer · top-{m.get('top_k', '?')}"))
        facts.append(("MTP head", "yes (self-drafting)" if (path / "mtp.safetensors").is_file()
                      else "no"))
        return facts
    cfg_path = path / "config.json"
    if not cfg_path.is_file():
        return []
    cfg = json.loads(cfg_path.read_text())
    text = cfg.get("text_config") or cfg
    q = cfg.get("quantization") or {}
    facts = [("precision", f"{q.get('bits', '?')}-bit (group {q.get('group_size', '?')})"),
             ("layers", str(text.get("num_hidden_layers", "?")))]
    experts = text.get("num_experts") or text.get("num_local_experts")
    if experts:
        top_k = text.get("num_experts_per_tok") or text.get("top_k_experts") or "?"
        facts.append(("experts", f"{experts} per layer · top-{top_k}"))
    return facts
