"""SWLP command-line interface — dispatch only.

    swlp chat MODEL             swlp pull MODEL
    swlp run MODEL "prompt"     swlp models
    swlp serve MODEL            swlp doctor [MODEL]
    swlp bench MODEL

The backend is chosen from the model (``cli_resolve.py``); presentation lives
in ``ui.py``; the parser in ``cli_args.py``.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import ui
from .cli_args import BACKENDS, build_parser, resolve_model
from .cli_resolve import Target, resolve_target, shard_dir_for
from .config import AppConfig, load_config
from .logging import configure_logging

LOGGER = logging.getLogger(__name__)

# Default answer length: until the model emits its end-of-answer token. A cap
# this size is never reached in practice (the token-id buffer it sizes is
# 8 bytes/token); Ctrl+C stops any answer.
UNTIL_DONE = 1 << 20

# Manifests and safetensors headers add a little on top of the raw weights.
_PREFLIGHT_HEADROOM = 1.15
# Activations, KV cache and macOS need room beside resident weights.
_RESIDENT_HEADROOM_GB = 1.5


def main(argv: Sequence[str] | None = None) -> int:
    _quiet_hub()
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.version:
        ui.console.print(f"swlp {_version()}")
        return 0
    if args.help or args.command is None:
        from .cli_help import print_swlp_help

        print_swlp_help()
        return 0
    if args.command in ("chat", "run", "serve", "bench"):
        if args.details or args.backend == "?":
            from .cli_explain import print_plan

            print_plan(args.model, args.quant, args.command)
            return 0
        if args.backend is not None and args.backend not in BACKENDS:
            ui.error(f"unknown backend {args.backend!r}.",
                     f"choose one of: {', '.join(BACKENDS)}   "
                     f"(swlp {args.command} {args.model} -d explains them)")
            return 2
    if args.command == "models" and args.details:
        from .cli_explain import print_model_details

        print_model_details()
        return 0
    handler = {
        "chat": _chat, "run": _run, "serve": _serve, "pull": _pull,
        "models": _models, "rm": _rm, "doctor": _doctor, "bench": _bench,
    }[args.command]
    try:
        return handler(args)
    except KeyboardInterrupt:
        ui.note("\n  interrupted")
        return 130
    except Exception as exc:  # one readable line, not a traceback (-v shows it)
        if getattr(args, "verbose", False):
            raise
        ui.error(_friendly(exc, args), "add -v to see the full error")
        return 1


# ── model commands ──────────────────────────────────────────────────────────

def _prepare(args: argparse.Namespace) -> tuple[AppConfig, Target] | None:
    """Resolve MODEL → backend and build the config; None (after explaining)
    when the model must be pulled first."""
    target = resolve_target(args.model, args.quant, args.backend)
    if not _fits_resident(target, args):
        return None
    if target.needs_pull:
        ui.error(f"{args.model} isn't prepared yet.",
                 f"swlp pull {args.model}   (lossless streaming — any size)\n"
                 f"  or: swlp {args.command} {args.model} -q int4   "
                 "(resident on MLX, if it fits in RAM)")
        return None
    config = load_config(args.config)
    config.model.model_id = target.model_id
    config.runtime.backend = target.backend
    config.runtime.shard_dir = target.shard_dir
    if target.local_path is not None:
        config.model.local_model_path = target.local_path
    if target.quant is not None:
        if target.backend == "mlx-moe":
            config.runtime.swlp_moe_quant = target.quant
        else:
            config.runtime.mlx_quant = target.quant
    config.runtime.swlp_mtp = target.mtp or config.runtime.swlp_mtp
    if not _apply_streaming_flags(config, target, args):
        return None
    config.generation.max_new_tokens = args.max_tokens or UNTIL_DONE
    configure_logging("INFO" if args.verbose else "WARNING", json_logs=False)
    if not args.verbose:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()  # "falling back to reference kernel" notes etc.
    return config, target


def _apply_streaming_flags(config: AppConfig, target: Target, args: argparse.Namespace) -> bool:
    """``--resident`` / ``--window``: only layer streaming uses them. An explicit
    resident count is honoured as-is by the runner, so one that cannot fit free
    RAM is refused: it swaps, which is slower than streaming those layers
    (measured: 30 × 761 MB on 16 GB → 7.7 GB of swap, no first token in 10 min)."""
    resident, window = args.resident, args.window
    if resident is None and window is None:
        return True
    if target.backend not in ("swlp", "speculative"):
        ui.warn(f"--resident/--window apply to layer streaming; {target.backend} ignores them "
                "(swlp COMMAND MODEL -d shows its settings)")
        return True
    if window is not None:
        if window < 1:
            ui.error("--window must be at least 1.")
            return False
        config.runtime.swlp_window_size = window
    if resident is None:
        return True
    resident = str(resident).lower()
    if resident not in ("auto", "off") and not resident.isdigit():
        ui.error(f"--resident {resident!r}: use a layer count, auto, or off.")
        return False
    config.runtime.swlp_residency = resident
    if resident.isdigit() and target.shard_dir is not None:
        import psutil

        from .model.shard import load_manifest

        layer_gb = load_manifest(target.shard_dir).layer_weight_mb / 1024
        free_gb = psutil.virtual_memory().available / 1024**3 - _RESIDENT_HEADROOM_GB
        fits = max(0, int(free_gb / layer_gb)) if layer_gb else int(resident)
        if int(resident) > fits:
            ui.error(f"{resident} resident layers need {int(resident) * layer_gb:.1f} GB; "
                     f"only {fits} fit in free RAM right now (they would swap: slower).",
                     f"use --resident {fits} (or fewer), or --resident auto")
            return False
    return True


# Bytes per full-precision (16-bit) byte at each MLX tier, incl. scales/biases.
_QUANT_FRACTION = {"int4": 0.28, "int8": 0.53, "bf16": 1.0}


def _fits_resident(target: Target, args: argparse.Namespace) -> bool:
    """Refuse a resident MLX run that cannot fit the GPU working set, before a
    long download/convert (27B at int4 is ~15 GB; an M5 16 GB holds 11.8)."""
    if target.backend != "mlx" or target.full_gb is None or target.quant not in _QUANT_FRACTION:
        return True
    try:
        import mlx.core as mx

        limit_gb = mx.device_info()["max_recommended_working_set_size"] / 1024**3
    except (ImportError, AttributeError, KeyError):
        return True
    need_gb = target.full_gb * _QUANT_FRACTION[target.quant]
    if need_gb <= limit_gb - _RESIDENT_HEADROOM_GB:
        return True
    ui.error(f"{args.model} at {target.quant} needs ~{need_gb:.0f} GB resident; "
             f"this Mac's GPU can hold {limit_gb:.1f} GB.",
             f"stream it instead (lossless):  swlp {args.command} {args.model}")
    return False


def _chat(args: argparse.Namespace) -> int:
    prepared = _prepare(args)
    if prepared is None:
        return 1
    from .chat import run_chat

    config, target = prepared
    run_chat(config, target.label, max_tokens=config.generation.max_new_tokens)
    return 0


def _run(args: argparse.Namespace) -> int:
    prompt = sys.stdin.read() if args.prompt in (None, "-") else args.prompt
    if not prompt.strip():
        ui.error("no prompt given.", f'swlp run {args.model} "your question"')
        return 1
    prepared = _prepare(args)
    if prepared is None:
        return 1
    config, target = prepared
    from .runner.base import build_runner

    runner = build_runner(config)
    if args.json:
        result = runner.run(prompt, profile=True)
        payload = {"prompt": prompt, "completion": result.completion,
                   "backend": target.backend, "metrics": result.metrics.to_dict()}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    from .chat import ChatSession, answer, load_with_header

    load_with_header(runner, config, target.label)
    answer(runner, ChatSession(), prompt, config.generation.max_new_tokens)
    return 0


def _serve(args: argparse.Namespace) -> int:
    prepared = _prepare(args)
    if prepared is None:
        return 1
    config, target = prepared
    from .serve import serve

    ui.header("swlp serve", [("model", config.model.model_id), ("backend", target.label),
                             ("endpoint", f"http://{args.host}:{args.port}/v1")])
    serve(config, args.host, args.port)
    return 0


def _bench(args: argparse.Namespace) -> int:
    prepared = _prepare(args)
    if prepared is None:
        return 1
    config, target = prepared
    from .benchmark.run import run_benchmark

    config.runtime.profile = True
    with ui.spinner(f"benchmarking {args.model} · {args.runs} runs"):
        run = run_benchmark(config, None, args.runs, "short", warmup_runs=1)
    overall = run.summary["overall"]
    if args.json:
        print(json.dumps({"backend": target.backend, "summary": run.summary}, indent=2))
        return 0

    def med(field: str, scale: float = 1.0, fmt: str = "{:.1f}") -> str:
        s = overall.get(field)
        return fmt.format(s["median"] * scale) if s else "—"

    ui.header("swlp bench", [("model", config.model.model_id), ("backend", target.label),
                             ("runs", f"{run.summary['runs']} (median shown)")])
    ui.table(["tok/s", "first token", "total", "peak RAM"], [[
        med("throughput_tokens_per_second"),
        med("time_to_first_token_seconds", fmt="{:.2f} s"),
        med("total_seconds", fmt="{:.1f} s"),
        med("ram_peak_bytes", 1 / 1e9, "{:.1f} GB"),
    ]])
    return 0


# ── setup commands ──────────────────────────────────────────────────────────

def _pull(args: argparse.Namespace) -> int:
    from .cli_resolve import hub_config, is_moe

    model_id = resolve_model(args.model)
    cfg = hub_config(model_id)
    if cfg and cfg.get("quantization"):
        # MLX-format checkpoints run in place — just download.
        from huggingface_hub import snapshot_download

        with ui.spinner(f"downloading {model_id}"):
            snapshot_download(model_id)
        kind = "MoE, experts streamed" if is_moe(cfg) else "resident"
        ui.ok(f"{model_id} ready ({kind}).  →  swlp chat {args.model}")
        return 0
    output_dir: Path = args.output_dir or shard_dir_for(args.model)
    if not _enough_disk(args.model, output_dir):
        return 1
    from .model.shard import shard_model_by_layer

    ui.header("swlp pull", [("model", model_id), ("into", str(output_dir)),
                            ("what", "download, then split into per-layer shards (one-time)")])
    with ui.progress() as bar:
        task = bar.add_task("sharding", total=None)

        def _on_layer(done: int, total: int, _mb: float) -> None:
            bar.update(task, completed=done, total=total, description=f"layer {done}/{total}")

        manifest = shard_model_by_layer(model_id, output_dir, progress=_on_layer)
    ui.ok(f"{manifest.num_layers} layers · {manifest.total_weight_mb / 1024:.1f} GB · "
          f"{manifest.weight_dtype}  →  swlp chat {args.model}")
    return 0


def _enough_disk(alias: str, output_dir: Path) -> bool:
    """Refuse a pull that cannot finish (only aliases with a known size)."""
    from .cli_doctor import KNOWN_FP16_GB

    known_gb = KNOWN_FP16_GB.get(alias)
    if known_gb is None:
        return True
    target = output_dir if output_dir.exists() else Path(".")
    free_gb = shutil.disk_usage(target).free / 1024**3
    needed_gb = known_gb * _PREFLIGHT_HEADROOM
    if free_gb >= needed_gb:
        return True
    ui.error(f"not enough disk for {alias}: needs ~{needed_gb:.0f} GB, {free_gb:.0f} GB free.",
             "use --output-dir on a bigger volume, or free some space")
    return False


def _models(args: argparse.Namespace) -> int:
    from .cli_models import print_models

    print_models()
    return 0


def _rm(args: argparse.Namespace) -> int:
    """Delete everything SWLP stored for a model — after showing it."""
    from .cli_models import model_artifacts

    items = model_artifacts(args.model)
    if not items:
        ui.note(f"  nothing on disk for {args.model}  (swlp models lists what is installed)")
        return 0
    total = sum(size for _, _, size in items)
    ui.table(["what", "size", "path"],
             [[kind, f"{size / 1024**3:.1f} GB", str(path)] for kind, path, size in items],
             title=f"{args.model}: {total / 1024**3:.1f} GB on disk")
    if not args.yes:
        try:
            answer = ui.console.input("\n  delete all of this? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer not in ("y", "yes"):
            ui.note("  kept everything.")
            return 0
    for _, path, _ in items:
        shutil.rmtree(path)
    ui.ok(f"removed {args.model} · freed {total / 1024**3:.1f} GB")
    return 0


def _doctor(args: argparse.Namespace) -> int:
    from .cli_doctor import print_doctor

    print_doctor(model=args.model)
    return 0


def _friendly(exc: Exception, args: argparse.Namespace) -> str:
    name = type(exc).__name__
    if name in ("RepositoryNotFoundError", "GatedRepoError"):
        model = getattr(args, "model", "the model")
        return (f"{model} was not found on Hugging Face (or needs a login). Use the full "
                "id, e.g. Qwen/Qwen2.5-7B-Instruct, or a name from  swlp models")
    first = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    return f"{name}: {first}" if first else name


def _quiet_hub() -> None:
    """Hub progress bars and server nags ("unauthenticated requests") would
    break the UI; SWLP draws its own progress. Runtime switches, because the
    package import already loaded huggingface_hub before main() runs."""
    from huggingface_hub.utils import disable_progress_bars
    from huggingface_hub.utils import logging as hf_logging

    disable_progress_bars()
    hf_logging.set_verbosity_error()


def _version() -> str:
    try:
        return version("swlp")
    except PackageNotFoundError:
        return "dev"


if __name__ == "__main__":
    raise SystemExit(main())
