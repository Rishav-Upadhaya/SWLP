"""Split a HuggingFace model into per-layer shard files on disk.

Each shard is a ``.safetensors`` file (Phase 17) or legacy ``.pt`` file.
``StreamingScheduler`` streams these from NVMe into RAM as needed.

Phase 6 — streaming sharder
---------------------------
``shard_model_by_layer`` streams weights directly from the model's safetensors
files one tensor at a time (via ``safetensors.safe_open``). It never holds the
full model in RAM, so models far larger than system memory (14B, 30B, ...) can
be sharded on a modest machine — the earlier full-model ``from_pretrained``
load would OOM a 16 GB machine on anything past ~7B.

Phase 17 — safetensors shard format
-------------------------------------
Layer shards are now written as ``.safetensors`` files (mmap-backed, zero-copy
load via ``safetensors.safe_open``). ``embed.pt`` / ``lm_head.pt`` stay as
``.pt`` (nested-dict structure; read only once at startup). Legacy shard
directories with ``.pt`` layer files continue to work — format is auto-detected
by extension in ``StreamingScheduler._read_shard``.  The manifest gains a
``shard_format`` field: ``"safetensors"`` (new default) or ``"pt"`` (legacy).
"""
from __future__ import annotations

import json
import logging
import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .. import codec

LOGGER = logging.getLogger(__name__)

MANIFEST_FILE = "shard_manifest.json"
# Native multi-token-prediction head, written when the checkpoint has mtp.* tensors.
MTP_FILE = "mtp.safetensors"


@dataclass
class ShardManifest:
    model_id: str
    num_layers: int
    layer_weight_mb: float
    total_weight_mb: float
    embed_file: str
    lm_head_file: str
    model_type: str
    # On-disk weight precision of the layer shards ("float16" | "bfloat16").
    weight_dtype: str = "float16"
    # File format for layer shards: "safetensors" (new) or "pt" (legacy).
    # Defaulted to "pt" so old manifests (missing this field) still load correctly.
    shard_format: str = "pt"
    # Lossless layer-shard compression: "none" or "swz" (zipnn container,
    # produced by compress_shards). Defaulted so old manifests still load.
    shard_compression: str = "none"
    # MoE: expert weights live in per-layer ``.experts.safetensors``
    # banks when ``expert_bank`` is true; ``num_experts``/``top_k`` describe the
    # routing. Zero/False for dense models — old manifests load unchanged.
    num_experts: int = 0
    top_k: int = 0
    expert_bank: bool = False
    # Precision of the checkpoint the shards came from when it was itself
    # quantized ("fp8" for block-FP8 releases): shards are exact w.r.t. that
    # release, not the original full-precision model.
    source_quant: str = ""
    # Round-1 audit: per-layer expert-bank bytes (MB). layer_weight_mb is
    # dense-only; the expert cache budgets against this field instead.
    expert_weight_mb: float = 0.0


@dataclass
class ShardIntegrityReport:
    """Result of ``verify_shards`` — a pre-flight check on a shard directory."""

    ok: bool
    missing: list[str] = field(default_factory=list)
    corrupt: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.ok:
            return "shard directory OK"
        parts = []
        if self.missing:
            parts.append(f"missing: {', '.join(self.missing)}")
        if self.corrupt:
            parts.append(f"corrupt: {', '.join(self.corrupt)}")
        return "; ".join(parts)


# ── sharding (streaming, no full-model RAM load) ──────────────────────────────

def shard_model_by_layer(
    model_id: str,
    output_dir: str | Path,
    dtype_str: str = "auto",
    cache_dir: str | None = None,
    progress: Callable[[int, int, float], None] | None = None,
) -> ShardManifest:
    """Split a HF model into per-layer ``.pt`` shards by streaming from safetensors.

    Weights are read one tensor at a time via ``safetensors.safe_open``, so the
    full model is never resident — a 30B model shards fine on a 16 GB machine.

    Output layout:
      output_dir/embed.pt           – embeddings (wte/wpe for GPT-2; embed_tokens+norm for Llama)
      output_dir/lm_head.pt         – language-model head
      output_dir/layer_000.pt ...   – one file per transformer block
      output_dir/shard_manifest.json
    """
    import torch
    from transformers import AutoConfig

    output_path = Path(output_dir)
    LOGGER.info("shard_start", extra={"model_id": model_id, "output_dir": str(output_path)})

    local_path = _resolve_model_files(model_id, cache_dir)
    weight_map = _build_weight_map(local_path)

    full_cfg = AutoConfig.from_pretrained(str(local_path))
    model_type = getattr(full_cfg, "model_type", "unknown")
    # Multimodal wrappers (e.g. qwen3_5) nest the decoder under text_config.
    cfg = full_cfg.get_text_config()
    fp8_block = _fp8_block_size(full_cfg, cfg)
    if dtype_str == "auto":
        dtype_str = _native_half_dtype(cfg, full_cfg)
    dtype = getattr(torch, dtype_str, torch.float16)
    LOGGER.info("shard_dtype", extra={"weight_dtype": dtype_str})
    num_layers = int(getattr(cfg, "num_hidden_layers", 0) or getattr(cfg, "n_layer", 0))
    if num_layers <= 0:
        raise ValueError(f"Could not determine layer count for {model_id}")

    is_gpt2 = any(k.startswith("transformer.h.") for k in weight_map)
    # Multimodal checkpoints keep the text decoder under model.language_model.*
    text_prefix = "model.language_model." if any(
        k.startswith("model.language_model.layers.") for k in weight_map
    ) else "model."
    layer_prefix = "transformer.h." if is_gpt2 else f"{text_prefix}layers."
    LOGGER.info(
        "shard_model_layout",
        extra={"model_type": model_type, "num_layers": num_layers, "gpt2_layout": is_gpt2},
    )

    # Create output directory here — after the download succeeds — so it is
    # always fresh and exists exactly at write time (creating it before a
    # potentially 38-minute download meant macOS could evict the empty dir).
    output_path.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    dense_bytes = 0
    expert_bytes_total = 0
    has_experts = False
    for i in range(num_layers):
        layer_state = _read_prefixed(weight_map, f"{layer_prefix}{i}.", dtype, fp8_block)
        if not layer_state:
            raise ValueError(f"No weights found for layer {i} ({layer_prefix}{i}.*)")
        # MoE expert tensors go to a separate bank file so the dense
        # stream never re-reads expert bytes and experts can be range-read.
        from .expert_bank import split_expert_tensors

        dense_state, expert_state = split_expert_tensors(layer_state)
        if expert_state:
            bank_path = output_path / f"layer_{i:03d}.experts.safetensors"
            _save_safetensors(expert_state, bank_path)
            has_experts = True
            bank_bytes = bank_path.stat().st_size
            total_bytes += bank_bytes
            expert_bytes_total += bank_bytes
        # Write as .safetensors for zero-copy mmap loading.
        layer_path = output_path / f"layer_{i:03d}.safetensors"
        _save_safetensors(dense_state, layer_path)
        layer_bytes = layer_path.stat().st_size
        total_bytes += layer_bytes
        dense_bytes += layer_bytes
        LOGGER.info(
            "shard_layer",
            extra={"layer": i, "total": num_layers, "mb": round(layer_bytes / 1e6, 1)},
        )
        if progress is not None:
            progress(i + 1, num_layers, layer_bytes / 1e6)
        del layer_state

    _stream_save_embed(weight_map, output_path / "embed.pt", is_gpt2, dtype, text_prefix,
                       fp8_block)
    LOGGER.info("shard_saved_embed")
    _stream_save_lm_head(weight_map, output_path / "lm_head.pt", dtype, text_prefix,
                         fp8_block)
    LOGGER.info("shard_saved_lm_head")
    # Native multi-token-prediction head (Qwen3.5/3.8): resident draft layer
    # for self-speculative decoding (runner/mtp.py). HF drops these weights.
    mtp_state = _read_prefixed(weight_map, "mtp.", dtype, fp8_block)
    if mtp_state:
        _save_safetensors(mtp_state, output_path / MTP_FILE)
        LOGGER.info("shard_saved_mtp", extra={"tensors": len(mtp_state)})

    # num_experts is also readable from routed-expert
    # configs that do not use the plain "num_experts" spelling.
    num_experts = int(
        getattr(cfg, "num_experts", 0)
        or getattr(cfg, "num_local_experts", 0)
        or getattr(cfg, "num_routed_experts", 0)
    )
    top_k = int(getattr(cfg, "num_experts_per_tok", 0) or 0)
    if has_experts and num_experts == 0:
        # Config lacked expert fields (unusual layout) — derive from layer 0.
        first = _read_prefixed(weight_map, f"{layer_prefix}0.", dtype, fp8_block)
        from .expert_bank import expert_count as _expert_count

        num_experts = _expert_count(first)
    if has_experts:
        from .expert_bank import EXPERT_INDEX_FILE, ExpertIndex

        index = ExpertIndex.build(output_path, num_layers)
        index.save(output_path / EXPERT_INDEX_FILE)
        LOGGER.info("shard_expert_banks", extra={"layers": len(index.layers),
                                                 "num_experts": num_experts})

    # Dense-only per-layer size: the streaming window never holds expert
    # bytes (banks are range-read by the ExpertScheduler), so residency and
    # feasibility must plan against the dense stream. Expert bytes are kept
    # separately for cache-budget reasoning.
    layer_weight_mb = (dense_bytes / num_layers) / 1e6 if num_layers else 0.0
    expert_weight_mb = (expert_bytes_total / num_layers) / 1e6 if num_layers else 0.0
    manifest = ShardManifest(
        model_id=model_id,
        num_layers=num_layers,
        layer_weight_mb=round(layer_weight_mb, 2),
        total_weight_mb=round(total_bytes / 1e6, 2),
        expert_weight_mb=round(expert_weight_mb, 2),
        embed_file="embed.pt",
        lm_head_file="lm_head.pt",
        model_type=model_type,
        weight_dtype=dtype_str,
        source_quant="fp8" if fp8_block else "",
        shard_format="safetensors",
        num_experts=num_experts,
        top_k=top_k,
        expert_bank=has_experts,
    )
    _write_manifest(output_path, manifest)
    LOGGER.info(
        "shard_complete",
        extra={"num_layers": num_layers, "total_mb": manifest.total_weight_mb},
    )
    return manifest


def compress_shards(
    shard_dir: str | Path,
    progress: Callable[[int, int, float], None] | None = None,
) -> ShardManifest:
    """Losslessly compress a shard directory's layer files in place.

    Each ``layer_XXX.safetensors`` becomes ``layer_XXX.safetensors.swz``
    (~31% smaller, bit-exact). Per layer the original is deleted only after
    ``codec.compress_bytes`` has roundtrip-verified the blob and the ``.swz``
    file is fully on disk — peak extra disk is one compressed layer, and an
    interrupted run resumes where it stopped (already-converted layers are
    skipped; readers auto-detect both extensions per file).

    ``embed.pt`` / ``lm_head.pt`` stay uncompressed: they are read once at
    startup and the layer shards dominate the directory size.
    """
    shard_path = Path(shard_dir)
    manifest = load_manifest(shard_path)
    if manifest.shard_format != "safetensors":
        raise ValueError(
            f"compress_shards requires safetensors shards, got {manifest.shard_format!r}"
        )
    if manifest.weight_dtype not in ("float16", "bfloat16"):
        raise ValueError(
            f"unsupported weight_dtype for swz compression: {manifest.weight_dtype!r}"
        )
    for i in range(manifest.num_layers):
        src = shard_path / f"layer_{i:03d}.safetensors"
        dst = codec.compressed_path(src)
        if dst.exists() and not src.exists():
            LOGGER.debug("compress_shards_skip_done", extra={"layer": i})
            continue
        raw = src.read_bytes()
        blob = codec.compress_bytes(raw)  # roundtrip-verified before src is deleted
        tmp = dst.with_name(dst.name + ".tmp")
        tmp.write_bytes(blob)
        tmp.replace(dst)
        src.unlink()
        ratio = len(blob) / len(raw)
        LOGGER.info(
            "compress_shards_layer",
            extra={"layer": i, "total": manifest.num_layers, "ratio": round(ratio, 4)},
        )
        if progress is not None:
            progress(i + 1, manifest.num_layers, ratio)
    manifest.shard_compression = "swz"
    _write_manifest(shard_path, manifest)
    LOGGER.info("compress_shards_complete", extra={"num_layers": manifest.num_layers})
    return manifest


def decompress_shards(
    shard_dir: str | Path,
    progress: Callable[[int, int, float], None] | None = None,
) -> ShardManifest:
    """Revert :func:`compress_shards`: restore plain ``.safetensors`` layers.

    The throughput off-ramp for fast-SSD machines (≳3.5 GB/s sequential read),
    where decompression contends with compute for memory bandwidth and costs
    more than the read bytes it saves — see the ``codec`` module docstring.

    Mirrors compress_shards' safety story: each layer is SHA-256-verified
    against the digest stored at compress time, written to a temp file, and
    atomically renamed before its ``.swz`` source is deleted. An interrupted
    run resumes where it stopped.
    """
    shard_path = Path(shard_dir)
    manifest = load_manifest(shard_path)
    if manifest.shard_format != "safetensors":
        raise ValueError(
            f"decompress_shards requires safetensors shards, got {manifest.shard_format!r}"
        )
    for i in range(manifest.num_layers):
        dst = shard_path / f"layer_{i:03d}.safetensors"
        src = codec.compressed_path(dst)
        if dst.exists() and _safetensors_file_ok(dst):
            src.unlink(missing_ok=True)  # already reverted; drop a leftover .swz
            LOGGER.debug("decompress_shards_skip_done", extra={"layer": i})
            continue
        blob = src.read_bytes()
        raw = codec.decompress_bytes(blob, check_sha=True)
        tmp = dst.with_name(dst.name + ".tmp")
        tmp.write_bytes(raw)
        tmp.replace(dst)
        src.unlink()
        ratio = len(blob) / len(raw)
        LOGGER.info(
            "decompress_shards_layer",
            extra={"layer": i, "total": manifest.num_layers, "ratio": round(ratio, 4)},
        )
        if progress is not None:
            progress(i + 1, manifest.num_layers, ratio)
    manifest.shard_compression = "none"
    _write_manifest(shard_path, manifest)
    LOGGER.info("decompress_shards_complete", extra={"num_layers": manifest.num_layers})
    return manifest


def _resolve_model_files(model_id: str, cache_dir: str | None) -> Path:
    """Return a local dir holding the model's safetensors + config.

    A local filesystem path is used as-is; otherwise the repo is fetched into
    the HF cache (only weights/config/tokenizer patterns).
    """
    p = Path(model_id)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        model_id,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json", "tokenizer*", "vocab.json", "merges.txt"],
    )
    return Path(local)


def _build_weight_map(local_path: Path) -> dict[str, Path]:
    """Map every weight key -> the safetensors file that holds it."""
    from safetensors import safe_open

    index_file = local_path / "model.safetensors.index.json"
    if index_file.exists():
        index = json.loads(index_file.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map", {})
        return {key: local_path / fname for key, fname in weight_map.items()}

    files = sorted(local_path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {local_path}")
    mapping: dict[str, Path] = {}
    for f in files:
        with safe_open(str(f), framework="pt") as handle:
            for key in handle.keys():
                mapping[key] = f
    return mapping


_FP8_SCALE_SUFFIX = "_scale_inv"


def _fp8_block_size(*cfgs) -> tuple[int, int] | None:
    """Block size of a block-FP8 checkpoint (dequantized at shard time); None
    for unquantized ones. Other quantized formats are refused: casting their
    packed weights would silently produce garbage shards."""
    for cfg in cfgs:
        q = getattr(cfg, "quantization_config", None)
        if not q:
            continue
        q = q if isinstance(q, dict) else q.to_dict()
        if q.get("quant_method") == "fp8" and q.get("weight_block_size"):
            b = q["weight_block_size"]
            return int(b[0]), int(b[1])
        raise ValueError(
            f"{q.get('quant_method', 'this')}-quantized checkpoint: SWLP shards "
            "full-precision (bf16/fp16) or block-FP8 checkpoints — pull the "
            "unquantized repo, or run an MLX-format repo directly"
        )
    return None


def _native_half_dtype(*cfgs) -> str:
    """The checkpoint's own half-precision dtype ("bfloat16" | "float16").

    bf16 -> fp16 is not lossless: fp16 loses values below 2**-17 and overflows
    above 65504 (Gated-DeltaNet state does), so bf16 checkpoints stay bf16.
    Anything else (fp32, unknown) shards to float16 as before.
    """
    for cfg in cfgs:  # text config first, then the multimodal wrapper
        native = getattr(cfg, "dtype", None) or getattr(cfg, "torch_dtype", None)
        name = str(native).replace("torch.", "") if native is not None else ""
        if name in ("bfloat16", "float16"):
            return name
    return "float16"


def _read_prefixed(weight_map: dict[str, Path], prefix: str, dtype,
                   fp8_block: tuple[int, int] | None = None) -> dict:
    """Read all tensors whose key starts with ``prefix``, keyed by the
    prefix-stripped (block-relative) name and cast to ``dtype``. FP8 weights
    are dequantized with their ``*_scale_inv`` block scales first."""
    from safetensors import safe_open

    by_file: dict[Path, list[str]] = {}
    for key in weight_map:
        if key.startswith(prefix):
            by_file.setdefault(weight_map[key], []).append(key)
    raw: dict = {}
    for fpath, fkeys in by_file.items():  # one open per file, not per tensor
        with safe_open(str(fpath), framework="pt", device="cpu") as handle:
            for key in fkeys:
                raw[key[len(prefix):]] = handle.get_tensor(key)
    return {name: _dequantized(name, t, raw, dtype, fp8_block)
            for name, t in raw.items() if not name.endswith(_FP8_SCALE_SUFFIX)}


def _read_one(weight_map: dict[str, Path], key: str, dtype,
              fp8_block: tuple[int, int] | None = None):
    """Read a single tensor by exact key; return None if the key is absent."""
    if key not in weight_map:
        return None
    raw = {key: _get_tensor(weight_map, key)}
    scale_key = key + _FP8_SCALE_SUFFIX
    if scale_key in weight_map:
        raw[scale_key] = _get_tensor(weight_map, scale_key)
    return _dequantized(key, raw[key], raw, dtype, fp8_block)


def _get_tensor(weight_map: dict[str, Path], key: str):
    from safetensors import safe_open

    with safe_open(str(weight_map[key]), framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def _dequantized(name: str, tensor, raw: dict, dtype, fp8_block: tuple[int, int] | None):
    """``tensor`` cast to ``dtype``; block-FP8 weights (DeepSeek/Qwen "fp8"
    format: e4m3 values + one ``weight_scale_inv`` per block) are multiplied by
    their block scale in fp32 first. Casting the raw FP8 values alone yields
    weights up to 448 — the e4m3 maximum — i.e. garbage."""
    scale = raw.get(name + _FP8_SCALE_SUFFIX)
    if scale is None:
        return tensor.to(dtype)
    if fp8_block is None:
        raise ValueError(f"{name}: FP8 scales present but no weight_block_size in the config")
    rows, cols = tensor.shape
    b_rows, b_cols = fp8_block
    full = scale.float().repeat_interleave(b_rows, 0)[:rows].repeat_interleave(b_cols, 1)[:, :cols]
    return (tensor.float() * full).to(dtype)


def _stream_save_embed(
    weight_map: dict[str, Path], path: Path, is_gpt2: bool, dtype, text_prefix: str = "model.",
    fp8_block: tuple[int, int] | None = None,
) -> None:
    import torch

    state: dict = {}
    if is_gpt2:
        wte = _read_one(weight_map, "transformer.wte.weight", dtype, fp8_block)
        wpe = _read_one(weight_map, "transformer.wpe.weight", dtype, fp8_block)
        if wte is not None:
            state["wte"] = {"weight": wte}
        if wpe is not None:
            state["wpe"] = {"weight": wpe}
        # Final LayerNorm is permanently on-device but belongs to no layer
        # shard — persist it here, or the loader materialises it from
        # uninitialized (freshly zeroed) memory: first-run zero logits.
        ln_f: dict = {}
        ln_w = _read_one(weight_map, "transformer.ln_f.weight", dtype, fp8_block)
        ln_b = _read_one(weight_map, "transformer.ln_f.bias", dtype, fp8_block)
        if ln_w is not None:
            ln_f["weight"] = ln_w
        if ln_b is not None:
            ln_f["bias"] = ln_b
        if ln_f:
            state["ln_f"] = ln_f
    else:
        embed = _read_one(weight_map, f"{text_prefix}embed_tokens.weight", dtype, fp8_block)
        norm = _read_one(weight_map, f"{text_prefix}norm.weight", dtype, fp8_block)
        if embed is not None:
            state["embed_tokens"] = {"weight": embed}
        if norm is not None:
            state["norm"] = {"weight": norm}
    torch.save(state, path)


def _stream_save_lm_head(
    weight_map: dict[str, Path], path: Path, dtype, text_prefix: str = "model.",
    fp8_block: tuple[int, int] | None = None,
) -> None:
    import torch

    # Untied models expose lm_head.weight directly; tied models reuse the input
    # embedding — fall back to it so the lm_head shard is always populated.
    weight = _read_one(weight_map, "lm_head.weight", dtype, fp8_block)
    if weight is None:
        weight = _read_one(weight_map, f"{text_prefix}embed_tokens.weight", dtype, fp8_block)
    if weight is None:
        weight = _read_one(weight_map, "transformer.wte.weight", dtype, fp8_block)
    torch.save({"weight": weight} if weight is not None else {}, path)


def _save_safetensors(state_dict: dict, path: Path) -> None:
    """Write a flat ``{name: tensor}`` layer state_dict to a .safetensors file."""
    import torch
    from safetensors.torch import save_file as _st_save

    safe_state: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        safe_state[k] = v.contiguous().cpu()
    _st_save(safe_state, str(path))


# ── manifest + shard-path helpers ─────────────────────────────────────────────

def _write_manifest(output_path: Path, manifest: ShardManifest) -> None:
    data = {
        "model_id": manifest.model_id,
        "num_layers": manifest.num_layers,
        "layer_weight_mb": manifest.layer_weight_mb,
        "total_weight_mb": manifest.total_weight_mb,
        "embed_file": manifest.embed_file,
        "lm_head_file": manifest.lm_head_file,
        "model_type": manifest.model_type,
        "weight_dtype": manifest.weight_dtype,
        "shard_format": manifest.shard_format,
        "shard_compression": manifest.shard_compression,
        "num_experts": manifest.num_experts,
        "top_k": manifest.top_k,
        "expert_bank": manifest.expert_bank,
        "expert_weight_mb": manifest.expert_weight_mb,
        "source_quant": manifest.source_quant,
    }
    (output_path / MANIFEST_FILE).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_manifest(shard_dir: str | Path) -> ShardManifest:
    path = Path(shard_dir) / MANIFEST_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    manifest = ShardManifest(**data)
    if manifest.weight_dtype == "float8":
        raise ValueError(
            f"{shard_dir}: FP8 shards are no longer supported; re-shard with `swlp pull`"
        )
    return manifest


def get_layer_path(
    shard_dir: str | Path,
    layer_idx: int,
    shard_format: str = "pt",
    shard_compression: str = "none",
) -> Path:
    """Return the path to a layer shard.

    Prefer the explicit ``shard_format`` / ``shard_compression`` when known.
    ``_shard_path`` in ``StreamingScheduler`` also auto-detects per file by
    trying all extensions.
    """
    ext = "safetensors" if shard_format == "safetensors" else "pt"
    path = Path(shard_dir) / f"layer_{layer_idx:03d}.{ext}"
    if shard_compression == "swz" and ext == "safetensors":
        return codec.compressed_path(path)
    return path


def list_layer_paths(shard_dir: str | Path) -> list[Path]:
    """Return sorted dense-layer shard paths (.safetensors, .swz, then .pt).

    Expert banks (``layer_XXX.experts.safetensors``) are excluded — they are
    range-read by the ExpertScheduler, never streamed whole.
    """
    d = Path(shard_dir)
    for pattern in (
        "layer_[0-9][0-9][0-9].safetensors",
        "layer_[0-9][0-9][0-9].safetensors.swz",
        "layer_[0-9][0-9][0-9].pt",
    ):
        paths = sorted(d.glob(pattern))
        if paths:
            return paths
    return []


# ── shard-integrity check ─────────────────────────────────────────────────────

# A torch ``.pt`` file is a ZIP archive — it must start with the "PK" magic.
_ZIP_MAGIC = b"PK"


def _pt_file_ok(path: Path) -> bool:
    """Cheap corruption check: file exists, is non-empty, and has the ZIP magic
    of a torch-serialised ``.pt`` archive. Does not load the tensors."""
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        with path.open("rb") as fh:
            return fh.read(2) == _ZIP_MAGIC
    except OSError:
        return False


def _safetensors_file_ok(path: Path) -> bool:
    """Cheap corruption check for a safetensors file.

    The safetensors format starts with an 8-byte LE uint64 header-length field.
    A valid file satisfies: ``header_len + 8 <= file_size`` and header_len > 0.
    """
    try:
        if not path.is_file():
            return False
        file_size = path.stat().st_size
        if file_size < 8:
            return False
        with path.open("rb") as fh:
            raw = fh.read(8)
        if len(raw) < 8:
            return False
        (header_len,) = struct.unpack("<Q", raw)
        return 0 < header_len <= file_size - 8
    except (OSError, struct.error):
        return False


def _swz_file_ok(path: Path) -> bool:
    """Cheap corruption check for a compressed ``.swz`` shard.

    Validates the container header (magic, positive raw size) and that a
    payload follows it. Does not decode — use ``codec.verify_blob`` for the
    full offline check.
    """
    try:
        if not path.is_file() or path.stat().st_size <= codec.HEADER_SIZE:
            return False
        with path.open("rb") as fh:
            head = fh.read(codec.HEADER_SIZE)
        return codec.decompressed_size(head) > 0
    except (OSError, codec.CodecError):
        return False


def _shard_file_ok(path: Path) -> bool:
    """Dispatch to the correct integrity check based on file extension."""
    if path.suffix == codec.COMPRESSED_SUFFIX:
        return _swz_file_ok(path)
    if path.suffix == ".safetensors":
        return _safetensors_file_ok(path)
    return _pt_file_ok(path)


def _existing_layer_path(shard_dir: Path, layer_idx: int, manifest: ShardManifest) -> Path | None:
    """First existing on-disk variant of a layer — manifest-preferred first.

    A directory mid-conversion (interrupted ``compress_shards``) legitimately
    holds a mix of ``.safetensors`` and ``.swz`` layers; readers auto-detect
    per file, so verification accepts either variant.
    """
    preferred = get_layer_path(
        shard_dir, layer_idx, manifest.shard_format, manifest.shard_compression
    )
    candidates = [preferred]
    if manifest.shard_format == "safetensors":
        st = get_layer_path(shard_dir, layer_idx, "safetensors")
        candidates.extend([st, codec.compressed_path(st)])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def verify_shards(shard_dir: str | Path) -> ShardIntegrityReport:
    """Verify a shard directory before use.

    Checks that the manifest is present and parseable, and that every layer
    shard plus ``embed.pt`` / ``lm_head.pt`` exists and is a non-empty,
    well-formed archive.  Supports both ``.safetensors`` (Phase 17) and legacy
    ``.pt`` layer shards.
    """
    shard_path = Path(shard_dir)
    manifest_path = shard_path / MANIFEST_FILE
    if not manifest_path.is_file():
        return ShardIntegrityReport(ok=False, missing=[MANIFEST_FILE])
    try:
        manifest = load_manifest(shard_path)
    except (json.JSONDecodeError, TypeError, KeyError, OSError):
        return ShardIntegrityReport(ok=False, corrupt=[MANIFEST_FILE])

    missing: list[str] = []
    corrupt: list[str] = []
    # embed + lm_head are always .pt (nested-dict format; not converted to safetensors).
    for name in (manifest.embed_file, manifest.lm_head_file):
        target = shard_path / name
        if not target.is_file():
            missing.append(name)
        elif not _pt_file_ok(target):
            corrupt.append(name)
    for i in range(manifest.num_layers):
        layer_path = _existing_layer_path(shard_path, i, manifest)
        if layer_path is None:
            expected = get_layer_path(
                shard_path, i, manifest.shard_format, manifest.shard_compression
            )
            missing.append(expected.name)
        elif not _shard_file_ok(layer_path):
            corrupt.append(layer_path.name)
        # Expert banks are separate files the loader range-reads.
        if manifest.expert_bank:
            bank = shard_path / f"layer_{i:03d}.experts.safetensors"
            if not bank.is_file():
                # v1 shard dirs (pre-split) keep experts inside the layer
                # file — verify that instead of trusting the layout blind.
                from .expert_bank import file_has_expert_tensors

                if (
                    layer_path is not None
                    and layer_path.suffix == ".safetensors"
                    and file_has_expert_tensors(layer_path)
                ):
                    continue
                missing.append(bank.name)
            elif not _safetensors_file_ok(bank):
                corrupt.append(bank.name)

    return ShardIntegrityReport(ok=not missing and not corrupt, missing=missing, corrupt=corrupt)
