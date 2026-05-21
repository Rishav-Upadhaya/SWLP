"""FP8 weight quantization for SWLP layer shards (Phase 7).

Layer shard files are torch ``.pt`` state dicts. The FP16 path stores tensors
directly; the FP8 path stores each 2-D weight matrix as a ``float8_e4m3``
tensor plus a per-output-channel FP16 scale, halving on-disk and in-RAM size.
Compute stays FP16 — dequantization happens when a layer is materialised into
its block module by ``StreamingScheduler``.

Per-output-channel-scaled FP8 weight quantization retains ~99-100% of FP16
benchmark performance (see Phase 7 notes / 2025 quantization literature), so it
is treated as the near-lossless default tier.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import torch

from .shard import (
    MANIFEST_FILE,
    ShardManifest,
    _save_safetensors,
    _write_manifest,
    get_layer_path,
    load_manifest,
)

LOGGER = logging.getLogger(__name__)

# float8_e4m3fn representable range is ±448.
_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = 448.0
# Marker key identifying a quantized shard dict.
_QUANT_KEY = "_swlp_quant"
_EPS = 1e-12

SUPPORTED_SCHEMES = ("float16", "float8")


def quantize_layer_state(state: dict, scheme: str = "float8") -> dict:
    """Quantize a flat layer state dict into a serializable dict.

    ``scheme="float16"`` returns the state unchanged (no quantization).
    ``scheme="float8"`` wraps each entry: 2-D weight matrices become
    ``float8_e4m3`` with a per-output-channel FP16 scale; other tensors
    (1-D norm weights / biases) are kept FP16 — they are tiny and quantizing
    them buys nothing.
    """
    if scheme == "float16":
        return dict(state)
    if scheme != "float8":
        raise ValueError(f"Unknown quantization scheme: {scheme!r}")
    weights = {name: _quantize_tensor(t) for name, t in state.items()}
    return {_QUANT_KEY: "float8", "weights": weights}


def is_quantized(raw: dict) -> bool:
    """True if ``raw`` is a quantized shard dict (vs. a plain FP16 state dict)."""
    return isinstance(raw, dict) and _QUANT_KEY in raw


def dequantize_layer_state(raw: dict) -> dict:
    """Reconstruct a flat FP16 state dict from a (possibly quantized) shard dict.

    A plain FP16 state dict is returned unchanged, so this is safe to call on
    legacy shards and on already-dequantized state.
    """
    if not is_quantized(raw):
        return raw
    weights = raw.get("weights", {})
    return {name: _dequantize_entry(entry) for name, entry in weights.items()}


def _quantize_tensor(tensor: torch.Tensor) -> dict:
    if tensor.dim() == 2:
        t = tensor.to(torch.float32)
        scale = (t.abs().amax(dim=1, keepdim=True) / _FP8_MAX).clamp(min=_EPS)
        q = (t / scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE)
        return {"data": q, "scale": scale.to(torch.float16)}
    # 1-D tensors are kept FP16; no "scale" key signals "already real".
    return {"data": tensor.to(torch.float16)}


def _dequantize_entry(entry: dict) -> torch.Tensor:
    data = entry["data"]
    scale = entry.get("scale")
    if scale is None:
        return data.to(torch.float16)
    return (data.to(torch.float16) * scale).to(torch.float16)


def requantize_shards(
    src_dir: str | Path,
    dst_dir: str | Path,
    scheme: str = "float8",
) -> ShardManifest:
    """Read an existing shard directory and write a re-quantized copy.

    Only ``layer_*.pt`` shards are quantized — ``embed.pt`` / ``lm_head.pt`` are
    copied verbatim (resident modules; quantizing them is a later sub-phase).
    The source may itself be quantized; layers are normalized to FP16 first.
    """
    if scheme not in SUPPORTED_SCHEMES:
        raise ValueError(f"Unknown quantization scheme: {scheme!r}")
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(src)
    LOGGER.info(
        "requantize_start",
        extra={"src": str(src), "dst": str(dst), "scheme": scheme,
               "num_layers": manifest.num_layers},
    )

    total_bytes = 0
    for i in range(manifest.num_layers):
        src_path = get_layer_path(src, i, manifest.shard_format)
        if src_path.suffix == ".safetensors":
            from safetensors import safe_open
            raw: dict = {}
            with safe_open(str(src_path), framework="pt", device="cpu") as handle:
                for k in handle.keys():
                    raw[k] = handle.get_tensor(k)
        else:
            raw = torch.load(str(src_path), map_location="cpu", weights_only=True)
        state = dequantize_layer_state(raw)
        # Phase 17: write FP8 shards as .safetensors.
        out_path = dst / f"layer_{i:03d}.safetensors"
        _save_safetensors(quantize_layer_state(state, scheme), out_path)
        total_bytes += out_path.stat().st_size
        LOGGER.info("requantize_layer", extra={"layer": i, "total": manifest.num_layers})

    for name in (manifest.embed_file, manifest.lm_head_file):
        shutil.copy2(src / name, dst / name)

    new_manifest = ShardManifest(
        model_id=manifest.model_id,
        num_layers=manifest.num_layers,
        layer_weight_mb=round((total_bytes / manifest.num_layers) / 1e6, 2)
        if manifest.num_layers else 0.0,
        total_weight_mb=round(total_bytes / 1e6, 2),
        embed_file=manifest.embed_file,
        lm_head_file=manifest.lm_head_file,
        model_type=manifest.model_type,
        weight_dtype=scheme,
        shard_format="safetensors",
    )
    _write_manifest(dst, new_manifest)
    LOGGER.info(
        "requantize_complete",
        extra={"total_mb": new_manifest.total_weight_mb,
               "layer_mb": new_manifest.layer_weight_mb},
    )
    # Carry the source manifest's auxiliary file names; touch MANIFEST_FILE name
    # only through _write_manifest so the on-disk format stays canonical.
    assert (dst / MANIFEST_FILE).is_file()
    return new_manifest
