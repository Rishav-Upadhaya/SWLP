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
from dataclasses import dataclass, field
from pathlib import Path

LOGGER = logging.getLogger(__name__)

MANIFEST_FILE = "shard_manifest.json"


@dataclass
class ShardManifest:
    model_id: str
    num_layers: int
    layer_weight_mb: float
    total_weight_mb: float
    embed_file: str
    lm_head_file: str
    model_type: str
    # On-disk weight precision of the layer shards: "float16" (default, legacy)
    # or "float8" (Phase 7 FP8 tier). Defaulted so old manifests still load.
    weight_dtype: str = "float16"
    # Phase 17: file format for layer shards: "safetensors" (new) or "pt" (legacy).
    # Defaulted to "pt" so old manifests (missing this field) still load correctly.
    shard_format: str = "pt"


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
    dtype_str: str = "float16",
    cache_dir: str | None = None,
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
    dtype = getattr(torch, dtype_str, torch.float16)
    LOGGER.info("shard_start", extra={"model_id": model_id, "output_dir": str(output_path)})

    local_path = _resolve_model_files(model_id, cache_dir)
    weight_map = _build_weight_map(local_path)

    cfg = AutoConfig.from_pretrained(str(local_path))
    model_type = getattr(cfg, "model_type", "unknown")
    num_layers = int(getattr(cfg, "num_hidden_layers", 0) or getattr(cfg, "n_layer", 0))
    if num_layers <= 0:
        raise ValueError(f"Could not determine layer count for {model_id}")

    is_gpt2 = any(k.startswith("transformer.h.") for k in weight_map)
    layer_prefix = "transformer.h." if is_gpt2 else "model.layers."
    LOGGER.info(
        "shard_model_layout",
        extra={"model_type": model_type, "num_layers": num_layers, "gpt2_layout": is_gpt2},
    )

    # Create output directory here — after the download succeeds — so it is
    # always fresh and exists exactly at write time (creating it before a
    # potentially 38-minute download meant macOS could evict the empty dir).
    output_path.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    for i in range(num_layers):
        layer_state = _read_prefixed(weight_map, f"{layer_prefix}{i}.", dtype)
        if not layer_state:
            raise ValueError(f"No weights found for layer {i} ({layer_prefix}{i}.*)")
        # Phase 17: write as .safetensors for zero-copy mmap loading.
        layer_path = output_path / f"layer_{i:03d}.safetensors"
        _save_safetensors(layer_state, layer_path)
        layer_bytes = layer_path.stat().st_size
        total_bytes += layer_bytes
        LOGGER.info(
            "shard_layer",
            extra={"layer": i, "total": num_layers, "mb": round(layer_bytes / 1e6, 1)},
        )
        del layer_state

    _stream_save_embed(weight_map, output_path / "embed.pt", is_gpt2, dtype)
    LOGGER.info("shard_saved_embed")
    _stream_save_lm_head(weight_map, output_path / "lm_head.pt", dtype)
    LOGGER.info("shard_saved_lm_head")

    layer_weight_mb = (total_bytes / num_layers) / 1e6 if num_layers else 0.0
    manifest = ShardManifest(
        model_id=model_id,
        num_layers=num_layers,
        layer_weight_mb=round(layer_weight_mb, 2),
        total_weight_mb=round(total_bytes / 1e6, 2),
        embed_file="embed.pt",
        lm_head_file="lm_head.pt",
        model_type=model_type,
        shard_format="safetensors",
    )
    _write_manifest(output_path, manifest)
    LOGGER.info(
        "shard_complete",
        extra={"num_layers": num_layers, "total_mb": manifest.total_weight_mb},
    )
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


def _read_prefixed(weight_map: dict[str, Path], prefix: str, dtype) -> dict:
    """Read all tensors whose key starts with ``prefix``, keyed by the
    prefix-stripped (block-relative) name and cast to ``dtype``."""
    from safetensors import safe_open

    by_file: dict[Path, list[str]] = {}
    for key in weight_map:
        if key.startswith(prefix):
            by_file.setdefault(weight_map[key], []).append(key)

    state: dict = {}
    for fpath, fkeys in by_file.items():
        with safe_open(str(fpath), framework="pt", device="cpu") as handle:
            for key in fkeys:
                state[key[len(prefix):]] = handle.get_tensor(key).to(dtype)
    return state


def _read_one(weight_map: dict[str, Path], key: str, dtype):
    """Read a single tensor by exact key; return None if the key is absent."""
    from safetensors import safe_open

    fpath = weight_map.get(key)
    if fpath is None:
        return None
    with safe_open(str(fpath), framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).to(dtype)


def _stream_save_embed(weight_map: dict[str, Path], path: Path, is_gpt2: bool, dtype) -> None:
    import torch

    state: dict = {}
    if is_gpt2:
        wte = _read_one(weight_map, "transformer.wte.weight", dtype)
        wpe = _read_one(weight_map, "transformer.wpe.weight", dtype)
        if wte is not None:
            state["wte"] = {"weight": wte}
        if wpe is not None:
            state["wpe"] = {"weight": wpe}
    else:
        embed = _read_one(weight_map, "model.embed_tokens.weight", dtype)
        norm = _read_one(weight_map, "model.norm.weight", dtype)
        if embed is not None:
            state["embed_tokens"] = {"weight": embed}
        if norm is not None:
            state["norm"] = {"weight": norm}
    torch.save(state, path)


def _stream_save_lm_head(weight_map: dict[str, Path], path: Path, dtype) -> None:
    import torch

    # Untied models expose lm_head.weight directly; tied models reuse the input
    # embedding — fall back to it so the lm_head shard is always populated.
    weight = _read_one(weight_map, "lm_head.weight", dtype)
    if weight is None:
        weight = _read_one(weight_map, "model.embed_tokens.weight", dtype)
    if weight is None:
        weight = _read_one(weight_map, "transformer.wte.weight", dtype)
    torch.save({"weight": weight} if weight is not None else {}, path)


def _save_safetensors(state_dict: dict, path: Path) -> None:
    """Write a layer state_dict to a .safetensors file.

    Handles two cases:
    - **Plain FP16/FP32** — a flat ``{name: tensor}`` dict is saved directly.
    - **FP8 quant** — the nested ``{"__swlp_quant__": "float8", "weights": ...}``
      format produced by ``quantize_layer_state()`` is flattened:
      2-D weights → ``{name}__fp8_data`` (fp8 tensor) + ``{name}__fp8_scale`` (fp16
      scale); 1-D biases → ``{name}`` (fp16, no scale). The scheme is stored in
      the safetensors metadata so ``_read_shard`` can reconstruct the nested format.
    """
    import torch
    from safetensors.torch import save_file as _st_save

    # Detect FP8 quant dicts (produced by quantize_layer_state("float8")).
    # _QUANT_KEY = "_swlp_quant" (single underscore, defined in quant.py).
    quant_scheme = state_dict.get("_swlp_quant", "")
    if isinstance(state_dict.get("weights"), dict) and quant_scheme:
        flat: dict[str, torch.Tensor] = {}
        metadata: dict[str, str] = {"__swlp_quant__": str(quant_scheme)}
        for name, entry in state_dict["weights"].items():
            data = entry["data"].contiguous().cpu()
            if "scale" in entry:
                flat[f"{name}__fp8_data"] = data
                flat[f"{name}__fp8_scale"] = entry["scale"].contiguous().cpu()
            else:
                # 1-D tensor (no scale) — stored directly.
                flat[name] = data
        _st_save(flat, str(path), metadata=metadata)
        return

    # Plain tensor dict (FP16/FP32 layer shards).
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
    }
    (output_path / MANIFEST_FILE).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_manifest(shard_dir: str | Path) -> ShardManifest:
    path = Path(shard_dir) / MANIFEST_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    return ShardManifest(**data)


def get_layer_path(shard_dir: str | Path, layer_idx: int, shard_format: str = "pt") -> Path:
    """Return the path to a layer shard.

    Prefer the explicit ``shard_format`` when known.  ``_read_shard`` in
    ``StreamingScheduler`` also auto-detects by trying both extensions.
    """
    ext = "safetensors" if shard_format == "safetensors" else "pt"
    return Path(shard_dir) / f"layer_{layer_idx:03d}.{ext}"


def list_layer_paths(shard_dir: str | Path) -> list[Path]:
    """Return sorted layer shard paths, preferring .safetensors over .pt."""
    d = Path(shard_dir)
    st_paths = sorted(d.glob("layer_*.safetensors"))
    if st_paths:
        return st_paths
    return sorted(d.glob("layer_*.pt"))


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


def _shard_file_ok(path: Path) -> bool:
    """Dispatch to the correct integrity check based on file extension."""
    if path.suffix == ".safetensors":
        return _safetensors_file_ok(path)
    return _pt_file_ok(path)


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
        layer_path = get_layer_path(shard_path, i, manifest.shard_format)
        if not layer_path.is_file():
            missing.append(layer_path.name)
        elif not _shard_file_ok(layer_path):
            corrupt.append(layer_path.name)

    return ShardIntegrityReport(ok=not missing and not corrupt, missing=missing, corrupt=corrupt)
