"""MoE expert bank — partial reads of expert weights from layer shards (Phase 25).

A MoE layer's expert weights are the bulk of its bytes (e.g. Qwen3-30B-A3B:
~93% of every layer). Streaming them whole with the dense part defeats the
point of expert routing, so the sharder (v2) splits them into a per-layer
**expert bank** file ``layer_{i:03d}.experts.safetensors`` and this module
reads *individual experts* out of it by byte range:

    safetensors header  →  per-tensor data_offsets  →  expert j's rows

Two on-disk layouts are supported, both normalized to ``(gate, up, down)``
slots by :func:`read_expert`:

- **stacked** (transformers ≥5 fused ``Experts`` modules): tensors named
  ``…experts.gate_up_proj`` ``[E, 2I, H]`` / ``…experts.down_proj`` ``[E, H, I]``
  (qwen3-moe family) or ``…experts.w1/w2/w3`` ``[E, H, I]`` (mixtral family).
  Expert *j* is the contiguous row range ``[j, j+1)`` of each stacked tensor.
- **per-expert** (original HF checkpoints): ``…experts.{j}.gate_proj /
  up_proj / down_proj`` (or ``w1/w2/w3``) — one whole tensor per expert.

Only ``.safetensors`` banks support ranged reads; legacy ``.pt`` MoE shards
fall back to whole-layer streaming (exact, but no expert caching).
Quantized banks (``__swlp_quant__`` metadata) are rejected — expert slots
must hold the model's compute dtype for exactness.
"""
from __future__ import annotations

import json
import logging
import os
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

LOGGER = logging.getLogger(__name__)

# Slot names experts are normalized to. "gate"+"up" are fused by the consumer
# (runner/experts.py) into a single [2I, H] gate_up slot, matching the fused
# matmul the reference Experts modules perform (bit-exactness requirement).
SLOT_KEYS = ("gate", "up", "down")

_PER_EXPERT_KEY = re.compile(
    r"(?P<stem>.+)\.experts\.(?P<idx>\d+)\.(?P<param>gate_proj|up_proj|down_proj|w1|w2|w3)"
    r"(?:\.weight)?$"
)
_STACKED_STEMS = {
    "gate_up_proj": "gate_up",   # qwen3-moe fused [E, 2I, H]
    "down_proj": "down",
    "w1": "gate",                # mixtral fused [E, H, I] × 3
    "w2": "down",
    "w3": "up",
}
_PER_EXPERT_PARAM_TO_SLOT = {
    "gate_proj": "gate", "up_proj": "up", "down_proj": "down",
    "w1": "gate", "w2": "down", "w3": "up",
}
_BANK_SUFFIX = ".experts.safetensors"
EXPERT_INDEX_FILE = "expert_index.json"

_ITEMSIZE = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8}


class ExpertBankError(RuntimeError):
    """The expert bank is unreadable, unsupported, or inconsistent."""


@dataclass(slots=True)
class ExpertSlice:
    """One contiguous byte range backing one slot of one expert."""

    slot: str            # "gate" | "up" | "down"
    offset: int          # absolute file offset
    nbytes: int
    dtype_str: str       # safetensors dtype ("F16" | "BF16" | …)
    rows: int
    cols: int
    # Absolute source file when it is not the layer's bank_file (MLX-format
    # checkpoints spread one layer's expert tensors across shard files).
    file: str = ""

    def to_dict(self) -> dict:
        d = {
            "slot": self.slot, "offset": self.offset, "nbytes": self.nbytes,
            "dtype": self.dtype_str, "rows": self.rows, "cols": self.cols,
        }
        if self.file:
            d["file"] = self.file
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ExpertSlice:
        return cls(slot=d["slot"], offset=int(d["offset"]), nbytes=int(d["nbytes"]),
                   dtype_str=d["dtype"], rows=int(d["rows"]), cols=int(d["cols"]),
                   file=d.get("file", ""))


@dataclass(slots=True)
class ExpertLayerIndex:
    """Ranged-read map of every expert in one layer's bank."""

    bank_file: str                     # path relative to the shard dir
    num_experts: int
    slices: list[list[ExpertSlice]]    # expert id → slot slices
    dtype_str: str = "F16"
    hidden: int = 0
    intermediate: int = 0

    def expert_bytes(self) -> int:
        if not self.slices:
            return 0
        return sum(s.nbytes for s in self.slices[0])

    def to_dict(self) -> dict:
        return {
            "bank_file": self.bank_file,
            "num_experts": self.num_experts,
            "dtype": self.dtype_str,
            "hidden": self.hidden,
            "intermediate": self.intermediate,
            "slices": [[s.to_dict() for s in exp] for exp in self.slices],
        }

    @classmethod
    def from_dict(cls, d: dict) -> ExpertLayerIndex:
        return cls(
            bank_file=d["bank_file"],
            num_experts=int(d["num_experts"]),
            slices=[[ExpertSlice.from_dict(s) for s in exp] for exp in d["slices"]],
            dtype_str=d.get("dtype", "F16"),
            hidden=int(d.get("hidden", 0)),
            intermediate=int(d.get("intermediate", 0)),
        )


@dataclass(slots=True)
class ExpertIndex:
    """All layers' expert banks in one shard directory."""

    layers: dict[int, ExpertLayerIndex] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.layers)

    def save(self, path: str | Path) -> None:
        payload = {str(k): v.to_dict() for k, v in sorted(self.layers.items())}
        Path(path).write_text(json.dumps(payload, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> ExpertIndex:
        data = json.loads(Path(path).read_text())
        return cls(layers={int(k): ExpertLayerIndex.from_dict(v) for k, v in data.items()})

    @classmethod
    def build(cls, shard_dir: str | Path, num_layers: int) -> ExpertIndex:
        """Scan a shard dir for per-layer expert banks and index them.

        Prefers v2 ``layer_{i}.experts.safetensors`` banks; a v1 fused
        ``layer_{i}.safetensors`` containing stacked expert tensors is indexed
        in place (works, but the dense stream re-reads expert bytes too).
        """
        root = Path(shard_dir)
        layers: dict[int, ExpertLayerIndex] = {}
        for i in range(num_layers):
            bank = root / f"layer_{i:03d}{_BANK_SUFFIX}"
            source = bank if bank.exists() else root / f"layer_{i:03d}.safetensors"
            if not source.exists():
                continue
            try:
                indexed = _index_safetensors(source)
            except ExpertBankError:
                raise
            except Exception as exc:  # unreadable header — treat as no bank
                LOGGER.warning(
                    "expert_bank_scan_failed",
                    extra={"file": str(source), "error": str(exc)},
                )
                continue
            if indexed is not None:
                if source is not bank:
                    LOGGER.info("expert_bank_v1_layer", extra={"layer": i})
                indexed.bank_file = source.relative_to(root).as_posix()
                layers[i] = indexed
        return cls(layers=layers)


def file_has_expert_tensors(path: str | Path) -> bool:
    """True when a safetensors layer file contains expert tensors (v1 fused
    MoE layers keep them inline; used by ``verify_shards`` to distinguish a
    genuine v1 layout from a deleted bank)."""
    try:
        header, _data_start = _parse_header(Path(path))
    except Exception:
        return False
    for key in header:
        if key == "__metadata__":
            continue
        if _PER_EXPERT_KEY.match(key):
            return True
        stem = key.rsplit(".", 1)[-1]
        if ".experts." in key and stem in _STACKED_STEMS:
            return True
    return False


def split_expert_tensors(state: dict) -> tuple[dict, dict]:
    """Split a layer state dict into (dense, experts) by key pattern.

    Used by the sharder to write MoE expert weights into their own bank file
    (v2 layout). Expert keys are the stacked fused tensors (``…experts.
    gate_up_proj/down_proj`` / ``…experts.w1/w2/w3``) and the per-expert
    originals (``…experts.{j}.gate_proj/…``); everything else is dense.
    """
    dense: dict = {}
    experts: dict = {}
    for key, value in state.items():
        stem = key.rsplit(".", 1)[-1]
        if _PER_EXPERT_KEY.match(key) or (".experts." in key and stem in _STACKED_STEMS):
            experts[key] = value
        else:
            dense[key] = value
    return dense, experts


def expert_count(state: dict) -> int:
    """Number of experts represented in a layer state dict (0 if dense)."""
    best = 0
    for key in state:
        m = _PER_EXPERT_KEY.match(key)
        if m:
            best = max(best, int(m.group("idx")) + 1)
            continue
        stem = key.rsplit(".", 1)[-1]
        if ".experts." in key and stem in _STACKED_STEMS:
            return int(state[key].shape[0])  # stacked: leading dim is E
    return best


def _parse_header(path: Path) -> tuple[dict, int]:
    """Return (header_dict, data_start) for a safetensors file."""
    with open(path, "rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len))
    return header, 8 + header_len


def _check_dtype(dtype_str: str, path: Path) -> str:
    if dtype_str not in _ITEMSIZE:
        raise ExpertBankError(f"unsupported expert dtype {dtype_str} in {path}")
    return dtype_str


def _stacked_slices(
    entries: dict[str, dict],
    data_start: int,
    path: Path,
) -> tuple[list[list[ExpertSlice]], int, int, int, str]:
    """Slices for stacked layouts; returns (slices, E, hidden, I, dtype)."""
    if "gate_up_proj" in entries:  # qwen3-moe fused pair
        gu, down = entries["gate_up_proj"], entries.get("down_proj")
        if down is None:
            raise ExpertBankError(f"gate_up_proj without down_proj in {path}")
        dtype_str = _check_dtype(gu["dtype"], path)
        num_experts, rows2i, hidden = (int(v) for v in gu["shape"])
        intermediate = rows2i // 2
        itemsize = _ITEMSIZE[dtype_str]
        gu_step = rows2i * hidden * itemsize        # [2I, H] per expert
        down_step = hidden * intermediate * itemsize  # [H, I] per expert
        gu0 = data_start + gu["data_offsets"][0]
        dn0 = data_start + down["data_offsets"][0]
        slices = []
        for j in range(num_experts):
            base = gu0 + j * gu_step
            half = gu_step // 2
            slices.append([
                ExpertSlice("gate", base, half, dtype_str, intermediate, hidden),
                ExpertSlice("up", base + half, half, dtype_str, intermediate, hidden),
                ExpertSlice("down", dn0 + j * down_step, down_step,
                            dtype_str, hidden, intermediate),
            ])
        return slices, num_experts, hidden, intermediate, dtype_str

    w1, w2, w3 = entries.get("w1"), entries.get("w2"), entries.get("w3")
    if not (w1 and w2 and w3):
        raise ExpertBankError(f"incomplete stacked expert set in {path}")
    dtype_str = _check_dtype(w1["dtype"], path)
    num_experts, hidden, intermediate = (int(v) for v in w1["shape"])
    step = hidden * intermediate * _ITEMSIZE[dtype_str]
    slices = []
    for j in range(num_experts):
        entry_list = []
        for stem, slot in (("w1", "gate"), ("w3", "up"), ("w2", "down")):
            off = data_start + entries[stem]["data_offsets"][0]
            entry_list.append(ExpertSlice(slot, off + j * step, step,
                                          dtype_str, hidden, intermediate))
        slices.append(entry_list)
    return slices, num_experts, hidden, intermediate, dtype_str


def _index_safetensors(path: Path) -> ExpertLayerIndex | None:
    """Index expert tensors in one safetensors file; None if none present."""
    header, data_start = _parse_header(path)
    meta = header.get("__metadata__") or {}
    if str(meta.get("__swlp_quant__", "")).strip().lower() not in ("", "none"):
        raise ExpertBankError(f"quantized expert bank not supported: {path}")

    stacked: dict[str, dict] = {}
    per_expert: dict[int, dict[str, dict]] = {}
    for key, entry in header.items():
        if key == "__metadata__":
            continue
        m = _PER_EXPERT_KEY.match(key)
        if m:
            per_expert.setdefault(int(m.group("idx")), {})[m.group("param")] = entry
            continue
        stem = key.rsplit(".", 1)[-1]
        if stem in _STACKED_STEMS and ".experts." in key:
            stacked[stem] = entry

    if not stacked and not per_expert:
        return None

    if stacked:
        slices, num_experts, hidden, intermediate, dtype_str = _stacked_slices(
            stacked, data_start, path
        )
        return ExpertLayerIndex(
            bank_file="", num_experts=num_experts, slices=slices,
            dtype_str=dtype_str, hidden=hidden, intermediate=intermediate,
        )

    # per-expert layout (original HF checkpoints)
    num_experts = max(per_expert) + 1
    first = next(iter(per_expert.values()))
    probe_key = "gate_proj" if "gate_proj" in first else "w1"
    dtype_str = _check_dtype(first[probe_key]["dtype"], path)
    slices: list[list[ExpertSlice]] = []
    hidden = intermediate = 0
    for j in range(num_experts):
        params = per_expert.get(j)
        if params is None or len(params) < 3:
            raise ExpertBankError(f"expert {j} incomplete in {path}")
        entry_list = []
        for param, entry in params.items():
            rows, cols = (int(v) for v in entry["shape"])
            start, end = entry["data_offsets"]
            _check_dtype(entry["dtype"], path)
            entry_list.append(ExpertSlice(
                _PER_EXPERT_PARAM_TO_SLOT[param], data_start + int(start),
                int(end) - int(start), entry["dtype"], rows, cols,
            ))
            if _PER_EXPERT_PARAM_TO_SLOT[param] == "gate":
                intermediate, hidden = rows, cols
        slices.append(entry_list)
    return ExpertLayerIndex(
        bank_file="", num_experts=num_experts, slices=slices,
        dtype_str=dtype_str, hidden=hidden, intermediate=intermediate,
    )


def _read_range(fd: int, offset: int, nbytes: int) -> bytearray:
    """Read an exact byte range via pread (POSIX, deterministic)."""
    buf = bytearray(nbytes)
    view = memoryview(buf)
    remaining = nbytes
    while remaining > 0:
        data = os.pread(fd, len(view), offset)
        if not data:
            raise ExpertBankError(f"short pread at {offset} ({remaining} of {nbytes} bytes left)")
        view[: len(data)] = data
        view = view[len(data):]
        offset += len(data)
        remaining -= len(data)
    return buf


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    import torch

    return {
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F64": torch.float64,
    }[dtype_str]


def read_expert(
    index: ExpertIndex,
    shard_dir: str | Path,
    layer: int,
    expert: int,
) -> dict[str, torch.Tensor]:
    """Read one expert's slots → ``{"gate": T, "up": T, "down": T}`` (CPU)."""
    import torch

    layer_idx = index.layers[layer]
    path = Path(shard_dir) / layer_idx.bank_file
    dtype = _resolve_dtype(layer_idx.dtype_str)
    out: dict[str, torch.Tensor] = {}
    fd = os.open(path, os.O_RDONLY)
    try:
        for sl in layer_idx.slices[expert]:
            raw = _read_range(fd, sl.offset, sl.nbytes)
            out[sl.slot] = torch.frombuffer(raw, dtype=dtype).reshape(sl.rows, sl.cols)
    finally:
        os.close(fd)
    return out
