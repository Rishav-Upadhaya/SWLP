"""Expert byte-range index for MLX-format MoE checkpoints (Phase 31).

``mlx_lm`` checkpoints (e.g. ``mlx-community/*-4bit``) store each MoE layer's
experts stacked on axis 0 — ``…{switch_glu|switch_mlp}.{gate,up,down}_proj.
{weight,scales,biases}`` — so expert *j* of every tensor is one contiguous row
block. This indexes those blocks straight from the downloaded safetensors (no
re-shard, no extra disk) for the ranged reads in
``runner/mlx_expert_cache.py``. Slot names are ``"<proj>.<part>"``
(``"gate.weight"``, ``"gate.scales"``, …); quantized weights stay packed.
"""
from __future__ import annotations

import re
from pathlib import Path

from .expert_bank import ExpertBankError, ExpertIndex, ExpertLayerIndex, ExpertSlice, _parse_header

_KEY = re.compile(
    r".*\.layers\.(?P<layer>\d+)\..*\.(?:switch_glu|switch_mlp)\."
    r"(?P<proj>gate|up|down)_proj\.(?P<part>weight|scales|biases)$"
)


def index_mlx_checkpoint(model_dir: str | Path) -> ExpertIndex:
    """Index every stacked expert tensor under ``model_dir`` (``*.safetensors``)."""
    tensors: dict[int, dict[str, tuple[Path, int, dict]]] = {}
    for path in sorted(Path(model_dir).glob("*.safetensors")):
        header, data_start = _parse_header(path)
        for key, entry in header.items():
            m = _KEY.match(key) if key != "__metadata__" else None
            if m:
                slot = f"{m['proj']}.{m['part']}"
                tensors.setdefault(int(m["layer"]), {})[slot] = (path.resolve(), data_start, entry)
    layers: dict[int, ExpertLayerIndex] = {}
    for layer, slots in sorted(tensors.items()):
        num_experts = {int(e["shape"][0]) for _, _, e in slots.values()}
        if len(num_experts) != 1:
            raise ExpertBankError(f"layer {layer}: inconsistent expert counts {num_experts}")
        n = num_experts.pop()
        per_expert: list[list[ExpertSlice]] = [[] for _ in range(n)]
        for slot, (path, data_start, entry) in sorted(slots.items()):
            start, end = (int(v) for v in entry["data_offsets"])
            if (end - start) % n:
                raise ExpertBankError(f"layer {layer} {slot}: bytes not divisible by {n}")
            step = (end - start) // n
            rows, cols = (int(v) for v in entry["shape"][1:3])
            for j in range(n):
                per_expert[j].append(ExpertSlice(
                    slot, data_start + start + j * step, step, entry["dtype"], rows, cols,
                    file=str(path),
                ))
        layers[layer] = ExpertLayerIndex(bank_file="", num_experts=n, slices=per_expert,
                                         dtype_str=slots["gate.weight"][2]["dtype"])
    return ExpertIndex(layers=layers)
