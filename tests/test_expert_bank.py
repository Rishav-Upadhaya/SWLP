"""Tests for swlp.model.expert_bank — ranged expert reads (Phase 25)."""
from __future__ import annotations

import json
import struct

import pytest
import torch
from safetensors.torch import save_file

from swlp.model.expert_bank import (
    EXPERT_INDEX_FILE,
    ExpertBankError,
    ExpertIndex,
    read_expert,
)

E, H, INTER = 4, 8, 6  # experts, hidden, intermediate dims


def _make_stacked_qwen(path, prefix="model.layers.0.mlp"):
    """qwen3-moe fused layout: gate_up_proj [E,2I,H], down_proj [E,H,I]."""
    gate_up = torch.randn(E, 2 * INTER, H, dtype=torch.float16)
    down = torch.randn(E, H, INTER, dtype=torch.float16)
    attn = torch.randn(H, H, dtype=torch.float16)
    save_file(
        {
            f"{prefix}.experts.gate_up_proj": gate_up,
            f"{prefix}.experts.down_proj": down,
            f"{prefix}.self_attn.q_proj": attn,
        },
        str(path),
    )
    return gate_up, down


def _make_stacked_mixtral(path, prefix="model.layers.0.block_sparse_moe"):
    """mixtral fused layout: w1/w3 [E,H,I] gate/up, w2 [E,H,I] down."""
    w1 = torch.randn(E, H, INTER, dtype=torch.float16)
    w2 = torch.randn(E, H, INTER, dtype=torch.float16)
    w3 = torch.randn(E, H, INTER, dtype=torch.float16)
    save_file(
        {f"{prefix}.experts.w1": w1, f"{prefix}.experts.w2": w2, f"{prefix}.experts.w3": w3},
        str(path),
    )
    return w1, w2, w3


def _make_per_expert(path, prefix="model.layers.0.mlp"):
    """Original HF per-expert layout: experts.{j}.gate/up/down_proj."""
    tensors: dict[str, torch.Tensor] = {}
    ref: list[dict[str, torch.Tensor]] = []
    for j in range(E):
        gate = torch.randn(INTER, H, dtype=torch.float16)
        up = torch.randn(INTER, H, dtype=torch.float16)
        down = torch.randn(H, INTER, dtype=torch.float16)
        tensors[f"{prefix}.experts.{j}.gate_proj"] = gate
        tensors[f"{prefix}.experts.{j}.up_proj"] = up
        tensors[f"{prefix}.experts.{j}.down_proj"] = down
        ref.append({"gate": gate, "up": up, "down": down})
    save_file(tensors, str(path))
    return ref


def test_stacked_qwen_layout_roundtrip(tmp_path):
    gate_up, down = _make_stacked_qwen(tmp_path / "layer_000.experts.safetensors")
    index = ExpertIndex.build(tmp_path, num_layers=1)
    assert index.layers and index.layers[0].num_experts == E

    for j in range(E):
        got = read_expert(index, tmp_path, 0, j)
        assert torch.equal(got["gate"], gate_up[j, :INTER, :])
        assert torch.equal(got["up"], gate_up[j, INTER:, :])
        assert torch.equal(got["down"], down[j])
        assert got["gate"].dtype == torch.float16


def test_stacked_mixtral_layout_roundtrip(tmp_path):
    w1, w2, w3 = _make_stacked_mixtral(tmp_path / "layer_000.experts.safetensors")
    index = ExpertIndex.build(tmp_path, num_layers=1)
    assert index.layers[0].num_experts == E
    j = 2
    got = read_expert(index, tmp_path, 0, j)
    assert torch.equal(got["gate"], w1[j])
    assert torch.equal(got["up"], w3[j])
    assert torch.equal(got["down"], w2[j])


def test_per_expert_layout_roundtrip(tmp_path):
    ref = _make_per_expert(tmp_path / "layer_000.experts.safetensors")
    index = ExpertIndex.build(tmp_path, num_layers=1)
    assert index.layers[0].num_experts == E
    for j in range(E):
        got = read_expert(index, tmp_path, 0, j)
        for slot in ("gate", "up", "down"):
            assert torch.equal(got[slot], ref[j][slot])


def test_dense_layer_yields_no_index(tmp_path):
    save_file(
        {"model.layers.0.self_attn.q_proj": torch.randn(4, 4, dtype=torch.float16)},
        str(tmp_path / "layer_000.safetensors"),
    )
    index = ExpertIndex.build(tmp_path, num_layers=1)
    assert not index


def test_quantized_bank_rejected(tmp_path):

    tensors = {"mlp.experts.gate_up_proj": torch.randn(E, 2 * INTER, H, dtype=torch.float16),
               "mlp.experts.down_proj": torch.randn(E, H, INTER, dtype=torch.float16)}
    save_file(tensors, str(tmp_path / "layer_000.experts.safetensors"),
              metadata={"__swlp_quant__": "float8"})
    with pytest.raises(ExpertBankError):
        ExpertIndex.build(tmp_path, num_layers=1)


def test_index_json_roundtrip(tmp_path):
    _make_stacked_qwen(tmp_path / "layer_000.experts.safetensors")
    index = ExpertIndex.build(tmp_path, num_layers=1)
    index.save(tmp_path / EXPERT_INDEX_FILE)
    loaded = ExpertIndex.load(tmp_path / EXPERT_INDEX_FILE)
    assert loaded.layers.keys() == index.layers.keys()
    a = read_expert(index, tmp_path, 0, 1)
    b = read_expert(loaded, tmp_path, 0, 1)
    assert torch.equal(a["gate"], b["gate"])
    # The saved JSON must be valid, sorted, plain data.
    data = json.loads((tmp_path / EXPERT_INDEX_FILE).read_text())
    assert data["0"]["num_experts"] == E


def test_v1_layer_file_indexed_in_place(tmp_path):
    """No .experts bank → the main layer file is indexed (v1 compatibility)."""
    gate_up, down = _make_stacked_qwen(tmp_path / "layer_000.safetensors")
    index = ExpertIndex.build(tmp_path, num_layers=1)
    assert index.layers[0].bank_file == "layer_000.safetensors"
    got = read_expert(index, tmp_path, 0, 3)
    assert torch.equal(got["down"], down[3])


def test_missing_layer_is_skipped(tmp_path):
    _make_stacked_qwen(tmp_path / "layer_000.experts.safetensors")
    index = ExpertIndex.build(tmp_path, num_layers=3)
    assert set(index.layers) == {0}


def test_safetensors_offsets_are_absolute_correct(tmp_path):
    """Slices must point at real data: mutate one expert, read it back."""
    gate_up, _ = _make_stacked_qwen(tmp_path / "layer_000.experts.safetensors")
    index = ExpertIndex.build(tmp_path, num_layers=1)
    # Rewrite the file with a modified expert 1 only.
    gate_up = gate_up.clone()
    gate_up[1] += 7.0

    save_file({"model.layers.0.mlp.experts.gate_up_proj": gate_up,
               "model.layers.0.mlp.experts.down_proj":
                   torch.randn(E, H, INTER, dtype=torch.float16),
               "model.layers.0.mlp.self_attn.q_proj": torch.randn(H, H, dtype=torch.float16)},
              str(tmp_path / "layer_000.experts.safetensors"))
    got = read_expert(index, tmp_path, 0, 1)
    assert torch.equal(got["gate"], gate_up[1, :INTER, :])
    got0 = read_expert(index, tmp_path, 0, 0)
    assert torch.equal(got0["gate"], gate_up[0, :INTER, :])


def test_header_parse_little_endian_u64(tmp_path):
    """Guard the raw format assumption: 8-byte LE length prefix."""
    p = tmp_path / "layer_000.safetensors"
    _make_stacked_qwen(p)
    with open(p, "rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len))
    assert "model.layers.0.mlp.experts.gate_up_proj" in header
    entry = header["model.layers.0.mlp.experts.gate_up_proj"]
    assert entry["shape"] == [E, 2 * INTER, H]
    assert entry["dtype"] == "F16"


def test_split_expert_tensors_and_count(tmp_path):
    from swlp.model.expert_bank import expert_count, split_expert_tensors

    state = {
        "mlp.experts.gate_up_proj": torch.randn(E, 2 * INTER, H, dtype=torch.float16),
        "mlp.experts.down_proj": torch.randn(E, H, INTER, dtype=torch.float16),
        "mlp.self_attn.q_proj": torch.randn(H, H, dtype=torch.float16),
    }
    dense, experts = split_expert_tensors(state)
    assert set(dense) == {"mlp.self_attn.q_proj"}
    assert len(experts) == 2
    assert expert_count(state) == E

    per_expert_state = {f"mlp.experts.{j}.w1": torch.randn(H, INTER) for j in range(3)}
    per_expert_state |= {f"mlp.experts.{j}.w2": torch.randn(INTER, H) for j in range(3)}
    per_expert_state |= {f"mlp.experts.{j}.w3": torch.randn(H, INTER) for j in range(3)}
    dense2, experts2 = split_expert_tensors(per_expert_state)
    assert not dense2 and len(experts2) == 9
    assert expert_count(per_expert_state) == 3

    assert expert_count({"a.b": torch.zeros(2)}) == 0
