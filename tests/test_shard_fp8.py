"""Block-FP8 checkpoints (e4m3 + per-block weight_scale_inv, e.g. Qwen3.8-27B-FP8)
must be dequantized at shard time; other quantized formats must be refused."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import LlamaConfig, LlamaForCausalLM

from swlp.model.shard import shard_model_by_layer

BLOCK = 16
KEY = "model.layers.0.self_attn.q_proj.weight"


def _checkpoint(tmp_path: Path, quant: dict) -> tuple[Path, torch.Tensor]:
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=40, intermediate_size=48, num_hidden_layers=1,
                      num_attention_heads=4, num_key_value_heads=4, tie_word_embeddings=False)
    hf_dir = tmp_path / "hf"
    LlamaForCausalLM(cfg).to(torch.bfloat16).save_pretrained(hf_dir)
    path = hf_dir / "model.safetensors"
    state = load_file(str(path))
    w = state[KEY].float()  # [40, 40]: not a multiple of the block — edge blocks
    rows, cols = w.shape
    nb_r, nb_c = -(-rows // BLOCK), -(-cols // BLOCK)
    scale = torch.empty(nb_r, nb_c)
    q = torch.empty_like(w)
    for i in range(nb_r):
        for j in range(nb_c):
            blk = w[i * BLOCK:(i + 1) * BLOCK, j * BLOCK:(j + 1) * BLOCK]
            s = blk.abs().max().clamp(min=1e-8) / 448.0
            scale[i, j] = s
            q[i * BLOCK:(i + 1) * BLOCK, j * BLOCK:(j + 1) * BLOCK] = blk / s
    state[KEY] = q.to(torch.float8_e4m3fn)
    state[KEY + "_scale_inv"] = scale
    save_file(state, str(path), metadata={"format": "pt"})
    c = json.loads((hf_dir / "config.json").read_text())
    c["quantization_config"] = quant
    (hf_dir / "config.json").write_text(json.dumps(c))
    return hf_dir, w


def test_fp8_weights_are_dequantized_with_block_scales(tmp_path: Path) -> None:
    hf_dir, original = _checkpoint(tmp_path, {"quant_method": "fp8", "fmt": "e4m3",
                                              "weight_block_size": [BLOCK, BLOCK]})
    shard_model_by_layer(str(hf_dir), tmp_path / "shards", dtype_str="bfloat16")
    with safe_open(str(tmp_path / "shards" / "layer_000.safetensors"), "pt") as f:
        keys = list(f.keys())
        got = f.get_tensor("self_attn.q_proj.weight").float()
    assert not any(k.endswith("_scale_inv") for k in keys)
    # Raw FP8 values reach ~448; dequantized ones match the source within FP8 error.
    assert got.abs().max() < 1.0
    assert torch.allclose(got, original, rtol=0.07, atol=1e-3)


def test_other_quantized_formats_are_refused(tmp_path: Path) -> None:
    hf_dir, _ = _checkpoint(tmp_path, {"quant_method": "gptq", "bits": 4})
    with pytest.raises(ValueError, match="gptq-quantized"):
        shard_model_by_layer(str(hf_dir), tmp_path / "shards")
