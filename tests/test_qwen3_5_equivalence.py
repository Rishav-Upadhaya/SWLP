"""Qwen3.5 (hybrid Gated-DeltaNet + full attention, multimodal wrapper):
shard the nested ``model.language_model.*`` layout and stream it losslessly."""
from __future__ import annotations

from pathlib import Path

import torch
from test_llama_equivalence import (
    NEW_TOKENS,
    TURN1_IDS,
    VOCAB,
    _assert_generation_equivalent,
    _cfg,
    _run_capturing,
)
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

from swlp.model.shard import load_manifest, shard_model_by_layer
from swlp.runner import build_runner


def _tiny_qwen3_5() -> Qwen3_5ForConditionalGeneration:
    cfg = Qwen3_5Config(
        text_config={
            "vocab_size": VOCAB, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 4, "full_attention_interval": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "linear_num_key_heads": 2, "linear_num_value_heads": 4,
            "linear_key_head_dim": 8, "linear_value_head_dim": 8,
            "tie_word_embeddings": False,
        },
        vision_config={"depth": 1, "hidden_size": 16, "num_heads": 2,
                       "intermediate_size": 32, "out_hidden_size": 32},
        tie_word_embeddings=False,
    )
    torch.manual_seed(7)
    return Qwen3_5ForConditionalGeneration(cfg).eval().to(torch.float32)


def test_qwen3_5_shards_and_streams_like_hf(tmp_path: Path) -> None:
    model = _tiny_qwen3_5()
    hf_dir = tmp_path / "hf"
    model.save_pretrained(hf_dir)
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    word_level = Tokenizer(models.WordLevel(vocab={"[UNK]": 0, "a": 1}, unk_token="[UNK]"))
    PreTrainedTokenizerFast(tokenizer_object=word_level).save_pretrained(hf_dir)
    shard_dir = tmp_path / "shards"
    manifest = shard_model_by_layer(str(hf_dir), shard_dir, dtype_str="float32")
    assert manifest.num_layers == 4

    # HF reference: greedy decode with the text model's own cache.
    with torch.no_grad():
        ref: list[torch.Tensor] = []
        out = model(torch.tensor([TURN1_IDS]), use_cache=True)
        for _ in range(NEW_TOKENS):
            ref.append(out.logits[:, -1, :].float())
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            out = model(nxt, past_key_values=out.past_key_values, use_cache=True)

    runner = build_runner(_cfg(hf_dir, shard_dir))
    runner.load()
    got, _ = _run_capturing(runner, TURN1_IDS)
    assert load_manifest(shard_dir).model_type == "qwen3_5"
    _assert_generation_equivalent(got, ref, "qwen3_5 streamed vs HF reference")
