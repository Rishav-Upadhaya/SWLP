"""MlxMoeRunner (runner/mlx_moe.py + runner/mlx_expert_cache.py): expert-streamed
MLX MoE must match mlx_lm's fully resident model on the same checkpoint."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from test_moe_equivalence import PROMPT_IDS, _tiny_moe  # noqa: E402
from tokenizers import Tokenizer, models  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from swlp.config import (  # noqa: E402
    AppConfig,
    CacheConfig,
    GenerationConfig,
    ModelConfig,
    RuntimeConfig,  # noqa: E402
)
from swlp.model.shard import shard_model_by_layer  # noqa: E402
from swlp.runner import build_runner  # noqa: E402

STEPS = 5


def _tiny_qwen3_5_moe():
    """Hybrid DeltaNet + attention MoE with a shared expert, multimodal wrapper
    (the Qwen3.5/3.6-35B-A3B layout, ``model.language_model.*`` prefix)."""
    import torch
    from transformers import Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration

    cfg = Qwen3_5MoeConfig(
        text_config={
            "vocab_size": 112, "hidden_size": 32, "num_hidden_layers": 4,
            "full_attention_interval": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 16,
            "linear_num_key_heads": 2, "linear_num_value_heads": 4,
            "linear_key_head_dim": 32, "linear_value_head_dim": 32,
            "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 24,
            "shared_expert_intermediate_size": 24, "tie_word_embeddings": False,
        },
        vision_config={"depth": 1, "hidden_size": 16, "num_heads": 2,
                       "intermediate_size": 32, "out_hidden_size": 32},
        tie_word_embeddings=False,
    )
    torch.manual_seed(11)
    return Qwen3_5MoeForConditionalGeneration(cfg).eval().to(torch.float32)


def _mlx_names(cfg: dict) -> None:
    """transformers 5 serializes num_local_experts / rope_parameters; Hub
    checkpoints (and mlx_lm's ModelArgs) use the original names."""
    if "num_local_experts" in cfg and "num_experts" not in cfg:
        cfg["num_experts"] = cfg["num_local_experts"]
    rope = cfg.get("rope_parameters") or {}
    if "rope_theta" in rope and "rope_theta" not in cfg:
        cfg["rope_theta"] = rope["rope_theta"]
    types = cfg.get("layer_types") or []
    if "full_attention" in types and "full_attention_interval" not in cfg:
        cfg["full_attention_interval"] = types.index("full_attention") + 1


def _fuse_experts_like_hub(hf_dir: Path) -> None:
    """Rewrite per-expert keys into the Hub's fused Qwen3.5/3.6 layout:
    ``experts.gate_up_proj`` [E, 2I, H] (gate first) + ``experts.down_proj``
    [E, H, I] — transformers 5 saves them unfused."""
    import re

    import torch
    from safetensors.torch import load_file, save_file

    path = hf_dir / "model.safetensors"
    state = load_file(str(path))
    per: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    fused: dict[str, torch.Tensor] = {}
    for key, t in state.items():
        m = re.match(r"(.+\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$", key)
        if m:
            per.setdefault(m.group(1), {}).setdefault(int(m.group(2)), {})[m.group(3)] = t
        else:
            fused[key] = t
    for stem, experts in per.items():
        ids = sorted(experts)
        fused[f"{stem}.gate_up_proj"] = torch.stack(
            [torch.cat([experts[i]["gate_proj"], experts[i]["up_proj"]]) for i in ids])
        fused[f"{stem}.down_proj"] = torch.stack([experts[i]["down_proj"] for i in ids])
    save_file(fused, str(path), metadata={"format": "pt"})


def _checkpoint(tmp_path: Path, arch: str = "qwen3_moe") -> tuple[Path, Path]:
    hf_dir = tmp_path / "hf"
    (_tiny_moe() if arch == "qwen3_moe" else _tiny_qwen3_5_moe()).save_pretrained(hf_dir)
    if arch == "qwen3_5_moe":
        _fuse_experts_like_hub(hf_dir)
    cfg_path = hf_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    _mlx_names(cfg)
    if "text_config" in cfg:
        _mlx_names(cfg["text_config"])
    cfg_path.write_text(json.dumps(cfg))
    vocab = {"[UNK]": 0, "a": 1}
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    ).save_pretrained(hf_dir)
    shard_dir = tmp_path / "shards"
    shard_model_by_layer(str(hf_dir), shard_dir, dtype_str="float32")
    return hf_dir, shard_dir


def _cfg(hf_dir: Path, shard_dir: Path, budget_mb: int) -> AppConfig:
    return AppConfig(
        model=ModelConfig(model_id=str(hf_dir), local_model_path=str(hf_dir)),
        cache=CacheConfig(),
        generation=GenerationConfig(max_new_tokens=4, temperature=0.0, do_sample=False),
        runtime=RuntimeConfig(backend="mlx-moe", shard_dir=str(shard_dir),
                              swlp_expert_cache_mb=budget_mb, log_level="WARNING"),
    )


def _greedy_logits(model, steps: int) -> list[np.ndarray]:
    """Prefill PROMPT_IDS, then feed back argmax tokens; logits per step."""
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    x = mx.array([PROMPT_IDS])
    out: list[np.ndarray] = []
    for _ in range(steps):
        logits = model(x, cache=cache)[:, -1, :]
        mx.eval(logits)
        out.append(np.array(logits.astype(mx.float32)))
        x = mx.argmax(logits, axis=-1)[:, None]
    return out


@pytest.mark.parametrize(("arch", "budget_mb"), [
    ("qwen3_moe", 1),     # 1 MB < 16 x 295 KB experts: evictions every step
    ("qwen3_moe", 64),
    ("qwen3_5_moe", 1),   # hybrid DeltaNet + shared expert + multimodal prefix
])
def test_mlx_moe_matches_resident_mlx_lm(tmp_path: Path, arch: str, budget_mb: int) -> None:
    from mlx_lm.utils import load_model

    hf_dir, shard_dir = _checkpoint(tmp_path, arch)
    ref_model, _ = load_model(hf_dir)
    runner = build_runner(_cfg(hf_dir, shard_dir, budget_mb))
    runner.load()
    from mlx.utils import tree_flatten

    # Untied fixtures: the vocab table must live in the CPU mmap, not in MLX.
    assert not any("embed_tokens" in k for k, _ in tree_flatten(runner.model.parameters()))

    ref = _greedy_logits(ref_model, STEPS)
    got = _greedy_logits(runner.model, STEPS)
    for step, (r, g) in enumerate(zip(ref, got, strict=True)):
        assert int(r.argmax()) == int(g.argmax()), f"greedy token differs at step {step}"
        np.testing.assert_allclose(g, r, rtol=1e-4, atol=1e-4, err_msg=f"step {step}")
    stats = runner.cache.stats()
    assert stats["misses"] + stats["prefetch_hits"] > 0
    if arch == "qwen3_moe" and budget_mb == 1:
        assert stats["evictions"] > 0
    runner.cache.close()


def test_mlx_moe_rejects_dense_shard_dir(tmp_path: Path) -> None:
    runner = build_runner(_cfg(tmp_path, tmp_path, 1))
    with pytest.raises(ValueError, match="expert_index.json"):
        runner.load()


def test_expert_cache_is_lfu_not_lru(tmp_path: Path) -> None:
    """A hot expert survives a scan of one-off experts (LRU would evict it —
    the decode-sweep pathology: 0% LRU hits at small caches on real traces)."""
    from swlp.model.expert_bank import EXPERT_INDEX_FILE, ExpertIndex
    from swlp.runner.mlx_expert_cache import MlxExpertCache

    _, shard_dir = _checkpoint(tmp_path)
    index = ExpertIndex.load(shard_dir / EXPERT_INDEX_FILE)
    per_expert = index.layers[0].expert_bytes()
    cache = MlxExpertCache(index, shard_dir, budget_bytes=2 * per_expert, workers=2)
    for _ in range(3):
        cache.get_many(0, [0])             # expert 0: hot
    for eid in (1, 2, 3, 4):
        cache.get_many(0, [eid])           # one-off scan
    assert (0, 0) in cache._cache          # LRU would have evicted it
    assert cache.stats()["hits"] == 2 and cache.evictions >= 3
    cache.close()


def test_expert_cache_never_exceeds_budget(tmp_path: Path, monkeypatch) -> None:
    """Budget invariant across heap rebuilds (a rebuild racing an insert once
    left keys unevictable, so the cache grew past its budget)."""
    import swlp.runner.mlx_expert_cache as mec
    from swlp.model.expert_bank import EXPERT_INDEX_FILE, ExpertIndex

    monkeypatch.setattr(mec, "_HEAP_SLACK", 0)  # rebuild on nearly every push
    _, shard_dir = _checkpoint(tmp_path)
    index = ExpertIndex.load(shard_dir / EXPERT_INDEX_FILE)
    budget = 3 * index.layers[0].expert_bytes()
    cache = mec.MlxExpertCache(index, shard_dir, budget_bytes=budget, workers=2)
    rng = np.random.default_rng(0)
    for _ in range(200):
        layer = int(rng.integers(0, len(index.layers)))
        cache.get_many(layer, sorted(set(rng.integers(0, 8, size=2).tolist())))
        assert cache.stats()["cached_bytes"] <= budget
    cache.close()


def _tiny_moe_64(tmp_path: Path) -> Path:
    """Tiny Qwen3-MoE with hidden 64 — mlx_lm quantizes routers at group 64."""
    import torch
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    cfg = Qwen3MoeConfig(
        vocab_size=112, hidden_size=64, intermediate_size=128, moe_intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        num_experts=8, num_experts_per_tok=2, max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    torch.manual_seed(5)
    hf_dir = tmp_path / "hf"
    Qwen3MoeForCausalLM(cfg).eval().to(torch.float32).save_pretrained(hf_dir)
    raw = json.loads((hf_dir / "config.json").read_text())
    _mlx_names(raw)
    (hf_dir / "config.json").write_text(json.dumps(raw))
    PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel(
        vocab={"[UNK]": 0, "a": 1}, unk_token="[UNK]"))).save_pretrained(hf_dir)
    return hf_dir


def test_mlx_moe_streams_4bit_mlx_checkpoint(tmp_path: Path) -> None:
    """MLX-format 4-bit checkpoint (no shard dir): experts read packed from the
    mlx-lm safetensors must match mlx_lm's fully resident quantized model."""
    from mlx_lm import convert
    from mlx_lm.utils import load_model

    hf_dir = _tiny_moe_64(tmp_path)
    mlx_dir = tmp_path / "mlx4"
    convert(str(hf_dir), mlx_path=str(mlx_dir), quantize=True, q_bits=4, q_group_size=32)
    PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel(
        vocab={"[UNK]": 0, "a": 1}, unk_token="[UNK]"))).save_pretrained(mlx_dir)
    ref_model, _ = load_model(mlx_dir)
    cfg = AppConfig(
        model=ModelConfig(model_id=str(mlx_dir), local_model_path=str(mlx_dir)),
        cache=CacheConfig(),
        generation=GenerationConfig(max_new_tokens=4, temperature=0.0, do_sample=False),
        runtime=RuntimeConfig(backend="mlx-moe", swlp_expert_cache_mb=1, log_level="WARNING"),
    )
    runner = build_runner(cfg)
    runner.load()
    ref = _greedy_logits(ref_model, STEPS)
    got = _greedy_logits(runner.model, STEPS)
    for step, (r, g) in enumerate(zip(ref, got, strict=True)):
        assert int(r.argmax()) == int(g.argmax()), f"greedy token differs at step {step}"
        np.testing.assert_allclose(g, r, rtol=1e-3, atol=1e-3, err_msg=f"step {step}")
    # Every expert came through the cache (tiny 4-bit experts all fit in 1 MB,
    # so eviction is covered by the full-precision test above).
    assert runner.cache.stats()["misses"] > 0
    runner.cache.close()


def test_quantize_on_load_matches_mlx_lm_convert(tmp_path: Path) -> None:
    """`-q int4` on full-precision MoE shards (dense quantized once, experts as
    they are cached) must equal mlx_lm's own quantize_model on the same
    checkpoint — same group size and per-layer predicate. The embedding table
    is left exact in the reference too: SWLP serves it from the CPU mmap."""
    from mlx_lm.utils import load_model, quantize_model

    hf_dir = _tiny_moe_64(tmp_path)
    shard_dir = tmp_path / "shards"
    shard_model_by_layer(str(hf_dir), shard_dir, dtype_str="float32")
    ref_model, ref_cfg = load_model(hf_dir)
    upstream = ref_model.quant_predicate

    def keep_embedding_exact(path, module):
        if path.endswith("embed_tokens"):
            return False
        return upstream(path, module) if upstream else True

    quantize_model(ref_model, ref_cfg, 64, 4, quant_predicate=keep_embedding_exact)

    cfg = _cfg(hf_dir, shard_dir, 1)
    cfg.runtime.swlp_moe_quant = "int4"
    runner = build_runner(cfg)
    runner.load()
    for step, (r, g) in enumerate(zip(_greedy_logits(ref_model, STEPS),
                                      _greedy_logits(runner.model, STEPS), strict=True)):
        assert int(r.argmax()) == int(g.argmax()), f"greedy token differs at step {step}"
        np.testing.assert_allclose(g, r, rtol=1e-3, atol=1e-3, err_msg=f"step {step}")
    cached = next(iter(runner.cache._cache.values()))
    assert "gate.scales" in cached  # experts really are cached packed
    runner.cache.close()
