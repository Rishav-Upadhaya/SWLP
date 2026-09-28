"""Phase 25 end-to-end MoE equivalence against the INSTALLED transformers.

The streamed path (shard v2 expert banks + ``SwlpCachedExperts`` slot cache +
``ExpertScheduler``) must reproduce the reference ``Qwen3MoeForCausalLM``
forward bit-for-bit on a tiny random model — logits at the first generated
position, the full greedy continuation, and survival across layer eviction
(window streaming meta-ifies every block each token).

The reference is the real installed implementation, not a hand-copied port:
if the decomposed forward diverges from transformers (routing dtype, top-k
normalization, gather/scatter order), these tests fail.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from swlp.config import AppConfig, CacheConfig, GenerationConfig, ModelConfig, RuntimeConfig
from swlp.model.shard import shard_model_by_layer
from swlp.runner import build_runner

VOCAB = 112
PROMPT_IDS = [7, 19, 3, 42, 88, 15]
NEW_TOKENS = 4


def _tiny_moe() -> Qwen3MoeForCausalLM:
    cfg = Qwen3MoeConfig(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, num_experts=8, num_experts_per_tok=2,
        max_position_embeddings=256, tie_word_embeddings=False,
    )
    # transformers ≥5 renamed torch_dtype → dtype; set both so the serialized
    # config and the re-instantiated model agree on float32.
    cfg.dtype = torch.float32
    cfg.torch_dtype = torch.float32
    torch.manual_seed(1234)
    return Qwen3MoeForCausalLM(cfg).eval().to(torch.float32)


def _runner_cfg(model_dir: Path, shard_dir: Path, dtype: str = "float32") -> AppConfig:
    return AppConfig(
        model=ModelConfig(model_id=str(model_dir)),
        cache=CacheConfig(),
        generation=GenerationConfig(
            max_new_tokens=NEW_TOKENS, temperature=0.0, do_sample=False, seed=42
        ),
        runtime=RuntimeConfig(
            device="cpu", dtype=dtype, backend="swlp", shard_dir=str(shard_dir),
            swlp_window_size=2, swlp_prefetch_depth=1, swlp_prefetch=True,
            swlp_residency="off", log_level="WARNING",
            swlp_fallback_to_baseline=False,
            # Small cache forces slot recycling across tokens (eviction path).
            swlp_expert_cache_mb=1,
        ),
    )


class _FixedIdsTokenizer:
    """Feeds exact token ids; decode yields placeholder text (unused)."""

    eos_token_id = None

    def __call__(self, prompt: str, return_tensors: str = "pt") -> dict:
        return {"input_ids": torch.tensor([PROMPT_IDS], dtype=torch.long)}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return ""


def test_streamed_moe_matches_hf_reference(tmp_path: Path) -> None:
    """First-token logits + full greedy continuation must match the installed
    transformers Qwen3Moe forward exactly."""
    from transformers.cache_utils import DynamicCache

    model = _tiny_moe()
    hf_dir = tmp_path / "hf"
    model.save_pretrained(hf_dir)
    # load_from_shards expects loadable tokenizer files next to the checkpoint;
    # a minimal WordLevel fast tokenizer needs no sentencepiece. The vocab is
    # irrelevant because the runner's tokenizer is stubbed below.
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    word_level = Tokenizer(models.WordLevel(vocab={"[UNK]": 0, "a": 1}, unk_token="[UNK]"))
    PreTrainedTokenizerFast(tokenizer_object=word_level).save_pretrained(hf_dir)
    shard_dir = tmp_path / "shards"
    shard_model_by_layer(str(hf_dir), shard_dir, dtype_str="float32")

    # Reference: full-model forward + manual greedy loop with KV cache,
    # capturing per-step logits (stronger than token equality — silently
    # zeroed experts would diverge at the logit level while possibly keeping
    # the same argmax on a tiny random model).
    ids = torch.tensor([PROMPT_IDS], dtype=torch.long)
    ref_logits_steps: list[torch.Tensor] = []
    with torch.no_grad():
        ref_logits_steps.append(model(ids).logits[:, -1, :].float())
        ref_first = int(ref_logits_steps[0].argmax())
        ref_tokens = [ref_first]
        next_id = torch.tensor([[ref_first]], dtype=torch.long)
        past = DynamicCache()
        model(ids, past_key_values=past, use_cache=True)
        for _ in range(NEW_TOKENS - 1):
            out = model(next_id, past_key_values=past, use_cache=True)
            ref_logits_steps.append(out.logits[:, -1, :].float())
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ref_tokens.append(int(next_id))

    runner = build_runner(_runner_cfg(hf_dir, shard_dir))
    runner.load()
    # Force real slot recycling: shrink the budget so each layer holds ONE
    # expert slot while routing activates two per token — every sweep evicts.
    # Also exercises the elastic set_budget resize path. Correctness must be
    # invariant to cache pressure (lossless guarantee).
    runner._expert_sched.set_budget(40 * 1024)
    runner.tokenizer = _FixedIdsTokenizer()
    picked: list[torch.Tensor] = []

    def _pick(logits: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
        picked.append(logits.detach().float().clone())
        return logits.argmax(dim=-1, keepdim=True)

    runner._select_next = _pick
    runner.run("unused — fixed-ids tokenizer injects PROMPT_IDS")

    got_tokens = [int(t.argmax()) for t in picked]
    assert len(picked) == NEW_TOKENS, f"expected {NEW_TOKENS} picks, got {len(picked)}"
    for step, (got, ref) in enumerate(zip(picked, ref_logits_steps, strict=True)):
        assert torch.allclose(got, ref, rtol=0.0, atol=1e-4), (
            f"streamed MoE logits diverged from HF reference at step {step}: "
            f"max|Δ|={(got - ref).abs().max().item():.3e}"
        )
    assert got_tokens == ref_tokens, (
        f"greedy continuation diverged: streamed={got_tokens} reference={ref_tokens}"
    )


def test_expert_slots_survive_layer_eviction(tmp_path: Path) -> None:
    """Direct regression for the fatal eviction bug: ``to_empty(meta)`` on the
    parent block must NOT destroy scheduler-owned expert slot arrays — at the
    REAL nesting depth (``mlp.experts``), not just as a direct child. Silent
    meta slots make ``F.linear`` return zeros (torch does not raise), so this
    must be asserted explicitly."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_experts import _make_bank, _make_scheduler  # noqa: E402

    from swlp.core.streaming import _to_meta_preserving_cached_experts  # noqa: E402

    _make_bank(tmp_path)
    sched, module = _make_scheduler(tmp_path, slots_override=2)

    hidden = torch.randn(2, module.hidden)
    idx = torch.tensor([[1, 0], [2, 1]])
    weights = torch.full((2, 2), 0.5)
    module(hidden, idx, weights)  # fills the two slots
    ref_out = module(hidden, idx, weights).clone()

    class _Mlp(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = module
            self.gate = torch.nn.Linear(module.hidden, 8)

    class _NestedBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = _Mlp()
            self.attn = torch.nn.Linear(module.hidden, module.hidden)

    block = _NestedBlock()
    _to_meta_preserving_cached_experts(block)
    assert block.attn.weight.is_meta, "dense weights should be on meta"
    assert block.mlp.gate.weight.is_meta, "nested dense weights should be on meta"
    assert not module.gate_up_slots.data.is_meta, "expert slots must survive eviction"
    assert module.has(1) and module.has(2), "slot occupancy must survive eviction"

    out_after = module(hidden, idx, weights)
    assert torch.equal(out_after, ref_out), "forward output changed after eviction"
    sched.cleanup()
