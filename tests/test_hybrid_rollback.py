"""Tests for swlp.runner.hybrid_rollback — exact DeltaNet state rollback after
speculative verification on a hybrid (linear + full attention) model."""
from __future__ import annotations

import pytest
import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.cache_utils import DynamicCache

from swlp.runner.hybrid_rollback import (
    is_hybrid_cache,
    record_linear_attention,
    rollback_hybrid_cache,
)

PROMPT = [5, 17, 99, 3, 42, 8, 61, 120, 11]
FED = [33, 7, 90, 14, 2]  # verify input: last token + 4 drafts (K+1 = 5)
TAIL = [70, 21, 45]  # tokens decoded after the rollback


def tiny_qwen3_5_text() -> Qwen3_5ForCausalLM:
    cfg = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=8, tie_word_embeddings=False,
    )
    torch.manual_seed(3)
    model = Qwen3_5ForCausalLM(cfg).eval().to(torch.float32)
    # Random init leaves A_log/dt/norms near-trivial; perturb every parameter so
    # the recurrent state genuinely carries history.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return model


def _step(model: Qwen3_5ForCausalLM, cache: DynamicCache, ids: list[int]) -> torch.Tensor:
    out = model(torch.tensor([ids]), past_key_values=cache, use_cache=True)
    return out.logits[0].float()


def _reference(model: Qwen3_5ForCausalLM, kept: list[int]) -> list[torch.Tensor]:
    cache = DynamicCache(config=model.config)
    _step(model, cache, PROMPT)
    for tok in kept:
        _step(model, cache, [tok])
    return [_step(model, cache, [tok])[-1] for tok in TAIL]


@pytest.mark.parametrize("keep", [1, 2, 4, 5])
def test_rollback_matches_token_by_token_decode(keep: int) -> None:
    model = tiny_qwen3_5_text()
    with torch.no_grad():
        ref = _reference(model, FED[:keep])
        cache = DynamicCache(config=model.config)
        _step(model, cache, PROMPT)
        assert is_hybrid_cache(cache)
        with record_linear_attention(model) as record:
            _step(model, cache, FED)
        assert len(record.conv_inputs) == len(record.delta_calls) == 3
        rollback_hybrid_cache(cache, record, keep=keep, fed=len(FED))
        assert cache.get_seq_length() == len(PROMPT) + keep
        got = [_step(model, cache, [tok])[-1] for tok in TAIL]
    for g, r in zip(got, ref, strict=True):
        assert torch.allclose(g, r, atol=1e-4), (g - r).abs().max()


def test_without_rollback_state_is_corrupted() -> None:
    """The bug is real: dropping only the attention KV leaves DeltaNet state
    polluted by rejected tokens, and later logits diverge."""
    model = tiny_qwen3_5_text()
    keep = 1
    with torch.no_grad():
        ref = _reference(model, FED[:keep])
        cache = DynamicCache(config=model.config)
        _step(model, cache, PROMPT)
        _step(model, cache, FED)
        for layer in cache.layers:
            if not hasattr(layer, "recurrent_states"):
                layer.crop(keep - len(FED))
        got = _step(model, cache, [TAIL[0]])[-1]
    assert not torch.allclose(got, ref[0], atol=1e-3)


def test_conv_state_is_exact_column_slice() -> None:
    """conv state after keeping m tokens == last `kernel` columns of
    cat(old_state, new[:m]) == x[..., m : m + kernel]."""
    model = tiny_qwen3_5_text()
    with torch.no_grad():
        for keep in range(1, len(FED) + 1):
            ref = DynamicCache(config=model.config)
            _step(model, ref, PROMPT + FED[:keep])
            cache = DynamicCache(config=model.config)
            _step(model, cache, PROMPT)
            with record_linear_attention(model) as record:
                _step(model, cache, FED)
            rollback_hybrid_cache(cache, record, keep=keep, fed=len(FED))
            for a, b in zip(cache.layers[:3], ref.layers[:3], strict=True):
                assert torch.allclose(a.conv_states[0], b.conv_states[0], atol=1e-5)


def test_rollback_noop_when_all_kept_and_restores_globals() -> None:
    model = tiny_qwen3_5_text()
    import transformers.models.qwen3_5.modeling_qwen3_5 as mod

    original = mod.torch_chunk_gated_delta_rule
    with torch.no_grad():
        cache = DynamicCache(config=model.config)
        _step(model, cache, PROMPT)
        with record_linear_attention(model) as record:
            _step(model, cache, FED)
        before = [layer.recurrent_states[0].clone() for layer in cache.layers[:3]]
        rollback_hybrid_cache(cache, record, keep=len(FED), fed=len(FED))
    assert mod.torch_chunk_gated_delta_rule is original
    for layer, b in zip(cache.layers[:3], before, strict=True):
        assert torch.equal(layer.recurrent_states[0], b)


def test_rollback_rejects_incomplete_record() -> None:
    model = tiny_qwen3_5_text()
    with torch.no_grad():
        cache = DynamicCache(config=model.config)
        _step(model, cache, PROMPT)
        with record_linear_attention(model) as record:
            _step(model, cache, FED)
    record.delta_calls.pop()
    with pytest.raises(RuntimeError, match="not recorded whole"):
        rollback_hybrid_cache(cache, record, keep=1, fed=len(FED))
