"""End-to-end losslessness tests on a tiny LLAMA-LIKE model.

Phase 26 round-2 audit fix: the previous prefix-cache e2e ran on GPT-2,
whose adapter never takes the seeding path (it requires a Llama-like
adapter + exact DynamicCache) — the test was vacuous. These tests use a
real in-process ``LlamaForCausalLM`` and ASSERT that cache hits happen,
so vacuity is impossible. They also cover the Llama chunked-prefill path
and the combined prefix-hit + chunking case.
"""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from swlp.config import AppConfig, CacheConfig, GenerationConfig, ModelConfig, RuntimeConfig
from swlp.core.prefix_cache import PrefixKVCache
from swlp.model.shard import shard_model_by_layer
from swlp.runner import build_runner

VOCAB = 112
NEW_TOKENS = 3
TURN1_IDS = [7, 19, 3, 42, 88, 15, 61, 30, 5, 77, 11, 94, 23, 50, 8, 66, 12, 39]
TURN2_EXTRA = [71, 4, 99, 27]  # turn2 shares the first 18 ids with turn1


def _tiny_llama() -> LlamaForCausalLM:
    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=256, tie_word_embeddings=False,
    )
    cfg.dtype = torch.float32
    torch.manual_seed(7)
    return LlamaForCausalLM(cfg).eval().to(torch.float32)


def _prepare(tmp_path: Path) -> tuple[Path, Path]:
    model = _tiny_llama()
    hf_dir = tmp_path / "hf"
    model.save_pretrained(hf_dir)
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    word_level = Tokenizer(models.WordLevel(vocab={"[UNK]": 0, "a": 1}, unk_token="[UNK]"))
    PreTrainedTokenizerFast(tokenizer_object=word_level).save_pretrained(hf_dir)
    shard_dir = tmp_path / "shards"
    shard_model_by_layer(str(hf_dir), shard_dir, dtype_str="float32")
    return hf_dir, shard_dir


def _cfg(model_dir: Path, shard_dir: Path, *, chunk: int = 0) -> AppConfig:
    return AppConfig(
        model=ModelConfig(model_id=str(model_dir)),
        cache=CacheConfig(),
        generation=GenerationConfig(
            max_new_tokens=NEW_TOKENS, temperature=0.0, do_sample=False, seed=42
        ),
        runtime=RuntimeConfig(
            device="cpu", dtype="float32", backend="swlp", shard_dir=str(shard_dir),
            swlp_window_size=2, swlp_prefetch_depth=1, swlp_prefetch=True,
            swlp_residency="off", log_level="WARNING",
            swlp_fallback_to_baseline=False, swlp_prefill_chunk=chunk,
        ),
    )


class _FixedIdsTokenizer:
    eos_token_id = None

    def __init__(self, ids: list[int]) -> None:
        self._ids = ids

    def __call__(self, prompt: str, return_tensors: str = "pt") -> dict:
        return {"input_ids": torch.tensor([self._ids], dtype=torch.long)}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return ""


def _freeze_load(runner) -> None:
    """Keep the loaded model across runs: run()'s cleanup nulls it and a
    plain no-op load would leave ``model=None``. Installed once, capturing
    the live model; later calls are no-ops."""
    if getattr(runner, "_load_frozen", False):
        return
    kept = runner.model
    assert kept is not None, "freeze_load must be called after an explicit load()"

    def _load() -> float:
        runner.model = kept
        return 0.0

    runner.load = _load
    runner._load_frozen = True


def _run_capturing(runner, ids: list[int]) -> tuple[list[torch.Tensor], object]:
    """Run once with fixed ids; return (per-step logits, RunResult)."""
    _freeze_load(runner)
    runner.tokenizer = _FixedIdsTokenizer(ids)
    picked: list[torch.Tensor] = []
    original = runner._select_next

    def wrapped(logits: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
        picked.append(logits.detach().float().clone())
        return original(logits, generated)

    runner._select_next = wrapped
    result = runner.run("unused — fixed-ids tokenizer injects ids")
    runner._select_next = original
    return picked, result



def _assert_generation_equivalent(got: list[torch.Tensor], other: list[torch.Tensor],
                                  label: str) -> None:
    """Per-step equivalence across multi-token generation.

    Step 0 (pure prefill) is held to 1e-4. Later steps compare greedy
    tokens strictly and logits at 5e-3: mathematically-equivalent sweep
    shapes differ at ~5e-6 per step in fp32, and that noise compounds
    through each generated token's attention over the growing KV.
    """
    assert len(got) == len(other), f"{label}: step count mismatch"
    for step, (g, o) in enumerate(zip(got, other, strict=True)):
        assert g.argmax().item() == o.argmax().item(), (
            f"{label}: greedy token diverged at step {step}"
        )
        atol = 1e-4 if step == 0 else 5e-3
        assert torch.allclose(g, o, rtol=0.0, atol=atol), (
            f"{label}: logits diverged at step {step}: "
            f"max|Δ|={(g - o).abs().max().item():.3e}"
        )


def test_prefix_cache_hit_is_real_and_lossless(tmp_path: Path) -> None:
    """Turn 2 must take a REAL seeding path (asserted via stats) and produce
    per-step logits identical to a cold no-cache run of the same ids."""
    model_dir, shard_dir = _prepare(tmp_path)
    reference_model = _tiny_llama()

    # Turn 1 seeds the cache; its generated tokens extend the stored
    # sequence (chat semantics: history includes the assistant reply).
    runner = build_runner(_cfg(model_dir, shard_dir))
    runner.load()
    cache = PrefixKVCache(max_entries=4)
    runner.set_prefix_cache(cache)
    picked1, _res1 = _run_capturing(runner, TURN1_IDS)
    gen_tokens = [int(t.argmax()) for t in picked1]
    assert cache.stats()["stored"] >= 1, "turn 1 must store a snapshot"

    # Turn 2 extends turn 1's full stored sequence with new ids → lookup
    # must HIT and seed. The HF reference and cold run use THESE ids.
    turn2_ids = [*TURN1_IDS, *gen_tokens, *TURN2_EXTRA]
    with torch.no_grad():
        ref_logits_steps: list[torch.Tensor] = []
        past = None
        for start in range(0, len(turn2_ids), 6):
            piece = torch.tensor([turn2_ids[start:start + 6]], dtype=torch.long)
            out = reference_model(piece, past_key_values=past, use_cache=True)
            past = out.past_key_values
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ref_logits_steps.append(out.logits[:, -1, :].float())
        for _ in range(NEW_TOKENS - 1):
            out = reference_model(next_id, past_key_values=past, use_cache=True)
            ref_logits_steps.append(out.logits[:, -1, :].float())
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    seeded, seeded_result = _run_capturing(runner, turn2_ids)
    stats = cache.stats()
    assert stats["hits"] >= 1, (
        f"prefix cache never hit — the seeding path was not exercised ({stats})"
    )

    # Cold reference for the same turn-2 ids, no cache.
    cold_runner = build_runner(_cfg(model_dir, shard_dir))
    cold_runner.load()
    cold, cold_result = _run_capturing(cold_runner, turn2_ids)

    _assert_generation_equivalent(seeded, cold, "prefix-seeded vs cold")
    _assert_generation_equivalent(cold, ref_logits_steps, "streamed vs HF reference")

    # run() public contract survives a hit: full-sequence accounting and an
    # identical completion to the cold run.
    assert seeded_result.metrics.generated_tokens == NEW_TOKENS, (
        f"hit-run undercounted generated tokens: "
        f"{seeded_result.metrics.generated_tokens}"
    )
    assert seeded_result.completion == cold_result.completion


def test_chunked_prefill_matches_single_sweep_llama(tmp_path: Path) -> None:
    """Llama-like chunked prefill must match a single full sweep (per-step
    logits), including the combined prefix-hit + chunking path."""
    model_dir, shard_dir = _prepare(tmp_path)

    # Establish the shared turn-2 sequence once: turn 1's greedy reply is
    # part of the stored prefix, so both sides must use the same ids.
    seed_runner = build_runner(_cfg(model_dir, shard_dir))
    seed_runner.load()
    picked1, _seed_res = _run_capturing(seed_runner, TURN1_IDS)
    gen_tokens = [int(t.argmax()) for t in picked1]
    turn2_ids = [*TURN1_IDS, *gen_tokens, *TURN2_EXTRA]

    plain_runner = build_runner(_cfg(model_dir, shard_dir))
    plain_runner.load()
    plain, _plain_res = _run_capturing(plain_runner, turn2_ids)

    chunked_runner = build_runner(_cfg(model_dir, shard_dir, chunk=5))
    chunked_runner.load()
    # Chunking COMBINED with a prefix hit: turn 1 seeds, turn 2 hits and
    # sweeps its suffix in slices.
    cache = PrefixKVCache(max_entries=4)
    chunked_runner.set_prefix_cache(cache)
    _run_capturing(chunked_runner, TURN1_IDS)
    chunked, _chunked_res = _run_capturing(chunked_runner, turn2_ids)
    assert cache.stats()["hits"] >= 1

    _assert_generation_equivalent(chunked, plain, "chunked+prefix vs single-sweep")
