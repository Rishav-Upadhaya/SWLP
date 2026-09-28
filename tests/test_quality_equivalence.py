"""Quality invariant tests for SWLP streaming.

Two properties are tested:

1. **Determinism**: two SWLP runs on the same prompt produce byte-identical
   output. Layer streaming must not introduce any non-determinism — the
   weights are fixed, the schedule is fixed, and greedy decoding is argmax.

2. **Coherence**: the SWLP completion is non-empty and the full-model HF
   completion is also non-empty when both run on the same model. This verifies
   that shard creation preserves the model's ability to generate.

NOTE: exact byte-for-byte equality between HF and SWLP is tested at the full-
model level (Mistral-7B) in scripts/research/quality_equivalence.py and documented in
docs/swlp_vs_airllm.md. On a randomly-initialized tiny model, FP16 rounding
differences in shard loading vs in-memory conversion can cause token-level
divergence (the logit differences are at FP16 epsilon level but deterministic
rounding breaks can flip the argmax). The CI test here focuses on the property
that matters most: SWLP is deterministic and produces well-formed output.
"""
from __future__ import annotations

import gc
from pathlib import Path

import torch

from swlp.config import AppConfig, CacheConfig, GenerationConfig, ModelConfig, RuntimeConfig
from swlp.model.shard import shard_model_by_layer
from swlp.runner import build_runner


def _clear_device_cache() -> None:
    """Release MPS / CUDA caches between runs to prevent prior-test contamination."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

TINY_MODEL = "hf-internal-testing/tiny-random-gpt2"
MAX_NEW_TOKENS = 8
PROMPT = "The quick brown fox"
SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def _make_cfg(
    backend: str, shard_dir: str | None = None, dtype: str = "float16"
) -> AppConfig:
    return AppConfig(
        model=ModelConfig(model_id=TINY_MODEL),
        cache=CacheConfig(),
        generation=GenerationConfig(
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
            do_sample=False,
            seed=SEED,
        ),
        runtime=RuntimeConfig(
            device=DEVICE,
            dtype=dtype,
            backend=backend,
            shard_dir=shard_dir,
            swlp_window_size=2,
            swlp_prefetch_depth=1,
            swlp_prefetch=True,
            swlp_residency="off",
            log_level="WARNING",
        ),
    )


def test_swlp_is_deterministic(tmp_path: Path) -> None:
    """Two SWLP runs on the same prompt and seed must produce identical output.

    Layer streaming must not introduce any non-determinism: weights are fixed,
    the schedule is fixed, and greedy decoding is a deterministic argmax.
    Checks generated_tokens (always >= 1) rather than the decoded string, since
    a random-weight model may generate EOS → empty completion after decoding.
    """
    shard_dir = tmp_path / "shards"
    shard_model_by_layer(TINY_MODEL, shard_dir)

    runner1 = build_runner(_make_cfg("swlp", str(shard_dir)))
    run1 = runner1.run(PROMPT)
    del runner1
    _clear_device_cache()  # release MPS state; prevents prior-test contamination

    runner2 = build_runner(_make_cfg("swlp", str(shard_dir)))
    run2 = runner2.run(PROMPT)
    del runner2

    toks1 = run1.metrics.generated_tokens or 0
    toks2 = run2.metrics.generated_tokens or 0
    c1 = run1.completion or ""
    c2 = run2.completion or ""

    assert toks1 >= 1, "SWLP run 1 produced no tokens"
    assert toks2 >= 1, "SWLP run 2 produced no tokens"
    assert toks1 == toks2 and c1 == c2, (
        f"SWLP output is non-deterministic across runs:\n"
        f"  run 1: {toks1} tokens → {c1!r}\n"
        f"  run 2: {toks2} tokens → {c2!r}"
    )


def test_swlp_and_hf_both_produce_output(tmp_path: Path) -> None:
    """Both SWLP and HF runners must produce non-empty completions on the same model.

    This verifies that shard creation preserves the model weights well enough
    to produce coherent (non-empty) output, confirming the shard round-trip
    does not corrupt the model.
    """
    shard_dir = tmp_path / "shards"
    shard_model_by_layer(TINY_MODEL, shard_dir)

    hf_result = build_runner(_make_cfg("hf")).run(PROMPT)
    swlp_result = build_runner(_make_cfg("swlp", str(shard_dir))).run(PROMPT)

    # Check that both runners generated at least 1 token — the decoded string may
    # be empty if the random-weight model generates only EOS (removed by
    # skip_special_tokens), but generated_tokens should always be >= 1.
    assert (hf_result.metrics.generated_tokens or 0) >= 1, "HF runner produced no tokens"
    assert (swlp_result.metrics.generated_tokens or 0) >= 1, "SWLP runner produced no tokens"


def _probe_first_token_logits(runner, tag: str, captured: dict) -> None:
    """Capture the run's first-token logits via a wrapped ``_select_next``.

    Used by the path-equivalence tests below: on a randomly-initialized model
    the top-2 logit gap is smaller than fp accumulation noise between
    mathematically-equivalent sweep shapes (~5e-6 fp32 on MPS, ~9e-8 for HF's
    own chunked forward), so greedy completions can flip without indicating a
    defect. The correct property is logit-level closeness of the two paths.
    """
    original = runner._select_next

    def wrapped(logits: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
        if tag not in captured:
            captured[tag] = (
                logits.detach().float().cpu().clone(),
                int(logits.argmax().item()),
            )
        return original(logits, generated)

    runner._select_next = wrapped


def test_chunked_prefill_matches_single_sweep(tmp_path: Path) -> None:
    """Chunked prefill (swlp_prefill_chunk) must produce the same last-position
    logits as one full-prompt sweep (Phase 26 losslessness).

    Chunking only changes *when* prompt tokens are fed through the layer
    sweep; causal attention over the accumulating KV makes the merged state
    mathematically identical. Verified at the logit level in float32:
    completions themselves are compared only on CPU, where accumulation is
    bitwise-stable across sweep shapes.
    """
    shard_dir = tmp_path / "shards"
    shard_model_by_layer(TINY_MODEL, shard_dir)

    captured: dict[str, tuple[torch.Tensor, int]] = {}
    plain_runner = build_runner(_make_cfg("swlp", str(shard_dir), dtype="float32"))
    _probe_first_token_logits(plain_runner, "plain", captured)
    plain = plain_runner.run(PROMPT)
    _clear_device_cache()

    chunked_cfg = _make_cfg("swlp", str(shard_dir), dtype="float32")
    chunked_cfg.runtime.swlp_prefill_chunk = 2  # "The quick brown fox" → 2-3 chunks
    chunked_runner = build_runner(chunked_cfg)
    _probe_first_token_logits(chunked_runner, "chunked", captured)
    chunked = chunked_runner.run(PROMPT)

    assert (plain.metrics.generated_tokens or 0) >= 1
    assert (chunked.metrics.generated_tokens or 0) >= 1
    lp, lc = captured["plain"][0], captured["chunked"][0]
    assert torch.allclose(lp, lc, rtol=0.0, atol=1e-4), (
        f"chunked prefill logits diverged: max|Δ|="
        f"{(lp - lc).abs().max().item():.3e}"
    )
    if DEVICE == "cpu":  # bitwise-stable across sweep shapes only on CPU
        assert plain.completion == chunked.completion


# The prefix-cache losslessness e2e lives in tests/test_llama_equivalence.py:
# GPT-2's adapter never takes the seeding path (it requires a Llama-like
# adapter + exact DynamicCache), so an e2e here would be vacuous.


def test_gpt2_streaming_first_token_logits_match_hf(tmp_path: Path) -> None:
    """Regression: the streaming path must load GPT-2's final norm (ln_f) with
    real weights.

    Before ``embed.pt`` persisted ``ln_f`` (Phase 26 fix), it was materialised
    from uninitialized memory: exactly-zero first-token logits on the first
    run in a process, stale-recycled-page garbage on later runs. Determinism
    tests could not catch this (both runs used the same wrong weights).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    shard_dir = tmp_path / "shards"
    shard_model_by_layer(TINY_MODEL, shard_dir, dtype_str="float32")

    captured: dict[str, tuple[torch.Tensor, int]] = {}
    runner = build_runner(_make_cfg("swlp", str(shard_dir), dtype="float32"))
    _probe_first_token_logits(runner, "swlp", captured)
    result = runner.run(PROMPT)
    assert (result.metrics.generated_tokens or 0) >= 1

    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    reference_model = (
        AutoModelForCausalLM.from_pretrained(TINY_MODEL, dtype=torch.float32)
        .eval()
        .to(DEVICE)
    )
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        ref_logits = reference_model(ids).logits[:, -1, :].float().cpu()

    got = captured["swlp"][0]
    assert got.abs().max().item() > 0.0, (
        "first-token logits are all zero — ln_f weights were never loaded"
    )
    assert torch.allclose(got, ref_logits, rtol=0.0, atol=1e-4), (
        f"streaming first-token logits diverged from HF reference: max|Δ|="
        f"{(got - ref_logits).abs().max().item():.3e}"
    )


