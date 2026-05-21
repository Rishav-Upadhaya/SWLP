"""Speculative-decoding primitives for SWLP (Phase 5).

Pure computation — no I/O, no model calls. The draft proposer (``NgramDrafter``)
and the greedy verifier (``verify_greedy``) here are unit-testable in isolation;
the orchestration that runs the streamed target model lives in
``runner/speculative.py``.

Prompt-lookup (n-gram) drafting
-------------------------------
Instead of a separate small draft model, ``NgramDrafter`` proposes continuation
tokens by finding the most recent earlier occurrence of the trailing ``n``-token
pattern in the sequence generated so far, and copying whatever followed it.

This is *draft-model-free* speculative decoding:

- zero extra RAM (no second model resident — critical on the memory-bound M5,
  see ``docs/phase5_design_decisions.md``),
- no tokenizer mismatch (it operates on the target model's own token IDs),
- lossless — every proposed token is still greedily verified by the target
  model, so output is bit-identical to plain greedy decoding.

Greedy verification
-------------------
The target model runs one forward over ``[last_token, *draft]`` and produces a
greedy pick at each position.  ``verify_greedy`` accepts draft tokens up to the
first mismatch, then appends exactly one target token (the correction at the
mismatch, or a bonus token when every draft token was accepted).  The result is
identical to what greedy decoding of the target model alone would produce.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SpeculativeConfig:
    """Tuning knobs for prompt-lookup speculative decoding."""

    ngram_size: int = 3   # length of the trailing pattern matched against history
    max_draft: int = 8    # K — maximum number of tokens proposed per sweep


class NgramDrafter:
    """Prompt-lookup token proposer.

    Given the sequence of token IDs produced so far, propose up to
    ``config.max_draft`` continuation tokens by locating the most recent earlier
    occurrence of the trailing ``config.ngram_size``-token pattern and copying
    the tokens that followed it.
    """

    def __init__(self, config: SpeculativeConfig) -> None:
        self._n = max(1, int(config.ngram_size))
        self._k = max(0, int(config.max_draft))

    @property
    def ngram_size(self) -> int:
        return self._n

    @property
    def max_draft(self) -> int:
        return self._k

    def propose(self, tokens: list[int]) -> list[int]:
        """Return up to K draft tokens, or ``[]`` when no n-gram match is found.

        Args:
            tokens: all token IDs generated so far (prompt + completion).

        Returns:
            A list of 0..K proposed continuation token IDs.
        """
        if self._k == 0 or len(tokens) <= self._n:
            return []
        pattern = tokens[-self._n:]
        # Search backwards for the most recent earlier occurrence of `pattern`,
        # excluding the trailing pattern itself.
        for start in range(len(tokens) - self._n - 1, -1, -1):
            if tokens[start:start + self._n] == pattern:
                draft = tokens[start + self._n: start + self._n + self._k]
                return list(draft)
        return []


def verify_greedy(
    draft_tokens: list[int],
    target_picks: list[int],
) -> tuple[list[int], int]:
    """Greedy speculative verification.

    Args:
        draft_tokens: the K tokens proposed by the drafter.
        target_picks: ``K + 1`` tokens — the target model's greedy choice at each
            verified position. ``target_picks[i]`` is what the target would emit
            given the confirmed prefix plus ``draft_tokens[:i]``.

    Returns:
        ``(new_tokens, n_accepted)`` where:

        - ``new_tokens`` is the list to append to the running output: the
          accepted draft tokens followed by exactly one target token (the
          correction at the first mismatch, or the bonus token when every draft
          token was accepted). ``len(new_tokens) == n_accepted + 1`` always.
        - ``n_accepted`` is how many draft tokens matched the target (0..K) — the
          caller uses it to crop the KV cache back to the confirmed prefix.

    Raises:
        ValueError: if ``target_picks`` does not have exactly ``len(draft_tokens) + 1``
            entries.
    """
    if len(target_picks) != len(draft_tokens) + 1:
        raise ValueError(
            f"target_picks must have len(draft_tokens)+1 entries; "
            f"got {len(target_picks)} for {len(draft_tokens)} draft tokens"
        )
    new_tokens: list[int] = []
    for i, drafted in enumerate(draft_tokens):
        predicted = target_picks[i]
        if predicted != drafted:
            # First mismatch — the target's pick replaces the rejected draft.
            new_tokens.append(predicted)
            return new_tokens, i
        new_tokens.append(drafted)
    # Every draft token was accepted — append the bonus token.
    new_tokens.append(target_picks[len(draft_tokens)])
    return new_tokens, len(draft_tokens)
