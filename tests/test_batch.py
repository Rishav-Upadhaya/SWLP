"""Tests for swlp.runner.batch — batched streaming helpers (Phase 10).

The full ``run_batch`` path needs a real streamed model and is measured
separately. These cover the pure logic: per-row token selection (the property
that keeps batched output bit-identical to batch-1), the growing attention
mask, completion decoding, and the left-padding position_ids formula.
"""
from __future__ import annotations

import torch

from swlp.config import AppConfig
from swlp.runner.batch import _decode_completion, _extend_mask, _select_batch
from swlp.runner.swlp import SWLPRunner


def _runner() -> SWLPRunner:
    config = AppConfig()
    config.runtime.backend = "swlp"
    return SWLPRunner(config)


def test_select_batch_is_row_independent():
    """Greedy selection per row must equal each row's own argmax — this is
    why a batched run produces output bit-identical to batch-1."""
    runner = _runner()
    logits = torch.tensor([[0.1, 0.9, 0.2], [0.7, 0.1, 0.2], [0.0, 0.0, 1.0]])
    generated = torch.zeros((3, 4), dtype=torch.long)
    picked = _select_batch(runner, logits, generated)
    assert picked.shape == (3, 1)
    assert picked.squeeze(-1).tolist() == [1, 0, 2]
    # Each row matches the same row selected on its own.
    for row in range(3):
        single = _select_batch(runner, logits[row : row + 1].clone(), generated[row : row + 1])
        assert int(single.item()) == int(picked[row].item())


def test_select_batch_repetition_penalty_per_row():
    runner = _runner()
    runner.config.generation.repetition_penalty = 2.0
    logits = torch.tensor([[5.0, 4.0, 0.0], [4.0, 5.0, 0.0]])
    # Row 0 has already produced token 0; row 1 has produced token 1.
    generated = torch.tensor([[0], [1]], dtype=torch.long)
    picked = _select_batch(runner, logits, generated).squeeze(-1).tolist()
    # The penalised token (5.0 -> 2.5) is demoted, flipping the winner each row.
    assert picked == [1, 0]


def test_extend_mask_appends_ones_column():
    mask = torch.tensor([[1, 1, 0], [0, 1, 1]])
    extended = _extend_mask(mask)
    assert extended.shape == (2, 4)
    assert extended[:, -1].tolist() == [1, 1]
    # Original columns are untouched.
    assert extended[:, :3].tolist() == mask.tolist()


class _FakeTokenizer:
    eos_token_id = 99

    def decode(self, ids, skip_special_tokens=False):  # noqa: ARG002
        keep = [i for i in ids if not (skip_special_tokens and i in (99, 0))]
        return " ".join(str(i) for i in keep)


def test_decode_completion_truncates_at_eos():
    tok = _FakeTokenizer()
    # prompt_len=2; generated tail = [5, 6, 99, 7]
    row = torch.tensor([1, 2, 5, 6, 99, 7])
    text, count = _decode_completion(tok, row, prompt_len=2)
    assert count == 3  # 5, 6, eos
    assert text == "5 6"  # eos skipped by skip_special_tokens


def test_decode_completion_no_eos_uses_full_tail():
    tok = _FakeTokenizer()
    row = torch.tensor([1, 2, 5, 6, 7])
    text, count = _decode_completion(tok, row, prompt_len=2)
    assert count == 3
    assert text == "5 6 7"


def test_left_pad_position_ids_formula():
    """position_ids = (mask.cumsum(-1) - 1).clamp(min=0): pad tokens get 0 and
    real tokens count from 0 — the formula LlamaLikeAdapter.prepare_step uses."""
    # Sequence 0: two pad tokens then 3 real; sequence 1: no padding.
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    pos = (mask.long().cumsum(-1) - 1).clamp(min=0)
    assert pos[0].tolist() == [0, 0, 0, 1, 2]
    assert pos[1].tolist() == [0, 1, 2, 3, 4]
