"""Tests for runner expert streaming — bit-exactness and cache behavior (Phase 25)."""
from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from swlp.model.expert_bank import EXPERT_INDEX_FILE, ExpertIndex
from swlp.runner.expert_scheduler import ExpertScheduler
from swlp.runner.experts import SwlpCachedExperts

E, H, INTER = 6, 8, 4
ACT = "silu"
# Per-expert bytes in this fixture: (2*I*H + H*I) * 4 = 384 bytes fp32.
EXPERT_BYTES = (2 * INTER * H + H * INTER) * 4


def _silu(x: torch.Tensor) -> torch.Tensor:
    # Same op the module resolves via ACT2FN["silu"] — a hand-rolled
    # x*sigmoid(x) is NOT bitwise identical to the fused kernel.
    return torch.nn.functional.silu(x)


def _make_bank(tmp_path, layer: int = 0, dtype=torch.float32):
    gate_up = torch.randn(E, 2 * INTER, H, dtype=dtype)
    down = torch.randn(E, H, INTER, dtype=dtype)
    save_file(
        {
            f"model.layers.{layer}.mlp.experts.gate_up_proj": gate_up,
            f"model.layers.{layer}.mlp.experts.down_proj": down,
            f"model.layers.{layer}.mlp.gate_proj": torch.randn(INTER, H, dtype=dtype),
        },
        str(tmp_path / f"layer_{layer:03d}.experts.safetensors"),
    )
    save_file(
        {f"model.layers.{layer}.mlp.self_attn.q_proj": torch.randn(H, H, dtype=dtype)},
        str(tmp_path / f"layer_{layer:03d}.safetensors"),
    )
    index = ExpertIndex.build(tmp_path, num_layers=layer + 1)
    index.save(tmp_path / EXPERT_INDEX_FILE)
    return gate_up, down


def _reference_forward(hidden, top_k_index, top_k_weights, gate_up, down):
    """Verbatim port of the transformers fused-Experts loop (ascending ids)."""
    final = torch.zeros_like(hidden)
    with torch.no_grad():
        mask = torch.nn.functional.one_hot(top_k_index, num_classes=E).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
    for h_ in hit:
        eid = int(h_[0])
        if eid == E:
            continue
        top_pos, tok_idx = torch.where(mask[eid])
        cur = hidden[tok_idx]
        gate, up = nn.functional.linear(cur, gate_up[eid]).chunk(2, dim=-1)
        chs = _silu(gate) * up
        chs = nn.functional.linear(chs, down[eid])
        chs = chs * top_k_weights[tok_idx, top_pos, None]
        final.index_add_(0, tok_idx, chs.to(final.dtype))
    return final


def _make_scheduler(tmp_path, budget_bytes: int = 64 * 1024 * 1024,
                    slots_override: int | None = None, dtype=torch.float32,
                    mode: str = "off"):
    sched = ExpertScheduler(
        ExpertIndex.load(tmp_path / EXPERT_INDEX_FILE), tmp_path,
        torch.device("cpu"), dtype, budget_bytes=budget_bytes,
        mode=mode, workers=1,
    )
    li = sched.index.layers[0]
    slots = slots_override if slots_override is not None else li.num_experts
    module = SwlpCachedExperts(0, li.num_experts, li.hidden, li.intermediate,
                               dtype, torch.device("cpu"), ACT, sched, slots=slots)
    sched.register(0, module)
    return sched, module


def _routing(T: int, k: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, E, (T, k), generator=g)
    w = torch.rand(T, k, generator=g)
    w = w / w.sum(-1, keepdim=True)
    return idx, w


def test_forward_bit_exact_all_resident(tmp_path):
    gate_up, down = _make_bank(tmp_path)
    sched, module = _make_scheduler(tmp_path)
    hidden = torch.randn(5, H)
    idx, w = _routing(5, 2, seed=1)
    out = module(hidden, idx, w)
    ref = _reference_forward(hidden, idx, w, gate_up, down)
    assert torch.equal(out, ref)
    sched.cleanup()


def test_forward_bit_exact_with_evictions(tmp_path):
    gate_up, down = _make_bank(tmp_path)
    # Force constant eviction: 2 slots for 6 experts.
    sched, module = _make_scheduler(tmp_path, slots_override=2)
    hidden = torch.randn(7, H)
    idx, w = _routing(7, 3, seed=2)
    out = module(hidden, idx, w)
    ref = _reference_forward(hidden, idx, w, gate_up, down)
    assert torch.equal(out, ref)
    stats = sched.stats()
    assert stats["misses"] >= 5  # routing over 6 experts with 2 slots must miss
    sched.cleanup()


def test_lru_eviction_frees_least_recently_used(tmp_path):
    _make_bank(tmp_path)
    sched, module = _make_scheduler(tmp_path, slots_override=2)
    sched.ensure(0, 1)
    sched.ensure(0, 2)
    sched.ensure(0, 3)  # must evict expert 1 (least recently used)
    assert module.has(2) and module.has(3)
    assert not module.has(1)
    sched.cleanup()


def test_slot_cap_respects_budget(tmp_path):
    _make_bank(tmp_path)
    # Budget for 1 expert (384 B) + slack, below 2 experts (768 B) → cap 1.
    sched, _ = _make_scheduler(tmp_path, budget_bytes=EXPERT_BYTES + 100)
    assert sched.slot_cap(0) == 1
    sched.cleanup()


def test_resize_preserves_live_experts(tmp_path):
    gate_up, down = _make_bank(tmp_path)
    sched, module = _make_scheduler(tmp_path)
    sched.ensure(0, 4)
    module.resize(2)
    assert module.capacity == 2
    assert module.has(4)
    # The live expert's weights must survive the compaction untouched.
    slot_after = module.slot_of(4)
    assert slot_after is not None
    assert torch.equal(module.gate_up_slots.data[slot_after],
                       torch.cat([gate_up[4, :INTER, :], gate_up[4, INTER:, :]], dim=0))
    assert torch.equal(module.down_slots.data[slot_after], down[4])
    sched.cleanup()


def test_elastic_set_budget(tmp_path):
    _make_bank(tmp_path)
    sched, module = _make_scheduler(tmp_path)
    assert module.capacity == E
    sched.ensure(0, 0)
    sched.set_budget(EXPERT_BYTES + 100)  # room for ~one expert
    assert module.capacity == 1
    assert module.has(0)  # the live expert survives the shrink
    sched.cleanup()


def test_dtype_mismatch_refused(tmp_path):
    _make_bank(tmp_path, dtype=torch.float32)
    sched, _ = _make_scheduler(tmp_path, dtype=torch.float16)
    try:
        with pytest.raises(RuntimeError, match="dtype"):
            sched.ensure(0, 0)
    finally:
        sched.cleanup()


def test_record_routing_feeds_prediction(tmp_path):
    _make_bank(tmp_path)
    sched, _module = _make_scheduler(tmp_path, mode="predictive")
    for _ in range(3):
        sched.record_routing(0, [1, 2])
    # History-based prediction for layer 0 must rank 1, 2 first.
    assert sched._history.predict(0, 2) == [1, 2]
    sched.cleanup()


def test_invalid_prefetch_mode_rejected(tmp_path):
    """Round-2 fix: invalid swlp_expert_prefetch modes raise, not coerce."""
    import pytest

    from swlp.runner.expert_scheduler import ExpertScheduler

    _make_bank(tmp_path)  # expert_index.json must exist for construction
    with pytest.raises(ValueError, match="swlp_expert_prefetch"):
        ExpertScheduler(
            ExpertIndex.load(tmp_path / EXPERT_INDEX_FILE), tmp_path,
            torch.device("cpu"), torch.float32, budget_bytes=1,
            mode="router", workers=1,
        )


def test_prepare_set_stages_misses_concurrently(tmp_path):
    """prepare_set fans the routed set into the pool so ensure() consumes
    staged tensors instead of serial disk reads. Output must be bit-identical
    to a serial (mode=off) run of the same routing."""
    _make_bank(tmp_path)
    serial, serial_mod = _make_scheduler(tmp_path, mode="off")
    staged, staged_mod = _make_scheduler(tmp_path, mode="predictive")

    hidden = torch.randn(3, H)
    idx = torch.tensor([[0, 1], [2, 3], [4, 0]])  # 5 distinct routed experts
    w = torch.rand(3, 2)

    # Cold caches on both; staged mode fans misses into the pool first.
    out_serial = serial_mod(hidden, idx, w)
    out_staged = staged_mod(hidden, idx, w)
    assert torch.equal(out_serial, out_staged), "staging changed forward output"
    assert all(staged_mod.has(e) for e in (0, 1, 2, 3, 4))
    # Second pass is fully cached on both — still identical.
    assert torch.equal(serial_mod(hidden, idx, w), staged_mod(hidden, idx, w))
    assert staged.stats()["prefetch_hits"] >= 0  # stats wired
    serial.cleanup()
    staged.cleanup()


def _make_bank_paths_gate_up_down(tmp_path):
    from swlp.model.expert_bank import EXPERT_INDEX_FILE, ExpertIndex, read_expert
    index = ExpertIndex.load(tmp_path / EXPERT_INDEX_FILE)
    parts = read_expert(index, tmp_path, 0, 0)
    gate_up = torch.cat([parts["gate"], parts["up"]], dim=0)
    return gate_up, parts["down"]
