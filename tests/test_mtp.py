"""Tests for swlp.runner.mtp — MTP head loading, MtpDrafter KV consistency, and
bit-identical speculative decoding (MTP drafter + hybrid rollback) on a tiny
streamed Qwen3.5 model."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from test_hybrid_rollback import tiny_qwen3_5_text
from test_llama_equivalence import TURN1_IDS, _cfg, _freeze_load
from test_qwen3_5_equivalence import _tiny_qwen3_5
from transformers import PretrainedConfig
from transformers.cache_utils import DynamicCache

from swlp.model.shard import MTP_FILE, shard_model_by_layer
from swlp.runner import build_runner
from swlp.runner.mtp import MtpDrafter, MtpHead, load_mtp_head

CPU = torch.device("cpu")
MAX_NEW = 12


def _random_head(text_config: PretrainedConfig, seed: int = 11) -> MtpHead:
    torch.manual_seed(seed)
    head = MtpHead(text_config).eval()
    with torch.no_grad():
        for p in head.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return head


def _write_head(head: MtpHead, shard_dir: Path) -> None:
    state = {f"mtp.{k}": v.contiguous() for k, v in head.state_dict().items()}
    save_file(state, str(shard_dir / MTP_FILE))


def test_head_matches_checkpoint_names_and_roundtrips(tmp_path: Path) -> None:
    model = tiny_qwen3_5_text()
    head = _random_head(model.config)
    names = set(head.state_dict())
    # The 15 tensors Qwen3.8-27B ships under mtp.* (prefix stripped).
    assert {"fc.weight", "norm.weight", "pre_fc_norm_embedding.weight",
            "pre_fc_norm_hidden.weight", "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.q_norm.weight", "layers.0.mlp.gate_proj.weight",
            "layers.0.input_layernorm.weight"} <= names
    assert len(names) == 15
    _write_head(head, tmp_path)
    loaded = load_mtp_head(tmp_path, model.config, torch.float32, CPU)
    for k, v in head.state_dict().items():
        assert torch.equal(loaded.state_dict()[k], v)


def test_load_mtp_head_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no MTP head"):
        load_mtp_head(tmp_path, tiny_qwen3_5_text().config, torch.float32, CPU)


def _target_hidden(model: torch.nn.Module, ids: list[int]) -> torch.Tensor:
    """Post-final-norm target hidden states for every position of ``ids``."""
    out = model.model(torch.tensor([ids]), past_key_values=DynamicCache(config=model.config),
                      use_cache=True)
    return out.last_hidden_state  # HF applies the final norm here


def _history(model: torch.nn.Module) -> tuple[list[int], torch.Tensor]:
    ids = [5, 17, 99, 3, 42, 8, 61, 120, 11, 33, 7, 90]
    with torch.no_grad():
        hidden = _target_hidden(model, ids)
    return ids, hidden


def test_propose_shapes_and_determinism() -> None:
    model = tiny_qwen3_5_text()
    head = _random_head(model.config)
    ids, hidden = _history(model)
    runs = []
    for _ in range(2):
        drafter = MtpDrafter(head, model, CPU, max_draft=4)
        drafter.extend(hidden[:, :-1], ids[1:])
        runs.append(drafter.propose(ids))
    assert len(runs[0]) == 4 and all(0 <= t < 128 for t in runs[0])
    assert runs[0] == runs[1]
    # Chain entries are cropped: the cache holds only confirmed positions.
    assert drafter._cache.get_seq_length() == len(ids) - 1


def test_extend_rejects_mismatched_lengths() -> None:
    model = tiny_qwen3_5_text()
    drafter = MtpDrafter(_random_head(model.config), model, CPU, max_draft=2)
    with pytest.raises(ValueError):
        drafter.extend(torch.zeros(1, 3, 64), [1, 2])


def test_incremental_cache_matches_fresh_history() -> None:
    """Drafting after N more confirmed tokens == a fresh MTP fed the same history."""
    model = tiny_qwen3_5_text()
    head = _random_head(model.config)
    ids, hidden = _history(model)
    split = 6
    inc = MtpDrafter(head, model, CPU, max_draft=3)
    inc.extend(hidden[:, :split - 1], ids[1:split])
    inc.propose(ids[:split])
    inc.extend(hidden[:, split - 1:-1], ids[split:])
    fresh = MtpDrafter(head, model, CPU, max_draft=3)
    fresh.extend(hidden[:, :-1], ids[1:])
    assert inc.propose(ids) == fresh.propose(ids)
    for a, b in zip(inc._cache.layers, fresh._cache.layers, strict=True):
        assert torch.allclose(a.keys, b.keys, atol=1e-5)


def test_zero_draft_or_no_pending_proposes_nothing() -> None:
    model = tiny_qwen3_5_text()
    ids, hidden = _history(model)
    drafter = MtpDrafter(_random_head(model.config), model, CPU, max_draft=0)
    drafter.extend(hidden[:, :-1], ids[1:])
    assert drafter.propose(ids) == []
    assert MtpDrafter(_random_head(model.config), model, CPU, max_draft=3).propose(ids) == []


# ── end-to-end: streamed hybrid model, speculative == plain greedy ────────────

class _IdsTokenizer:
    eos_token_id = None

    def __init__(self, ids: list[int]) -> None:
        self._ids = ids

    def __call__(self, prompt: str, return_tensors: str = "pt") -> dict:
        return {"input_ids": torch.tensor([self._ids], dtype=torch.long)}

    def decode(self, ids: object, skip_special_tokens: bool = True) -> str:
        return " ".join(str(int(i)) for i in ids)


def _prepare(tmp_path: Path) -> tuple[Path, Path]:
    model = _tiny_qwen3_5()
    # Default init (std 0.02) makes greedy output nearly context-blind; perturb
    # so a corrupted cache actually changes tokens.
    torch.manual_seed(5)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    hf_dir = tmp_path / "hf"
    model.save_pretrained(hf_dir)
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    word_level = Tokenizer(models.WordLevel(vocab={"[UNK]": 0, "a": 1}, unk_token="[UNK]"))
    PreTrainedTokenizerFast(tokenizer_object=word_level).save_pretrained(hf_dir)
    shard_dir = tmp_path / "shards"
    shard_model_by_layer(str(hf_dir), shard_dir, dtype_str="float32")
    _write_head(_random_head(model.config.get_text_config()), shard_dir)
    return hf_dir, shard_dir


def _generate(hf_dir: Path, shard_dir: Path, backend: str, *, mtp: bool,
              oracle: list[int] | None = None) -> tuple[str, object]:
    cfg = _cfg(hf_dir, shard_dir)
    cfg.runtime.backend = backend
    cfg.runtime.swlp_mtp = mtp
    cfg.runtime.swlp_spec_max_draft = 4
    cfg.generation.max_new_tokens = MAX_NEW
    runner = build_runner(cfg)
    runner.load()
    _freeze_load(runner)
    runner.tokenizer = _IdsTokenizer(TURN1_IDS)
    if oracle is not None:
        # Drafts right for 2 tokens then wrong: forces partial acceptance so
        # the hybrid rollback runs mid-sequence on every sweep.
        class _Oracle:
            def propose(self, tokens: list[int]) -> list[int]:
                n = len(tokens) - len(TURN1_IDS)
                good = oracle[n:n + 2]
                if n + 2 >= len(oracle):
                    return good
                return [*good, (oracle[n + 2] + 1) % 100]

        runner._build_drafter = lambda: _Oracle()
    return runner.run("unused").completion, runner


def test_mtp_speculative_is_bit_identical_to_greedy(tmp_path: Path) -> None:
    hf_dir, shard_dir = _prepare(tmp_path)
    plain, _ = _generate(hf_dir, shard_dir, "swlp", mtp=False)
    spec, runner = _generate(hf_dir, shard_dir, "speculative", mtp=True)
    assert spec == plain
    assert runner._mtp_head is not None
    assert not runner.degradations


def test_partial_acceptance_rollback_is_bit_identical(tmp_path: Path) -> None:
    hf_dir, shard_dir = _prepare(tmp_path)
    plain, _ = _generate(hf_dir, shard_dir, "swlp", mtp=False)
    ref_ids = [int(t) for t in plain.split()[len(TURN1_IDS):]]
    spec, runner = _generate(hf_dir, shard_dir, "speculative", mtp=False, oracle=ref_ids)
    assert spec == plain
    assert not runner.degradations


def test_mtp_and_draft_model_are_mutually_exclusive(tmp_path: Path) -> None:
    hf_dir, shard_dir = _prepare(tmp_path)
    cfg = _cfg(hf_dir, shard_dir)
    cfg.runtime.backend = "speculative"
    cfg.runtime.swlp_mtp = True
    cfg.runtime.swlp_draft_model = "some/draft"
    runner = build_runner(cfg)
    with pytest.raises(ValueError, match="mutually exclusive"):
        runner._build_drafter()


def test_partial_acceptance_without_rollback_diverges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guard the guard: with the hybrid rollback disabled the same oracle run
    drifts from greedy, so the test above genuinely exercises the rollback."""
    import swlp.runner.speculative as spec_mod

    hf_dir, shard_dir = _prepare(tmp_path)
    plain, _ = _generate(hf_dir, shard_dir, "swlp", mtp=False)
    ref_ids = [int(t) for t in plain.split()[len(TURN1_IDS):]]
    monkeypatch.setattr(spec_mod, "rollback_hybrid_cache", lambda *a, **k: None)
    spec, _ = _generate(hf_dir, shard_dir, "speculative", mtp=False, oracle=ref_ids)
    assert spec != plain
