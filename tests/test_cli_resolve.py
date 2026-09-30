"""cli_resolve.resolve_target: users name a model, SWLP picks the backend."""
from __future__ import annotations

import json
from pathlib import Path

from swlp.cli_resolve import is_moe, resolve_target, shard_dir_for


def _shards(tmp: Path, *, experts: bool = False, mtp: bool = False) -> Path:
    d = tmp / "shards" / "m"
    d.mkdir(parents=True)
    (d / "shard_manifest.json").write_text(json.dumps(
        {"model_id": "org/m", "weight_dtype": "bfloat16"}))
    if experts:
        (d / "expert_index.json").write_text("{}")
    if mtp:
        (d / "mtp.safetensors").write_bytes(b"")
    return d


def _mlx_dir(tmp: Path, moe: bool) -> Path:
    d = tmp / "mlx"
    d.mkdir()
    cfg = {"quantization": {"bits": 4, "group_size": 64}}
    if moe:
        cfg["text_config"] = {"num_experts": 128}
    (d / "config.json").write_text(json.dumps(cfg))
    return d


def test_mock():
    assert resolve_target("x", backend="mock").backend == "mock"


def test_dense_shards_stream_losslessly(tmp_path):
    t = resolve_target(str(_shards(tmp_path)))
    assert (t.backend, t.model_id, t.mtp) == ("swlp", "org/m", False)
    assert "lossless" in t.label


def test_moe_shards_use_expert_streaming(tmp_path):
    t = resolve_target(str(_shards(tmp_path, experts=True)))
    assert t.backend == "mlx-moe" and t.shard_dir is not None


def test_mtp_head_enables_self_speculation(tmp_path):
    t = resolve_target(str(_shards(tmp_path, mtp=True)))
    assert t.backend == "speculative" and t.mtp


def test_pulled_alias_finds_its_shards(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _shards(tmp_path)
    assert resolve_target("m").backend == "swlp"  # shards/m, found by name


def test_mlx_checkpoint_dirs(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    moe = resolve_target(str(_mlx_dir(tmp_path / "a", moe=True)))
    assert moe.backend == "mlx-moe" and moe.local_path is not None and "4-bit" in moe.label
    dense = resolve_target(str(_mlx_dir(tmp_path / "b", moe=False)))
    assert dense.backend == "mlx" and dense.quant == "bf16"  # load the checkpoint as-is


def test_quant_means_resident_mlx():
    t = resolve_target("org/some-model", quant="int4", known_config={})
    assert (t.backend, t.quant) == ("mlx", "int4")


def test_hub_mlx_moe_repo():
    cfg = {"quantization": {"bits": 4}, "num_experts": 128}
    assert resolve_target("mlx-community/x-4bit", known_config=cfg).backend == "mlx-moe"


def test_plain_hub_model_needs_pull():
    t = resolve_target("org/big-model", known_config={"num_hidden_layers": 64})
    assert t.needs_pull and t.backend == "swlp" and t.shard_dir == shard_dir_for("org/big-model")


def test_backend_override_wins(tmp_path):
    assert resolve_target(str(_shards(tmp_path)), backend="hf").backend == "hf"


def test_is_moe():
    assert is_moe({"num_local_experts": 8})
    assert is_moe({"text_config": {"n_routed_experts": 64}})
    assert not is_moe({"num_hidden_layers": 32})


def test_quant_on_moe_shards_quantizes_on_load(tmp_path):
    t = resolve_target(str(_shards(tmp_path, experts=True)), quant="int4")
    assert (t.backend, t.quant) == ("mlx-moe", "int4") and "lossy" in t.label


def test_quant_on_dense_shards_goes_resident(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _shards(tmp_path)  # dense bf16 shards/m: re-read per token, so -q means MLX
    t = resolve_target("m", quant="int4", known_config={})
    assert (t.backend, t.quant) == ("mlx", "int4")


def test_quant_on_already_quantized_repo_is_ignored():
    cfg = {"quantization": {"bits": 4}, "num_experts": 128}
    t = resolve_target("mlx-community/x-4bit", quant="int8", known_config=cfg)
    assert t.backend == "mlx-moe" and "-q int8 ignored" in t.label


def test_quant_on_dense_shards_uses_the_manifest_model_id(tmp_path):
    t = resolve_target(str(_shards(tmp_path)), quant="int4")  # typed a local name
    assert t.model_id == "org/m"  # not the typed name (was: 401 from the Hub)
