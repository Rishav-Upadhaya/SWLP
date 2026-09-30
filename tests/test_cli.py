"""The `swlp` CLI surface: seven commands, end-to-end on the mock backend."""
from __future__ import annotations

import io
import json

import pytest

from swlp.cli import main
from swlp.cli_args import build_parser, resolve_model

COMMANDS = ["chat", "run", "serve", "pull", "models", "rm", "doctor", "bench"]


def test_bare_invocation_prints_help(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    for cmd in COMMANDS:
        assert f"swlp {cmd}" in out


def test_version(capsys):
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.startswith("swlp ")


@pytest.mark.parametrize("removed", ["simulate", "suite", "package", "layer", "download",
                                     "compress-shards", "profile", "report", "help"])
def test_removed_commands_are_gone(removed):
    with pytest.raises(SystemExit):
        build_parser().parse_args([removed])


def test_every_command_parses():
    p = build_parser()
    assert p.parse_args(["chat", "m", "-q", "int4"]).quant == "int4"
    assert p.parse_args(["run", "m", "hi", "-n", "7"]).max_tokens == 7
    assert p.parse_args(["serve", "m", "--port", "9"]).port == 9
    assert p.parse_args(["pull", "m"]).model == "m"
    assert p.parse_args(["models"]).command == "models"
    assert p.parse_args(["doctor"]).model is None
    assert p.parse_args(["bench", "m", "--runs", "2"]).runs == 2


def test_run_streams_answer(capsys):
    assert main(["run", "anything", "hello there", "--backend", "mock"]) == 0
    out = capsys.readouterr().out
    assert "hello there" in out and "tok/s" in out


def test_run_json(capsys):
    assert main(["run", "anything", "hi json", "--backend", "mock", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["backend"] == "mock"
    assert "hi json" in payload["completion"]
    assert "throughput_tokens_per_second" in payload["metrics"]


def test_run_reads_prompt_from_stdin(capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("from stdin"))
    assert main(["run", "anything", "-", "--backend", "mock"]) == 0
    assert "from stdin" in capsys.readouterr().out


def test_run_without_prompt_explains(capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert main(["run", "anything", "--backend", "mock"]) == 1
    assert "no prompt" in capsys.readouterr().err


def test_unprepared_model_suggests_pull(capsys, monkeypatch):
    monkeypatch.setattr("swlp.cli_resolve.hub_config", lambda _id: None)
    assert main(["run", "some/unpulled-model", "hi"]) == 1
    err = capsys.readouterr().err
    assert "swlp pull some/unpulled-model" in err and "-q int4" in err


def test_chat_exits_on_slash_exit(capsys, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *_: "/exit")
    assert main(["chat", "anything", "--backend", "mock"]) == 0
    assert "bye" in capsys.readouterr().out


def test_bench_json(capsys):
    assert main(["bench", "anything", "--backend", "mock", "--runs", "1", "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)["summary"]
    assert summary["runs"] >= 1


def test_models_lists_aliases(capsys):
    assert main(["models"]) == 0
    assert "gemma4-26b" in capsys.readouterr().out


def test_model_alias_resolution():
    assert resolve_model("mistral-7b") == "unsloth/mistral-7b-instruct-v0.2"
    assert resolve_model("some/org-model") == "some/org-model"


def test_pull_refuses_when_disk_too_small(tmp_path, monkeypatch, capsys):
    from swlp import cli

    monkeypatch.setattr("swlp.cli_doctor.KNOWN_FP16_GB", {"big": 100.0})
    monkeypatch.setattr("shutil.disk_usage", lambda _p: type("U", (), {"free": 10 * 1024**3})())
    assert cli._enough_disk("big", tmp_path) is False
    assert "not enough disk" in capsys.readouterr().err
    assert cli._enough_disk("unknown-alias", tmp_path) is True


def test_max_tokens_spellings_and_until_done_default():
    p = build_parser()
    for flag in ("-n", "--max-tokens", "--max_tokens", "--max-new-tokens"):
        assert p.parse_args(["chat", "m", flag, "18000"]).max_tokens == 18000
    from swlp.cli import UNTIL_DONE, _prepare

    config, _ = _prepare(p.parse_args(["chat", "m", "--backend", "mock"]))
    assert config.generation.max_new_tokens == UNTIL_DONE  # no 32-token cut-off


def test_ctrl_c_at_prompt_exits(capsys, monkeypatch):
    def interrupt(*_):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupt)
    assert main(["chat", "anything", "--backend", "mock"]) == 0
    assert "bye" in capsys.readouterr().out


def _fake_model(tmp_path, monkeypatch):
    """shards/<name> + an empty HF cache, in a scratch cwd."""
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "shards" / "demo"
    d.mkdir(parents=True)
    (d / "shard_manifest.json").write_text("{}")
    (d / "layer_000.safetensors").write_bytes(b"x" * 1024)
    monkeypatch.setattr("huggingface_hub.scan_cache_dir",
                        lambda *_: type("I", (), {"repos": []})())
    return d


def test_rm_lists_then_deletes_with_yes(tmp_path, monkeypatch, capsys):
    d = _fake_model(tmp_path, monkeypatch)
    assert main(["rm", "demo", "--yes"]) == 0
    assert not d.exists()
    assert "freed" in capsys.readouterr().out


def test_rm_keeps_everything_when_declined(tmp_path, monkeypatch, capsys):
    d = _fake_model(tmp_path, monkeypatch)
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    assert main(["rm", "demo"]) == 0
    assert d.exists()


def test_rm_unknown_model_is_a_noop(tmp_path, monkeypatch, capsys):
    _fake_model(tmp_path, monkeypatch)
    assert main(["rm", "not-installed"]) == 0
    assert "nothing on disk" in capsys.readouterr().out


def test_details_explains_without_loading(capsys, tmp_path, monkeypatch):
    """`-d` and bare `--backend` print the plan (backends + settings), load nothing."""
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "shards" / "m"
    d.mkdir(parents=True)
    (d / "shard_manifest.json").write_text('{"model_id": "org/m", "weight_dtype": "bfloat16"}')
    for argv in (["chat", "m", "-d"], ["chat", "m", "--backend"], ["run", "m", "--backend", "-d"]):
        assert main(argv) == 0
        out = capsys.readouterr().out
        assert "Backends" in out and "SWLP_WINDOW_SIZE" in out and "SWLP_RESIDENCY" in out


def test_unknown_backend_lists_the_options(capsys):
    assert main(["chat", "m", "--backend", "turbo"]) == 2
    assert "mlx-moe" in capsys.readouterr().err


def test_models_details(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "shards" / "m"
    d.mkdir(parents=True)
    (d / "shard_manifest.json").write_text(json.dumps(
        {"model_id": "org/m", "weight_dtype": "bfloat16", "num_layers": 4,
         "layer_weight_mb": 10, "num_experts": 8, "top_k": 2}))
    (d / "expert_index.json").write_text("{}")
    monkeypatch.setattr("swlp.cli_models._HF_HUB", tmp_path / "no-hub")
    assert main(["models", "-d"]) == 0
    out = capsys.readouterr().out
    assert "8 per layer · top-2" in out and "swlp rm m" in out


def test_too_big_for_resident_mlx_is_refused_early(capsys, monkeypatch):
    from swlp import cli
    from swlp.cli_resolve import Target

    monkeypatch.setattr("mlx.core.device_info",
                        lambda: {"max_recommended_working_set_size": int(11.8 * 1024**3)})
    big = Target("mlx", "org/27b", "MLX int4 · resident", quant="int4", full_gb=48.0)
    args = build_parser().parse_args(["chat", "27b", "-q", "int4"])
    assert cli._fits_resident(big, args) is False
    assert "stream it instead" in capsys.readouterr().err
    small = Target("mlx", "org/7b", "MLX int4 · resident", quant="int4", full_gb=14.0)
    assert cli._fits_resident(small, args) is True


def test_errors_are_one_line_not_a_traceback(capsys, monkeypatch):
    def boom(_args):
        raise RuntimeError("disk on fire\nsecond line")

    monkeypatch.setattr("swlp.cli._run", boom)
    assert main(["run", "m", "hi"]) == 1
    err = capsys.readouterr().err
    assert "RuntimeError: disk on fire" in err and "second line" not in err and "-v" in err


def test_line_breaks_survive_markdown_but_code_is_untouched():
    from swlp.ui import _hard_breaks

    assert _hard_breaks("Red\nBlue") == "Red  \nBlue"
    assert _hard_breaks("a\n```\nx\ny\n```\nb") == "a  \n```\nx\ny\n```  \nb"


def _dense_shards(tmp_path, monkeypatch, layer_mb=761.0):
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "shards" / "m"
    d.mkdir(parents=True)
    (d / "shard_manifest.json").write_text(json.dumps({
        "model_id": "org/m", "num_layers": 64, "layer_weight_mb": layer_mb,
        "total_weight_mb": 64 * layer_mb, "embed_file": "embed.pt",
        "lm_head_file": "lm_head.pt", "model_type": "llama", "weight_dtype": "bfloat16"}))


def test_resident_and_window_reach_the_config(tmp_path, monkeypatch):
    from swlp.cli import _prepare

    _dense_shards(tmp_path, monkeypatch, layer_mb=1.0)
    config, target = _prepare(build_parser().parse_args(
        ["chat", "m", "--resident", "4", "--window", "3"]))
    assert target.backend == "swlp"
    assert config.runtime.swlp_residency == "4" and config.runtime.swlp_window_size == 3


def test_resident_that_cannot_fit_is_refused(tmp_path, monkeypatch, capsys):
    from swlp.cli import _prepare

    _dense_shards(tmp_path, monkeypatch)
    monkeypatch.setattr("psutil.virtual_memory",
                        lambda: type("V", (), {"available": 8 * 1024**3})())
    args = build_parser().parse_args(["chat", "m", "--resident", "30"])
    assert _prepare(args) is None
    err = capsys.readouterr().err
    assert "only 8 fit" in err and "--resident 8" in err


def test_bad_resident_value(tmp_path, monkeypatch, capsys):
    from swlp.cli import _prepare

    _dense_shards(tmp_path, monkeypatch)
    assert _prepare(build_parser().parse_args(["chat", "m", "--resident", "lots"])) is None
    assert "layer count, auto, or off" in capsys.readouterr().err
