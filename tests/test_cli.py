"""Tests for swlp.cli — the flag-based CLI and tool subcommands."""
from swlp.cli import main
from swlp.cli_args import build_parser, resolve_model


def test_run_via_backend_flag(capsys):
    exit_code = main(["--backend", "mock", "--prompt", "Hello swlp", "--json"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Hello swlp" in captured.out
    assert "mock" in captured.out


def test_run_via_env_backend(capsys, monkeypatch):
    monkeypatch.setenv("SWLP_BACKEND", "mock")
    exit_code = main(["--prompt", "Hello swlp"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Hello swlp" in captured.out


def test_friendly_summary_is_printed(capsys, monkeypatch):
    monkeypatch.setenv("SWLP_BACKEND", "mock")
    main(["--prompt", "Hi", "--profile"])
    captured = capsys.readouterr()
    assert "Completion:" in captured.out
    assert "backend=mock" in captured.out
    assert "tok/s" in captured.out


def test_bare_invocation_prints_help(capsys):
    exit_code = main([])
    captured = capsys.readouterr()
    assert exit_code == 0
    # Custom help — not argparse's "usage: swlp" header.
    assert "SWLP" in captured.out
    assert "TWO WAYS TO RUN" in captured.out      # resident MLX vs FP16 streaming
    assert "DOWNLOAD" in captured.out


def test_help_command_prints_all_sections(capsys):
    exit_code = main(["help"])
    captured = capsys.readouterr()
    assert exit_code == 0
    for section in ("QUICK START", "DOWNLOAD", "CHAT", "FLAGS", "BENCHMARKING"):
        assert section in captured.out, f"Missing section: {section}"
    # Key commands must appear.
    assert "swlp download --model" in captured.out
    assert "swlp chat" in captured.out
    assert "--shard-dir" in captured.out


def test_model_alias_resolution():
    assert resolve_model("mistral-7b") == "unsloth/mistral-7b-instruct-v0.2"
    assert resolve_model("qwen-14b") == "Qwen/Qwen2.5-14B-Instruct"
    # An unknown name (a real HF id) is passed through unchanged.
    assert resolve_model("org/some-model") == "org/some-model"


def test_quant_flag_implies_mlx_backend():
    parser = build_parser()
    args = parser.parse_args(["--model", "mistral-7b", "--quant", "int8"])
    assert args.quant == "int8"
    assert args.backend is None  # backend is inferred later, not parsed


def test_window_alias_accepted():
    parser = build_parser()
    args = parser.parse_args(["--swlp-window-size", "4"])
    assert args.window == 4
    args = parser.parse_args(["--window", "6"])
    assert args.window == 6


def test_benchmark_creates_metrics_file(tmp_path, monkeypatch):
    monkeypatch.setenv("SWLP_BACKEND", "mock")
    output_path = tmp_path / "benchmark.json"
    exit_code = main(
        ["benchmark", "--prompt", "Hello swlp", "--output", str(output_path), "--format", "json"]
    )
    assert exit_code == 0
    payload = output_path.read_text(encoding="utf-8")
    assert "runs" in payload
    assert "time_to_first_token_seconds" in payload


# ── swlp pull disk preflight (Phase 24) ──────────────────────────────────────

def test_preflight_passes_for_unknown_model(tmp_path):
    from swlp.cli import _preflight_disk_space
    # Arbitrary HF ids have no known size — preflight stays silent.
    assert _preflight_disk_space("some-org/some-model", tmp_path / "out") is True


def test_preflight_blocks_when_disk_too_small(tmp_path, monkeypatch):
    import shutil as _shutil

    from swlp.cli import _preflight_disk_space

    free = _shutil.disk_usage  # keep a handle for the real call

    def _tiny(path):
        usage = free(path)
        return _shutil._ntuple_diskusage(usage.total, usage.used, 0)  # zero free

    monkeypatch.setattr("shutil.disk_usage", _tiny)
    # mistral-7b is a known 14 GB alias → must refuse with zero free bytes.
    assert _preflight_disk_space("mistral-7b", tmp_path / "out") is False


def test_preflight_passes_when_disk_sufficient(tmp_path):
    # tmp filesystems in CI generally have > 14 GB × 1.15 free; if the runner
    # is genuinely that constrained the unknown-model branch above still holds.
    import shutil

    from swlp.cli import _preflight_disk_space
    free_gb = shutil.disk_usage(tmp_path).free / 1024**3
    expected = free_gb >= 14 * 1.15
    assert _preflight_disk_space("mistral-7b", tmp_path / "out") is expected


def test_pull_subcommand_parses_like_download():
    from swlp.cli_args import build_parser

    parser = build_parser()
    args = parser.parse_args(["pull", "--model", "mistral-7b"])
    assert args.command == "pull"
    assert args.model == "mistral-7b"
    assert args.output_dir is None
