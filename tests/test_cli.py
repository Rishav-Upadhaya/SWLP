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
    assert "INFERENCE" in captured.out
    assert "DOWNLOAD" in captured.out


def test_help_command_prints_all_sections(capsys):
    exit_code = main(["help"])
    captured = capsys.readouterr()
    assert exit_code == 0
    for section in ("INFERENCE", "CHAT", "DOWNLOAD", "BENCHMARK", "SIMULATION",
                    "MODEL ALIASES", "BACKENDS", "GLOBAL FLAGS"):
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
