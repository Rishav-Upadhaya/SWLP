from pathlib import Path

from swlp.config import load_config


def test_load_config_from_file(tmp_path, monkeypatch):
    config_file = tmp_path / "custom.toml"
    config_file.write_text(
        """
[model]
model_id = "custom/model"

[cache]
cache_dir = ".cache/custom"

[generation]
max_new_tokens = 7
prompt = "Hello from config"

[runtime]
backend = "mock"
""",
        encoding="utf-8",
    )

    monkeypatch.delenv("SWLP_MODEL_ID", raising=False)
    config = load_config(config_file)

    assert config.model.model_id == "custom/model"
    assert config.cache.cache_dir == Path(".cache/custom").resolve()
    assert config.generation.max_new_tokens == 7
    assert config.runtime.backend == "mock"


def test_env_overrides_config(tmp_path, monkeypatch):
    config_file = tmp_path / "custom.toml"
    config_file.write_text(
        """
[model]
model_id = "custom/model"
""",
        encoding="utf-8",
    )

    monkeypatch.setenv("SWLP_MODEL_ID", "override/model")
    monkeypatch.setenv("SWLP_CACHE_OFFLINE", "true")
    monkeypatch.setenv("SWLP_WINDOW_SIZE", "3")
    monkeypatch.setenv("SWLP_PREFETCH_DEPTH", "4")
    monkeypatch.setenv("SWLP_PREFETCH", "false")
    monkeypatch.setenv("SWLP_KV_BUDGET_MB", "256")
    monkeypatch.setenv("SWLP_KV_COMPRESSION", "true")
    monkeypatch.setenv("SWLP_KV_TIERING", "true")

    config = load_config(config_file)

    assert config.model.model_id == "override/model"
    assert config.cache.offline is True
    assert config.runtime.swlp_window_size == 3
    assert config.runtime.swlp_prefetch_depth == 4
    assert config.runtime.swlp_prefetch is False
    assert config.runtime.kv_memory_budget_mb == 256
    assert config.runtime.kv_compression is True
    assert config.runtime.kv_tiering is True
