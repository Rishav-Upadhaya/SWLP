from pathlib import Path

from swlp.benchmark.suite import _prepare_prompt, load_suite


class DummyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        return [len(part) for part in text.split()]

    def decode(self, ids, skip_special_tokens: bool = True):
        return " ".join("x" * value for value in ids)


def test_prepare_prompt_pads_to_target():
    tokenizer = DummyTokenizer()
    prompt = "hello world"
    prepared = _prepare_prompt(tokenizer, prompt, 5)
    assert len(tokenizer.encode(prepared)) == 5


def test_load_suite(tmp_path: Path):
    config = tmp_path / "suite.toml"
    config.write_text(
        """
[suite]
name = "test-suite"
prompts = ["hello"]
context_lengths = [32]
runs_per_case = 1
window_sizes = [1]
prefetch_depths = [1]
prefetch_enabled = [true]
double_buffer_enabled = [false]
kv_memory_budget_mb = [256]
kv_compression = [false]
kv_tiering = [false]
""",
        encoding="utf-8",
    )
    suite = load_suite(config)
    assert suite.name == "test-suite"
    assert suite.context_lengths == [32]
