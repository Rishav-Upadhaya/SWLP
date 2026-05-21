"""Tests for swlp.chat — ChatSession, format_chat_prompt, run_chat via mock."""
from __future__ import annotations

from swlp.chat import ChatSession, format_chat_prompt
from swlp.config import load_config
from swlp.runner.mock import MockRunner

# ── ChatSession ───────────────────────────────────────────────────────────────

def test_session_starts_empty():
    s = ChatSession()
    assert s.messages == []


def test_session_add_user_and_assistant():
    s = ChatSession()
    s.add_user("Hello")
    s.add_assistant("Hi there!")
    assert len(s.messages) == 2
    assert s.messages[0] == {"role": "user", "content": "Hello"}
    assert s.messages[1] == {"role": "assistant", "content": "Hi there!"}


def test_session_clear():
    s = ChatSession()
    s.add_user("X")
    s.add_assistant("Y")
    s.clear()
    assert s.messages == []


def test_session_multi_turn_order():
    s = ChatSession()
    s.add_user("Turn 1")
    s.add_assistant("Response 1")
    s.add_user("Turn 2")
    assert s.messages[-1]["role"] == "user"
    assert s.messages[-1]["content"] == "Turn 2"


# ── format_chat_prompt ────────────────────────────────────────────────────────

def test_format_prompt_no_history_no_tokenizer():
    s = ChatSession()
    prompt = format_chat_prompt(s, tokenizer=None, user_input="Hi there")
    assert "Hi there" in prompt
    assert "User:" in prompt or "user" in prompt.lower()


def test_format_prompt_includes_history():
    s = ChatSession()
    s.add_user("What is Python?")
    s.add_assistant("Python is a programming language.")
    prompt = format_chat_prompt(s, tokenizer=None, user_input="Tell me more")
    assert "Python is a programming language." in prompt
    assert "Tell me more" in prompt


def test_format_prompt_new_user_input_at_end():
    s = ChatSession()
    prompt = format_chat_prompt(s, tokenizer=None, user_input="New question")
    # The new user input must appear AFTER any history.
    assert prompt.index("New question") > 0


def test_format_prompt_does_not_modify_session():
    """format_chat_prompt must not mutate the session (pure function)."""
    s = ChatSession()
    s.add_user("Old")
    before = len(s.messages)
    format_chat_prompt(s, tokenizer=None, user_input="New")
    assert len(s.messages) == before


# ── MockRunner.stream_tokens ──────────────────────────────────────────────────

def test_mock_stream_yields_tokens():
    config = load_config(None)
    config.runtime.backend = "mock"
    runner = MockRunner(config)
    tokens = list(runner.stream_tokens("Hello", max_tokens=512))
    assert len(tokens) > 0
    full_text = "".join(tokens)
    assert len(full_text) > 0


def test_mock_stream_contains_prompt_echo():
    config = load_config(None)
    runner = MockRunner(config)
    tokens = list(runner.stream_tokens("my unique prompt xyz", max_tokens=512))
    full_text = "".join(tokens)
    assert "my unique prompt xyz" in full_text


def test_mock_stream_is_deterministic():
    config = load_config(None)
    runner = MockRunner(config)
    tokens_a = list(runner.stream_tokens("test prompt"))
    tokens_b = list(runner.stream_tokens("test prompt"))
    assert "".join(tokens_a) == "".join(tokens_b)


# ── CLI chat subcommand wiring ────────────────────────────────────────────────

def test_chat_subcommand_parses():
    from swlp.cli_args import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["chat", "--backend", "mock", "--model", "tiny-gpt2", "--max-chat-tokens", "256"]
    )
    assert args.command == "chat"
    assert args.backend == "mock"
    assert args.max_chat_tokens == 256


def test_chat_subcommand_default_max_tokens():
    from swlp.cli_args import build_parser

    parser = build_parser()
    args = parser.parse_args(["chat", "--backend", "mock"])
    assert args.max_chat_tokens == 512
