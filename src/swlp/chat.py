"""Interactive chat REPL for SWLP (Phase 10).

Usage::

    swlp chat --model mistral-7b --backend mlx --quant int8

Tokens stream to the terminal as they are generated.  Conversation history
is maintained across turns so the model sees the full context.

Slash commands (at the ``You:`` prompt):
  /quit   — exit the session
  /clear  — reset history (start a fresh conversation)
  /help   — print this help
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

from .config import AppConfig

# ── ANSI colour helpers ───────────────────────────────────────────────────────

_COLOUR_SUPPORT = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _COLOUR_SUPPORT:
        return text
    return f"\033[{code}m{text}\033[0m"


def _bold(t: str) -> str:
    return _c("1", t)


def _dim(t: str) -> str:
    return _c("2", t)


def _green(t: str) -> str:
    return _c("32", t)


def _blue(t: str) -> str:
    return _c("34", t)


def _cyan(t: str) -> str:
    return _c("36", t)


def _yellow(t: str) -> str:
    return _c("33", t)


# ── Chat session ──────────────────────────────────────────────────────────────

@dataclass
class ChatSession:
    """Mutable conversation history for a single chat session."""

    messages: list[dict[str, str]] = field(default_factory=list)

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def clear(self) -> None:
        self.messages.clear()


def format_chat_prompt(session: ChatSession, tokenizer, user_input: str) -> str:
    """Build the full prompt string for the next generation step.

    Adds the new user turn and applies the model's chat template when the
    tokenizer supports it.  Falls back to a simple ``User:/Assistant:`` format
    for models without a template (e.g. tiny-gpt2).
    """
    # Temporarily append the new user message to build the prompt.
    history = session.messages + [{"role": "user", "content": user_input}]

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass  # template failed — fall through to plain format

    # Plain fallback format (works for any model, just less instruction-tuned).
    parts: list[str] = []
    for msg in history:
        role = msg["role"].capitalize()
        parts.append(f"{role}: {msg['content']}")
    parts.append("Assistant:")
    return "\n".join(parts)


# ── Terminal helpers ──────────────────────────────────────────────────────────

def _banner(model_id: str, backend: str, quant: str | None) -> None:
    quant_str = f"/{quant}" if quant and backend == "mlx" else ""
    title = f" SWLP Chat  ·  {model_id}  ·  {backend}{quant_str} "
    hint = " /quit  exit   /clear  reset   /help  commands "
    width = max(len(title), len(hint)) + 2
    bar = "─" * width
    print(_dim(f"╭{bar}╮"))
    print(_dim("│") + _bold(_cyan(title.center(width))) + _dim("│"))
    print(_dim("│") + _dim(hint.center(width)) + _dim("│"))
    print(_dim(f"╰{bar}╯"))
    print()


def _user_prompt() -> str:
    """Print the 'You' label and return stripped user input."""
    try:
        raw = input(_bold(_blue(" You  ")))
    except EOFError:
        return "/quit"
    return raw.strip()


def _print_assistant_label() -> None:
    print(_bold(_green("\n Assistant  ")), end="", flush=True)


def _print_status(tps: float, elapsed: float, n_tokens: int) -> None:
    print(
        _dim(f"\n\n  {tps:.1f} tok/s  ·  {n_tokens} tokens  ·  {elapsed:.1f}s"),
        flush=True,
    )


# ── Main REPL ─────────────────────────────────────────────────────────────────

def run_chat(config: AppConfig, max_tokens: int = 512) -> None:
    """Start the interactive chat loop."""
    from .runner.base import build_runner

    runner = build_runner(config)
    backend = config.runtime.backend
    model_id = config.model.model_id.split("/")[-1]  # short name for display
    quant = config.runtime.mlx_quant if backend == "mlx" else None

    # --quant is an MLX-only flag; warn early so users know it has no effect here.
    if backend != "mlx" and config.runtime.mlx_quant:
        print(
            _yellow(
                f"  Warning: --quant {config.runtime.mlx_quant!r} is only supported "
                f"with --backend mlx and is ignored for backend={backend!r}.\n"
                f"  Use: swlp chat --backend mlx --quant {config.runtime.mlx_quant}\n"
            )
        )

    # When SWLP is used without --shard-dir the full model loads into RAM.  That
    # works for small models but OOMs for 7B+ on 16 GB.  Give a heads-up.
    if backend == "swlp" and not config.runtime.shard_dir:
        print(
            _yellow(
                "  Note: --backend swlp without --shard-dir loads the full model into RAM.\n"
                "  For 7B+ models on 16 GB, run 'swlp download --model <name>' first,\n"
                "  then: swlp chat --shard-dir ./shards/<name> --window 2\n"
            )
        )

    _banner(model_id, backend, quant)

    # Pre-load the model once so the first turn isn't slow and the tokenizer
    # is available for chat-template formatting before the REPL starts.
    # mock has no model to load; all real backends (mlx, hf, swlp) expose .load().
    if backend != "mock":
        print(_dim("  Loading model…"), end="\r", flush=True)
        runner.load()  # type: ignore[union-attr]
        print(" " * 30, end="\r", flush=True)  # clear the loading line

    # Grab tokenizer for chat template (all real runners expose .tokenizer after load()).
    tokenizer = getattr(runner, "tokenizer", None)

    session = ChatSession()

    while True:
        # ── read user input ────────────────────────────────────────────────
        try:
            user_input = _user_prompt()
        except KeyboardInterrupt:
            print()
            continue

        if not user_input:
            continue

        # ── slash commands ─────────────────────────────────────────────────
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd in ("/quit", "/exit", "/q"):
                print(_dim("\n  Goodbye.\n"))
                break
            if cmd in ("/clear", "/reset"):
                session.clear()
                print(_dim("  History cleared.\n"))
                continue
            if cmd in ("/help", "/?"):
                print(
                    _dim(
                        "\n  Commands:\n"
                        "    /quit    — exit\n"
                        "    /clear   — reset conversation history\n"
                        "    /help    — show this message\n"
                        "\n"
                        "  Start a new chat with a different model:\n"
                        "    swlp chat --model smollm-1.7b  --backend mlx --quant int4\n"
                        "    swlp chat --model qwen-7b       --backend mlx --quant int4\n"
                        "    swlp chat --model phi-3.5       --backend mlx --quant int4\n"
                        "    swlp chat --model mistral-7b    --backend mlx --quant int4\n"
                        "    swlp chat --model mistral-24b   --backend mlx --quant int4\n"
                        "\n"
                        "  Stream a model larger than RAM (FP16, lossless):\n"
                        "    swlp download --model mistral-7b    # one-time shard to ./shards/\n"
                        "    swlp download --model mistral-24b\n"
                        "    swlp download --model qwen-7b\n"
                        "    swlp download --model qwen-14b\n"
                        "    swlp chat --shard-dir ./shards/<name> --window 2\n"
                        "\n"
                        "  Any HuggingFace model id also works:\n"
                        "    swlp chat --model google/gemma-2-2b-it --backend mlx --quant int4\n"
                    )
                )
                continue
            print(_yellow(f"  Unknown command: {user_input}  (try /help)\n"))
            continue

        # ── build prompt with history ──────────────────────────────────────
        prompt = format_chat_prompt(session, tokenizer, user_input)

        # ── stream the response ────────────────────────────────────────────
        _print_assistant_label()
        response_parts: list[str] = []
        token_count = 0
        gen_start = time.perf_counter()

        try:
            for token_text in runner.stream_tokens(prompt, max_tokens=max_tokens):  # type: ignore[union-attr]
                print(token_text, end="", flush=True)
                response_parts.append(token_text)
                token_count += 1
        except KeyboardInterrupt:
            # Ctrl+C stops the current generation but keeps the session alive.
            print(_yellow("  [interrupted]"), end="")

        elapsed = time.perf_counter() - gen_start
        tps = token_count / elapsed if elapsed > 0 else 0.0
        _print_status(tps, elapsed, token_count)

        # ── store turns in history ─────────────────────────────────────────
        session.add_user(user_input)
        session.add_assistant("".join(response_parts))
        print()
