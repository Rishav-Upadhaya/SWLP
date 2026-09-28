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

import time
from dataclasses import dataclass, field

from . import tui
from .config import AppConfig

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

def _banner(model_id: str, backend: str, quant: str | None, extra: list[str]) -> None:
    """Session header: what is loaded, and how to drive it."""
    label = f"{backend}/{quant}" if quant and backend == "mlx" else backend
    lines = [
        tui.kv("model", tui.bold(model_id), 9),
        tui.kv("backend", label, 9),
    ]
    lines += [tui.kv(k, v, 9) for k, v in (e.split("=", 1) for e in extra if "=" in e)]
    lines.append("")
    lines.append(
        tui.dim("/quit  ") + tui.dim("exit") + tui.dim("     /clear  ")
        + tui.dim("reset") + tui.dim("     /help  ") + tui.dim("commands")
    )
    print()
    print(tui.box(lines, title="SWLP chat"))
    print()


def _user_prompt() -> str:
    """Print the 'you' label and return stripped user input."""
    try:
        raw = input(tui.bold(tui.blue("you  ")) + tui.dim("› "))
    except EOFError:
        return "/quit"
    return raw.strip()


def _print_assistant_label() -> None:
    print(tui.bold(tui.green("\nswlp ")) + tui.dim("›"))


def _print_status(tps: float, elapsed: float, n_tokens: int) -> None:
    bits = f"{tps:.1f} tok/s  ·  {n_tokens} tokens  ·  {elapsed:.1f}s"
    print("\n\n" + tui.dim(f"      {bits}"), flush=True)


def _print_help() -> None:
    """Slash-command reference plus a few ready-to-paste commands."""
    print()
    print(tui.rule("commands"))
    for cmd, desc in [
        ("/quit", "end the session"),
        ("/clear", "forget the conversation so far"),
        ("/stats", "memory and tuning for this session"),
        ("/help", "this list"),
    ]:
        print(tui.kv(cmd, tui.dim(desc), 10))
    print()
    print(tui.rule("fast, resident (fits in RAM)"))
    print(tui.dim("      swlp chat qwen-7b --quant int4"))
    print(tui.dim("      swlp chat mistral-7b --quant int8      # byte-identical to fp16"))
    print()
    print(tui.rule("lossless, streamed (bigger than RAM)"))
    print(tui.dim("      swlp pull qwen-14b                      # one-time"))
    print(tui.dim("      swlp chat --shard-dir ./shards/qwen-14b --window 2"))
    print()


def _print_stats(runner: object) -> None:
    """Live memory picture — the number that decides what you can run."""
    print()
    print(tui.rule("session"))
    try:
        import mlx.core as mx

        print(tui.kv("mlx active", f"{mx.get_active_memory() / 1024 ** 3:.2f} GB"))
        print(tui.kv("mlx peak", f"{mx.get_peak_memory() / 1024 ** 3:.2f} GB"))
        print(tui.kv("mlx cache", f"{mx.get_cache_memory() / 1024 ** 3:.2f} GB"))
    except Exception:
        pass
    try:
        import psutil

        vm = psutil.virtual_memory()
        print(tui.kv("system RAM", f"{vm.used / 1024 ** 3:.1f} / {vm.total / 1024 ** 3:.0f} GB"))
    except Exception:
        pass
    tuning = getattr(runner, "tuning", None)
    if tuning is not None:
        print(tui.kv("tuning", tuning.summary()))
        for n in tuning.notes:
            print(tui.kv("", tui.dim(n)))
    degraded = getattr(runner, "degradations", None)
    if degraded:
        print(tui.kv("degradations", tui.yellow(str(len(degraded)))))
    print()


# ── Main REPL ─────────────────────────────────────────────────────────────────

def run_chat(config: AppConfig, max_tokens: int = 512) -> None:
    """Start the interactive chat loop."""
    from .runner.base import build_runner

    runner = build_runner(config)
    backend = config.runtime.backend
    model_id = config.model.model_id.split("/")[-1]  # short name for display
    quant = config.runtime.mlx_quant if backend == "mlx" else None

    # NOTE: there is deliberately no "--quant ignored" warning here. `--quant`
    # already implies `--backend mlx` in _resolve_backend(), so the only way to
    # reach this point with a non-mlx backend is the *default* mlx_quant value
    # — warning about it fired on every mock/hf/swlp session and meant nothing.

    # When SWLP is used without --shard-dir the full model loads into RAM.  That
    # works for small models but OOMs for 7B+ on 16 GB.  Give a heads-up.
    if backend == "swlp" and not config.runtime.shard_dir:
        tui.warn("--backend swlp without --shard-dir loads the whole model into RAM.")
        tui.note("for 7B+ on 16 GB: swlp pull <model>, then --shard-dir ./shards/<model>")

    extra: list[str] = []
    if config.runtime.shard_dir:
        extra.append(f"shards={config.runtime.shard_dir}")
        extra.append(f"window={config.runtime.swlp_window_size}")
    if backend == "mlx" and config.runtime.mlx_kv_bits in (4, 8):
        extra.append(f"kv={config.runtime.mlx_kv_bits}-bit")
    if config.runtime.mlx_draft_model or config.runtime.swlp_draft_model:
        extra.append(f"draft={config.runtime.mlx_draft_model or config.runtime.swlp_draft_model}")

    _banner(model_id, backend, quant, extra)

    # Pre-load the model once so the first turn isn't slow and the tokenizer
    # is available for chat-template formatting before the REPL starts.
    # mock has no model to load; all real backends (mlx, hf, swlp) expose .load().
    if backend != "mock":
        with tui.Spinner(f"loading {model_id}") as sp:
            runner.load()  # type: ignore[union-attr]
        tui.note(f"ready in {sp.elapsed:.1f}s")
        tuning = getattr(runner, "tuning", None)
        if tuning is not None and tuning.wired_limit_mb:
            tui.note(tuning.summary().lower())
        print()

    # Phase 26: prefix KV reuse across turns — each turn's prefill then only
    # streams the suffix (lossless; identical prefixes ⇒ identical KV).
    if hasattr(runner, "set_prefix_cache") and backend == "swlp":
        from .core.prefix_cache import PrefixKVCache

        runner.set_prefix_cache(PrefixKVCache())  # type: ignore[union-attr]

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
                print(tui.dim("\n  bye.\n"))
                break
            if cmd in ("/clear", "/reset"):
                session.clear()
                tui.ok("history cleared")
                print()
                continue
            if cmd in ("/help", "/?"):
                _print_help()
                continue
            if cmd in ("/stats", "/mem"):
                _print_stats(runner)
                continue
            tui.warn(f"unknown command {user_input} — try /help")
            print()
            continue

        # ── build prompt with history ──────────────────────────────────────
        prompt = format_chat_prompt(session, tokenizer, user_input)

        # ── stream the response ────────────────────────────────────────────
        _print_assistant_label()
        response_parts: list[str] = []
        token_count = 0
        gen_start = time.perf_counter()

        wrapper = tui.StreamWrapper(indent="      ")
        try:
            for token_text in runner.stream_tokens(prompt, max_tokens=max_tokens):  # type: ignore[union-attr]
                piece = wrapper.feed(token_text)
                if piece:
                    print(piece, end="", flush=True)
                response_parts.append(token_text)
                token_count += 1
            tail = wrapper.finish()
            if tail:
                print(tail, end="", flush=True)
        except KeyboardInterrupt:
            # Ctrl+C stops the current generation but keeps the session alive.
            print(tui.yellow(" [interrupted]"), end="")

        elapsed = time.perf_counter() - gen_start
        tps = token_count / elapsed if elapsed > 0 else 0.0
        _print_status(tps, elapsed, token_count)

        # ── store turns in history ─────────────────────────────────────────
        session.add_user(user_input)
        session.add_assistant("".join(response_parts))
        print()
