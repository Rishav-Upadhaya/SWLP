"""Interactive chat for ``swlp chat`` (and the answer rendering ``swlp run`` reuses).

History is kept across turns; answers stream as rendered markdown with a status
line (tok/s, ms/token, and the expert-cache hit rate on MoE models).
Slash commands: /help /clear /stats /exit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import ui
from .config import AppConfig

# Below this the MoE expert cache is too small for good hit rates (Phase 31:
# Gemma 4 26B A4B ran 14.5 tok/s at 6.5 GB, 7.2 at 3 GB).
_LOW_CACHE_BYTES = 3 * 1024**3

_SLASH_HELP = [
    ("/help", "this list"),
    ("/clear", "forget the conversation so far"),
    ("/think", "toggle the model's visible reasoning (off by default)"),
    ("/stats", "memory and expert-cache numbers for this session"),
    ("/exit", "leave  (also /quit, Ctrl+C or Ctrl+D at the prompt)"),
]


@dataclass
class ChatSession:
    """Conversation history for one chat."""

    messages: list[dict[str, str]] = field(default_factory=list)
    thinking: bool = False  # reasoning models: show the thought channel

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def clear(self) -> None:
        self.messages.clear()


def format_chat_prompt(session: ChatSession, tokenizer: Any, user_input: str) -> str:
    """History + the new turn, through the model's chat template when it has
    one (instruct models need it), else a plain ``User:/Assistant:`` script."""
    history = session.messages + [{"role": "user", "content": user_input}]
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        try:
            # enable_thinking: honoured by reasoning templates (Qwen3.x,
            # Gemma 4), ignored by the rest.
            return tokenizer.apply_chat_template(history, tokenize=False,
                                                 add_generation_prompt=True,
                                                 enable_thinking=session.thinking)
        except Exception:
            pass  # malformed template → plain format below
    lines = [f"{m['role'].capitalize()}: {m['content']}" for m in history]
    return "\n".join([*lines, "Assistant:"])


def load_with_header(runner: Any, config: AppConfig, label: str) -> None:
    """Header panel (model · backend · machine), then load with a spinner."""
    ui.header("swlp", [
        ("model", config.model.model_id),
        ("backend", label),
        ("machine", _machine()),
    ])
    if config.runtime.backend == "mock":
        return
    with ui.spinner("loading model") as timing:
        runner.load()
    cache = getattr(runner, "cache", None)
    budget = getattr(cache, "budget_bytes", 0)
    extra = f" · expert cache {budget / 1024**3:.1f} GB" if budget else ""
    ui.note(f"  ready in {timing['elapsed']:.1f}s{extra}")
    if cache is not None and budget < _LOW_CACHE_BYTES:
        import psutil

        free = psutil.virtual_memory().available / 1024**3
        ui.warn(f"only {free:.1f} GB of RAM is free, so the expert cache is small and "
                "answers will be slower — close other apps for more speed.")
    ui.console.print()


def answer(runner: Any, session: ChatSession, user_input: str, max_tokens: int) -> str:
    """Stream one answer as markdown, print the status line, record the turn."""
    prompt = format_chat_prompt(session, getattr(runner, "tokenizer", None), user_input)
    cache = getattr(runner, "cache", None)
    before = cache.stats() if cache is not None else None
    stream = runner.stream_tokens(prompt, max_tokens=max_tokens)
    text, n_chunks, seconds = ui.stream_markdown(stream)
    ui.console.print()
    n_tokens = _count_tokens(runner, text, n_chunks)
    parts = [f"{n_tokens / seconds:.1f} tok/s" if seconds > 0 else "",
             f"{seconds * 1000 / n_tokens:.0f} ms/tok" if n_tokens else "",
             f"{n_tokens} tokens"]
    if cache is not None and before is not None:
        parts.append(_turn_hit_rate(before, cache.stats()))
    ui.status(parts)
    session.add_user(user_input)
    session.add_assistant(text)
    return text


def run_chat(config: AppConfig, label: str, max_tokens: int = 512) -> None:
    """The REPL: read → answer → repeat, until /exit or Ctrl+D."""
    from .runner.base import build_runner

    runner = build_runner(config)
    load_with_header(runner, config, label)
    if hasattr(runner, "set_prefix_cache") and config.runtime.backend == "swlp":
        from .core.prefix_cache import PrefixKVCache

        runner.set_prefix_cache(PrefixKVCache())  # lossless prefill reuse across turns
    ui.note("  /help for commands  ·  Ctrl+C stops an answer; at the prompt it exits\n")
    session = ChatSession()
    while True:
        try:
            line = ui.console.input("[accent]›[/] ").strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl+D / Ctrl+C at the prompt: leave
            ui.console.print()
            line = "/exit"
        if not line:
            continue
        if line.startswith("/"):
            if not _slash(line.lower(), session, runner):
                ui.note("  bye.")
                return
            continue
        ui.console.print()
        answer(runner, session, line, max_tokens)
        ui.console.print()


def _slash(cmd: str, session: ChatSession, runner: Any) -> bool:
    """Handle a slash command; False means leave the chat."""
    if cmd in ("/exit", "/quit", "/q"):
        return False
    if cmd in ("/clear", "/reset"):
        session.clear()
        ui.ok("conversation cleared")
    elif cmd == "/think":
        session.thinking = not session.thinking
        ui.ok(f"reasoning {'shown' if session.thinking else 'hidden'}")
    elif cmd in ("/stats", "/mem"):
        _print_stats(runner)
    elif cmd in ("/help", "/?"):
        ui.commands(_SLASH_HELP)
    else:
        ui.warn(f"unknown command {cmd} — /help lists them")
    return True


def _print_stats(runner: Any) -> None:
    rows: list[list[str]] = []
    try:
        import mlx.core as mx

        rows.append(["GPU memory (MLX)", f"{mx.get_active_memory() / 1024**3:.2f} GB"])
        rows.append(["GPU peak", f"{mx.get_peak_memory() / 1024**3:.2f} GB"])
    except ImportError:
        pass
    import psutil

    vm = psutil.virtual_memory()
    rows.append(["system RAM", f"{vm.used / 1024**3:.1f} / {vm.total / 1024**3:.0f} GB"])
    cache = getattr(runner, "cache", None)
    if cache is not None:
        s = cache.stats()
        rows.append(["expert hit rate",
                     f"{s['hit_rate']:.0%}  ({s['hits']} hits, {s['misses']} misses)"])
        rows.append(["experts cached",
                     f"{s['cached_experts']}  ·  {s['cached_bytes'] / 1024**3:.1f} GB"])
    degraded = getattr(runner, "degradations", None)
    if degraded:
        rows.append(["degradations", str(len(degraded))])
    ui.table(["", ""], rows)


def _count_tokens(runner: Any, text: str, n_chunks: int) -> int:
    """Real token count: a chunk can hold several tokens (speculative decoding
    streams every accepted draft of a sweep at once)."""
    tokenizer = getattr(runner, "tokenizer", None)
    if tokenizer is None or not text:
        return n_chunks
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return n_chunks


def _turn_hit_rate(before: dict, after: dict) -> str:
    hits = (after["hits"] + after["prefetch_hits"]) - (before["hits"] + before["prefetch_hits"])
    total = hits + after["misses"] - before["misses"]
    return f"expert hits {hits / total:.0%}" if total else ""


def _machine() -> str:
    from .hardware.detect import detect_hardware

    hw = detect_hardware()
    return f"{hw.chip_name} · {hw.memory_gb:.0f} GB · SSD {hw.ssd_bandwidth_gbps:.1f} GB/s"
