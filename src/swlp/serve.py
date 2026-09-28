"""``swlp serve`` — OpenAI-compatible HTTP server (stdlib only).

Endpoints (OpenAI API shape):
    POST /v1/chat/completions   messages[] → completion (stream=true → SSE)
    POST /v1/completions        prompt     → completion (stream=true → SSE)
    GET  /v1/models             model listing (OpenAI client discovery)
    GET  /health                liveness probe

No new dependencies: http.server + json. Requests are serialized by a
generation lock (the streaming scheduler is single-sequence); concurrent
clients queue on the lock rather than thrash the sliding window.

N.B.: ThreadingHTTPServer with a generation lock — a real serving stack
(vLLM-style continuous batching) is out of scope until multi-user demand.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .chat import ChatSession, format_chat_prompt
from .config import AppConfig

LOGGER = logging.getLogger(__name__)

_MAX_TOKENS_DEFAULT = 256


class ModelHandle:
    """One loaded runner shared by all requests, serialized by a lock."""

    def __init__(self, config: AppConfig) -> None:
        from .runner.base import build_runner

        self.config = config
        self.runner = build_runner(config)
        if config.runtime.backend != "mock":
            self.runner.load()  # type: ignore[union-attr]
        self.lock = threading.Lock()

    def complete(self, prompt: str, max_tokens: int) -> Iterator[str]:
        # The streaming scheduler is single-sequence; serialize generation.
        with self.lock:
            yield from self.runner.stream_tokens(prompt, max_tokens=max_tokens)  # type: ignore[union-attr]


def _sse_chunk(chunk_id: str, model: str, text: str, finish: str | None,
               *, chat: bool = True) -> str:
    if chat:
        payload = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish}],
        }
    else:
        # /v1/completions streams text_completion chunks with a `text` field.
        payload = {
            "id": chunk_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "text": text, "finish_reason": finish}],
        }
    return f"data: {json.dumps(payload)}\n\n"


def _count_tokens(handle: ModelHandle, text: str) -> int:
    """Token count via the loaded tokenizer; word-count estimate fallback."""
    tokenizer = getattr(handle.runner, "tokenizer", None)
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            LOGGER.exception("serve_token_count_failed")
    return max(1, len(text.split()))


def _completion_response(body: str, model: str, prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "id": f"cmpl-{int(time.time() * 1000)}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "text": body, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _chat_response(body: str, model: str) -> dict:
    return {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": body},
            "finish_reason": "stop",
        }],
    }


def make_handler(handle: ModelHandle) -> type:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002 — stdlib signature
            LOGGER.debug("http_request", extra={"line": format % args})

        def _json(self, code: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802 — http.server API
            if self.path == "/health":
                self._json(200, {"status": "ok", "backend": handle.config.runtime.backend})
            elif self.path == "/v1/models":
                self._json(200, {
                    "object": "list",
                    "data": [{
                        "id": handle.config.model.model_id,
                        "object": "model",
                        "owned_by": "swlp",
                    }],
                })
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:  # noqa: N802 — http.server API
            try:
                length = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self._json(400, {"error": {"message": "invalid JSON body"}})
                return

            is_chat = self.path == "/v1/chat/completions"
            if not is_chat and self.path != "/v1/completions":
                self._json(404, {"error": {"message": f"unknown path {self.path}"}})
                return

            # max_tokens: 0 is a valid OpenAI value (generate nothing);
            # missing falls back; non-numeric is a 400; large values are capped.
            raw_max = req.get("max_tokens")
            try:
                max_tokens = _MAX_TOKENS_DEFAULT if raw_max is None else int(raw_max)
            except (ValueError, TypeError):
                self._json(400, {"error": {"message": "max_tokens must be an integer"}})
                return
            max_tokens = max(0, min(max_tokens, 8192))
            stream = bool(req.get("stream", False))
            model_name = str(req.get("model") or handle.config.model.model_id)

            try:
                if is_chat:
                    messages = req.get("messages") or []
                    if not isinstance(messages, list) or not messages:
                        raise ValueError("messages must be a non-empty list")
                    for m in messages:
                        if not isinstance(m, dict) or not isinstance(m.get("content"), str):
                            raise ValueError("each message needs string 'content'")
                    session = ChatSession()
                    session.messages = [m for m in messages if m.get("role") != "system"]
                    system = next(
                        (m["content"] for m in messages if m.get("role") == "system"), None
                    )
                    prompt = format_chat_prompt(
                        session, getattr(handle.runner, "tokenizer", None),
                        messages[-1]["content"],
                    )
                    if system:
                        prompt = f"{system}\n\n{prompt}"
                else:
                    prompt = str(req.get("prompt") or "")
            except (ValueError, TypeError, KeyError) as exc:
                self._json(400, {"error": {"message": f"invalid request: {exc}"}})
                return

            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                chunk_id = f"{'chatcmpl' if is_chat else 'cmpl'}-{int(time.time() * 1000)}"
                try:
                    for token_text in handle.complete(prompt, max_tokens):
                        chunk = _sse_chunk(chunk_id, model_name, token_text, None,
                                           chat=is_chat)
                        self.wfile.write(chunk.encode())
                        self.wfile.flush()
                    self.wfile.write(
                        _sse_chunk(chunk_id, model_name, "", "stop", chat=is_chat).encode()
                    )
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except BrokenPipeError:
                    pass  # client disconnected mid-stream
                except Exception:
                    LOGGER.exception("serve_stream_failed")
            else:
                try:
                    body = "".join(handle.complete(prompt, max_tokens))
                except Exception:
                    LOGGER.exception("serve_generate_failed")
                    self._json(500, {"error": {"message": "generation failed"}})
                    return
                if is_chat:
                    self._json(200, _chat_response(body, model_name))
                else:
                    self._json(200, _completion_response(
                        body, model_name,
                        _count_tokens(handle, prompt),
                        _count_tokens(handle, body),
                    ))

    return Handler


def serve(config: AppConfig, host: str, port: int) -> None:
    """Block serving requests until Ctrl-C."""
    handle = ModelHandle(config)
    server = ThreadingHTTPServer((host, port), make_handler(handle))
    model_name = config.model.model_id.split("/")[-1]
    print(f"  swlp serve  ·  {model_name}  ·  {config.runtime.backend}")
    print(f"  http://{host}:{port}/v1/chat/completions   (OpenAI-compatible)")
    print(f"  http://{host}:{port}/health")
    print("  Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped.")
    finally:
        server.server_close()
