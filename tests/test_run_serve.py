"""Tests for `swlp serve` (OpenAI-compatible HTTP API)."""
import json
import threading
import urllib.error
import urllib.request

import pytest

from swlp.config import load_config
from swlp.serve import ModelHandle, make_handler

# ── swlp run ────────────────────────────────────────────────────────────────


# ── serve internals ─────────────────────────────────────────────────────────


@pytest.fixture()
def server_url():
    config = load_config(None)
    config.runtime.backend = "mock"
    config.model.model_id = "mock-test"
    handle = ModelHandle(config)
    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(handle))
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    yield url
    server.shutdown()
    server.server_close()


def _post(url: str, path: str, payload: dict):
    req = urllib.request.Request(
        url + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=30)


def test_health(server_url):
    with urllib.request.urlopen(server_url + "/health", timeout=10) as r:
        body = json.loads(r.read())
    assert body["status"] == "ok"


def test_chat_completions_non_stream(server_url):
    with _post(server_url, "/v1/chat/completions",
               {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}) as r:
        body = json.loads(r.read())
    assert r.status == 200
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert len(body["choices"][0]["message"]["content"]) > 0


def test_completions_non_stream(server_url):
    with _post(server_url, "/v1/completions", {"prompt": "hi", "max_tokens": 8}) as r:
        body = json.loads(r.read())
    assert body["object"] == "text_completion"
    assert body["choices"][0]["text"]


def test_chat_completions_stream_sse(server_url):
    with _post(server_url, "/v1/chat/completions",
               {"messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8, "stream": True}) as r:
        raw = r.read().decode()
    assert raw.startswith("data: ")
    assert "chat.completion.chunk" in raw
    assert raw.rstrip().endswith("[DONE]")
    # Every data line before [DONE] must parse as the OpenAI chunk shape.
    chunks = [json.loads(line[len("data: "):]) for line in raw.splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert all("delta" in c["choices"][0] for c in chunks)


def test_unknown_path_404(server_url):
    try:
        with _post(server_url, "/v1/nope", {}):
            pass
        raise AssertionError("expected 404")
    except urllib.error.HTTPError as e:
        assert e.code == 404


def test_invalid_json_400(server_url):
    req = urllib.request.Request(server_url + "/v1/completions", data=b"{not json",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10):
            pass
        raise AssertionError("expected 400")
    except urllib.error.HTTPError as e:
        assert e.code == 400


def test_serve_cli_smoke():
    """`swlp serve --help` must not crash and must mention the endpoints."""
    from swlp.cli_args import build_parser
    parser = build_parser()
    args = parser.parse_args(["serve", "mistral-7b", "--port", "9999"])
    assert args.command == "serve"
    assert args.model == "mistral-7b"
    assert args.port == 9999


def test_models_endpoint(server_url):
    """GET /v1/models lists the served model for OpenAI client discovery."""
    with urllib.request.urlopen(server_url + "/v1/models", timeout=10) as r:
        body = json.loads(r.read())
    assert r.status == 200
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "mock-test"
    assert body["data"][0]["object"] == "model"


def test_completions_stream_sse_uses_text_completion_shape(server_url):
    """Round-2 fix: /v1/completions SSE chunks must be text_completion
    objects with a `text` field — not chat.completion.chunk deltas."""
    with _post(server_url, "/v1/completions",
               {"prompt": "hi", "max_tokens": 8, "stream": True}) as r:
        raw = r.read().decode()
    assert r.status == 200
    chunks = [json.loads(line[len("data: "):]) for line in raw.splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    assert chunks, "no SSE chunks received"
    for chunk in chunks:
        assert chunk["object"] == "text_completion"
        assert "text" in chunk["choices"][0]
        assert "delta" not in chunk["choices"][0]
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert raw.rstrip().endswith("[DONE]")


def test_chat_completions_rejects_non_string_content(server_url):
    req = urllib.request.Request(
        server_url + "/v1/chat/completions",
        data=json.dumps({"messages": [{"role": "user", "content": 42}]}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(req, timeout=10)
    assert excinfo.value.code == 400


def test_completions_rejects_non_numeric_max_tokens(server_url):
    req = urllib.request.Request(
        server_url + "/v1/completions",
        data=json.dumps({"prompt": "hi", "max_tokens": "lots"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(req, timeout=10)
    assert excinfo.value.code == 400
