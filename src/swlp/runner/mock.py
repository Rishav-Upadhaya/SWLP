from __future__ import annotations

import hashlib
import time
from collections.abc import Iterator

from ..config import AppConfig
from ..metrics import RunMetrics, RunResult


class MockRunner:
    backend = "mock"

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def stream_tokens(self, prompt: str, max_tokens: int = 512) -> Iterator[str]:
        """Yield words from the mock response one at a time with a tiny delay."""
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        completion = (
            f"[mock:{digest}] {prompt.strip()}\n"
            "This is an offline baseline response from the mock runner."
        )
        for word in completion.split():
            yield word + " "
            time.sleep(0.01)  # simulate token latency

    def run(self, prompt: str, profile: bool = False) -> RunResult:
        started = time.perf_counter()
        prompt_tokens = max(1, len(prompt.split()))
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        completion = (
            f"[mock:{digest}] {prompt.strip()}\n"
            f"This is an offline baseline response from the mock runner."
        )
        generate_seconds = time.perf_counter() - started
        per_token_latency = [generate_seconds] if profile else None
        metrics = RunMetrics(
            model_id="mock",
            backend=self.backend,
            device=self.config.runtime.device,
            load_seconds=0.0,
            input_tokens=prompt_tokens,
            output_tokens=max(1, len(completion.split())),
            preprocess_seconds=0.0 if profile else None,
            forward_seconds=0.0 if profile else None,
            generate_seconds=generate_seconds,
            total_seconds=generate_seconds,
            time_to_first_token_seconds=generate_seconds if profile else None,
            per_token_latency_seconds=per_token_latency,
            throughput_tokens_per_second=(
                max(1, len(completion.split())) / generate_seconds if generate_seconds > 0 else None
            ),
            generated_tokens=max(1, len(completion.split())),
        )
        return RunResult(prompt=prompt, completion=completion, metrics=metrics)
