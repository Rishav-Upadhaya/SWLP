from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator

import torch
import torch.nn.functional as F

from ..config import AppConfig
from ..metrics import RunMetrics, RunResult

LOGGER = logging.getLogger(__name__)


class HuggingFaceRunner:
    backend = "hf"

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = self._resolve_device()
        self.dtype = self._resolve_dtype()

    def _resolve_device(self) -> torch.device:
        configured = self.config.runtime.device.lower()
        if configured == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(configured)

    def _resolve_dtype(self) -> torch.dtype:
        configured = self.config.runtime.dtype.lower()
        if configured == "auto":
            if self.device.type == "cuda":
                return torch.float16
            return torch.float32
        return getattr(torch, configured)

    def load(self) -> float:
        # Idempotent: if the model is already in memory (e.g. run_chat pre-loaded it)
        # skip the from_pretrained call so stream_tokens / run() don't double-load.
        if self.model is not None:
            return 0.0
        started = time.perf_counter()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_source = self.config.model.local_model_path or self.config.model.model_id
        LOGGER.info(
            "loading_model",
            extra={
                "model_source": str(model_source),
                "cache_dir": str(self.config.cache.cache_dir),
                "device": self.device.type,
                "dtype": str(self.dtype),
            },
        )
        attempts = int(os.getenv("SWLP_RETRIES", "2"))
        last_exc: Exception | None = None
        for attempt in range(attempts + 1):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_source,
                    cache_dir=str(self.config.cache.cache_dir),
                    trust_remote_code=self.config.model.trust_remote_code,
                    revision=self.config.model.revision or None,
                )
                if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                model = AutoModelForCausalLM.from_pretrained(
                    model_source,
                    cache_dir=str(self.config.cache.cache_dir),
                    trust_remote_code=self.config.model.trust_remote_code,
                    revision=self.config.model.revision or None,
                    torch_dtype=self.dtype,
                )
                model.eval()
                if self.device.type != "cpu":
                    model.to(self.device)
                self.model = model
                return time.perf_counter() - started
            except Exception as exc:
                last_exc = exc
                LOGGER.warning(
                    "model_load_attempt_failed",
                    extra={"attempt": attempt, "error": str(exc)},
                )
                import time as _t
                _t.sleep(0.5 * (attempt + 1))
                continue
        if last_exc:
            raise last_exc
        return time.perf_counter() - started

    def _select_next_token(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.config.generation.do_sample or self.config.generation.temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        temperature = max(self.config.generation.temperature, 1e-5)
        scores = logits / temperature
        if self.config.generation.top_p < 1.0:
            sorted_scores, sorted_indices = torch.sort(scores, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_scores, dim=-1), dim=-1)
            cutoff = cumulative_probs > self.config.generation.top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False
            sorted_scores = sorted_scores.masked_fill(cutoff, float("-inf"))
            scores = torch.zeros_like(scores).scatter(-1, sorted_indices, sorted_scores)
        probs = F.softmax(scores, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        if penalty == 1.0:
            return logits
        for token_id in generated[0].tolist():
            token_id = int(token_id)
            if logits[0, token_id] < 0:
                logits[0, token_id] *= penalty
            else:
                logits[0, token_id] /= penalty
        return logits

    def _sync_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()

    def _generate_with_timings(
        self,
        encoded: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, list[float], float]:
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        max_new_tokens = self.config.generation.max_new_tokens
        per_token_latency: list[float] = []
        forward_seconds = 0.0
        generated = input_ids
        past_key_values = None
        eos_token_id = self.tokenizer.eos_token_id if self.tokenizer else None

        with torch.no_grad():
            for _ in range(max_new_tokens):
                self._sync_device()
                step_start = time.perf_counter()
                if past_key_values is None:
                    step_input_ids = generated
                    step_attention = attention_mask
                else:
                    step_input_ids = generated[:, -1:]
                    step_attention = None
                outputs = self.model(
                    input_ids=step_input_ids,
                    attention_mask=step_attention,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                self._sync_device()
                forward_seconds += time.perf_counter() - step_start
                logits = outputs.logits[:, -1, :]
                logits = self._apply_repetition_penalty(
                    logits, generated, self.config.generation.repetition_penalty
                )
                next_token = self._select_next_token(logits)
                step_end = time.perf_counter()
                per_token_latency.append(step_end - step_start)
                generated = torch.cat([generated, next_token], dim=-1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_token, device=attention_mask.device)],
                    dim=-1,
                )
                past_key_values = outputs.past_key_values
                if eos_token_id is not None and int(next_token.item()) == eos_token_id:
                    break

        return generated, per_token_latency, forward_seconds

    def stream_tokens(self, prompt: str, max_tokens: int = 512) -> Iterator[str]:
        """Yield decoded token strings one at a time as they are generated.

        Uses the same manual decode loop as ``_generate_with_timings`` so the
        sliding KV cache is maintained correctly — no extra threading needed.
        """
        self.load()
        assert self.model is not None
        assert self.tokenizer is not None

        torch.manual_seed(self.config.generation.seed)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        generated = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask", torch.ones_like(generated))
        past_key_values = None
        eos_id = self.tokenizer.eos_token_id

        with torch.no_grad():
            for _ in range(max_tokens):
                step_input = generated[:, -1:] if past_key_values is not None else generated
                step_mask = attention_mask if past_key_values is None else None
                outputs = self.model(
                    input_ids=step_input,
                    attention_mask=step_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                next_token = self._select_next_token(outputs.logits[:, -1, :])
                token_id = int(next_token.item())
                token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                if token_text:
                    yield token_text
                generated = torch.cat([generated, next_token], dim=-1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones(1, 1, device=attention_mask.device)], dim=-1
                )
                past_key_values = outputs.past_key_values
                if eos_id is not None and token_id == int(eos_id):
                    break

    def run(self, prompt: str, profile: bool = False) -> RunResult:
        import psutil

        torch.manual_seed(self.config.generation.seed)

        memory_tracker = psutil.Process()
        load_seconds = self.load()
        peak_rss_bytes = memory_tracker.memory_info().rss

        assert self.model is not None
        assert self.tokenizer is not None

        preprocess_start = time.perf_counter()
        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        preprocess_seconds = time.perf_counter() - preprocess_start
        prompt_tokens = int(encoded["input_ids"].shape[-1])

        generation_start = time.perf_counter()
        generated, per_token_latency, forward_seconds = self._generate_with_timings(encoded)
        generate_seconds = time.perf_counter() - generation_start
        peak_rss_bytes = max(peak_rss_bytes, memory_tracker.memory_info().rss)

        total_seconds = load_seconds + preprocess_seconds + generate_seconds
        output_tokens = int(generated.shape[-1])
        generated_tokens = max(output_tokens - prompt_tokens, 0)

        completion_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        completion = (
            completion_text[len(prompt):].lstrip()
            if completion_text.startswith(prompt)
            else completion_text
        )

        ttft = per_token_latency[0] if per_token_latency else None
        vram_peak_bytes = (
            int(torch.cuda.max_memory_allocated(self.device))
            if profile and self.device.type == "cuda" and torch.cuda.is_available()
            else None
        )

        metrics = RunMetrics(
            model_id=self.config.model.model_id,
            backend=self.backend,
            device=self.device.type,
            load_seconds=load_seconds,
            input_tokens=prompt_tokens,
            output_tokens=output_tokens,
            preprocess_seconds=preprocess_seconds if profile else None,
            forward_seconds=forward_seconds if profile else None,
            generate_seconds=generate_seconds,
            total_seconds=total_seconds,
            time_to_first_token_seconds=ttft if profile else None,
            per_token_latency_seconds=per_token_latency if profile else None,
            throughput_tokens_per_second=(
                generated_tokens / generate_seconds if generate_seconds > 0 else None
            ),
            generated_tokens=generated_tokens,
            vram_peak_bytes=vram_peak_bytes,
            ram_peak_bytes=peak_rss_bytes if profile else None,
        )
        return RunResult(prompt=prompt, completion=completion, metrics=metrics)
