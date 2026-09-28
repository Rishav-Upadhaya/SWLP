"""MlxRunner — native MLX inference for Apple Silicon (Phase 8).

The Phase 7 FP8 spike proved weight-streaming precision tricks cannot reach
interactive speed on M5: MPS has no fast quantized matmul, so every streamed
layer is dequantized to FP16 per token. MLX *does* have native quantized
matmul, so a quantized model runs resident at full GPU speed instead of
streaming layer-by-layer from disk.

``MlxRunner`` is interchangeable with the other runners via ``build_runner()``
— it returns a ``RunResult``. It is the interactive-speed counterpart to the
lossless SWLP streaming runners: SWLP-streaming stays the FP16 big-model
feasibility tool, ``MlxRunner`` is the fast tool.

Quality dial (``runtime.mlx_quant``): ``bf16`` (lossless) | ``int8``
(near-lossless, default) | ``int4`` (fast tier, mild quality cost).

Apple Silicon tuning is centralised in ``runner/mlx_tune.py`` and wired here:

- **wired-memory ceiling** — raised at load time so a model that macOS would
  otherwise push into swap stays resident. Biggest single win on 16 GB.
- **KV quantization** (``mlx_kv_bits``) — 4-bit KV is measured *faster* than
  fp16 on unified memory, because decode is bandwidth-bound.
- **prompt cache** — the shared prefix of a chat is prefilled once, not once
  per turn. Exact reuse, no quality cost.
- **speculative decoding** (``mlx_draft_model``) — 1.9-2.1x measured on
  M4/M5 with a same-family draft model.
- **prefill chunking** (``mlx_prefill_step_size``) — bounds TTFT memory.
"""
from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..metrics import RunMetrics, RunResult
from .mlx_tune import apply_memory_tuning, generation_kwargs

LOGGER = logging.getLogger(__name__)

# mlx_quant value -> quantization bit-width for mlx_lm.convert.
_QUANT_BITS = {"int4": 4, "int8": 8}
_VALID_QUANT = ("bf16", "int8", "int4")


class MlxRunner:
    backend = "mlx"

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.mlx_quant = str(config.runtime.mlx_quant).lower()
        if self.mlx_quant not in _VALID_QUANT:
            raise ValueError(
                f"mlx_quant must be one of {_VALID_QUANT}, got {self.mlx_quant!r}"
            )
        # Cached after first load; None until _ensure_loaded() is called.
        self._mlx_model = None
        self._mlx_draft_model = None
        self.tokenizer = None  # exposed so run_chat() can build the chat template
        # Prompt cache reused across turns in a chat session, plus the token
        # ids it currently holds so we only reuse it on an exact prefix match.
        self._prompt_cache = None
        self._prompt_cache_tokens: list[int] = []
        self.tuning = None  # TuningReport, set on first load
        self.degradations: list[str] = []

    def _resolve_model_path(self) -> str:
        """Return a path/repo for ``mlx_lm.load``.

        ``bf16`` uses the HF model id directly (mlx_lm loads the weights as
        bf16 MLX arrays). ``int4`` / ``int8`` produce a quantized MLX copy once
        under the cache dir via ``mlx_lm.convert``, then reuse it.
        """
        model_id = self.config.model.local_model_path or self.config.model.model_id
        if self.mlx_quant == "bf16":
            return str(model_id)

        bits = _QUANT_BITS[self.mlx_quant]
        slug = re.sub(r"[^A-Za-z0-9._-]", "_", str(model_id))
        mlx_path = Path(self.config.cache.cache_dir) / f"mlx-{self.mlx_quant}-{slug}"
        if not (mlx_path / "config.json").is_file():
            from mlx_lm import convert

            LOGGER.info(
                "mlx_converting",
                extra={"model": str(model_id), "bits": bits, "out": str(mlx_path)},
            )
            mlx_path.parent.mkdir(parents=True, exist_ok=True)
            convert(
                hf_path=str(model_id),
                mlx_path=str(mlx_path),
                quantize=True,
                q_bits=bits,
            )
        return str(mlx_path)

    def _ensure_loaded(self) -> None:
        """Load and cache the MLX model and tokenizer; no-op if already loaded."""
        if self._mlx_model is not None:
            return
        import psutil
        from mlx_lm import load

        # Raise the wired-memory ceiling BEFORE the weights land, so a model
        # that would otherwise be pushed into swap is admitted to RAM.
        self.tuning = apply_memory_tuning(
            self.config.runtime.mlx_wired_limit,
            psutil.virtual_memory().total / (1024 ** 3),
        )

        model_path = self._resolve_model_path()
        LOGGER.info("mlx_loading", extra={"model_path": model_path, "quant": self.mlx_quant})
        self._mlx_model, self.tokenizer = load(model_path)

        # Load the draft model for speculative decoding if configured.
        draft_id = self.config.runtime.mlx_draft_model
        if draft_id:
            LOGGER.info("mlx_loading_draft", extra={"draft_model": draft_id})
            try:
                draft_model, _ = load(draft_id)
                self._mlx_draft_model = draft_model
                LOGGER.info("mlx_draft_loaded", extra={"draft_model": draft_id})
            except Exception as exc:
                LOGGER.warning(
                    "mlx_draft_load_failed",
                    extra={"draft_model": draft_id, "error": str(exc)},
                )
                self._mlx_draft_model = None

    def load(self) -> None:
        """Pre-load the MLX model and tokenizer (used by run_chat for warm-start)."""
        self._ensure_loaded()

    def _gen_kwargs(self) -> dict[str, Any]:
        """Build mlx-lm generation kwargs from the active config.

        ``mlx_kv_bits`` is the explicit MLX control; the shared ``kv_quant
        = int4`` setting maps onto it too, so a config written for the
        streaming backend does the sane thing here.
        """
        rt = self.config.runtime
        kv_bits = int(rt.mlx_kv_bits)
        if kv_bits == 0 and rt.kv_quant == "int4":
            kv_bits = 4
        return generation_kwargs(
            kv_bits=kv_bits,
            kv_group_size=int(rt.mlx_kv_group_size),
            quantized_kv_start=int(rt.mlx_quantized_kv_start),
            max_kv_size=int(rt.kv_window),
            prefill_step_size=int(rt.mlx_prefill_step_size),
            num_draft_tokens=int(rt.mlx_num_draft_tokens),
            has_draft_model=self._mlx_draft_model is not None,
        )

    def _resolve_prompt_cache(self, prompt: str) -> Any:
        """Return a prompt cache to pass to mlx-lm, reusing it across turns.

        mlx-lm mutates the cache in place as it generates, so after turn N it
        holds the KV for "turn N prompt + turn N answer". Turn N+1 re-sends
        exactly that as its prefix, which is why an exact prefix match is both
        common and safe. On a mismatch we start fresh rather than guess.
        """
        if not self.config.runtime.mlx_prompt_cache:
            return None
        from mlx_lm.models.cache import make_prompt_cache

        ids = self.tokenizer.encode(prompt)
        cached = self._prompt_cache_tokens
        if self._prompt_cache is not None and ids[: len(cached)] == cached and cached:
            LOGGER.info("mlx_prompt_cache_hit", extra={"reused_tokens": len(cached)})
            self._prompt_cache_tokens = ids
            return self._prompt_cache

        self._prompt_cache = make_prompt_cache(self._mlx_model)
        self._prompt_cache_tokens = ids
        return self._prompt_cache

    def stream_tokens(self, prompt: str, max_tokens: int = 512) -> Iterator[str]:
        """Yield text fragments from ``mlx_lm.stream_generate`` as they arrive."""
        from mlx_lm import stream_generate

        self._ensure_loaded()
        kwargs = self._gen_kwargs()
        cache = self._resolve_prompt_cache(prompt)
        if cache is not None:
            kwargs["prompt_cache"] = cache
        for resp in stream_generate(
            self._mlx_model,
            self.tokenizer,
            prompt,
            max_tokens=max_tokens,
            draft_model=self._mlx_draft_model,
            **kwargs,
        ):
            if resp.text:
                yield resp.text

    def run(self, prompt: str, profile: bool = False) -> RunResult:
        import psutil
        from mlx_lm import stream_generate

        tracker = psutil.Process()

        # _ensure_loaded is idempotent: near-zero if already warm (e.g. run_chat reuse).
        # Conversion (one-time, for quantized tiers) happens inside _resolve_model_path().
        load_start = time.perf_counter()
        self._ensure_loaded()
        load_seconds = time.perf_counter() - load_start
        peak_rss = tracker.memory_info().rss

        max_new = self.config.generation.max_new_tokens
        pieces: list[str] = []
        per_token_latency: list[float] = []
        generated_tokens = 0
        gen_tps: float | None = None

        kwargs = self._gen_kwargs()
        cache = self._resolve_prompt_cache(prompt)
        if cache is not None:
            kwargs["prompt_cache"] = cache
        gen_start = time.perf_counter()
        last = gen_start
        for resp in stream_generate(
            self._mlx_model,
            self.tokenizer,
            prompt,
            max_tokens=max_new,
            draft_model=self._mlx_draft_model,
            **kwargs,
        ):
            now = time.perf_counter()
            per_token_latency.append(now - last)
            last = now
            pieces.append(resp.text)
            generated_tokens = resp.generation_tokens
            gen_tps = resp.generation_tps
        generate_seconds = time.perf_counter() - gen_start
        peak_rss = max(peak_rss, tracker.memory_info().rss)

        completion = "".join(pieces)
        prompt_tokens = len(self.tokenizer.encode(prompt))
        total_seconds = load_seconds + generate_seconds
        throughput = gen_tps if gen_tps else (
            generated_tokens / generate_seconds if generate_seconds > 0 else None
        )
        ttft = per_token_latency[0] if per_token_latency else None

        metrics = RunMetrics(
            model_id=self.config.model.model_id,
            backend=self.backend,
            device="mps",
            load_seconds=load_seconds,
            input_tokens=prompt_tokens,
            output_tokens=prompt_tokens + generated_tokens,
            generate_seconds=generate_seconds,
            total_seconds=total_seconds,
            time_to_first_token_seconds=ttft if profile else None,
            per_token_latency_seconds=per_token_latency if profile else None,
            throughput_tokens_per_second=throughput,
            generated_tokens=generated_tokens,
            ram_peak_bytes=peak_rss if profile else None,
        )
        return RunResult(prompt=prompt, completion=completion, metrics=metrics)
