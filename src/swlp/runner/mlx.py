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

M5 optimisations wired through to mlx-lm kwargs:
- ``max_kv_size``: sliding KV window via RotatingKVCache (from ``kv_window``).
- ``kv_bits``:     native MLX KV-cache quantisation  (from ``kv_quant``).
- ``draft_model``: speculative decoding — 2-4× throughput on suitable prompts
                   (from ``mlx_draft_model``; ignored when kv_window is set).
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
        from mlx_lm import load

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

        Maps SWLP runtime settings to the kwargs accepted by
        ``mlx_lm.stream_generate`` / ``generate_step``:

        - ``kv_window > 0``  →  ``max_kv_size`` (RotatingKVCache sliding window)
        - ``kv_quant int4``  →  ``kv_bits=4``   (native MLX KV quantisation)

        Note: mlx-lm drops ``max_kv_size`` silently when ``draft_model`` is
        provided, so both can always be passed — speculative decoding takes
        precedence over the KV window.
        """
        kwargs: dict[str, Any] = {}
        kv_window = self.config.runtime.kv_window
        if kv_window > 0:
            kwargs["max_kv_size"] = kv_window
        if self.config.runtime.kv_quant == "int4":
            kwargs["kv_bits"] = 4
        return kwargs

    def stream_tokens(self, prompt: str, max_tokens: int = 512) -> Iterator[str]:
        """Yield text fragments from ``mlx_lm.stream_generate`` as they arrive."""
        from mlx_lm import stream_generate

        self._ensure_loaded()
        kwargs = self._gen_kwargs()
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
