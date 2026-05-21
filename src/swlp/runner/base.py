"""Runner factory and top-level execute_baseline entry point."""
from __future__ import annotations

import logging
import sys

from ..config import AppConfig
from ..metrics import RunResult
from .hf import HuggingFaceRunner
from .mlx import MlxRunner
from .mock import MockRunner
from .speculative import SpeculativeRunner
from .swlp import SWLPRunner

LOGGER = logging.getLogger(__name__)

_TINY_MODELS = {"sshleifer/tiny-gpt2", "mock"}


def check_hf_oom(config: AppConfig) -> None:
    """Abort early with a helpful message when ``backend=hf`` would OOM.

    Estimates FP16 model size from the HF config.  If the model won't fit
    available RAM, prints actionable alternatives and raises ``SystemExit(1)``
    so the user sees a clean message rather than a kernel kill.
    """
    if config.runtime.backend != "hf":
        return
    model_id = config.model.model_id
    if not model_id or model_id in _TINY_MODELS:
        return
    try:
        from transformers import AutoConfig as _AC

        from ..cli_args import MODEL_ALIASES
        from ..hardware.detect import detect_hardware, fits_in_memory

        cfg = _AC.from_pretrained(model_id, cache_dir=config.cache.cache_dir or None)
        num_layers = int(getattr(cfg, "num_hidden_layers", 0) or getattr(cfg, "n_layer", 0))
        hidden = int(getattr(cfg, "hidden_size", 0) or getattr(cfg, "n_embd", 0))
        intermediate = int(getattr(cfg, "intermediate_size", hidden * 4))
        params_per_layer = 4 * hidden * hidden + 3 * hidden * intermediate
        total_bytes = num_layers * params_per_layer * 2  # FP16 = 2 bytes/param
        if total_bytes <= 0:
            return
        hw = detect_hardware()
        if fits_in_memory(total_bytes, hw):
            return
        size_gb = total_bytes / 1e9
        short_alias = next((k for k, v in MODEL_ALIASES.items() if v == model_id), None)
        shard_dir_name = short_alias or model_id.split("/")[-1]
        model_ref = short_alias or model_id
        print(
            f"\nError: model '{model_id}' is ~{size_gb:.1f} GB FP16 "
            f"but only {hw.memory_gb:.0f} GB RAM is available — "
            "full load would OOM.\n\n"
            "Suggested alternatives:\n"
            f"  swlp --model {model_ref} --backend mlx --quant int8 --prompt \"...\"\n"
            "      (MLX int8 lossless on Apple Silicon — ~16 tok/s)\n"
            f"  swlp --shard-dir ./shards/{shard_dir_name} --window 2 --prompt \"...\"\n"
            "      (SWLP streaming FP16 — run a model larger than RAM)\n"
            f"      Shard first if needed:  swlp package <checkpoint> ./shards/{shard_dir_name}\n",
            file=sys.stderr,
        )
        raise SystemExit(1)
    except SystemExit:
        raise
    except Exception:
        pass  # detection failure → let the runner handle it naturally


def build_runner(config: AppConfig) -> MockRunner | HuggingFaceRunner | MlxRunner:
    backend = config.runtime.backend.lower()
    if backend == "mock":
        return MockRunner(config)
    if backend == "mlx":
        return MlxRunner(config)
    if backend == "speculative":
        return SpeculativeRunner(config)
    if backend == "swlp":
        return SWLPRunner(config)
    try:
        return HuggingFaceRunner(config)
    except Exception as exc:
        LOGGER.exception("runner_initialization_failed", extra={"backend": backend})
        if config.runtime.allow_mock_fallback:
            LOGGER.warning("falling_back_to_mock_runner", extra={"reason": str(exc)})
            return MockRunner(config)
        raise


def execute_baseline(config: AppConfig, prompt: str | None = None) -> RunResult:
    runner = build_runner(config)
    runtime_prompt = prompt or config.generation.prompt

    if isinstance(runner, HuggingFaceRunner):
        try:
            return runner.run(runtime_prompt, profile=config.runtime.profile)
        except Exception as exc:
            LOGGER.exception("baseline_run_failed", extra={"backend": runner.backend})
            if isinstance(runner, SWLPRunner) and config.runtime.swlp_fallback_to_baseline:
                LOGGER.warning("swlp_fallback_to_baseline", extra={"reason": str(exc)})
                return HuggingFaceRunner(config).run(runtime_prompt, profile=config.runtime.profile)
            if config.runtime.allow_mock_fallback:
                LOGGER.warning("falling_back_to_mock_runner", extra={"reason": str(exc)})
                return MockRunner(config).run(runtime_prompt, profile=config.runtime.profile)
            raise

    return runner.run(runtime_prompt, profile=config.runtime.profile)
