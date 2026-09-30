"""MlxMoeRunner — MoE expert streaming on native MLX (Phase 30).

Builds the stock ``mlx_lm`` model for the checkpoint's ``model_type``
(``qwen3_moe``, ``qwen3_5_moe``, ``olmoe``, …), keeps every non-expert weight
resident (attention / DeltaNet / router / shared expert / norms / lm_head —
a few GB for A3B models), and swaps each MoE block's ``switch_mlp`` for a
:class:`CachedSwitchGLU` over one global byte-budgeted expert LRU fed from the
shard dir's expert banks (``model/expert_bank.py``). The result is an ordinary
``mlx_lm`` model, so generation is ``mlx_lm.stream_generate`` unchanged.

Lossless: expert weights are read at the checkpoint's dtype, and routing and
outputs are the model's own; the cache and prefetch only decide *when* bytes
are read.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psutil

from ..config import AppConfig
from ..metrics import RunMetrics, RunResult
from ..model.expert_bank import EXPERT_INDEX_FILE, ExpertIndex
from ..model.shard import load_manifest
from .mlx_moe_load import (
    dense_weights,
    swap_in_cached_experts,
    swap_in_mmap_embedding,
)

LOGGER = logging.getLogger(__name__)

# Parallel expert reads in flight (slotstream, M5 Pro: whole-expert preads
# scale to queue depth ~8, flat beyond).
_EXPERT_READ_WORKERS = 8
# RAM left for macOS + activations when the expert budget is auto-sized.
_AUTO_BUDGET_RESERVE_BYTES = 2 * 1024**3
# Auto-sized budgets never drop below this (explicit budgets are honoured).
_MIN_AUTO_BUDGET_BYTES = 256 * 1024**2
# Quantize-on-load (``swlp_moe_quant``): mlx_lm's affine defaults.
_QUANT_BITS = {"int4": 4, "int8": 8}
_QUANT_GROUP = 64
# Tokenizer/config files fetched when only shards are local.
_META_PATTERNS = ["*.json", "*.txt", "*.model", "*.tiktoken", "*.jinja"]


class MlxMoeRunner:
    backend = "mlx-moe"

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.model: Any = None
        self.tokenizer: Any = None
        self.cache = None

    # ── loading ───────────────────────────────────────────────────────────

    def load(self) -> float:
        """Load from an SWLP shard dir (``--shard-dir``, full-precision expert
        banks) or, with no shard dir, straight from an MLX-format checkpoint
        (``--model mlx-community/...-4bit``: experts read in place, packed)."""
        if self.model is not None:
            return 0.0
        started = time.perf_counter()
        shard_dir = self.config.runtime.shard_dir
        if shard_dir:
            model, cache, meta_path, blocks = self._load_from_shards(Path(shard_dir))
        else:
            model, cache, meta_path, blocks = self._load_mlx_checkpoint()
        if str(self.config.runtime.swlp_expert_prefetch) == "predictive":
            for layer, nxt in zip(sorted(blocks), sorted(blocks)[1:], strict=False):
                parent, attr = blocks[layer]
                gate = getattr(blocks[nxt][0], "gate", None)
                if gate is not None:
                    getattr(parent, attr).set_next_router(nxt, gate)
        cache.budget_bytes = self._budget_bytes()
        self.model, self.cache = model, cache
        from mlx_lm.utils import load_config, load_tokenizer

        # Stop tokens come from config.json, as in mlx_lm.load: Gemma 4 ends a
        # turn with <turn|>, which is not the tokenizer's own eos — without it
        # generation runs past the answer.
        eos_ids = load_config(meta_path).get("eos_token_id")
        self.tokenizer = load_tokenizer(meta_path, eos_token_ids=eos_ids)
        LOGGER.info("mlx_moe_loaded", extra={"moe_layers": len(blocks),
                                             "expert_budget_mb": cache.budget_bytes >> 20})
        return time.perf_counter() - started

    def _load_from_shards(self, shard_dir: Path) -> tuple[Any, Any, Path, dict]:
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_lm.utils import _get_classes, load_config

        from .mlx_expert_cache import MlxExpertCache

        index_path = shard_dir / EXPERT_INDEX_FILE
        if not index_path.is_file():
            raise ValueError(
                f"--backend mlx-moe needs an MoE shard dir with {EXPERT_INDEX_FILE}; "
                f"got {shard_dir}"
            )
        manifest = load_manifest(shard_dir)
        meta_path = self._meta_path(manifest.model_id)
        hf_config = load_config(meta_path)
        model_cls, args_cls = _get_classes(hf_config)
        model = model_cls(args_cls.from_dict(hf_config))
        # Budget is sized after the resident load so it sees the true free RAM.
        cache = MlxExpertCache(ExpertIndex.load(index_path), shard_dir, 0, _EXPERT_READ_WORKERS)
        bits = _QUANT_BITS.get(str(self.config.runtime.swlp_moe_quant).lower())
        if bits:
            cache.quant = (_QUANT_GROUP, bits)  # experts quantized as they are cached
        blocks = swap_in_cached_experts(
            model, cache, (_QUANT_GROUP, bits, "affine") if bits else None)

        cpu_embed = swap_in_mmap_embedding(model, shard_dir)
        weights = model.sanitize(
            dense_weights(shard_dir, manifest.num_layers, hf_config, include_embed=not cpu_embed)
        )
        missing = set(dict(tree_flatten(model.parameters()))) - set(weights)
        if missing:
            raise ValueError(f"shards lack {len(missing)} weights, e.g. {sorted(missing)[:3]}")
        model.load_weights(list(weights.items()), strict=False)
        if bits:
            from mlx_lm.utils import quantize_model

            # Same math and per-layer predicate (routers at 8-bit) as
            # mlx_lm.convert; the cached-expert and mmap-embedding modules
            # have no to_quantized() and are skipped.
            quantize_model(model, hf_config, _QUANT_GROUP, bits)
        mx.eval(model.parameters())
        model.eval()
        return model, cache, meta_path, blocks

    def _load_mlx_checkpoint(self) -> tuple[Any, Any, Path, dict]:
        """mlx_lm's own lazy loader (quantization + sanitize stay upstream);
        the stacked expert tensors are dropped before anything is evaluated,
        so only non-expert weights are ever read into memory."""
        import mlx.core as mx
        from huggingface_hub import snapshot_download
        from mlx_lm.utils import load_model

        from ..model.mlx_expert_index import index_mlx_checkpoint
        from .mlx_expert_cache import MlxExpertCache

        local = self.config.model.local_model_path
        cache_dir = self.config.cache.cache_dir
        path = Path(local) if local else Path(snapshot_download(
            self.config.model.model_id, cache_dir=str(cache_dir) if cache_dir else None))
        model, _ = load_model(path, lazy=True)
        cache = MlxExpertCache(index_mlx_checkpoint(path), path, 0, _EXPERT_READ_WORKERS)
        blocks = swap_in_cached_experts(model, cache)
        mx.eval(model.parameters())
        model.eval()
        return model, cache, path, blocks

    def _meta_path(self, model_id: str) -> Path:
        local = self.config.model.local_model_path
        if local:
            return Path(local)
        from huggingface_hub import snapshot_download

        cache_dir = self.config.cache.cache_dir
        return Path(snapshot_download(
            self.config.model.model_id or model_id,
            allow_patterns=_META_PATTERNS,
            cache_dir=str(cache_dir) if cache_dir else None,
        ))

    def stream_tokens(self, prompt: str, max_tokens: int = 512) -> Iterator[str]:
        """Yield text as it is generated. ``prompt`` is used verbatim — the chat
        REPL passes an already chat-templated conversation."""
        from mlx_lm import stream_generate

        self.load()
        for resp in stream_generate(self.model, self.tokenizer, prompt, max_tokens=max_tokens):
            if resp.text:
                yield resp.text

    def _format(self, prompt: str) -> str:
        """Chat-template the prompt when the tokenizer has one, as mlx_lm's own
        CLI does by default. Instruct checkpoints need it: raw Gemma 4 text
        lacks <bos> and degenerates (measured: " of of of …")."""
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False,
                enable_thinking=False)
        return prompt

    def _budget_bytes(self) -> int:
        """Expert-cache bytes: configured or free RAM, always clamped to what
        Metal will actually hold. Exceeding the GPU working set aborts the
        process (kIOGPUCommandBufferCallbackErrorOutOfMemory, measured with a
        14 GB budget on the 16 GB M5 whose working set is 11.8 GB)."""
        import mlx.core as mx

        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        gpu_room = (int(info["max_recommended_working_set_size"])
                    - int(mx.get_active_memory()) - _AUTO_BUDGET_RESERVE_BYTES)
        configured = int(self.config.runtime.swlp_expert_cache_mb) * 1024**2
        wanted = configured if configured > 0 else max(
            _MIN_AUTO_BUDGET_BYTES,
            int(psutil.virtual_memory().available) - _AUTO_BUDGET_RESERVE_BYTES,
        )
        budget = max(0, min(wanted, gpu_room))
        if configured > budget:
            LOGGER.warning("mlx_moe_budget_clamped", extra={
                "requested_mb": configured >> 20, "budget_mb": budget >> 20})
        return budget

    # ── generation ────────────────────────────────────────────────────────

    def run(self, prompt: str, profile: bool = False) -> RunResult:
        from mlx_lm import stream_generate

        load_seconds = self.load()
        pieces: list[str] = []
        latencies: list[float] = []
        generated = 0
        gen_tps: float | None = None
        start = last = time.perf_counter()
        for resp in stream_generate(self.model, self.tokenizer, self._format(prompt),
                                    max_tokens=self.config.generation.max_new_tokens):
            now = time.perf_counter()
            latencies.append(now - last)
            last = now
            pieces.append(resp.text)
            generated = resp.generation_tokens
            gen_tps = resp.generation_tps
        generate_seconds = time.perf_counter() - start
        stats = self.cache.stats()
        LOGGER.info("mlx_moe_expert_stats", extra=stats)
        prompt_tokens = len(self.tokenizer.encode(prompt))
        metrics = RunMetrics(
            model_id=self.config.model.model_id,
            backend=self.backend,
            device="mps",
            load_seconds=load_seconds,
            input_tokens=prompt_tokens,
            output_tokens=prompt_tokens + generated,
            generate_seconds=generate_seconds,
            total_seconds=load_seconds + generate_seconds,
            time_to_first_token_seconds=latencies[0] if latencies else None,
            per_token_latency_seconds=(
                sum(latencies[1:]) / (len(latencies) - 1) if len(latencies) > 1 else None
            ),
            throughput_tokens_per_second=gen_tps,
            generated_tokens=generated,
            ram_peak_bytes=int(psutil.Process().memory_info().rss) if profile else None,
        )
        return RunResult(prompt=prompt, completion="".join(pieces), metrics=metrics)
