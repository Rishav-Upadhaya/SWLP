"""Model loading paths for SWLPRunner.

Two strategies:

- ``load_full_model`` — standard HF ``from_pretrained`` with retries. The full
  model lands in CPU RAM; the scheduler swaps blocks CPU<->device.
- ``load_from_shards`` — empty-weight init via ``accelerate.init_empty_weights``;
  only embeddings, final norm, and lm_head are materialized at load time. The
  per-layer ``MistralDecoderLayer`` / ``LlamaDecoderLayer`` instances stay on
  the meta device until ``StreamingScheduler`` reads their shards on demand.
"""
from __future__ import annotations

import logging
import os
import time as _time
from pathlib import Path
from typing import Any

import torch

from ..model.shard import load_manifest, verify_shards

LOGGER = logging.getLogger(__name__)


def load_full_model(runner: Any) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_source = runner.config.model.local_model_path or runner.config.model.model_id
    LOGGER.info(
        "swlp_loading_model",
        extra={
            "model_source": str(model_source),
            "device": runner.device.type,
            "dtype": str(runner.dtype),
        },
    )
    attempts = int(os.getenv("SWLP_RETRIES", "2"))
    last_exc: Exception | None = None
    for attempt in range(attempts + 1):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                cache_dir=str(runner.config.cache.cache_dir),
                trust_remote_code=runner.config.model.trust_remote_code,
                revision=runner.config.model.revision or None,
            )
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_source,
                cache_dir=str(runner.config.cache.cache_dir),
                trust_remote_code=runner.config.model.trust_remote_code,
                revision=runner.config.model.revision or None,
                dtype=runner.dtype,
                low_cpu_mem_usage=True,
            )
            model.eval()
            try:
                model.to("cpu")
            except Exception:
                pass
            return model, tokenizer
        except Exception as exc:
            last_exc = exc
            LOGGER.warning(
                "swlp_model_load_attempt_failed",
                extra={"attempt": attempt, "error": str(exc)},
            )
            _time.sleep(0.5 * (attempt + 1))
    assert last_exc is not None
    raise last_exc


def load_from_shards(runner: Any, shard_dir: Path) -> tuple[Any, Any]:
    """Build an empty model and load only embed/norm/lm_head from shards."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    report = verify_shards(shard_dir)
    if not report.ok:
        raise RuntimeError(
            f"Shard directory {shard_dir} failed its integrity check — "
            f"{report.summary()}. Re-run sharding to regenerate the affected files."
        )

    manifest = load_manifest(shard_dir)
    model_source = runner.config.model.model_id or manifest.model_id
    LOGGER.info(
        "swlp_loading_from_shards",
        extra={
            "shard_dir": str(shard_dir),
            "model_source": model_source,
            "num_layers": manifest.num_layers,
            "layer_weight_mb": manifest.layer_weight_mb,
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        cache_dir=str(runner.config.cache.cache_dir),
        trust_remote_code=runner.config.model.trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    cfg = AutoConfig.from_pretrained(
        model_source,
        cache_dir=str(runner.config.cache.cache_dir),
        trust_remote_code=runner.config.model.trust_remote_code,
    )
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg, dtype=runner.dtype)
    model.eval()

    # mmap: the token-embedding table (2.5 GB for 248k-vocab models) stays
    # file-backed — clean, reclaimable pages; a lookup touches one row/token.
    embed_state = torch.load(
        str(shard_dir / "embed.pt"), map_location="cpu", weights_only=True, mmap=True
    )
    lm_head_state = torch.load(
        str(shard_dir / "lm_head.pt"), map_location="cpu", weights_only=True
    )
    _materialize_embeddings(model, embed_state, manifest.model_type, runner.device)
    model.lm_head.to_empty(device=runner.device)
    model.lm_head.load_state_dict(lm_head_state, strict=False, assign=True)
    _wire_expert_streaming(runner, model, manifest, shard_dir)
    return model, tokenizer


def _wire_expert_streaming(runner: Any, model: Any, manifest: Any, shard_dir: Path) -> None:
    """Swap fused Experts modules for slot-cached ones when the shard dir has
    expert banks (Phase 25). Dense models are untouched.

    The swapped modules keep the reference forward math (bit-exact output);
    the global ``ExpertScheduler`` lands on ``runner._expert_sched`` for the
    per-layer prefetch hook and teardown.
    """
    from ..model.expert_bank import EXPERT_INDEX_FILE, ExpertIndex
    from .expert_scheduler import ExpertScheduler
    from .experts import SwlpCachedExperts

    if manifest.num_experts <= 0:
        return
    index_path = shard_dir / EXPERT_INDEX_FILE
    if index_path.is_file():
        index = ExpertIndex.load(index_path)
    else:
        # v1 shard dir (experts inline in the layer files): index in place.
        # Honest trade-off: the dense stream still reads the whole layer file
        # (expert bytes included) and the scheduler range-reads experts again
        # — a double read of expert bytes, kept for v1 compatibility only.
        # Re-shard to get the v2 split.
        index = ExpertIndex.build(shard_dir, manifest.num_layers)
        if index:
            index.save(index_path)
    if not index:
        return

    budget_mb = int(runner.config.runtime.swlp_expert_cache_mb)
    if budget_mb <= 0:
        import psutil

        budget_mb = int(min(4 * 1024, 0.25 * psutil.virtual_memory().available / 1024**2))
    sched = ExpertScheduler(
        index, shard_dir, runner.device, runner.dtype,
        budget_bytes=budget_mb * 1024 * 1024,
        mode=str(runner.config.runtime.swlp_expert_prefetch),
    )
    act_fn = str(getattr(model.config, "hidden_act", "silu"))
    blocks = model.model.layers if hasattr(model, "model") else []
    wired = 0
    for layer, block in enumerate(blocks):
        parent_name, parent = _find_experts_parent(block)
        if parent is None:
            continue
        li = index.layers.get(layer)
        if li is None:
            LOGGER.warning("expert_bank_layer_missing", extra={"layer": layer})
            continue
        # Construct at the FINAL slot capacity directly — an interim
        # full-bank allocation (~all expert weights on device) would OOM the
        # memory-constrained machines this engine targets.
        module = SwlpCachedExperts(
            layer, li.num_experts, li.hidden, li.intermediate,
            runner.dtype, runner.device, act_fn, sched, slots=sched.slot_cap(layer),
        )
        sched.register(layer, module)
        parent.experts = module
        wired += 1
    runner._expert_sched = sched
    LOGGER.info(
        "expert_streaming_wired",
        extra={"layers": wired, "budget_mb": budget_mb,
               "mode": str(runner.config.runtime.swlp_expert_prefetch)},
    )


def _find_experts_parent(block: Any) -> tuple[str, Any]:
    """Name and module of the child holding a fused ``experts`` container."""
    for name, child in block.named_children():
        experts = getattr(child, "experts", None)
        if experts is not None and experts.__class__.__name__.endswith("Experts"):
            return name, child
    return "", None


def _materialize_embeddings(
    model: Any, embed_state: dict, model_type: str, device: torch.device
) -> None:
    if hasattr(model, "transformer") and "wte" in embed_state:
        t = model.transformer
        t.wte.to_empty(device=device)
        t.wte.load_state_dict(embed_state["wte"], assign=True)
        if "wpe" in embed_state:
            t.wpe.to_empty(device=device)
            t.wpe.load_state_dict(embed_state["wpe"], assign=True)
        t.drop.to_empty(device=device)
        if "ln_f" in embed_state:
            t.ln_f.to_empty(device=device)
            t.ln_f.load_state_dict(embed_state["ln_f"], assign=True)
        else:
            # Legacy shard dirs (pre-ln_f persistence): to_empty leaves the
            # final norm in uninitialized memory — zeroed on a fresh process,
            # stale recycled pages afterwards. The only correct fix is a
            # re-shard; warn loudly rather than fail silently.
            t.ln_f.to_empty(device=device)
            LOGGER.warning(
                "embed_state_missing_ln_f",
                extra={"hint": "re-shard this model to persist ln_f in embed.pt"},
            )
        return
    if hasattr(model, "model") and "embed_tokens" in embed_state:
        inner = model.model
        # Stays on CPU (mmap-backed): LlamaLikeAdapter.prepare_step looks rows
        # up on CPU and moves only [N, T, hidden] to the device.
        inner.embed_tokens.load_state_dict(embed_state["embed_tokens"], assign=True)
        if "norm" in embed_state:
            inner.norm.to_empty(device=device)
            inner.norm.load_state_dict(embed_state["norm"], assign=True)
        if hasattr(inner, "rotary_emb"):
            # RoPE's inv_freq is a computed (non-persistent) buffer — re-instantiate
            # rather than copy from meta, since init_empty_weights left it uninitialized.
            rope_cls = type(inner.rotary_emb)
            try:
                inner.rotary_emb = rope_cls(config=model.config).to(device=device)
            except TypeError:
                inner.rotary_emb = rope_cls(model.config).to(device=device)
        return
    LOGGER.warning("embeddings_layout_unknown", extra={"model_type": model_type})
