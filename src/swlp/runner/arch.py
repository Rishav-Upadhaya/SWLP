"""Model-architecture dispatch for SWLPRunner.

Different model families expose their transformer blocks under different module
paths and require different forward-pass plumbing (positional encoding, masks,
KV cache type). This module abstracts those differences behind a single
``ArchAdapter`` interface so SWLPRunner stays family-agnostic.

Supported families:
- GPT-2: ``model.transformer.{wte,wpe,drop,ln_f,h}`` — legacy per-layer KV tuples.
- Llama/Mistral (and HF lookalikes): ``model.model.{embed_tokens,norm,layers,rotary_emb}``
  with HF ``DynamicCache``, RoPE position embeddings, and causal mask.
"""
from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import nn

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class StepContext:
    """Per-forward-step state shared across all layers in a single token step."""

    hidden_states: torch.Tensor
    position_ids: torch.Tensor | None
    causal_mask: torch.Tensor | None
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None
    past_state: Any  # DynamicCache (gpt2 and llama/mistral)
    padding_mask: torch.Tensor | None = None  # [N, total_len] 1/0 mask for batched runs


class ArchAdapter(Protocol):
    name: str

    def get_blocks(self, model: nn.Module) -> list[nn.Module]: ...

    def device_modules(self, model: nn.Module) -> list[nn.Module]:
        """Modules that always stay on device (embed, norm, lm_head, rotary)."""
        ...

    def init_past_state(self, model: nn.Module, num_layers: int) -> Any: ...

    def prepare_step(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        past_state: Any,
        device: torch.device,
        position_offset: int,
        attention_mask: torch.Tensor | None = None,
    ) -> StepContext:
        """Build the per-step context.

        ``attention_mask`` is the full ``[N, total_len]`` 1/0 padding mask
        (prompt + tokens generated so far). When given, position_ids are
        derived per-sequence from it so left-padded batches stay correct;
        when ``None`` the single-sequence ``position_offset`` path is used.
        """
        ...

    def call_block(
        self, block: nn.Module, ctx: StepContext, layer_idx: int
    ) -> tuple[torch.Tensor, Any]:
        """Run one block. Returns (new_hidden_states, per_layer_present_kv_or_None)."""
        ...

    def final_norm(self, model: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor: ...

    def estimate_kv_bytes_per_token(self, model: nn.Module, dtype: torch.dtype) -> int: ...


class GPT2Adapter:
    name = "gpt2"

    def get_blocks(self, model: nn.Module) -> list[nn.Module]:
        return list(model.transformer.h)

    def device_modules(self, model: nn.Module) -> list[nn.Module]:
        t = model.transformer
        return [t.wte, t.wpe, t.drop, t.ln_f, model.lm_head]

    def init_past_state(self, model: nn.Module, num_layers: int) -> Any:
        from transformers.cache_utils import DynamicCache

        # transformers ≥5 GPT-2 blocks mutate a shared Cache in place via
        # cache.update(...) and return only hidden states — the legacy
        # per-layer tuple protocol no longer exists. A bare DynamicCache()
        # (no config pre-sizing) matches the Llama-like path and is safe for
        # the batch-1 streaming this adapter serves.
        return DynamicCache()

    def prepare_step(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        past_state: Any,
        device: torch.device,
        position_offset: int,
        attention_mask: torch.Tensor | None = None,
    ) -> StepContext:
        # GPT-2 is the tiny-test path only; batched streaming targets the
        # Llama-family models, so the attention_mask is accepted but unused.
        t = model.transformer
        position_ids = torch.arange(
            position_offset,
            position_offset + input_ids.shape[-1],
            device=input_ids.device,
            dtype=torch.long,
        ).unsqueeze(0)
        inputs_embeds = t.wte(input_ids) + t.wpe(position_ids)
        hidden_states = t.drop(inputs_embeds)
        return StepContext(
            hidden_states=hidden_states,
            position_ids=position_ids,
            causal_mask=None,
            position_embeddings=None,
            past_state=past_state,
            padding_mask=None,
        )

    def call_block(
        self, block: nn.Module, ctx: StepContext, layer_idx: int
    ) -> tuple[torch.Tensor, Any]:
        attention_mask = None
        query_len = int(ctx.hidden_states.shape[1])
        cache = ctx.past_state
        # Per-layer past length BEFORE this block's in-place update: earlier
        # layers of the same sweep have already appended their keys to the
        # shared cache, so the global length would overcount this layer's KV.
        past_len = int(cache.get_seq_length(layer_idx)) if cache is not None else 0
        if past_len > 0 and query_len > 1:
            # Chunked prefill / prefix-suffix: queries > 1 over a non-empty
            # past need the bottom-right-aligned causal mask. With no mask,
            # sdpa's default top-left triangle wrongly hides the chunk's own
            # prefix from its earlier queries. Query i (global position
            # past+i) attends kv j <= i + past_len, so block j - i > past_len.
            min_value = torch.finfo(ctx.hidden_states.dtype).min
            full = torch.full((query_len, past_len + query_len), min_value,
                              device=ctx.hidden_states.device,
                              dtype=ctx.hidden_states.dtype)
            causal = torch.triu(full, diagonal=1 + past_len)
            attention_mask = causal[None, None, :, :]
        outputs = block(
            ctx.hidden_states,
            past_key_values=cache,
            attention_mask=attention_mask,
            use_cache=True,
        )
        # transformers ≥5 GPT2Block returns bare hidden states; the KV cache
        # was updated in place inside attention. Keep the tuple shape of the
        # ArchAdapter contract with a None present.
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs
        return hidden_states, None

    def final_norm(self, model: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        return model.transformer.ln_f(hidden_states)

    def estimate_kv_bytes_per_token(self, model: nn.Module, dtype: torch.dtype) -> int:
        n_layer = int(getattr(model.config, "n_layer", 0))
        n_embd = int(getattr(model.config, "n_embd", 0))
        bytes_per = torch.tensor([], dtype=dtype).element_size()
        return 2 * n_layer * n_embd * bytes_per


class LlamaLikeAdapter:
    """Llama / Mistral / similar HF architectures using DynamicCache + RoPE."""

    name = "llama_like"

    def get_blocks(self, model: nn.Module) -> list[nn.Module]:
        return list(model.model.layers)

    def device_modules(self, model: nn.Module) -> list[nn.Module]:
        inner = model.model
        # embed_tokens is deliberately absent: it stays wherever the loader
        # put it (CPU mmap for shard streaming) — see prepare_step.
        mods = [inner.norm, model.lm_head]
        if hasattr(inner, "rotary_emb"):
            mods.append(inner.rotary_emb)
        return mods

    def init_past_state(self, model: nn.Module, num_layers: int) -> Any:
        from transformers.cache_utils import DynamicCache

        # Hybrid models (qwen3_5: Gated-DeltaNet + full attention) need the
        # config-structured cache — linear-attention layers hold conv/recurrent
        # state, not K/V. ponytail: batched (N>1) streaming unverified for these.
        if "linear_attention" in (getattr(model.config, "layer_types", None) or ()):
            return DynamicCache(config=model.config)
        # Plain DynamicCache() — grows dynamically and handles any batch size.
        # DynamicCache(config=...) pre-structures the per-layer cache for the
        # config's layout and silently corrupts batched (N>1) K/V writes, so
        # batch-1 always worked but batched streaming produced garbage.
        return DynamicCache()

    def prepare_step(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        past_state: Any,
        device: torch.device,
        position_offset: int,
        attention_mask: torch.Tensor | None = None,
    ) -> StepContext:
        inner = model.model
        embed = inner.embed_tokens
        inputs_embeds = embed(input_ids.to(embed.weight.device)).to(device)
        seq_len = input_ids.shape[-1]

        if attention_mask is not None:
            # Batched / padded path: position_ids are per-sequence. The full
            # mask covers prompt + tokens generated so far; cumsum-1 yields the
            # true position of each token (pad tokens do not advance it). Slice
            # the trailing seq_len so it matches the tokens fed this step.
            pos = attention_mask.long().cumsum(-1) - 1
            position_ids = pos.clamp(min=0)[:, -seq_len:].to(inputs_embeds.device)
        else:
            position_ids = torch.arange(
                position_offset,
                position_offset + seq_len,
                device=inputs_embeds.device,
                dtype=torch.long,
            ).unsqueeze(0)

        position_embeddings = rope_embeddings(inner, inputs_embeds, position_ids)

        causal_mask = _build_causal_mask(
            model.config, inputs_embeds, past_state, position_ids, attention_mask
        )

        return StepContext(
            hidden_states=inputs_embeds,
            position_ids=position_ids,
            causal_mask=causal_mask,
            position_embeddings=position_embeddings,
            past_state=past_state,
            padding_mask=attention_mask,
        )

    def call_block(
        self, block: nn.Module, ctx: StepContext, layer_idx: int
    ) -> tuple[torch.Tensor, Any]:
        mask = ctx.causal_mask
        # transformers 5.8 names it ``block_type`` (``layer_type`` before).
        kind = getattr(block, "block_type", None) or getattr(block, "layer_type", None)
        if kind == "linear_attention":
            mask = _linear_attn_mask(ctx)
        outputs = block(
            ctx.hidden_states,
            attention_mask=mask,
            position_ids=ctx.position_ids,
            past_key_values=ctx.past_state,
            use_cache=True,
            position_embeddings=ctx.position_embeddings,
        )
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs
        # For Llama-like models, KV mutates DynamicCache in place; no per-layer return.
        return hidden_states, None

    def final_norm(self, model: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        return model.model.norm(hidden_states)

    def estimate_kv_bytes_per_token(self, model: nn.Module, dtype: torch.dtype) -> int:
        cfg = model.config
        n_layer = int(getattr(cfg, "num_hidden_layers", 0))
        n_kv_heads = int(
            getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", 0))
        )
        head_dim = int(
            getattr(cfg, "head_dim", 0)
            or (getattr(cfg, "hidden_size", 0) // max(getattr(cfg, "num_attention_heads", 1), 1))
        )
        bytes_per = torch.tensor([], dtype=dtype).element_size()
        return 2 * n_layer * n_kv_heads * head_dim * bytes_per


def rope_embeddings(
    inner: nn.Module, inputs_embeds: torch.Tensor, position_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """(cos, sin) from the decoder's rotary module, or None if it has none."""
    if not hasattr(inner, "rotary_emb"):
        return None
    rope_ids = position_ids
    if hasattr(inner.rotary_emb, "mrope_section"):
        # mRoPE (qwen3_5) rotates over 3 axes; for text-only input every
        # axis is the token position, which is what HF's forward builds.
        rope_ids = position_ids[None].expand(3, *position_ids.shape)
    return inner.rotary_emb(inputs_embeds, position_ids=rope_ids)


def _linear_attn_mask(ctx: StepContext) -> torch.Tensor | None:
    """Padding mask for linear-attention layers — mirrors HF's
    ``_update_linear_attn_mask``: only needed on an uncached, padded prefill."""
    mask = ctx.padding_mask
    if mask is None or bool(torch.all(mask == 1)) or ctx.past_state.has_previous_state():
        return None
    return mask


def _build_causal_mask(
    config: Any,
    inputs_embeds: torch.Tensor,
    past_state: Any,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Build the causal attention mask, deferring to transformers helpers when present.

    ``attention_mask`` (the ``[N, total_len]`` 1/0 padding mask) is forwarded so
    padded positions in a batch are excluded from attention; ``None`` keeps the
    plain causal mask used by single-sequence runs.
    """
    try:
        from transformers.masking_utils import (
            create_causal_mask,
            create_sliding_window_causal_mask,
        )
    except ImportError:
        return None

    use_sliding = getattr(config, "sliding_window", None) is not None
    mask_fn = create_sliding_window_causal_mask if use_sliding else create_causal_mask

    # The parameter name changed across transformers versions:
    #   ≤ 5.x:  inputs_embeds  (plural)
    #   newer:  input_embeds   (singular)
    # Detect the correct name from the function signature to be forward- and
    # backward-compatible without hard-coding either variant.
    sig_params = inspect.signature(mask_fn).parameters
    if "inputs_embeds" in sig_params:
        embeds_kwarg = "inputs_embeds"
    else:
        embeds_kwarg = "input_embeds"

    try:
        return mask_fn(
            config=config,
            **{embeds_kwarg: inputs_embeds},
            attention_mask=attention_mask,
            past_key_values=past_state,
            position_ids=position_ids,
        )
    except Exception:
        LOGGER.exception("causal_mask_build_failed")
        return None


_GPT2_TYPES = {"gpt2"}
_LLAMA_LIKE_TYPES = {
    "llama",
    "mistral",
    "qwen2",
    "qwen3",
    "qwen3_5",
    "qwen3_5_text",
    "phi",
    "phi3",
    "gemma",
    "gemma2",
    "mixtral",
}


def get_adapter(model_type: str) -> ArchAdapter:
    mt = (model_type or "").lower()
    if mt in _GPT2_TYPES:
        return GPT2Adapter()
    if mt in _LLAMA_LIKE_TYPES:
        return LlamaLikeAdapter()
    # Fallback: structural detection happens at runtime; default to llama-like
    # since all modern decoder-only models follow that layout.
    LOGGER.warning("arch_unknown_defaulting_llama_like", extra={"model_type": model_type})
    return LlamaLikeAdapter()

