"""Model-assembly helpers for :class:`swlp.runner.mlx_moe.MlxMoeRunner`.

Split out to keep the runner under the file budget: swapping stacked expert
modules for :class:`CachedSwitchGLU`, serving untied embeddings from a CPU mmap,
and collecting the non-expert weights from SWLP shards.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def swap_in_cached_experts(model: Any, cache: Any,
                            quant: tuple[int, int, str] | None = None,
                            ) -> dict[int, tuple[Any, str]]:
    """Replace every MoE block's stacked expert module (``switch_mlp`` in
    qwen/olmoe, ``switch_glu`` in gemma4) with a :class:`CachedSwitchGLU`
    that keeps the original's activation and quantization parameters
    (``quant`` overrides them for quantize-on-load)."""
    from .mlx_switch import CachedSwitchGLU

    blocks: dict[int, tuple[Any, str]] = {}
    for i, layer in enumerate(model.layers):
        for child in layer.children().values():
            attr = next((a for a in ("switch_mlp", "switch_glu") if hasattr(child, a)), None)
            if attr is None:
                continue
            orig = getattr(child, attr)
            proj = orig.gate_proj
            native = ((int(proj.group_size), int(proj.bits), str(getattr(proj, "mode", "affine")))
                      if hasattr(proj, "bits") else None)
            q = quant or native
            setattr(child, attr, CachedSwitchGLU(
                i, cache, int(getattr(child, "top_k", 0) or 0), orig.activation, q))
            blocks[i] = (child, attr)
    return blocks


def swap_in_mmap_embedding(model: Any, shard_dir: Path) -> bool:
    """Serve token embeddings from the mmap-backed ``embed.pt`` on CPU.

    A decode step looks up one row, yet the MLX table would pin the whole
    vocabulary in the Metal working set (1.0 GB for a 248k-vocab A3B model —
    ~20% of its expert budget on 16 GB). Skipped for tied embeddings, where the
    table doubles as ``lm_head`` and must stay on the GPU.
    """
    import torch
    from mlx.utils import tree_flatten, tree_unflatten

    from .mlx_switch import MmapEmbedding

    params = dict(tree_flatten(model.parameters()))
    if not any(k.endswith("lm_head.weight") for k in params):
        return False
    name = next((k[: -len(".weight")] for k in params if k.endswith("embed_tokens.weight")), None)
    if name is None:
        return False
    state = torch.load(shard_dir / "embed.pt", map_location="cpu", weights_only=True, mmap=True)
    model.update_modules(tree_unflatten([(name, MmapEmbedding(state["embed_tokens"]["weight"]))]))
    return True



def dense_weights(shard_dir: Path, num_layers: int, hf_config: dict,
                   include_embed: bool = True) -> dict:
    """Non-expert weights under their original HF names (for ``Model.sanitize``)."""
    import mlx.core as mx
    import torch

    prefix = "model.language_model." if "text_config" in hf_config else "model."
    weights: dict = {}
    for i in range(num_layers):
        for key, arr in mx.load(str(shard_dir / f"layer_{i:03d}.safetensors")).items():
            weights[f"{prefix}layers.{i}.{key}"] = arr
    embed = torch.load(shard_dir / "embed.pt", map_location="cpu", weights_only=True, mmap=True)
    if include_embed:
        weights[f"{prefix}embed_tokens.weight"] = _torch_to_mx(embed["embed_tokens"]["weight"])
    weights[f"{prefix}norm.weight"] = _torch_to_mx(embed["norm"]["weight"])
    lm_head = torch.load(shard_dir / "lm_head.pt", map_location="cpu", weights_only=True)
    weights["lm_head.weight"] = _torch_to_mx(lm_head["weight"])
    return weights


def _torch_to_mx(t: Any) -> Any:
    import mlx.core as mx
    import torch

    if t.dtype == torch.bfloat16:
        return mx.array(t.view(torch.int16).numpy()).view(mx.bfloat16)
    return mx.array(t.numpy())
