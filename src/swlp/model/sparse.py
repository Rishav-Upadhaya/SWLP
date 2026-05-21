"""Sparse weight codec for SWLP shard files (Phase 13).

Stores high-sparsity weight tensors in COO (coordinate) format to reduce
on-disk size and per-token streaming bytes.  Each sparse tensor is split into
three auxiliary keys:

    ``{name}__sparse_indices``  — int32 COO indices, shape ``[ndim, nnz]``
    ``{name}__sparse_values``   — float values, shape ``[nnz]``
    ``{name}__sparse_shape``    — int64 original shape, shape ``[ndim]``

A dict is considered *sparse-encoded* if at least one key ends with
``__sparse_indices``.  ``decode_sparse`` is a **no-op** on plain dense dicts,
so wiring it into the streaming path is safe for existing shard directories.

Practical note on transformer weights
---------------------------------------
Standard, un-pruned transformer FFN / attention weights have near-zero
sparsity (< 1 %).  COO encoding only saves space when sparsity ≥ ~50 %.
Below that threshold ``encode_sparse`` keeps the tensor dense.  Users with
magnitude-pruned or SparseGPT-sparsified checkpoints can run
``sparsify_shards()`` as a one-time conversion step to benefit from reduced
streaming traffic.  Quality is **bit-exact** — the decode is ``to_dense()``
on a lossless COO representation.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

LOGGER = logging.getLogger(__name__)

# Suffix triplet that marks a sparse-encoded tensor.
_IDX_SUFFIX = "__sparse_indices"
_VAL_SUFFIX = "__sparse_values"
_SHP_SUFFIX = "__sparse_shape"


# ── public API ────────────────────────────────────────────────────────────────


def sparsity(tensor: torch.Tensor) -> float:
    """Return the fraction of exactly-zero elements in ``tensor``."""
    if tensor.numel() == 0:
        return 0.0
    return float((tensor == 0).sum().item()) / tensor.numel()


def encode_sparse(
    state_dict: dict[str, torch.Tensor],
    threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Encode weight tensors with zero-fraction ≥ ``threshold`` as COO sparse.

    Only 2-D+ float tensors are candidates (1-D bias/norm vectors are too small
    to benefit and their COO overhead exceeds any saving).  Tensors below the
    threshold are stored dense.  Returns a new dict; the input is not modified.
    """
    result: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if not tensor.is_floating_point() or tensor.ndim < 2:
            result[name] = tensor
            continue
        sp = sparsity(tensor)
        if sp >= threshold:
            sparse = tensor.to_sparse_coo().coalesce()
            result[f"{name}{_IDX_SUFFIX}"] = sparse.indices().to(torch.int32)
            result[f"{name}{_VAL_SUFFIX}"] = sparse.values()
            result[f"{name}{_SHP_SUFFIX}"] = torch.tensor(
                list(tensor.shape), dtype=torch.int64
            )
            LOGGER.debug(
                "sparse_encoded",
                extra={"name": name, "sparsity": round(sp, 4), "nnz": int(sparse._nnz())},
            )
        else:
            result[name] = tensor
    return result


def decode_sparse(state_dict: dict) -> dict:
    """Reconstruct dense tensors from COO sparse encoding.

    Detects sparse keys by the ``__sparse_indices`` suffix.  Returns the
    original dict unchanged if no sparse keys are present — safe to call on
    any shard dict without overhead.
    """
    sparse_names = {
        k[: -len(_IDX_SUFFIX)]
        for k in state_dict
        if k.endswith(_IDX_SUFFIX)
    }
    if not sparse_names:
        return state_dict

    result: dict = {}
    sparse_aux_keys = {
        f"{n}{s}"
        for n in sparse_names
        for s in (_IDX_SUFFIX, _VAL_SUFFIX, _SHP_SUFFIX)
    }
    for k, v in state_dict.items():
        if k not in sparse_aux_keys:
            result[k] = v

    for name in sparse_names:
        try:
            indices = state_dict[f"{name}{_IDX_SUFFIX}"].to(torch.int64)
            values = state_dict[f"{name}{_VAL_SUFFIX}"]
            shape = state_dict[f"{name}{_SHP_SUFFIX}"].tolist()
            sparse_t = torch.sparse_coo_tensor(indices, values, shape, check_invariants=False)
            result[name] = sparse_t.to_dense()
        except Exception:
            LOGGER.exception("sparse_decode_failed", extra={"name": name})
    return result


def is_sparse_encoded(state_dict: dict) -> bool:
    """Return True if ``state_dict`` contains at least one COO-encoded tensor."""
    return any(k.endswith(_IDX_SUFFIX) for k in state_dict)


# ── shard conversion utility ──────────────────────────────────────────────────


def sparsify_shards(
    shard_dir: str | Path,
    output_dir: str | Path,
    threshold: float = 0.5,
) -> dict[str, int]:
    """Convert an existing shard directory to sparse-encoded shards.

    Reads every ``layer_*.pt`` file from ``shard_dir``, applies
    ``encode_sparse(threshold=threshold)``, and writes the result to
    ``output_dir``.  Non-layer files (``embed.pt``, ``lm_head.pt``,
    ``shard_manifest.json``) are copied unchanged.

    Returns a summary: ``{"layers": int, "tensors_sparsified": int}``.
    """
    shard_dir = Path(shard_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = 0
    tensors_sparsified = 0

    for pt_file in sorted(shard_dir.glob("*.pt")):
        state = torch.load(str(pt_file), map_location="cpu", weights_only=True)
        dense_count = sum(1 for k in state if not k.endswith(_IDX_SUFFIX))
        encoded = encode_sparse(state, threshold=threshold)
        sparse_count = sum(1 for k in encoded if k.endswith(_IDX_SUFFIX))
        tensors_sparsified += sparse_count
        torch.save(encoded, str(output_dir / pt_file.name))
        layers += 1
        LOGGER.info(
            "sparsify_shard",
            extra={
                "file": pt_file.name,
                "dense_tensors": dense_count,
                "sparse_tensors": sparse_count,
            },
        )

    # Copy the manifest if present.
    manifest = shard_dir / "shard_manifest.json"
    if manifest.exists():
        import shutil
        shutil.copy(manifest, output_dir / "shard_manifest.json")

    LOGGER.info(
        "sparsify_shards_done",
        extra={"layers": layers, "tensors_sparsified": tensors_sparsified},
    )
    return {"layers": layers, "tensors_sparsified": tensors_sparsified}
