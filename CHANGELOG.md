# Changelog

All notable changes to SWLP are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-09-30

First release on PyPI. SWLP is now Apple Silicon only.

### Added
- **FP16/BF16 layer streaming** (`--shard-dir`, backend `swlp`): run models
  larger than RAM by streaming transformer layers SSD to unified memory through
  a sliding window (Qwen2.5-14B, 26 GB, in 1.7 GB peak RAM on a 16 GB M5).
  Zero-copy shard reads into reusable buffers with a prefetch worker pool;
  `SWLP_DIRECT_IO=auto|on|off` page-cache policy.
- **Speculative decoding** (backend `speculative`), lossless: n-gram prompt
  lookup (multi-resolution), a resident draft model (`--draft-model`, e.g.
  Qwen2.5-0.5B for Qwen2.5-14B, adaptive draft length), and the checkpoint's
  own multi-token-prediction head (`--mtp`).
- **Qwen3.5-style hybrid (Gated DeltaNet) streaming**, including exact
  speculative rollback of recurrent state; the sharder keeps the checkpoint's
  native dtype (bf16 stays bf16).
- **MoE expert streaming**: shard format v2 expert banks, decomposed MoE
  forward (bit-exact vs Hugging Face), `ExpertScheduler` with a global expert
  cache (`SWLP_EXPERT_CACHE_MB`, `SWLP_EXPERT_PREFETCH=off|lru|predictive`,
  default `lru`).
- **`mlx-moe` backend**: dense weights resident in MLX, experts streamed
  through an LFU cache with parallel reads; also streams 4-bit MLX-format
  checkpoints directly.
- **`mlx` backend**: native MLX compute (`--quant bf16|int8|int4`) with Apple
  tuning: wired-memory limit, KV quantization (`--kv-bits`), draft-model
  speculation, prompt cache.
- **`swlp serve`**: OpenAI-compatible HTTP server (`/v1/chat/completions`,
  `/v1/completions`, `/v1/models`, SSE streaming), bound to 127.0.0.1 by default.
- **Lossless `.swz` shard codec** (`swlp compress-shards`, `--revert`;
  `swlp[codec]` extra): bit-exact ~31% disk saving, CRC-gated. Slower than
  plain shards on fast SSDs, so off by default.
- **Prefix-KV caching** across chat turns, lossless.
- Multi-volume shard striping (`SWLP_SHARD_VOLUMES`) and chunked prefill
  (`SWLP_PREFILL_CHUNK`).
- KV cache manager: RAM budget, zlib compression, host/disk tiers, KV window,
  opt-in lossy INT4 KV (`--kv-quant int4`).
- CLI: `swlp run`, `chat`, `download`/`pull`, `doctor`, `models`,
  `benchmark`, `suite`, `simulate`, `report`, `profile`, `package`,
  `validate-package`, `layer`; measured SSD-bandwidth probe.
- Batched streaming (per-layer disk cost is flat in batch size).
- CI on macOS (pytest + ruff), PyPI Trusted Publishing release workflow,
  `CITATION.cff`, PEP 561 `py.typed`.

### Fixed
- GPT-2 final norm (`ln_f`) was never persisted or loaded on the streaming
  path and ran on uninitialized memory.
- GPT-2 adapter used the tuple-KV protocol removed in transformers 5, silently
  dropping per-token KV.
- `HuggingFaceRunner.stream_tokens` skipped the repetition penalty.

### Removed
- CUDA/NVIDIA support: CUDA scheduler, pinned-memory staging, `torch.cuda`
  branches, `pynvml` and the `gpu` extra, `--device cuda`.
- FP8 shard tier (measured slower) and sparse (COO) shards.
- Lossy early-exit and layer-pruning options.
- `pin_memory` / `double_buffer` knobs (meaningless on unified memory).
- Research simulator subcommands (`sim`, `analyze`, `sweep`, `evaluate`,
  `policy-report`); they moved to `python -m scripts.research.simtools`
  in the repository.
