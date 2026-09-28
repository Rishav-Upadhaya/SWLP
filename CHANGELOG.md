# Changelog

All notable changes to SWLP are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Concurrent expert-miss staging**: `prepare_set` fans a token's whole routed
  set into the fetch pool before sequential consumption (condition-variable
  handoff, no duplicate reads); prediction leads with the last token's routed
  set (token-to-token repetition signal). `scripts/research/expert_fetch_bench.py`
  measures fetch strategies; at DSV4-Flash-scale experts 4-worker staging
  matches/beats serial cold fetch and warm repeats are served by the OS page
  cache at RAM speed.
- **MoE expert-streaming engine (Phase 25)**: Mixture-of-Experts models run
  with expert-selective sweeps — per-token bytes scale with *active* params
  while disk scales with total. Shard format v2 (per-layer dense shard +
  `layer_XXX.experts.safetensors` bank), decomposed MoE forward (bit-exact vs
  HF), `ExpertScheduler` with a global LRU expert cache, predictive routing
  prefetch, and a set_budget resize API (elastic reallocation between
  generation safe points) (`SWLP_EXPERT_CACHE_MB`,
  `SWLP_EXPERT_PREFETCH` = `off|lru|predictive`). Targets: Qwen3-30B-A3B, Mixtral-8x7B;
  DeepSeek-V4-Flash (284B, 13B active) is the feasibility class. Design
  informed by FreeToken (arXiv:2608.16157) and Mixtral-offloading
  (arXiv:2312.17238).
- **Prefix KV cache (Phase 26)**: exact-match prefix snapshots — full
  sequence plus interior interval slices, bounded by count and bytes —
  skip re-prefilling shared history in chat (FreeToken-inspired),
  lossless by construction. Wired via `SWLPRunner.set_prefix_cache`.
- **Multi-volume striping**: `SWLP_SHARD_VOLUMES` spreads layer shards across
  SSDs; parallel reads aggregate bandwidth.
- **Chunked prefill**: `SWLP_PREFILL_CHUNK` bounds activation RAM on long
  prompts; lossless (causal attention over accumulating KV).
- Multi-resolution n-gram drafting (full-context match with backoff);
  `SWLP_SPEC_MAX_DRAFT` default 8 → 16.
- Read-ahead: `madvise(MADV_WILLNEED)` on the mmap path;
  `posix_fadvise(WILLNEED)` / macOS `F_READAHEAD` on the legacy reader.
- `swlp pull <alias>` — one-command download + shard with progress and
  disk-space preflight; `swlp serve` gains `/v1/models`.
- Measured SSD bandwidth probe cached in `~/.cache/swlp/hardware.json` and consumed
  by hardware detection (hardcoded class-of-hardware values remain fallback).
- `swlp doctor` MoE streaming advisory; `scripts/research/moe_sweep.py`
  expert-cache budget sweep harness.
- `swlp doctor` — hardware check with a per-model backend recommendation.
- `swlp models` — list supported model aliases and backend support.
- Progress bar during `swlp download` sharding.
- CI (pytest + ruff on Python 3.11–3.13), `CITATION.cff`, PEP 561 `py.typed`.
- `SWLP_DIRECT_IO=auto|on|off` — page-cache bypass is now policy, not
  hardwired; models that fit in RAM get OS page-cache residency for free.
- `core/shard_io.py` — single-read shard loading into reusable buffers with
  zero-copy safetensors parsing.
- `bench_common.summarize_runs()` — median ± IQR statistics for headline
  benchmark numbers.
- Draft-model speculative decoding (Phase 21): `--draft-model` /
  `SWLP_DRAFT_MODEL` loads a small resident same-tokenizer model (e.g.
  Qwen2.5-0.5B for a Qwen2.5-14B target) that drafts on every step — unlike
  n-gram prompt-lookup, it accelerates novel text. Draft length adapts to
  acceptance (AIMD), so low-agreement text costs ~nothing instead of
  regressing. Output stays byte-identical to greedy SWLP. Measured on M5:
  Qwen-14B FP16 streaming 0.19 → **0.55–1.16 tok/s** (2.9×–5.9×, acceptance
  48–90%). `--shard-dir` + `--draft-model` auto-selects the speculative
  backend; new profile `configs/swlp_qwen_draft_mps.toml`.

### Fixed
- **GPT-2 final norm (`ln_f`) was never persisted or loaded on the streaming
  path** — it ran on uninitialized memory: exactly-zero first-token logits on
  the first run in a process, recycled-page garbage afterwards. Determinism
  tests passed because both runs were identically wrong; the new
  HF-reference logits regression catches it. `embed.pt` now persists `ln_f`
  (legacy dirs load with a loud re-shard warning).
- **GPT-2 adapter used the removed tuple-KV protocol** — transformers ≥5
  `GPT2Block` mutates a shared `DynamicCache` in place and returns bare
  hidden states, so per-token KV was silently dropped and chunked queries
  mis-masked. The adapter now passes the shared cache and builds the
  bottom-right-aligned causal mask (`triu(diagonal=1+past_len)`) for
  multi-query steps over a non-empty past.
- `swlp profile` non-JSON output crashed on a `result.comulsion` typo.
- `HuggingFaceRunner.stream_tokens` skipped the repetition penalty that
  `run()` applies.
- Early-exit entropy was computed over hidden states instead of vocabulary
  logits.

### Changed
- Research/phase scripts moved from `scripts/` to `scripts/research/`;
  user-facing utilities (`bootstrap.sh`, `shard_*.py`, `package_model.py`,
  `phase0_hardware_check.py`) remain in `scripts/`.
- Streaming hot path (Phase 20): prefetch worker pool prepares device-ready
  layers; the compute thread only pointer-assigns. Mistral-7B FP16 direct-I/O
  streaming: 0.218 → **0.370 tok/s** (+70%), prefill 2.5× faster, output
  byte-identical. Qwen2.5-0.5B: 5.47 → 8.39 tok/s (9.85 with cached reads).
- Scheduler eviction no longer copies unchanged weights device→host
  (CPU-master pointer restore) — halves PCIe traffic on the CUDA path.
- `swlp_prefetch_depth` now actually extends the prefetch lookahead beyond
  the window (previously unused).
- Per-token detokenization cost no longer scales with prompt length
  (8-token anchor incremental decode).

## [0.1.0] — 2026-06-10

Initial public release.

### Added
- FP16 layer streaming (`--shard-dir`): run models larger than RAM by
  streaming transformer layers SSD → RAM through a sliding window.
  26 GB model in 1.7 GB peak RAM; 44 GB in 3.8 GB.
- Native MLX backend for Apple Silicon (`--backend mlx`): 16 tok/s lossless
  int8 (byte-identical to FP16 on Mistral-7B), up to 28 tok/s int4.
- Prompt-lookup speculative decoding (`--backend speculative`): up to 3.3×
  on repetitive / long-context output, no draft model.
- Batched streaming: per-layer disk cost is flat in batch size (~18.5×
  aggregate throughput at batch 16).
- Interactive chat REPL (`swlp chat`) with streaming tokens.
- Benchmark / suite / simulation subcommands with JSON + terminal reports.
- KV cache manager with budget, zlib compression, host/disk offload tiers,
  and opt-in INT4 KV quantization.
- SWLP package format: per-layer `.safetensors` shards with manifest,
  integrity validation, and independent layer loading.
- 154-test suite that runs without GPU or model downloads.
