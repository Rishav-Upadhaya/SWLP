# Architecture

SWLP runs a transformer one layer at a time and streams the weights from SSD, so only a window
of `W` layers is in memory at any moment. This page covers how that path is built. For settings,
see [configuration.md](configuration.md).

## Package layout

| Package | Responsibility |
|---|---|
| `swlp/config.py`, `metrics.py`, `logging.py`, `codec.py` | Leaf modules: `AppConfig`, `RunMetrics` and `RunResult`, logging setup, and the `.swz` codec. |
| `swlp/core/` | Streaming algorithms: schedulers, shard I/O, residency planning, KV cache, speculative helpers, profiler. No model loading. |
| `swlp/hardware/` | `detect_hardware()`: chip, RAM, and measured SSD bandwidth. Read-only. |
| `swlp/model/` | Disk formats: sharding, expert banks, layer packages. |
| `swlp/runner/` | Runners behind one `build_runner(config)` factory. |
| `swlp/benchmark/` | `swlp bench` measurement (plus the suite/simulator libraries used by tests and `scripts/research`). |
| `swlp/cli*.py`, `chat.py`, `serve.py`, `tui.py` | CLI, chat REPL, OpenAI-compatible server, terminal UI. |

The research-only scheduling simulators, the trace analyzer and the policy evaluator are in
`scripts/research/simtools/`, not in the package:

```bash
python -m scripts.research.simtools --help   # policy-report | sim | analyze | sweep | evaluate
```

## Streaming path

`SWLPRunner` (`runner/swlp.py`) keeps the embeddings, `lm_head` and final norm loaded
permanently. Only transformer blocks stream. `SWLPRunner._build_scheduler()` picks one of two
schedulers:

| Scheduler | Module | When |
|---|---|---|
| `StreamingScheduler` | `core/streaming.py` | A shard directory exists. This is the main path for models larger than RAM. |
| `ThreadedScheduler` | `core/scheduler.py` | The model is already in CPU RAM. It swaps blocks between CPU and MPS with a thread pool. |

### Per-token loop (`SWLPRunner._run_blocks`)

```
lookahead = max(window_size, prefetch_depth)
warm-up:  prefetch layers 0 .. lookahead-1
for each layer i:
    prefetch(i+1 .. i+lookahead)     # non-blocking
    block = ensure(i)                # may wait on the prefetch
    hidden = call_block(block, ...)  # compute on MPS
    evict(i)                         # block → meta device
```

At most `lookahead + 1` layers are materialized, so peak RAM is about `(lookahead + 1) × layer size`.

### Two-stage prefetch pipeline

`StreamingScheduler` runs two persistent `ThreadPoolExecutor` pools, so one layer's disk read
overlaps another layer's host-to-device copy:

```
layer N    [ read + parse ][ upload ]
layer N+1                  [ read + parse ][ upload ]
layer N+2                                  [ read + parse ][ upload ]
```

- **Read** (`_pool`): `core/shard_io.py` reads the shard once with `readinto()` into a reusable
  per-worker buffer and decompresses `.swz` shards. Tensors are zero-copy views into that buffer.
- **Upload** (`_upload_pool`): makes the one host-to-device copy.
- **`ensure()`** on the compute thread only calls `load_state_dict(assign=True)`, which swaps
  pointers.

`ensure()` classifies each layer, and `overlap_stats()` reports the counts:

| Outcome | Meaning |
|---|---|
| `hit` | Prefetch finished before `ensure()`, so there was no wait. |
| `wait` | Prefetch was still in flight, so compute blocked. |
| `miss` | Nothing was in flight, so the read ran synchronously on the compute thread. |
| `resident` | The layer came from the CPU-RAM residency cache, with no SSD read. |

Overlap rate = hits / (hits + waits + misses). The first token always misses on cold layers.

### Eviction and residency

- `evict()` calls `to_empty(device="meta")` on the block. If every layer is resident, eviction is
  skipped.
- **Residency** (`core/residency.py::plan_residency`) keeps `resident_count` layers as state
  dicts in CPU RAM, never on MPS. Keeping them on MPS fragments the Metal allocator. The planner
  applies a full-model-fit guard: residency turns on only when the whole model fits the usable
  budget. Partial residency pushes the streaming layers out of the page cache and triggers the
  macOS memory compressor (see [results.md](results.md#adaptive-residency)).

  ```
  usable = (total RAM − 4 GB OS reserve − 2 GB working reserve) × 0.75
  ```

- **Direct I/O** (`core/streaming.py::resolve_direct_io`): in `auto` mode, reads bypass the page
  cache with `F_NOCACHE` only when the model is larger than 60% of available RAM. Smaller models
  get page-cache residency, which macOS can reclaim under pressure.
- **Striping**: `SWLP_SHARD_VOLUMES` assigns layers round-robin across directories, so reads run
  in parallel across SSDs.

### Planning chain

`swlp doctor` and `swlp_residency = "auto"` run the same chain:

```
detect_hardware()             hardware/detect.py     chip, RAM, measured SSD GB/s
  → pipeline ratio            core/pipeline_model.py, core/residency.py   I/O ÷ compute per layer
  → resident count            core/resident_policy.py   (ratio, free RAM) → layers to keep
  → plan_residency            core/residency.py         memory clamp + reasoning chain
  → confidence                core/confidence.py        score in [0, 1] with factor breakdown
```

```
pipeline ratio = (SSD read + deserialize + upload) / block compute
  > 1  I/O-bound: prefetch overlap and speculation pay off
  < 1  compute-bound: streaming overhead is small
```

## Profiling

Profiling (`SWLP_PROFILE=1`) attaches a `LayerProfiler` (`core/profiler.py`) that timestamps
every stage of every layer: `read`, `deserialize`, `upload`, `ready`, `compute` and `evict`. It
exports a JSON trace with hardware metadata:

```bash
SWLP_PROFILE=1 swlp run mistral-7b "Hi" -n 8        # writes layer_traces.json
python -m scripts.research.simtools analyze layer_traces.json
```

The measured per-layer stages for Mistral-7B on M5 were: read about 120 ms (SSD-limited) and
host-to-device about 39 ms, with direct I/O and `W=4` ([results.md](results.md#lossless-shard-codec-swz)).
End-to-end tok/s also includes costs the pipeline model leaves out: attention, KV bookkeeping,
Python, Metal dispatch and synchronization.

## Speculative decoding

`SpeculativeRunner(SWLPRunner)` (`runner/speculative.py`) verifies up to K drafted tokens in a
single disk sweep. Verification is greedy, so the output is identical to plain greedy `swlp`.

| Drafter | Module | Selected by |
|---|---|---|
| N-gram prompt lookup (multi-resolution) | `core/speculative.py::NgramDrafter` | default |
| Resident small model (same tokenizer) | `runner/draft.py::DraftModelDrafter` | `SWLP_DRAFT_MODEL` |
| Checkpoint MTP head (dense models) | `runner/mtp.py` | automatic when `mtp.safetensors` exists |

Draft length adapts to acceptance (AIMD). For hybrid Gated-DeltaNet models, which cannot be
rolled back with `DynamicCache.crop()`, `runner/hybrid_rollback.py` replays only the accepted
prefix.

## KV cache

`core/kv_cache.py::KVCacheManager` holds KV in tiers: device, host, zlib-compressed, then disk
spill. `core/compressed_cache.py` exposes it as a transformers `Cache`. INT4 KV
(`core/kv_quant.py`) is opt-in and lossy. `core/prefix_cache.py::PrefixKVCache` reuses the exact
prefix KV across chat turns.

## Mixture-of-Experts

- **Torch path** (`swlp` backend, shard-format v2): `runner/experts.py::SwlpCachedExperts`
  replaces the fused Experts module. `runner/expert_scheduler.py::ExpertScheduler` range-reads
  single experts from the expert banks into a global LRU cache with a byte budget.
- **MLX path** (`mlx-moe`): `runner/mlx_moe.py` keeps the dense weights resident and replaces
  each `switch_mlp` with `runner/mlx_switch.py::CachedSwitchGLU`, backed by
  `runner/mlx_expert_cache.py::MlxExpertCache`. That cache holds one MLX array per expert,
  uses LFU eviction, and reads with parallel `pread`. Expert byte ranges in MLX-format
  checkpoints come from `model/mlx_expert_index.py`.

## MLX dense backend

`runner/mlx.py::MlxRunner` loads the whole model through `mlx_lm` with int8, int4 or bf16
weights. `runner/mlx_tune.py` raises the Metal wired limit (clamped to the recommended working
set) and applies KV quantization, speculative-decoding and prefill-chunk settings. Use it when
the model fits in RAM. Use `swlp` when it doesn't.
