# SWLP — Sliding Window Layer Pipeline

> **Run FP16 LLMs that don't fit your RAM — no quantization, no quality loss.**

SWLP streams a model's transformer layers through a sliding window in RAM, one shard at a time. Only *W* layers live in memory at once; the rest stay on disk. A 26 GB model runs on a 16 GB machine with 1.7 GB peak RAM. The model is never quantized — every weight stays full FP16.

For interactive speed, SWLP also ships a native **MLX backend** for Apple Silicon (~16 tok/s, lossless int8).

---

## Why SWLP?

| Problem | Usual fix | SWLP's fix |
|---------|-----------|------------|
| Model (26 GB) > RAM (16 GB) | Quantize to 4-bit | Stream layers from SSD — full FP16 |
| Slow streaming (~0.09 tok/s with AirLLM) | — | Async prefetch overlaps I/O with compute — **2.5× faster** |
| Interactive speed needed | Quantize | MLX native backend — ~16–28 tok/s |
| 44 GB model on a 16 GB machine | Not possible | SWLP: 1.7 GB RAM peak ✅ |

**Design principle: zero quality compromise is a first-class constraint.**

---

## Benchmarks

All measured on **Apple M5, 16 GB unified memory**, Mistral-7B FP16, 32 output tokens, greedy decoding.

### SWLP vs AirLLM — FP16 streaming (equal quality)

| Runtime | tok/s | TTFT | Peak RAM | Notes |
|---------|------:|-----:|---------:|-------|
| **SWLP W=2** | **0.21** | **5.3 s** | **1.21 GB** | Async prefetch — I/O overlaps compute |
| AirLLM | 0.09 | 13.5 s | 1.83 GB | Sequential — I/O blocks compute |
| Full-model FP16 (HF/MLX) | ❌ OOM | ❌ OOM | ~14 GB needed | Doesn't fit 16 GB |

**SWLP is 2.5× faster and 1.5× lower RAM than AirLLM at identical quality.**
AirLLM also cannot run Mistral-Small-24B (architecture incompatibility); SWLP runs it correctly.

### Interactive speed — MLX backend (Apple Silicon)

| Model | Backend | tok/s | vs SWLP FP16 | Quality |
|-------|---------|------:|:------------:|---------|
| Mistral-7B | SWLP FP16 streaming | 0.50 | 1× | Lossless |
| **Mistral-7B** | **MLX int8** | **16.0** | **32×** | **Byte-identical to FP16** ✅ |
| Mistral-7B | MLX int4 | 27.9 | 56× | Minor wording drift |
| **Qwen2.5-14B** | **MLX int4** | **13.8** | **71×** | Minor wording drift |

### Model scale on 16 GB M5

| Model | Disk size | tok/s | TTFT | Peak RAM | Feasible? |
|-------|----------:|------:|-----:|---------:|:---------:|
| Mistral-7B | 14 GB | 0.42 | 13.6 ms | 1.13 GB | ✅ SWLP |
| Qwen2.5-14B | 26 GB | 0.19 | 24.8 ms | 1.66 GB | ✅ SWLP |
| Mistral-Small-24B | 44 GB | 0.08 | 12.7 s | 3.76 GB | ✅ SWLP |
| Any of the above (full-load) | — | ❌ | ❌ | > RAM | ❌ OOM |

---

## Installation

**Requirements:** Python ≥ 3.11, macOS (Apple Silicon) or Linux (NVIDIA)

```bash
git clone https://github.com/rishavupadhaya/swlp.git
cd swlp

bash scripts/bootstrap.sh    # creates .venv and installs swlp[dev]
source .venv/bin/activate
```

**Optional extras:**

```bash
pip install swlp[apple]      # MLX backend — Apple Silicon native compute
pip install swlp[gpu]        # NVIDIA VRAM tracking (pynvml)
```

---

## Quick Start

```bash
# No model needed — offline smoke test to validate swlp is working or not
swlp --backend mock --prompt "What can you do?"

# Standard HuggingFace inference (model must fit RAM)
swlp --model mistral-7b --prompt "Explain transformers in one sentence."

# Apple Silicon — interactive speed, lossless int8 (~16 tok/s)
swlp --model mistral-7b --backend mlx --quant int8 --prompt "Hello"

# Stream a model bigger than RAM from per-layer shards
swlp --shard-dir ./shards/mistral-7b --window 2 --prompt "Hello"
```

The `--model` flag accepts short aliases (`mistral-7b`, `qwen-14b`, `tiny-gpt2`) or any HuggingFace model id.
Backend is auto-selected: `--quant` → MLX, `--shard-dir` → SWLP streaming, else HuggingFace.

---

## Usage

### Run inference

```bash
swlp --model <name> --prompt "<text>" [options]

Options:
  --backend   hf | mlx | swlp | speculative | mock   (default: auto)
  --quant     bf16 | int8 | int4                      (MLX only; implies --backend mlx)
  --shard-dir <path>     per-layer shard directory    (implies --backend swlp)
  --window    <int>      sliding-window depth          (default: 2)
  --max-tokens <int>     max new tokens                (default: 128)
  --device    cuda | mps | cpu | auto                  (default: auto)
  --json                 print full metrics as JSON
  --config    <file>     optional TOML config file
```

### Choosing a backend

| Backend | Speed | Quality | Best for |
|---------|------:|---------|----------|
| `mlx` | ~16–28 tok/s | int8 lossless / int4 near-lossless | Interactive speed on Apple Silicon |
| `swlp` | 0.1–0.5 tok/s | FP16 exact | Model doesn't fit RAM |
| `speculative` | up to 3.3× over SWLP | FP16 exact | Repetitive / long-context output |
| `hf` | GPU speed | FP16 exact | Small models that fit RAM |
| `mock` | instant | deterministic | Testing, CI, offline dev |

### Shard a model (one-time setup for SWLP streaming)

```bash
# Auto-shards on first run when shard-dir is empty:
swlp --shard-dir ./shards/mistral-7b --model mistral-7b --prompt "Hello"

# Or shard manually ahead of time:
python scripts/shard_mistral.py    # → ./shards/mistral-7b  (~14 GB, ~5 min)
python scripts/shard_qwen.py       # → ./shards/qwen2.5-14b (~26 GB, ~10 min)
```

Sharding streams weights block-by-block — it never loads the full model into RAM.

### Speculative decoding

Proposes up to K tokens per sweep using n-gram matching against the prompt context. Accepted tokens are lossless (byte-identical to greedy output). No draft model needed — zero extra RAM.

```bash
swlp --shard-dir ./shards/mistral-7b --backend speculative --prompt "..."
# or
swlp --config configs/swlp_speculative_mps.toml --prompt "..."
```

Speedup: ~1× on novel text, up to **3.3× on repetitive/long-context output**.

### Benchmarking

```bash
# Timed N-run benchmark with statistics
swlp benchmark --runs 5 --warmup-runs 1 --report

# Sweep prompt sets and window configs side-by-side
swlp suite --suite configs/bench_suite.toml --report

# Physics simulation — no model needed
swlp simulate --scenario configs/sim_m5.toml --report

# Print a previously saved result
swlp report benchmarks/<timestamp>.json
swlp suite-report benchmarks/suite-<timestamp>.json
```

### Batch throughput

SWLP supports batched inference — the per-layer disk cost is flat regardless of batch size, so aggregate throughput scales linearly:

```
Batch 1  → 3.5 tok/s
Batch 4  → 15 tok/s
Batch 16 → 65 tok/s   (~18.5× aggregate, same ~0.28 s/sweep wall time)
```

---

## Configuration

Three ways to configure (in order of precedence):

1. **CLI flags** — `--device mps --window 2`
2. **Environment variables** — `SWLP_DEVICE=mps SWLP_WINDOW_SIZE=2`
3. **TOML config file** — `--config configs/swlp_mps.toml`

### Environment variables

| Variable | Example | Description |
|----------|---------|-------------|
| `SWLP_MODEL_ID` | `mistralai/Mistral-7B-Instruct-v0.2` | HuggingFace model id |
| `SWLP_DEVICE` | `mps` | `cuda` · `mps` · `cpu` · `auto` |
| `SWLP_BACKEND` | `swlp` | `hf` · `mlx` · `swlp` · `speculative` · `mock` |
| `SWLP_WINDOW_SIZE` | `2` | Sliding-window depth |
| `SWLP_SHARD_DIR` | `./shards/mistral-7b` | Per-layer shard directory |
| `SWLP_MLX_QUANT` | `int8` | MLX precision: `bf16` · `int8` · `int4` |
| `SWLP_KV_BUDGET_MB` | `4096` | KV cache RAM budget in MB |
| `SWLP_KV_COMPRESSION` | `true` | Enable zlib KV compression (lossless) |
| `SWLP_KV_QUANT` | `none` | KV quantization: `none` (default) · `int4` (lossy) |
| `SWLP_KV_WINDOW` | `4096` | Keep only last N KV positions (0 = unbounded) |
| `SWLP_SPEC_NGRAM` | `3` | Speculative: n-gram match size |
| `SWLP_SPEC_MAX_DRAFT` | `8` | Speculative: max draft tokens per sweep |
| `SWLP_RESIDENCY` | `auto` | `auto` · `off` · `<integer>` layer count |
| `SWLP_LOG_LEVEL` | `INFO` | `DEBUG` · `INFO` · `WARNING` |
| `SWLP_PROFILE` | `1` | Collect detailed per-layer timings |

### Config profiles

| File | Device | Backend | Use for |
|------|--------|---------|---------|
| `configs/baseline.toml` | auto | hf | Standard HF baseline |
| `configs/swlp_mps.toml` | mps | swlp | M5 SWLP streaming |
| `configs/swlp_mistral_mps.toml` | mps | swlp | Mistral-7B streaming |
| `configs/swlp_qwen_mps.toml` | mps | swlp | Qwen2.5-14B streaming |
| `configs/swlp_qwen32b_mps.toml` | mps | swlp | Qwen2.5-32B streaming |
| `configs/swlp_mlx_mps.toml` | mps | mlx | MLX native backend |
| `configs/swlp_speculative_mps.toml` | mps | speculative | Speculative decoding |
| `configs/sim_m5.toml` | — | — | M5 physics simulation |

---

## How It Works

### SWLP Streaming (lossless FP16)

A transformer computes one layer at a time. SWLP exploits this:

```
NVMe SSD ──[background thread]──▶ CPU RAM window (W layers live)
                                            │
                                    MPS / CUDA compute
                                            │
                                  evict to meta (0 bytes)
                                            │
                                    next layer prefetch fires
```

- A **background prefetch thread** loads layer N+1 while layer N computes — I/O and compute overlap.
- Each layer is evicted to a PyTorch `meta` tensor (zero bytes) immediately after its forward pass.
- At W=2: only `2 × layer_size` RAM is ever live. For Mistral-7B (436 MB/layer) that is **870 MB**.
- Embeddings, layer norms, and the LM head are tiny — kept permanently on-device.

This is why SWLP beats AirLLM: AirLLM serializes load→compute→discard with no overlap. SWLP keeps the SSD pipeline and the GPU both busy simultaneously.

### MLX Backend (interactive speed)

When the model fits in compressed form (int8 ≈ half size, int4 ≈ quarter size), SWLP loads it fully via MLX into Apple's unified memory and runs native quantized matmul — no per-token disk reads.

```
HuggingFace weights ──▶ MLX quantize ──▶ fully resident in unified memory
                                                     │
                                         native int8 / int4 matmul
                                             16–28 tok/s
```

MLX int8 is byte-identical to FP16 on Mistral-7B (measured). It is the default recommended tier.

### Speculative Decoding

Proposes K tokens by matching the trailing n-gram against earlier context (prompt lookup). The streamed model verifies all K in **one** 32-layer disk sweep. Accepted tokens are lossless. Token throughput becomes `(accepted + 1) / one_sweep_cost` — up to **4 tok/sweep** (3.3× speedup) on repetitive output.

---

## Project Structure

```
swlp/
├── src/swlp/
│   ├── cli.py             CLI entry point
│   ├── cli_args.py        Argument parser construction
│   ├── config.py          AppConfig + load_config()
│   ├── metrics.py         RunMetrics + RunResult
│   ├── logging.py         configure_logging()
│   ├── core/
│   │   ├── scheduler.py   SWLPScheduler (CUDA) + ThreadedScheduler (MPS/CPU)
│   │   ├── pipeline.py    ThreadedPipeline — background prefetch
│   │   ├── streaming.py   StreamingScheduler — per-layer shard materialization
│   │   ├── kv_cache.py    KVCacheManager — 4-tier KV storage
│   │   ├── compressed_cache.py  CompressedDynamicCache
│   │   ├── residency.py   plan_residency() — adaptive layer residency
│   │   ├── speculative.py NgramDrafter + verify_greedy()
│   │   └── kv_quant.py    INT4 KV quantization (optional lossy tier)
│   ├── hardware/
│   │   └── detect.py      detect_hardware(), fits_in_memory(), window_size_recommendation()
│   ├── model/
│   │   ├── shard.py       shard_model_by_layer() — streaming shard writer
│   │   ├── package.py     SWLP package format (safetensors)
│   │   ├── quant.py       FP8 weight quantization
│   │   └── sparse.py      COO sparse weight codec
│   ├── runner/
│   │   ├── base.py        build_runner() factory
│   │   ├── hf.py          HuggingFaceRunner
│   │   ├── swlp.py        SWLPRunner (sliding-window streaming)
│   │   ├── mlx.py         MlxRunner (native Apple Silicon)
│   │   ├── speculative.py SpeculativeRunner
│   │   ├── batch.py       run_batch() — batched column-wise execution
│   │   ├── arch.py        GPT2 / Llama / Mistral dispatch
│   │   ├── load.py        full-model + shard loading
│   │   └── mock.py        MockRunner (deterministic, offline)
│   ├── benchmark/
│   │   ├── run.py         run_benchmark()
│   │   ├── suite.py       run_suite()
│   │   └── simulator.py   simulate_scenario() — pure-Python bottleneck math
│   └── reporting/
│       ├── run_report.py
│       ├── sim_report.py
│       └── suite_report.py
│
├── configs/               TOML config profiles
├── scripts/               Diagnostic and sharding utilities
├── tests/                 pytest suite (154 tests)
└── docs/                  Benchmark results and design docs
```

---

## Development

```bash
# Run all tests (no model download required)
pytest

# Single file
pytest tests/test_simulator.py

# Single test
pytest -k test_plan_residency

# Lint
ruff check src/

# Auto-fix lint issues
ruff check --fix src/

# Hardware baseline (run once per machine)
python scripts/phase0_hardware_check.py
```

All unit tests use `MockRunner` or pure-math functions — no GPU or model download needed. The full suite of 154 tests completes in seconds.

---

## Supported Models

Any HuggingFace Llama / Mistral family model works with the SWLP streaming backend. Tested:

| Model | Alias | Size | SWLP | MLX |
|-------|-------|-----:|:----:|:---:|
| `unsloth/mistral-7b-instruct-v0.2` | `mistral-7b` | 14 GB | ✅ | ✅ |
| `mistralai/Mistral-Small-24B-Instruct-2501` | — | 44 GB | ✅ | — |
| `Qwen/Qwen2.5-14B-Instruct` | `qwen-14b` | 26 GB | ✅ | ✅ |
| `Qwen/Qwen2.5-32B-Instruct` | — | ~60 GB | ✅ | — |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | — | 720 MB | ✅ | — |
| `openai-community/gpt2` | `tiny-gpt2` | 548 MB | ✅ | — |

---

## Limitations

- **Throughput ceiling:** SWLP FP16 is bounded by `SSD_bandwidth / model_bytes_per_token`. On M5 (6.93 GB/s) with Mistral-7B this is ~0.50 tok/s. Use `--backend mlx` for interactive speed.
- **MLX is Apple Silicon only:** `--backend mlx` requires macOS + M-series chip.
- **Speculative decoding speedup is workload-dependent:** 1× on novel text, up to 3.3× on repetitive output.
- **NVIDIA path:** CUDA async-PCIe streaming is implemented but not yet measured — CUDA hardware benchmarks pending.
- **int4 KV quantization is lossy:** `SWLP_KV_QUANT=int4` trades ~0.5–2% quality for 4× smaller KV. Off by default; always labelled.

---

## Documentation

| Doc | Description |
|-----|-------------|
| [`docs/results.md`](docs/results.md) | Full benchmark tables across all phases |
| [`docs/swlp_vs_airllm.md`](docs/swlp_vs_airllm.md) | SWLP vs AirLLM rigorous comparison |
| [`docs/hardware_baseline.md`](docs/hardware_baseline.md) | M5 measured SSD/MPS/RAM numbers |
| [`docs/phase5_design_decisions.md`](docs/phase5_design_decisions.md) | Speculative decoding design rationale |
| [`docs/model_packaging.md`](docs/model_packaging.md) | SWLP package format spec |
| [`.claude/CLAUDE.md`](.claude/CLAUDE.md) | Full architecture + 18-phase build history |

---

## License

MIT © 2026 Rishav Upadhaya
