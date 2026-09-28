# SWLP — Sliding Window Layer Pipeline

[![CI](https://github.com/Rishav-Upadhaya/SWLP/actions/workflows/ci.yml/badge.svg)](https://github.com/Rishav-Upadhaya/SWLP/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> **Run a 26 GB model on a 16 GB machine. No quantization. 1.7 GB peak RAM.**

That's not a typo. SWLP streams transformer layers one at a time — SSD → RAM → compute → evict. Only *W* layers ever live in memory. The rest stay on disk. A 26 GB model needs only **1.7 GB RAM**. A 44 GB model needs **3.8 GB**. The weights are never quantized. Every parameter stays full FP16.

**Who this is for:** ML engineers and researchers who need exact FP16 inference — for evaluation, paper reproducibility, or benchmarking — on a machine where the model literally doesn't fit.

For **interactive speed** on Apple Silicon (when the model fits compressed), SWLP also ships a native MLX backend: **16 tok/s lossless int8, up to 28 tok/s with int4** — no streaming required.

---

## Two Modes, One Tool

| | FP16 Streaming | MLX Interactive |
|---|---|---|
| **When to use** | Model exceeds your RAM, even quantized | Apple Silicon, need conversational speed |
| **Speed** | ~0.17–0.21 tok/s | 16–28 tok/s |
| **Quality** | Exact FP16 — zero compromise | int8: byte-identical to FP16 |
| **RAM needed** | ~2 × layer size (e.g. 1.7 GB for a 26 GB model; 3.8 GB for a 44 GB model) | ~half the model size |
| **How to invoke** | `--shard-dir ./shards/model` | `--backend mlx --quant int8` |

---

## Start Here

1. [Install SWLP](#installation).
2. One command — download, shard, and run: `swlp run mistral-7b --prompt "Hello"`.
   Already-sharded models are detected and reused. `swlp pull mistral-7b`
   does the same without running.
3. Or serve it OpenAI-style: `swlp serve mistral-7b` → `http://127.0.0.1:8080/v1/chat/completions`.
4. Run `swlp doctor` to get the recommended command for your hardware.

## Contents

- [Benchmarks](#benchmarks)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Supported Models](#supported-models)
- [Development](#development)
- [Limitations](#limitations)
- [Platform Support](#platform-support)
- [Documentation](#documentation)
- [Support and Contributing](#support-and-contributing)
- [Citation](#citation)
- [License](#license)

## Benchmarks

Measured on **Apple M5, 16 GB unified memory**, greedy decoding, Mistral-7B FP16.

### How SWLP compares to the tools you already know

| Tool | tok/s | Quality | Notes |
|------|------:|---------|-------|
| Ollama (Q4_K_M) | 28 | 4-bit quantized | Fastest — but not FP16 |
| **SWLP MLX int8** | **16** | **Byte-identical to FP16 ✅** | Native quantized matmul on Apple Silicon |
| SWLP MLX int4 | 28 | Near-lossless | Faster; minor wording drift |
| **SWLP FP16 streaming** | **~0.17–0.21** † | **Exact FP16 ✅** | Prefetch overlaps disk I/O with compute; bounded peak RAM |
| AirLLM FP16 streaming | ~0.085 † | Exact FP16 | **~2.5× slower**; more resident RAM |
| Full FP16 load (HF / MLX naive) | ❌ OOM | — | 14 GB model doesn't fit 16 GB |

SWLP MLX int8 is byte-identical to FP16 on Mistral-7B (measured). Ollama Q4_K_M is 4-bit — a different quality tier. For FP16 models that exceed RAM, the alternatives are layer streaming (AirLLM), disk offload (HF Accelerate `device_map="auto"` + `offload_folder`), and mmap-backed loading (llama.cpp with an F16 GGUF, paged through the OS cache). SWLP's contribution is an explicit sliding window with prefetch, giving a bounded, predictable peak RSS; a head-to-head against llama.cpp F16 + mmap and Accelerate offload is on the roadmap.

> † FP16-streaming figures are **warm-cache medians** (M5, Mistral-7B, W=2, greedy, airllm 2.11.0), reproduced 2026-05-29. The physics ceiling is `SSD_bw / model_bytes` = **0.496 tok/s** (cold). The authoritative cold-SSD median is **0.174 tok/s** (Mistral-7B, 35% of the ceiling); the ~0.21 above is the warm-cache upper bound, and AirLLM's ~0.085 is its warm-cache median — see [`docs/benchmark_methodology.md`](docs/benchmark_methodology.md). Earlier drafts cited 0.42/0.50 tok/s for SWLP; those were single-run / warm-cache artifacts and are superseded.
>
> **Phase 20 update (2026-06-11):** a hot-path overhaul (single-read buffers, zero-copy shard parsing, worker-side device transfer) raised Mistral-7B direct-I/O streaming from **0.218 → 0.370 tok/s (+70%, 75% of the cold ceiling)** with byte-identical output; prefill is 2.5× faster. Formal multi-run cold/warm medians are being re-measured — see `docs/results.md` Phase 20.
>
> **Phase 21 update (2026-06-11):** draft-model speculative decoding (`--draft-model qwen-0.5b`): a ~1 GB resident drafter proposes tokens that the streamed target verifies in one disk sweep, with draft length adapting to acceptance. Qwen2.5-14B (28 GB FP16) went from 0.19 to **0.55–1.16 tok/s on a 16 GB M5 (2.9×–5.9×, workload-dependent)** — still byte-identical to plain greedy decoding. See `docs/results.md` Phase 21.

### Models larger than RAM, on a 16 GB machine

| Model | Disk size | tok/s † | Peak RAM | Full FP16 load? |
|-------|----------:|------:|--------:|:--------------------------:|
| Mistral-7B | 14 GB | ~0.21 | 1.1 GB | ❌ OOM on naive full load |
| Qwen2.5-14B | 26 GB | 0.19 | 1.7 GB | ❌ OOM on naive full load |
| Mistral-Small-24B | 44 GB | 0.08 | 3.8 GB | ❌ OOM on naive full load |
| Qwen2.5-32B | ~60 GB | — | — | ❌ OOM on naive full load |

The first three rows are not quantized — actual FP16 weights, measured on an M5 16 GB machine (warm-cache; see the methodology note above). Qwen2.5-32B is architecture-verified but per-token measurements are pending (model download required). **The point of these rows is *feasibility*, not speed** — none of these fit in 16 GB as a full FP16 load, and SWLP runs them with a bounded peak RAM; the peak-RAM column is the headline, not tok/s.

### Batch throughput — the scaling principle

SWLP's per-layer disk cost is **flat regardless of batch size** — load once, run N sequences through the same weights. Measured on SmolLM2-360M FP16 shards (M5, W=2):

| Batch size | Aggregate tok/s | Sweep wall time |
|-----------:|----------------:|----------------:|
| 1 | 3.5 | ~0.28 s |
| 4 | 15 | ~0.28 s |
| 8 | 24 | ~0.28 s |
| **16** | **65** | **~0.28 s** |

The sweep wall time is flat — batch-independent — so aggregate throughput scales ~linearly (~18.5× at batch 16). This principle is architecture-independent; absolute tok/s will differ for larger models (7B, 14B) proportional to per-layer compute. 7B/14B batch numbers are pending model re-sharding.

---

## Installation

**Requirements:** Python ≥ 3.11. The mock backend needs no model or accelerator.
Real-model commands download checkpoints from Hugging Face, so they also need network access and
free disk space at least equal to the model size shown in [Supported Models](#supported-models).

### uv (recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/Rishav-Upadhaya/SWLP.git
cd SWLP

uv sync --extra dev                              # core + dev tools
# uv sync --extra dev --extra apple             # + MLX backend (Apple Silicon)
# uv sync --extra dev --extra gpu               # + NVIDIA VRAM tracking

source .venv/bin/activate
```

### pip

```bash
git clone https://github.com/Rishav-Upadhaya/SWLP.git
cd SWLP

python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

pip install -e ".[dev]"
# pip install -e ".[dev,apple]"    # + MLX (Apple Silicon)
# pip install -e ".[dev,gpu]"      # + NVIDIA VRAM tracking
```

### requirements.txt

```bash
pip install -r requirements.txt
```

The file includes commented optional sections — just uncomment the lines you need:

```
# ── if: Apple Silicon MLX backend ────────────────────────────────────────
# Uncomment if you are on Apple Silicon and want --backend mlx (~16 tok/s)
# mlx==0.31.2
# mlx-lm==0.31.3

# ── if: NVIDIA VRAM tracking ─────────────────────────────────────────────
# Uncomment if you have an NVIDIA GPU and want VRAM usage reported
# pynvml>=11.5
```

### Verify

```bash
swlp --backend mock --prompt "Hello, does SWLP work?"   # no model or GPU needed
swlp doctor                                              # what can THIS machine run?
python scripts/phase0_hardware_check.py                  # SSD bandwidth + hardware check
```

`swlp doctor` detects your chip, RAM, and MLX availability, then prints the exact command to use for each supported model on your machine — start there if you're unsure which mode you need.

---

## Quick Start

```bash
# ① Smoke test — no model, no GPU needed
swlp --backend mock --prompt "What can you do?"

# ② Apple Silicon — interactive speed, lossless int8 (~16 tok/s)
swlp --model mistral-7b --backend mlx --quant int8 --prompt "Explain transformers."

# ③ Stream a model bigger than your RAM (auto-shards on first run)
swlp --shard-dir ./shards/mistral-7b --model mistral-7b --prompt "Explain transformers."

# ④ Standard inference (model must fit RAM)
swlp --model mistral-7b --prompt "Explain transformers."
```

`--model` accepts short aliases (`mistral-7b`, `qwen-14b`, `tiny-gpt2`) or any HuggingFace model ID.  
Backend is auto-selected: `--quant` → MLX, `--shard-dir` → SWLP streaming, else HuggingFace.
Steps 2–4 download their model checkpoint; the mock command remains the fastest installation check.

---

## Usage

Run `swlp --help` for the complete CLI reference, `swlp chat --help` for interactive chat, and
`swlp doctor` for a hardware-specific recommendation.

### Inference flags

```bash
swlp --model <name> --prompt "<text>" [options]

  --backend    hf | mlx | swlp | speculative | mock   (default: auto)
  --quant      bf16 | int8 | int4                      (MLX only)
  --shard-dir  <path>    per-layer shard directory     (SWLP streaming)
  --window     <int>     sliding-window depth           (default: 2)
  --max-tokens <int>     max new tokens                 (default: 128)
  --device     cuda | mps | cpu | auto
  --json                 print full metrics as JSON
  --config     <file>    optional TOML config file
```

### Which backend to use

Not sure? Run `swlp doctor` — it answers this table for your actual hardware. `swlp models` lists all aliases with sizes.

| Your situation | Use |
|----------------|-----|
| Apple Silicon, model fits when int8 quantized | `--backend mlx --quant int8` |
| Model exceeds your RAM — need exact FP16 | `--shard-dir ./shards/model` |
| NVIDIA GPU, model fits VRAM | `--backend hf` |
| Repetitive or long-context output + SWLP | `--backend speculative` |
| CI / offline / no model | `--backend mock` |

### Streaming setup (one-time per model)

SWLP auto-shards on first run — just point at an empty directory:

```bash
swlp --shard-dir ./shards/mistral-7b --model mistral-7b --prompt "Hello"
```

Or shard manually ahead of time:

```bash
swlp download --model mistral-7b   # → ./shards/mistral-7b (~14 GB)
swlp download --model qwen-14b     # → ./shards/qwen-14b   (~26 GB)
```

Sharding streams weights block-by-block — the full model is never loaded into RAM.

### Speculative decoding

Proposes up to K tokens per sweep via n-gram matching against the prompt. No draft model, no extra RAM. Accepted tokens are byte-identical to greedy output.

```bash
swlp --shard-dir ./shards/mistral-7b --backend speculative --prompt "..."
```

Speedup: ~1× on novel text, up to **3.3× on repetitive / long-context output**.

### Benchmarking

```bash
swlp benchmark --runs 5 --warmup-runs 1 --report
swlp suite     --suite configs/bench_suite.toml --report
swlp simulate  --scenario configs/sim_m5.toml --report
swlp report    benchmarks/<timestamp>.json
```

---

## How It Works

### FP16 Streaming

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

- A **background thread prefetches** layer N+1 while layer N computes — I/O and compute overlap.
- Each layer is **evicted to a `meta` tensor** (zero bytes) immediately after its forward pass.
- At W=2: `2 × layer_size` RAM ever live. Mistral-7B (436 MB/layer) → **870 MB**.
- Embeddings, layer norms, and the LM head are tiny — kept permanently on-device.

**Why SWLP beats AirLLM 2.11.0:** SWLP uses ~1.5× less RAM (immediate per-block eviction) and runs models with an explicit `head_dim` that crash AirLLM (e.g. Mistral-Small-24B). We pin AirLLM 2.11.0 and run its MLX class (`AirLLMLlamaMlx`); we do **not** attribute the throughput gap to prefetch, since that path was not profiled for I/O overlap (see paper §7.4).

**Why batch throughput scales linearly:** The per-layer disk cost is the same whether 1 or 16 sequences pass through it. Load once, run N — sweep wall time stays flat (~0.28 s), aggregate tok/s scales with N.

### MLX Backend

When the model fits compressed (int8 ≈ half size, int4 ≈ quarter size), SWLP loads it fully into Apple unified memory via MLX and runs native quantized matmul — no per-token disk reads.

```
HuggingFace weights ──▶ MLX quantize ──▶ resident in unified memory ──▶ 16–28 tok/s
```

MLX int8 is byte-identical to FP16 on Mistral-7B (measured). It is the default recommended tier.

### Speculative Decoding

Proposes K tokens by matching the trailing n-gram against earlier context. The streamed model verifies all K in **one** disk sweep. Accepted tokens are lossless. Throughput becomes `(accepted + 1) / one_sweep_cost` — up to **4 tok/sweep** on repetitive output.

---

## Configuration

Three ways, in order of precedence:

1. **CLI flags** — `--window 2 --device mps`
2. **Environment variables** — `SWLP_WINDOW_SIZE=2 SWLP_DEVICE=mps`
3. **TOML config file** — `--config configs/swlp_mps.toml`

### Key environment variables

| Variable | Example | Description |
|----------|---------|-------------|
| `SWLP_BACKEND` | `swlp` | `hf` · `mlx` · `swlp` · `speculative` · `mock` |
| `SWLP_MODEL_ID` | `mistralai/Mistral-7B-Instruct-v0.2` | HuggingFace model ID |
| `SWLP_SHARD_DIR` | `./shards/mistral-7b` | Per-layer shard directory |
| `SWLP_WINDOW_SIZE` | `2` | Sliding-window depth |
| `SWLP_DEVICE` | `mps` | `cuda` · `mps` · `cpu` · `auto` |
| `SWLP_MLX_QUANT` | `int8` | MLX precision: `bf16` · `int8` · `int4` |
| `SWLP_KV_BUDGET_MB` | `4096` | KV cache RAM budget in MB |
| `SWLP_KV_QUANT` | `none` | KV quantization: `none` (default) · `int4` (lossy) |
| `SWLP_KV_WINDOW` | `4096` | Keep only last N KV positions (0 = unbounded) |
| `SWLP_SPEC_MAX_DRAFT` | `8` | Speculative: max draft tokens per sweep |
| `SWLP_RESIDENCY` | `auto` | `auto` · `off` · `<integer>` layer count |
| `SWLP_SHARD_VOLUMES` | — | Comma-separated extra shard dirs; layers stripe across SSDs |
| `SWLP_PREFILL_CHUNK` | `0` | Prompt tokens per prefill sweep slice (0 = single sweep) |
| `SWLP_EXPERT_CACHE_MB` | `0` | MoE expert-cache RAM budget (0 = auto) |
| `SWLP_EXPERT_PREFETCH` | `predictive` | MoE expert prefetch: `off` · `router` · `predictive` |
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

## Supported Models

Any HuggingFace Llama / Mistral family model works with SWLP streaming. Tested:

| Model | Alias | Disk size | SWLP | MLX |
|-------|-------|----------:|:----:|:---:|
| `unsloth/mistral-7b-instruct-v0.2` | `mistral-7b` | 14 GB | ✅ | ✅ |
| `Qwen/Qwen2.5-14B-Instruct` | `qwen-14b` | 26 GB | ✅ | ✅ |
| `mistralai/Mistral-Small-24B-Instruct-2501` | — | 44 GB | ✅ | — |
| `Qwen/Qwen2.5-32B-Instruct` | — | ~60 GB | ✅ | — |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | `qwen3-30b-a3b` | 61 GB | ✅ MoE | — |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | `mixtral-8x7b` | 93 GB | ✅ MoE | — |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | — | 720 MB | ✅ | — |
| `openai-community/gpt2` | `tiny-gpt2` | 548 MB | ✅ | — |

**MoE models (Phase 25)** stream *selectively*: each sweep reads the dense
parts plus only the routed experts the tokens activate, so per-token bytes
scale with active parameters (Qwen3-30B-A3B: 3.4B of 30B) while disk holds the
full model. Quality is exact — routing is part of the forward pass. Tune the
expert cache with `SWLP_EXPERT_CACHE_MB`; see `swlp doctor` for guidance and
`scripts/research/moe_sweep.py` for budget sweeps.

---

## Development

```bash
pytest                             # full suite — no GPU or model download needed
pytest tests/test_simulator.py    # single file
pytest -k test_plan_residency     # single test by name
ruff check src/                   # lint
ruff check --fix src/             # lint + autofix
python scripts/phase0_hardware_check.py   # SSD bandwidth + hardware baseline
```

All tests use `MockRunner` or pure-math functions. The full suite completes in seconds.

---

## Limitations

- **FP16 throughput ceiling:** `tok/s ≤ SSD_bandwidth / model_bytes_per_token`. On M5 (6.93 GB/s) with Mistral-7B this caps at **0.496 tok/s** (measured cold throughput ~0.17 tok/s). Use `--backend mlx` when you need speed and the model fits.
- **Streaming is a feasibility tool, not a speed tool.** If your model fits RAM, `--backend hf` or `--backend mlx` will be faster. SWLP streaming wins only when those options OOM.
- **MLX is Apple Silicon only.** `--backend mlx` requires macOS + M-series chip.
- **Speculative speedup is workload-dependent.** ~1× on novel text, up to 3.3× on repetitive output.
- **NVIDIA path is built but not yet benchmarked.** CUDA async-PCIe streaming is wired; hardware numbers pending.
- **INT4 KV quantization is lossy.** `SWLP_KV_QUANT=int4` trades ~0.5–2% quality for 4× smaller KV cache. Off by default; always labelled.

---

## Platform Support

| Platform | Hardware | Status |
|----------|----------|--------|
| macOS (Apple Silicon) | M1 / M2 / M3 / M5 | ✅ Fully tested — streaming, MLX int8/int4, speculative |
| Linux (NVIDIA GPU) | MX230, RTX series | 🔧 CUDA path built; hardware benchmarks pending |
| Linux (CPU only) | Any | 🧪 Streaming works; MLX backend unavailable |
| Windows | Any | ❌ Untested — no known blockers |

Developed and benchmarked on **macOS M5 (16 GB)**. If you run it on Linux/NVIDIA and collect numbers, please open a PR — the CUDA path is ready.

---

## Documentation

| Doc | Description |
|-----|-------------|
| [`docs/results.md`](docs/results.md) | Full benchmark tables across all phases |
| [`docs/benchmark_methodology.md`](docs/benchmark_methodology.md) | How benchmarks are run: cold-vs-warm cache protocol, provenance, fair head-to-head rules |
| [`docs/swlp_vs_airllm.md`](docs/swlp_vs_airllm.md) | Detailed SWLP vs AirLLM comparison |
| [`docs/hardware_baseline.md`](docs/hardware_baseline.md) | M5 measured SSD / MPS / RAM numbers |
| [`docs/architecture.md`](docs/architecture.md) | System architecture and pipeline analysis |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Phase history and open research work |
| [`docs/phase5_design_decisions.md`](docs/phase5_design_decisions.md) | Speculative decoding design rationale |
| [`CHANGELOG.md`](CHANGELOG.md) | Release history and notable changes |

---

## Support and Contributing

- For setup questions, bugs, or feature requests, [search or open an issue](https://github.com/Rishav-Upadhaya/SWLP/issues).
- For code, documentation, or hardware-benchmark contributions, follow the [contribution guide](CONTRIBUTING.md).

---

## Citation

If SWLP contributes to your research, please cite it using [CITATION.cff](CITATION.cff).

---

## License

MIT © 2026 Rishav Upadhaya
