# SWLP — Sliding Window Layer Pipeline

> **Run a 26 GB model on a 16 GB machine. No quantization. 1.7 GB peak RAM.**

That's not a typo. SWLP streams transformer layers one at a time — SSD → RAM → compute → evict. Only *W* layers ever live in memory. The rest stay on disk. A 26 GB model needs only **1.7 GB RAM**. A 44 GB model needs **3.8 GB**. The weights are never quantized. Every parameter stays full FP16.

**Who this is for:** ML engineers and researchers who need exact FP16 inference — for evaluation, paper reproducibility, or benchmarking — on a machine where the model literally doesn't fit.

For **interactive speed** on Apple Silicon (when the model fits compressed), SWLP also ships a native MLX backend: **16 tok/s lossless int8, up to 28 tok/s with int4** — no streaming required.

---

## Two Modes, One Tool

| | FP16 Streaming | MLX Interactive |
|---|---|---|
| **When to use** | Model exceeds your RAM, even quantized | Apple Silicon, need conversational speed |
| **Speed** | 0.1–0.5 tok/s | 16–28 tok/s |
| **Quality** | Exact FP16 — zero compromise | int8: byte-identical to FP16 |
| **RAM needed** | ~2 × layer size (e.g. 1.7 GB for a 26 GB model; 3.8 GB for a 44 GB model) | ~half the model size |
| **How to invoke** | `--shard-dir ./shards/model` | `--backend mlx --quant int8` |

---

## Benchmarks

Measured on **Apple M5, 16 GB unified memory**, greedy decoding, Mistral-7B FP16.

### How SWLP compares to the tools you already know

| Tool | tok/s | Quality | Notes |
|------|------:|---------|-------|
| Ollama (Q4_K_M) | 28 | 4-bit quantized | Fastest — but not FP16 |
| **SWLP MLX int8** | **16** | **Byte-identical to FP16 ✅** | Native quantized matmul on Apple Silicon |
| SWLP MLX int4 | 28 | Near-lossless | Faster; minor wording drift |
| **SWLP FP16 streaming** | **0.50** | **Exact FP16 ✅** | The only option when model exceeds RAM |
| AirLLM FP16 streaming | 0.21 | Exact FP16 | No prefetch — I/O blocks compute |
| Full FP16 load (HF / MLX naive) | ❌ OOM | — | 14 GB model doesn't fit 16 GB |

SWLP MLX int8 is byte-identical to FP16 on Mistral-7B (measured). Ollama Q4_K_M is 4-bit — a different quality tier. For models that don't fit at all, **SWLP streaming is the only viable FP16 option**.

### The models nobody else can run on a 16 GB machine

| Model | Disk size | tok/s | Peak RAM | Feasible with other tools? |
|-------|----------:|------:|--------:|:--------------------------:|
| Mistral-7B | 14 GB | 0.50 | 1.1 GB | ❌ FP16 OOM everywhere |
| Qwen2.5-14B | 26 GB | 0.19 | 1.7 GB | ❌ FP16 OOM everywhere |
| Mistral-Small-24B | 44 GB | 0.08 | 3.8 GB | ❌ FP16 OOM everywhere |
| Qwen2.5-32B | ~60 GB | — | — | ❌ FP16 OOM everywhere |

The first three rows are not quantized — actual FP16 weights, measured on an M5 16 GB machine. Qwen2.5-32B is architecture-verified but per-token measurements are pending (model download required).

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

**Requirements:** Python ≥ 3.11

### uv (recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/rishavupadhaya/swlp.git
cd swlp

uv sync --extra dev                              # core + dev tools
# uv sync --extra dev --extra apple             # + MLX backend (Apple Silicon)
# uv sync --extra dev --extra gpu               # + NVIDIA VRAM tracking

source .venv/bin/activate
```

### pip

```bash
git clone https://github.com/rishavupadhaya/swlp.git
cd swlp

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
python scripts/phase0_hardware_check.py                  # SSD bandwidth + hardware check
```

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

---

## Usage

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
python scripts/shard_mistral.py    # → ./shards/mistral-7b  (~14 GB, ~5 min)
python scripts/shard_qwen.py       # → ./shards/qwen2.5-14b (~26 GB, ~10 min)
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

**Why SWLP is 2× faster than AirLLM:** AirLLM serializes load → compute → discard with no overlap. SWLP keeps the SSD pipeline and the compute unit busy simultaneously.

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
| `HuggingFaceTB/SmolLM2-360M-Instruct` | — | 720 MB | ✅ | — |
| `openai-community/gpt2` | `tiny-gpt2` | 548 MB | ✅ | — |

---

## Development

```bash
pytest                             # all 154 tests — no GPU or model download needed
pytest tests/test_simulator.py    # single file
pytest -k test_plan_residency     # single test by name
ruff check src/                   # lint
ruff check --fix src/             # lint + autofix
python scripts/phase0_hardware_check.py   # SSD bandwidth + hardware baseline
```

All tests use `MockRunner` or pure-math functions. The full suite completes in seconds.

---

## Limitations

- **FP16 throughput ceiling:** `tok/s ≤ SSD_bandwidth / model_bytes_per_token`. On M5 (6.93 GB/s) with Mistral-7B this caps at ~0.50 tok/s. Use `--backend mlx` when you need speed and the model fits.
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
| [`docs/swlp_vs_airllm.md`](docs/swlp_vs_airllm.md) | Detailed SWLP vs AirLLM comparison |
| [`docs/hardware_baseline.md`](docs/hardware_baseline.md) | M5 measured SSD / MPS / RAM numbers |
| [`docs/phase5_design_decisions.md`](docs/phase5_design_decisions.md) | Speculative decoding design rationale |

---

## License

MIT © 2026 Rishav Upadhaya
