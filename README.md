# SWLP: Sliding Window Layer Pipeline

Run LLMs larger than your RAM on Apple Silicon, at full FP16/BF16 precision, by streaming transformer layers from SSD.

[![PyPI](https://img.shields.io/pypi/v/swlp.svg)](https://pypi.org/project/swlp/)
[![Python](https://img.shields.io/pypi/pyversions/swlp.svg)](https://pypi.org/project/swlp/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/Rishav-Upadhaya/SWLP/blob/main/LICENSE)
[![CI](https://github.com/Rishav-Upadhaya/SWLP/actions/workflows/ci.yml/badge.svg)](https://github.com/Rishav-Upadhaya/SWLP/actions/workflows/ci.yml)

## What it is

A transformer computes one layer at a time, so only a small window of layers ever needs to be in memory. SWLP streams each layer from NVMe SSD into unified memory through a sliding window of W layers, prefetching the next layers while the current one computes, and evicts them afterwards. The weights are never quantized, pruned, or distilled: streamed output is bit-identical to a full-precision greedy run, and every speedup (prefetch, speculative decoding, prefix-KV reuse, expert caching) preserves that. SWLP targets Apple Silicon only.

## Headline results

Apple M5, 16 GB unified memory, greedy decoding. Full tables, run counts and methodology: [docs/results.md](https://github.com/Rishav-Upadhaya/SWLP/blob/main/docs/results.md).

| Model (size on disk) | Mode | tok/s | Peak RAM | Output |
|---|---|---:|---:|---|
| Mistral-7B FP16 (14 GB) | `swlp` streaming, W=2, direct I/O | 0.37 | 1.2 GB | exact FP16 |
| Qwen2.5-14B FP16 (26 GB) | `swlp` streaming, W=2 | 0.19 | 1.7 GB | exact FP16 |
| Qwen2.5-14B FP16 (26 GB) | `speculative`, Qwen2.5-0.5B draft | 0.55-1.16 | - | identical to greedy |
| Qwen3.8-27B hybrid (49 GB) | `speculative`, native MTP head | 0.50 | - | identical to greedy |
| Qwen3.6-35B-A3B MoE BF16 (66 GB) | `mlx-moe` expert streaming | 4.4-5.0 | - | exact BF16 |
| Mistral-7B | `mlx`, int8 (opt-in quantized tier) | 16.0 | - | matched FP16 on the test prompt |

None of the streamed models fit in 16 GB as a full-precision load. Streaming numbers are development runs (single or median-of-3); the draft-model speedup depends on acceptance rate (2.9x on open-ended text, 5.9x on constrained output). Against AirLLM on identical FP16 weights, SWLP measured about 2x faster (warm cache); see [docs/benchmarking.md](https://github.com/Rishav-Upadhaya/SWLP/blob/main/docs/benchmarking.md) for cold/warm protocol.

## Requirements

- macOS on an Apple Silicon (M-series) Mac. There is no CUDA or Linux GPU path.
- Python 3.11 or newer.
- A fast internal NVMe SSD: streaming throughput is bounded by `SSD bandwidth / bytes read per token`.
- Free disk space at least equal to the model's FP16 size (`swlp pull` checks this first). RAM needs are roughly two layers plus embeddings and KV cache.

## Install

```bash
pip install "swlp[apple]"      # recommended: adds MLX (mlx, mlx-lm) for the mlx and mlx-moe backends
pip install swlp               # streaming (swlp/speculative), hf and mock backends only
pip install "swlp[codec]"      # optional: lossless .swz shard codec (zipnn)
```

From source:

```bash
git clone https://github.com/Rishav-Upadhaya/SWLP.git && cd SWLP
pip install -e ".[dev]"        # add ,apple or ,codec as needed
```

## Quickstart

```bash
swlp doctor                     # this Mac: chip, RAM, SSD — and what it can run, with the command
swlp chat gemma4-26b            # 26B MoE on 16 GB (~14 tok/s): 4-bit experts streamed from SSD
swlp chat qwen-7b -q int4       # fits in RAM → resident on MLX, fastest
swlp pull qwen3.6-35b           # bigger than RAM → download + shard once (lossless bf16)
swlp chat qwen3.6-35b
```

Eight commands:

```bash
swlp chat MODEL                 # talk to a model  (/help /clear /think /stats /exit)
swlp run MODEL "prompt"         # one answer and exit  (--json for full metrics; "-" reads stdin)
swlp serve MODEL                # OpenAI-compatible API on http://127.0.0.1:8080/v1
swlp pull MODEL                 # download + prepare for streaming
swlp models                     # installed models and how each runs here; the aliases
swlp rm MODEL                   # delete a model: shards, downloads, converted copies (asks first)
swlp doctor [MODEL]             # machine check + what you can run
swlp bench MODEL                # measure tok/s (median of --runs)
```

You name a model (an alias from `swlp models`, a HuggingFace id, or a local
directory); SWLP picks the backend: MoE expert streaming, lossless layer
streaming (with MTP self-drafting when the checkpoint has an MTP head), or
resident MLX with `-q int4|int8|bf16` (on pulled MoE shards, `-q` quantizes
on load: lossy, ~3.5x more experts cached). Answers run until the model
finishes (`-n N` caps them); Ctrl+C stops an answer, and at the prompt exits. `--backend NAME` overrides it; advanced
tuning lives in `SWLP_*` environment variables or a `--config` TOML.
`swlp chat MODEL -d` (or a bare `--backend`) shows which backends can run that
model and each backend's settings with their current values;
`swlp models -d` shows every installed model's details.

## Python API

```python
from swlp import build_runner
from swlp.config import load_config

config = load_config()                  # defaults, optional TOML, then SWLP_* env vars
config.runtime.backend = "mock"         # or "swlp" with config.runtime.shard_dir = Path("shards/mistral-7b")
config.generation.max_new_tokens = 32

runner = build_runner(config)
result = runner.run("Explain sliding-window layer streaming.")
print(result.completion)
print(result.metrics.throughput_tokens_per_second, result.metrics.ram_peak_bytes)
```

Every backend returns the same `RunResult` (`prompt`, `completion`, `metrics`).

## Supported models and backends

`swlp models` lists the built-in aliases; any Hugging Face model ID also works.

| Family | Aliases |
|---|---|
| Qwen2.5 | `qwen-0.5b`, `qwen-1.5b`, `qwen-3b`, `qwen-7b`, `qwen-14b` |
| Mistral | `mistral-7b`, `mistral-24b` |
| Others | `phi-3.5`, `smollm-360m`, `smollm-1.7b`, `tiny-gpt2` |
| MoE | `qwen3-30b-a3b`, `mixtral-8x7b`, `deepseek-v4-flash` |

Streaming supports Llama/Mistral/Qwen-style decoders, GPT-2, Qwen3.5-style hybrid (Gated DeltaNet) models, and MoE models (only routed experts are read per token).

| Backend | What it does |
|---|---|
| `swlp` | FP16/BF16 layer streaming from a shard directory |
| `speculative` | `swlp` plus draft verification in one disk sweep (n-gram, `--draft-model`, or `--mtp`); lossless |
| `mlx` | Model fully resident in MLX (`--quant bf16`, `int8`, `int4`) for interactive speed when it fits |
| `mlx-moe` | MoE on MLX: dense weights resident, experts streamed through an LFU cache |
| `hf` | Plain Hugging Face `transformers` full load (model must fit in RAM) |
| `mock` | Deterministic offline responses for tests and CI |

## How it works

```
NVMe SSD --[prefetch workers]--> unified RAM window (W layers) --> MPS compute
    ^                                                                  |
    +------------- evict layer, prefetch layer i+W <-------------------+

Always resident: embeddings, final norm, LM head, KV cache
```

Per-layer shards are read into reusable buffers by a worker pool and exposed as zero-copy tensors; the compute thread only swaps pointers. With W=2 about two layers are live at a time, so Mistral-7B streams in ~1.2 GB. Speculative decoding amortises each full disk sweep over several verified tokens. Details: [docs/architecture.md](https://github.com/Rishav-Upadhaya/SWLP/blob/main/docs/architecture.md).

## Documentation

| Page | Contents |
|---|---|
| [Configuration](https://github.com/Rishav-Upadhaya/SWLP/blob/main/docs/configuration.md) | Backends, CLI flags, `SWLP_*` environment variables, TOML profiles |
| [Architecture](https://github.com/Rishav-Upadhaya/SWLP/blob/main/docs/architecture.md) | Streaming pipeline, schedulers, KV tiers, MoE expert cache |
| [Formats](https://github.com/Rishav-Upadhaya/SWLP/blob/main/docs/formats.md) | Shard directory, expert banks, `.swz` codec, package format |
| [Benchmarking](https://github.com/Rishav-Upadhaya/SWLP/blob/main/docs/benchmarking.md) | Methodology and how to reproduce the numbers |
| [Results](https://github.com/Rishav-Upadhaya/SWLP/blob/main/docs/results.md) | All measured numbers |
| [Roadmap](https://github.com/Rishav-Upadhaya/SWLP/blob/main/docs/ROADMAP.md) | Project history and open work |
| [Changelog](https://github.com/Rishav-Upadhaya/SWLP/blob/main/CHANGELOG.md) | Release notes |

The TOML profiles in `configs/` and the tools in `scripts/` (hardware check, benchmark harnesses) live in the repository only; they are not part of the installed package.

## Limitations

- **Apple Silicon only.** The CUDA/NVIDIA path was removed; Linux and Windows are unsupported.
- **Streaming is slow by design.** It makes oversized models feasible; it does not make them interactive. Throughput is capped at SSD bandwidth divided by bytes per token (about 0.5 tok/s for Mistral-7B FP16 on a 6.9 GB/s SSD). If a model fits in memory, `mlx` is far faster.
- **The `.swz` codec saves ~31% disk but is slower on fast SSDs.** Decompression competes for unified-memory bandwidth; it only helps below roughly 3.5 GB/s read speed. Plain shards stay the default.
- **Lossy tiers are opt-in and off by default:** MLX `int8`/`int4` weights, MLX KV quantization (`--kv-bits`), INT4 KV (`--kv-quant int4`), and `--max-kv-size`.
- Speculative speedups depend on how often drafts are accepted; n-gram drafting gives little on novel text.

## Citation

```bibtex
@software{upadhaya_swlp_2026,
  author  = {Upadhaya, Rishav},
  title   = {{SWLP: Sliding Window Layer Pipeline}},
  version = {0.1.0},
  year    = {2026},
  url     = {https://github.com/Rishav-Upadhaya/SWLP}
}
```

See [CITATION.cff](https://github.com/Rishav-Upadhaya/SWLP/blob/main/CITATION.cff).

## Contributing and license

Bug reports, hardware measurements from other M-series chips, and pull requests are welcome; see [CONTRIBUTING.md](https://github.com/Rishav-Upadhaya/SWLP/blob/main/CONTRIBUTING.md) and report vulnerabilities per [SECURITY.md](https://github.com/Rishav-Upadhaya/SWLP/blob/main/SECURITY.md).

MIT License, copyright 2026 Rishav Upadhaya. See [LICENSE](https://github.com/Rishav-Upadhaya/SWLP/blob/main/LICENSE).
