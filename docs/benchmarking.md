# Benchmarking

This page covers how SWLP numbers are measured, how to reproduce the published results, and how
to validate the residency predictor. The measured numbers themselves are in
[results.md](results.md).

## Methodology

### Page-cache state

For any layer-streaming runtime, throughput is capped by how fast the SSD can deliver the bytes
streamed per token:

```
max tok/s = SSD read bandwidth / bytes streamed per token
Mistral-7B FP16 on M5:  6.93 GB/s / 13.96 GB = 0.496 tok/s
```

A cold run must come in **below** this ceiling. If a run meets or beats it, the shards were
already in the OS page cache and the run read from RAM, not the SSD. That happened here once: a
0.505 tok/s "streaming" result turned out to be a warm-cache read. Cold and warm numbers are both
valid, but they answer different questions:

| Mode | What it measures | How to report it |
|---|---|---|
| **Cold** (page cache dropped before each timed run) | True SSD streaming | The headline number, and the basis for head-to-head comparisons |
| **Warm** (shards already cached) | Best case when the OS kept the model cached | An upper bound. Label it and never make it the headline. |

### Cold protocol

The comparison harnesses take `--cold`. In that mode they drop the page cache **after** the
warm-up and **before each** timed run, so kernels and the tokenizer stay warm and only the
weights are cold. On macOS, `bench_common.drop_page_cache()` runs `purge`, which needs `sudo`.

```bash
sudo .venv/bin/python scripts/research/compare_airllm_swlp.py --models 7b --cold
sudo .venv/bin/python scripts/research/phase3_baselines.py --baseline all --cold
```

If the cache couldn't be dropped, the result records `cache_state = "warm (<reason>)"`, so a
harness can't label a warm run as cold. Every result has a `cache_state` field and every report
has a top-level `cache_mode`.

Streaming runs should also pin `SWLP_DIRECT_IO=on`. The harnesses already do this, so that
shard reads bypass the page cache.

### Provenance

Every benchmark JSON contains a `provenance` block from `bench_common.provenance()` with the UTC
timestamp, git commit, platform, Python version, hardware (chip, RAM, SSD bandwidth) and library
versions. The AirLLM comparisons used **airllm 2.11.0**. Its behaviour changes between releases,
so a comparison is only valid if it names the version.

### Rules for a fair comparison

1. **Same harness, same process, same prompts, same `cache_mode`** for both systems. Never
   pair numbers from different harnesses.
2. **Same cache state**: `--cold` drops the cache before both SWLP's timed runs and the
   competitor's.
3. **Headline numbers:** at least 5 timed runs with 1 warm-up discarded, reported as median and
   IQR (`bench_common.summarize_runs`). Always state the prompt set.
4. **Same precision:** the headline compares FP16 with FP16. Quantized systems (Ollama Q4_K_M,
   MLX 4-bit) go in a separate, labelled reference tier.
5. Set `SWLP_STRICT=1` so a degraded hot path fails the run instead of silently slowing it
   down.

### Related systems

| System | Approach | Lossless | Status here |
|---|---|---|---|
| AirLLM | Load, compute, discard for each layer | FP16 | Benchmarked (2.11.0) |
| oLLM | Layer and KV SSD streaming, no quantization | FP16/BF16 | Not yet benchmarked. It runs on Apple Silicon but loses its flash-attn long-context path. |
| FlexGen, DeepSpeed ZeRO-Inference, PowerInfer | Offload and batching; neuron sparsity | Mixed | CUDA only. Related work. |
| LLM in a Flash (Apple) | Sparsity-driven flash loading | Needs ReLU sparsity | Related work |

AirLLM and the other competitors are benchmark-only dependencies. Install them in the venv
yourself. They aren't in `pyproject.toml`.

## Reproducing the results

Requirements: an Apple Silicon Mac. The reference machine is an M5 with 16 GB. You also need
about 50 GB of free disk for Mistral-7B and Qwen2.5-14B shards, and a source checkout, because
the harnesses and `configs/` aren't in the wheel.

```bash
git clone <repo-url> swlp && cd swlp
bash scripts/bootstrap.sh && source .venv/bin/activate
pip install -e '.[apple]'
```

**1. Hardware baseline.** This measures SSD bandwidth and caches it in
`~/.cache/swlp/hardware.json`. Use a quiet machine: the probe reads a 256 MB file, and a warm
cache inflates the number. `SWLP_SSD_BW_GBPS` overrides the cached value.

```bash
python scripts/phase0_hardware_check.py
swlp doctor mistral-7b          # pipeline ratio, predicted residency, confidence
```

**2. Shard the models.**

```bash
swlp pull mistral-7b    # ./shards/mistral-7b, 13.96 GB
swlp pull qwen-14b      # ./shards/qwen-14b
swlp pull qwen-0.5b     # draft model for speculative runs
```

**3. Streaming throughput** (cold, same harness as AirLLM):

```bash
sudo .venv/bin/python scripts/research/compare_airllm_swlp.py --models 7b --runs 5 --cold
sudo .venv/bin/python scripts/research/compare_airllm_swlp.py --models 7b --runs 5 --cold --skip-airllm
```

**4. Baseline table** (SWLP, AirLLM, MLX-lm, Ollama). Start `ollama serve` first.

```bash
sudo .venv/bin/python scripts/research/phase3_baselines.py --baseline all --runs 5 --cold --out benchmarks/phase3.json
```

**5. Per-configuration CLI benchmarks.**

```bash
SWLP_DIRECT_IO=on SWLP_STRICT=1 swlp bench mistral-7b --window 2 --runs 5    # median tok/s, TTFT, peak RAM
SWLP_DRAFT_MODEL=qwen-0.5b swlp run qwen-14b "List ten prime numbers."        # draft-model speculation
swlp bench mistral-7b --runs 5 --json > benchmarks/mistral-7b.json             # machine-readable summary
```

`--window` takes one integer; to sweep windows, run once per value. Batched streaming and
prompt-set suites are library APIs (`SWLPRunner.run_batch`, `swlp.benchmark.suite.run_suite`).

**6. Quality equivalence.** Check that streaming output matches the HF reference.

```bash
python scripts/research/quality_equivalence.py --model unsloth/mistral-7b-instruct-v0.2 --shard-dir ./shards/mistral-7b
```

**7. MoE expert-cache sweep.**

```bash
python scripts/research/moe_sweep.py --shard-dir ./shards/qwen3-30b-a3b \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 --budgets 0,2048,6144 --runs 3 --max-tokens 32
```

**8. Simulation and figures** (no model needed).

```bash
python scripts/run_full_benchmark.py                        # discrete-event model × feature tables
python -m scripts.research.simtools sim --layers 32 --layer-size-mb 436 --window 2 --prefetch 4 --report
python scripts/generate_figures.py --output-dir figures/
```

**9. Profiling a slow run.**

```bash
SWLP_PROFILE=1 swlp run mistral-7b "Hi" -n 8    # per-layer timeline + summary; writes layer_traces.json
python -m scripts.research.simtools analyze layer_traces.json
```

Raw outputs go to `benchmarks/` (CLI and harness JSON), `experiments/` (sweeps and
cross-machine data), and `figures/`.

## Policy validation

This checks how well `swlp doctor` and `swlp_residency = "auto"` predict the best resident-layer
count across different machines.

**1. Collect rows.** Run this on each machine. It sweeps resident counts and writes to
`experiments/cross_machine/<chip>_<RAM>GB/`:

```bash
python scripts/collect_cross_machine.py --full
```

**2. Build a matrix** as CSV or JSON, using `experiments/policy_matrix_template.csv` as the
template.

| Column | Required |
|---|---|
| `machine`, `model`, `pipeline_ratio`, `free_ram_gb`, `resident_count`, `throughput_tokens_per_second` | yes |
| `workload`, `quant` | no |

**3. Score it:**

```bash
python -m scripts.research.simtools policy-report experiments/policy_matrix.csv
```

The report prints the resident-count MAE in layers, the percentage of predictions within ±2
layers, and the average throughput regret.

**Acceptance criteria:**

| Metric | Target |
|---|---|
| Resident MAE | ≤ 2.0 layers |
| Within ±2 layers | ≥ 90% |
| Mean throughput regret | ≤ 5% |

**Design:** use at least 3 machine classes and at least 2 model sizes. Sweep `resident_count`
around the predicted value and average several runs for each point. To compare scheduling
policies in simulation:

```bash
python -m scripts.research.simtools evaluate --policies baseline,resident_4,resident_8 --report
python -m scripts.research.simtools sweep --layers 32,48 --ram 8,16,24,32 --window 2,4 --csv
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| A "streaming" number at or above the SSD ceiling | The cache was warm. Rerun with `--cold` under `sudo` and `SWLP_DIRECT_IO=on`. |
| `cache_state` says `warm (purge needs sudo …)` | Run the harness with `sudo`. |
| Low throughput or swapping | Run `swlp doctor`, try `--window 1`, and check that `swlp_residency` resolved to `0` for models larger than RAM. |
| `mlx` not available | `pip install 'swlp[apple]'` |
