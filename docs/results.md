# Results

All measured SWLP numbers are collected here. Unless a row says otherwise, the machine is an
**Apple M5 with 16 GB unified memory** (LPDDR5X, 153 GB/s) and the decoding is greedy. Phase
numbers point to the matching entries in [ROADMAP.md](ROADMAP.md). For how the numbers were
measured, see [benchmarking.md](benchmarking.md).

## How to read these numbers

- **Cold and warm.** In a *cold* run the page cache was dropped before every timed run, so the
  SSD was really in the loop. In a *warm* run the shards may have been served from RAM, so the
  number is an upper bound. *Direct I/O* means the shard reads bypassed the page cache with
  `F_NOCACHE`, which forces SSD reads without a `purge`.
- **Before and after Phase 20.** Phase 20 (2026-06-11) removed three full-shard copies per
  layer per token. Streaming numbers from before it and after it are not comparable.
- **Precision tiers.** FP16/BF16 streaming is lossless. MLX `int8` gave output byte-identical to
  FP16 on Mistral-7B. `int4` and 4-bit checkpoints are a separate, lossy reference tier.
- **Single runs.** Most rows are development runs with n = 1 to 3. The paper-grade
  re-measurement (at least 5 runs, median and IQR, cold and warm) is still open work.

## Summary

| Model (on-disk size) | Configuration | tok/s | Conditions |
|---|---|---:|---|
| Mistral-7B FP16 (13.96 GB) | `swlp`, W=2 | **0.372** | Direct I/O, after Phase 20, median of 3, 8 tokens |
| Mistral-7B FP16 | `swlp`, W=2 | 0.174 | Cold (`purge`), before Phase 20, 3-prompt mean |
| Mistral-7B FP16 | `swlp`, W=2 | 0.210 | Warm, before Phase 20, mean of 3 per-prompt medians |
| Mistral-7B FP16 | AirLLM 2.11.0 | 0.085 | Warm, same harness and prompts as the 0.210 row |
| Qwen2.5-14B FP16 (26.4 GB) | `swlp`, W=2 | 0.187–0.197 | Direct I/O, after Phase 20 |
| Qwen2.5-14B FP16 | `speculative` + Qwen2.5-0.5B draft | **0.545–1.158** | Direct I/O. Depends on acceptance. Output identical. |
| Qwen3.8-27B fp16 (48.7 GB) | `speculative --mtp` | 0.498 | 8–24 tokens. Output identical. |
| Mistral-Small-24B FP16 (~44 GB) | `swlp`, W=2 | 0.081 | Warm, before Phase 20, 3.76 GB peak RAM |
| Mistral-7B | `mlx --quant int8` | 16.0 | Byte-identical to FP16 |
| Mistral-7B | `mlx --quant int4` | 27.9 | Lossy |
| Qwen2.5-14B | `mlx --quant int4` | 13.8 | Lossy. int8 runs out of memory. |
| Gemma 4 26B A4B, 4-bit (15.3 GB) | `mlx-moe` | 14.5 | Steady state, 95% expert hits. Stock mlx_lm runs out of memory. |
| Qwen3.6-35B-A3B BF16 (66 GB) | `mlx-moe` | 4.4–5.0 | Steady state. About 5 tok/s is the BF16 ceiling on 16 GB. |

**Streaming ceiling.** Throughput is capped at `tok/s ≤ SSD bandwidth / bytes per token`.
Phase 0 measured the SSD at 6.93 GB/s, which gives 0.496 tok/s for Mistral-7B and 0.158 tok/s
for 24B. The paper's cold analysis used the 6.5 GB/s that `detect_hardware()` reported at the
time, which gives a 0.465 tok/s ceiling for 7B. Against 0.496:

| Measurement | Fraction of the 7B ceiling |
|---|---|
| Cold 0.174 | 35% |
| Direct-I/O 0.372 (after Phase 20) | 75% |

## Hardware baseline (Phase 0)

| Metric | M5 16 GB |
|---|---|
| SSD sequential read / write | 6.93 / 4.48 GB/s (`scripts/phase0_hardware_check.py`) |
| MPS 1k×1k matmul | 4.2 ms |
| MLX 1k×1k matmul | 3.5 ms |
| Unified memory bandwidth | 153 GB/s (spec) |
| tiny-gpt2, `hf` | 100.3 tok/s, 4.35 s load |
| tiny-gpt2, `swlp` W=4 | 111.1 tok/s, TTFT 5.85 ms (fits in RAM, so SSD streaming isn't exercised) |

## Mistral-7B FP16 streaming

`unsloth/mistral-7b-instruct-v0.2`: 32 layers of 436 MB, 13.96 GB in total. Every SWLP number
ever reported for this configuration is listed below with its conditions. Only the unmarked
rows should be quoted.

| tok/s | Date / phase | Conditions | Status |
|---:|---|---|---|
| 0.40 | Phase 1 | 16 tokens, cache state not controlled | Historical |
| 0.422 | Phase 3 (`benchmarks/phase3.json`) | 1 run, 1 prompt, cache state not controlled | Superseded |
| 0.502 / 0.505 | Phase 3/4 multi-run | Warm-up left the model in the page cache | **Invalid as streaming**: above the 0.496 ceiling |
| 0.210 | 2026-05-21 | Warm, 3 prompts × 2 runs, same harness as AirLLM | Warm reference |
| ~0.21 (0.240 / 0.218 / 0.185) | 2026-05-29, Phase 19 | Warm, hardened harness, reproduced | Warm reference |
| 0.174 (0.133 / 0.186 / 0.203) | 2026-05-30 | **Cold**: `purge` before each run, `--skip-airllm`, TTFT 5.3 s, 0.78 GB RAM | Cold reference before Phase 20 |
| 0.218 → **0.372** | 2026-06-11, Phase 20 | Direct I/O, 8 tokens, median of 3 (0.360 / 0.372 / 0.379). Prefill 5.86 → 2.30 s. RAM 1.17 GB. | Current |
| 0.390 | 2026-06-11, Phase 22 | Direct I/O, W=4, prefetch depth 2, 6-token probe | Current |

The 0.174 cold run is recorded in the paper notes (`research/research.md`, Table VI). ROADMAP
Phase 19 still lists the cold run as pending. The same cold measurement hasn't been repeated
since Phase 20.

### Window size (Phase 1, 16 tokens)

| W | tok/s | TTFT | RAM peak |
|---|---:|---:|---:|
| **2** | **0.40** | 6 ms | 1.40 GB |
| 4 | 0.29 | 14 ms | 1.03 GB |
| 6 | 0.25 | 22 ms | 0.93 GB |

W=2 was the fastest window end to end. Larger windows increase unified-memory contention
between prefetch and MPS compute. A synthetic pipeline with 80 ms of compute per layer showed
the opposite trend in overlap gain (W=2: 18.9%, W=4: 30.0%, W=6: 36.3%), so the synthetic
result did not transfer to real runs.

## SWLP vs AirLLM

This is a same-harness, warm-cache comparison from 2026-05-21, with AirLLM 2.11.0, FP16, 32 new
tokens, 2 timed runs per prompt (median) and 1 warm-up. It was run before Phase 20.

| Prompt | SWLP tok/s | AirLLM tok/s | Speed-up | SWLP TTFT | AirLLM TTFT | SWLP RAM | AirLLM RAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 `What is 2+2?` | 0.202 | 0.099 | 2.04× | 4.71 s | 10.43 s | 0.79 GB | 1.83 GB |
| P2 attention paragraph | 0.214 | 0.102 | 2.10× | 6.02 s | 9.38 s | 0.97 GB | 1.70 GB |
| P3 solar system | 0.213 | 0.053 ¹ | 4.02× | 5.15 s | 20.58 s | 1.21 GB | 1.52 GB |
| **Mean** | **0.210** | **0.085** | **2.5×** | 5.29 s | 13.47 s | 1.21 GB peak | 1.83 GB peak |

¹ AirLLM slowed down on P3 after two consecutive 14 GB sweeps because the macOS memory compressor
kicked in. The two numbers quoted elsewhere for AirLLM are both from this table: 0.099 is P1
alone, and **0.085 is the 3-prompt mean**. On P1 and P2, where AirLLM wasn't degraded, the
speed-up is about 2.0–2.1×.

- **Quality.** Both systems produce semantically equivalent completions. Wording diverges
  because PyTorch/MPS and MLX round FP16 differently.
- **Cold SWLP vs warm AirLLM.** SWLP cold (0.174) against AirLLM warm (0.085) is 2.05×. AirLLM
  has not been measured cold.
- **Earlier comparison.** The Phase 3 comparison (SWLP 0.422 vs AirLLM 0.208, 2.03×, TTFT
  13.6 ms vs 6.54 s) used single uncontrolled-cache runs. It's superseded by the table above.
- **Attribution.** The RAM advantage and the `head_dim` compatibility (below) are robust
  results. The paper does not attribute the throughput gap only to prefetch, because AirLLM's
  MLX path wasn't isolated.

### Mistral-Small-24B (FP16, about 44 GB)

| | SWLP (W=2, warm) | AirLLM |
|---|---|---|
| Runs? | Yes | **No.** It crashes with `[rope] dims must not exceed … (128) but got 160` |
| tok/s | 0.081 (P1/P2/P3: 0.078 / 0.081 / 0.085), 51% of the 0.158 ceiling | — |
| TTFT | 12.65 s | — |
| Peak RAM | 3.76 GB | — |
| Disk needed | about 44 GB of shards | HF weights plus `.mlx` shards, about 2× the model size |

AirLLM computes `rope_dims = hidden_size / num_heads = 160` and ignores the config's explicit
`head_dim = 128`. SWLP reads `head_dim` from the config.

## Quantized reference tier (Phase 3, Mistral-7B)

This tier is not equal quality. It fits in RAM, so it isn't streaming.

| System | Quant | tok/s | TTFT | Peak RAM |
|---|---|---:|---:|---:|
| Ollama (llama.cpp) | Q4_K_M | 28.25 | 61 ms warm, 15.9 s cold | — |
| MLX-lm | 4-bit | 29.96 | 1.11 s | 4.36 GB |
| MLX-lm / HF transformers | FP16 full load | — | — | Out of memory (~14 GB model) |

## Adaptive residency

Phase 4, Mistral-7B FP16, 16 GB. The comparison baseline was a warm run.

| Configuration | tok/s | RAM peak |
|---|---:|---:|
| All streaming (reference, warm) | 0.502 | 1.13 GB |
| 17 layers resident on MPS | 0.041 | about 9 GB. The Metal allocator fragmented. |
| 17 layers resident in CPU RAM | 0.080 | about 9 GB. The macOS memory compressor fired. |
| `auto` with the full-model-fit guard (0 resident) | 0.505 | 1.47 GB |

Locking 7.4 GB of resident layers pushes the page cache out for the 6.5 GB of streamed shards.
`plan_residency()` therefore turns residency on only when the whole model fits in
`(RAM − 6 GB) × 0.75`. For 7B FP16 that needs at least 32 GB of RAM. Models that fit get
reclaimable page-cache residency instead, through `SWLP_DIRECT_IO=auto` (Phase 20). For
example, Qwen2.5-0.5B reached 9.85 tok/s with a 0.195 s TTFT.

## Speculative decoding

### Prompt-lookup n-gram drafter (Phase 5, Mistral-7B)

The baseline was the warm 0.505 run.

| Workload | Drafts accepted / proposed | Tokens per sweep | tok/s | vs baseline |
|---|---|---:|---:|---:|
| Novel text | 0 / 0 | 1.03 | 0.471 | 0.93× |
| Mildly repetitive | 6 / 6 | 1.28 | 0.578 | 1.15× |
| Repetition-heavy (48 tokens) | 35 / 35 | 4.00 | 1.66 | 3.29× |

When a draft fires, all of it is accepted. The speed-up depends on how often the output repeats
earlier context. No extra RAM is used.

### Draft model (Phase 21, Qwen2.5-14B, Qwen2.5-0.5B drafter)

Direct I/O, W=2, 32 tokens.

| Workload | Baseline | Draft-spec | Speed-up | Acceptance | Tokens per sweep |
|---|---:|---:|---:|---:|---:|
| Open-ended sentence | 0.187 | **0.545** | 2.9× | 47.7% | 3.2 |
| Constrained list | 0.197 | **1.158** | 5.9× | 90.0% | 8.0 |

Output is byte-identical to plain greedy streaming. Two things did not help:

- A fixed K=8 without adaptation gave 0.151 tok/s, which is 19% slower than the baseline.
- A 1.5B drafter got the same acceptance as the 0.5B drafter at three times the memory.

Mistral-7B has no small draft model with the same tokenizer.

### Native MTP head (Phase 29, Qwen3.8-27B, fp16 shards, 64 layers)

| Run | tok/s |
|---|---:|
| Baseline defaults, before the fix (10 GB footprint, swapping) | 0.035 |
| Window 1, prefetch 1, residency off | 0.100 |
| Defaults with the embedding kept on CPU via mmap (8.4 GB peak) | 0.135 |
| `--mtp`, max draft 2 / 4 / 16 | 0.339 / 0.461 / **0.498** |

Output is identical to plain decoding. Splitting shard reads into 2 parallel `pread` calls made
the pipeline 3× slower, even though it was faster in isolation.

## Model ladder (FP16 streaming, W=2)

| Model | Layers × size | Total | tok/s | TTFT | RAM peak |
|---|---|---:|---:|---:|---:|
| Mistral-7B | 32 × 436 MB | 13.96 GB | see [above](#mistral-7b-fp16-streaming) | | 0.8–1.2 GB |
| Qwen2.5-14B (Phase 6) | 48 × 551 MB | 26.4 GB | 0.194 (before Phase 20) | 24.8 ms | 1.66 GB |
| Mistral-Small-24B | 40 × ~850 MB | ~44 GB | 0.081 (warm) | 12.65 s | 3.76 GB |
| Qwen3.8-27B (hybrid DeltaNet) | 64 layers | 48.7 GB | 0.135; 0.498 with MTP | | 8.4 GB |

Peak RAM grows with layer size, not with the number of layers. None of these models can be
fully loaded on this machine.

## MLX backend (Phase 8)

32 tokens, same prompt as the Phase 3 runs.

| Model | Backend | tok/s | Completion vs FP16 |
|---|---|---:|---|
| Mistral-7B | `mlx --quant int8` | **16.0** | Byte-identical |
| Mistral-7B | `mlx --quant int4` | 27.9 | Minor wording drift |
| Qwen2.5-14B | `mlx --quant int8` | — | Out of memory (~14 GB) |
| Qwen2.5-14B | `mlx --quant int4` | **13.8** | Minor wording drift |

MLX RAM figures come from psutil RSS, which under-reports wired Metal memory.

## MoE expert streaming

### OLMoE-1B-7B BF16 (Phase 30; 64 experts, top-8)

| Configuration | tok/s | Expert hit rate |
|---|---:|---:|
| torch `swlp` MoE path, 9 GB budget | 0.55 | — |
| `mlx-moe`, 12% of experts cached | 8.8 | 37% |
| `mlx-moe`, 49% cached | 10.4 | 78% |
| `mlx-moe`, 70% cached | 19.9 | 93% |
| `mlx-moe`, 70% cached: LFU vs LRU (same-session A/B) | 16.6 vs 11.7 | 93% |

Predictive prefetch was slower at every budget: 6.5 vs 9.6 tok/s at 12% cached. LRU scores
about 0% hits at small budgets because decode touches every layer's top-k once per token. Run to
run variation is large (8.8 vs 6.2 tok/s for the same point an hour apart), so only compare
numbers from the same session.

### Qwen3.6-35B-A3B BF16 (Phase 30; 256 experts, top-8; 256 tokens)

| Expert budget | tok/s steady | Hit rate |
|---|---:|---:|
| 2.5 GB (LFU) | 4.4 | 45% |
| auto (3.3 GB) | 4.5 | 52% |
| max (6.4 GB, clamped to Metal) | 4.7–5.0 | 67% |
| max, predictive prefetch | 2.8 | 71% |

About 5 tok/s is the BF16 ceiling on 16 GB. The routing sync costs about 120 ms per token, and
roughly 1 GB of expert misses per token dominates the rest.

### Gemma 4 26B A4B, 4-bit MLX checkpoint (Phase 31)

| Configuration | tok/s | Hit rate |
|---|---:|---:|
| Stock mlx_lm, full load | Crash (Metal out of memory) | — |
| `mlx-moe`, auto budget 6.5 GB, 256 tokens, steady | **14.5** (p50 58 ms) | 95.3% |
| `mlx-moe` CLI, 128 tokens including cold start | 11.7 (TTFT 1.8 s) | 91.5% |
| `mlx-moe`, 3 GB budget | 7.2 | 77.7% |

## Batched streaming (Phase 10, SmolLM2-360M FP16, W=2)

| Batch | Sweep time | Aggregate tok/s |
|---:|---:|---:|
| 1 | 0.282 s | 3.54 |
| 4 | 0.264 s | 15.13 |
| 16 | 0.243 s | 65.75 |

Sweep time stays flat as the batch grows, so aggregate throughput scales about linearly: batch
16 is 18.5× batch 1. Each row is bit-identical to a batch-1 run. The 7B and 14B batched runs are
still pending.

## Lossless shard codec (.swz)

Phase 22, Mistral-7B FP16, W=4, direct I/O. Compressed size ratio 0.673 (13.96 → 9.40 GB).

| Variant | tok/s | Per-layer worker stages |
|---|---:|---|
| Plain `.safetensors` | **0.390** | read 120 ms + host-to-device 39 ms |
| `.swz`, one decompress at a time | 0.264 | decompress 59 ms |
| `.swz`, one decompress at a time, prefetch depth 4 | 0.278 | |

Compression costs throughput on fast SSDs. Standalone, the decoder runs at 15 GB/s, but it drops
to about 7 GB/s under load while competing for unified-memory bandwidth. It only wins below
about 3.5 GB/s of SSD read. The disk saving of about 31% applies either way.

## FP8 shards (Phase 7): negative result, format removed

| Run | tok/s | vs FP16 |
|---|---:|---:|
| Mistral-7B FP8 (fully resident) | 0.436 | 0.86× |
| Qwen2.5-14B FP8 (streaming) | 0.102 | 0.53× |

Halving the bytes on disk didn't help. Converting FP8 back to FP16 on the CPU for every token
cost as much as the read it saved. The completions were byte-identical. The FP8 format has since
been removed, and loading an FP8 manifest raises an error.

## KV cache (Phase 2)

On Mistral-7B, lossless zlib compression of the KV cache gives only about 1.10× (level 1:
1.104× at 29.0 s; level 9: 1.108× at 37.0 s), because FP16 activations are high-entropy.
Completions are identical and generation is about 3% slower. The zlib tier is useful for
offloading cold layers to host RAM, not for its compression ratio. INT4 KV (`--kv-quant int4`)
is about 4× smaller, but its perplexity cost has not been measured yet.

## Simulation vs end to end

`swlp simulate` and `scripts/research/simtools` model only the scheduling pipeline: read,
deserialize, upload, compute and evict. They leave out attention, KV bookkeeping, Python,
Metal dispatch and synchronization, so real tok/s comes in well below the simulated value. Use
the simulators to compare scheduling strategies, not to predict absolute throughput.

The `swlp suite` tooling is validated on tiny-gpt2 (`benchmarks/suite-20260520T044433Z.json`),
because the suite's full-model HF baseline can't load Mistral-7B on 16 GB.

## Pending measurements

- Paper-grade re-measurement after Phase 20: at least 5 runs, median and IQR, cold and warm,
  SWLP and AirLLM in the same harness, plus a set of prompts for draft-model speculation.
- AirLLM cold, and oLLM on the M5.
- Batched runs on 7B and 14B. Qwen2.5-32B at W=2. Long context at 4K and 8K.
- Cold-read latency of safetensors vs `.pt` shards. Perplexity cost of INT4 KV.
- Qwen3-30B-A3B MoE sweep and DeepSeek-V4-Flash feasibility (Phase 28 targets). Qwen3.8-27B
  re-sharded as bf16.
