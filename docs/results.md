# Results

## Two measurement tiers

This document reports two distinct types of measurements. They measure
different systems and should not be directly compared as if they were
the same thing.

### Pipeline simulator (discrete-event)

Models the **scheduling pipeline** only: SSD → Deserialize → Upload → Compute → Evict.
Does **not** include tokenizer overhead, attention kernel cost, Python/framework
overhead, synchronization, kernel launch latency, or OS scheduling.

Useful for: comparing scheduling strategies, understanding overlap efficiency,
measuring GPU idle ratio, predicting which strategy wins on a given hardware
configuration.

**Pipeline ratio** (used throughout) is defined as:

```
Pipeline Ratio = (SSD Read + Deserialize + Upload) / Block Compute
```

This is the ratio of I/O time to compute time per transformer block. A ratio > 1
means I/O-bound (streaming helps). A ratio < 1 means compute-bound (streaming
has limited benefit).

### End-to-end inference

Measures **actual tok/s** from prompt to final token, including all overhead:
tokenizer, attention, KV cache management, Python runtime, Metal/CUDA kernel
launch, synchronization, and framework bookkeeping.

Useful for: real-world performance claims, comparison with other systems
(MLX-lm, Ollama, HF transformers, AirLLM).

The gap between simulator and end-to-end is the **runtime overhead** —
typically 5–15× on Apple Silicon due to per-token materialization, Metal
allocator fragmentation, and Python GIL contention.

---

## Summary of measured results

### Mistral-7B FP16 on M5 16 GB (end-to-end)

> ## ⚠️ Benchmark-integrity update (Phase 19, 2026-05-29)
>
> An audit found the Mistral-7B W=2 FP16 throughput cited as **three different
> numbers** (0.21 / 0.42 / 0.50 tok/s) across the docs, used interchangeably. The
> tables below are **historical record** — read them with these corrections:
>
> - **0.505 tok/s is not a cold-streaming number.** It exceeds the cold-SSD
>   ceiling (0.496 tok/s = 6.93 GB/s ÷ 13.96 GB), which is physically impossible
>   for streaming from disk — it was measured from **warm OS page cache**.
> - **0.422 tok/s** (phase3.json) was a single uncontrolled-cache run.
> - **Reproduced warm-cache median (2026-05-29, hardened same-harness):**
>   SWLP **~0.21 tok/s** (0.240 / 0.218 / 0.185 across P1/P2/P3),
>   AirLLM **~0.10 tok/s** (prior same-harness) → **SWLP ~2× faster, warm**.
> - **Authoritative cold-SSD numbers are pending** a `sudo` re-measurement
>   (`sudo .venv/bin/python scripts/research/compare_airllm_swlp.py --models 7b --cold`).
>
> Methodology, provenance rules, and the full reconciliation are in
> [`benchmark_methodology.md`](benchmark_methodology.md). The SWLP-vs-AirLLM
> *direction* (≈2× faster, lower TTFT, lower RAM) is robust across all runs; only
> the absolute SWLP number was inflated by warm cache.

Comparison table for the paper. All numbers measured on **M5 (Apple Silicon,
16 GB unified memory)** with **Mistral-7B** (`unsloth/mistral-7b-instruct-v0.2`),
32 new tokens, greedy decoding, identical prompt:

> "Explain in one sentence what makes a Macbook Air good for development."

Raw data: `benchmarks/phase3.json` (harness: `scripts/research/phase3_baselines.py`).
The MX230 / NVIDIA column is **pending** — to be filled on the Pop!_OS machine.

---

## How to read this table — two tiers

SWLP's first-class constraint is **zero quality compromise** (FP16, no
quantization). A fair speed comparison must therefore separate two tiers:

- **FP16 / lossless tier** — bit-exact FP16 weights. This is the apples-to-apples
  comparison and where the paper's headline speedup is computed.
- **Quantized reference tier** — 4-bit weights. Faster, but *not* equal quality.
  Listed as a reference ceiling, not a like-for-like competitor.

---

## FP16 / lossless tier (equal quality)

| System | Backend | tok/s | TTFT | Generate (32 tok) | Peak RAM | Feasible on 16 GB? |
|---|---|---|---|---|---|---|
| **SWLP** (W=2) | sliding-window SSD→RAM streaming | **0.422** | **13.6 ms** | **75.9 s** | **1.13 GB** | ✅ yes |
| AirLLM | layer-by-layer streaming | 0.208 | 6.54 s | 153.8 s | — | ✅ yes |
| MLX-lm (FP16) | naive full-model load | — | — | — | ~14 GB | ❌ **OOM** |
| HF transformers (FP16) | naive full-model load | — | — | — | ~14 GB | ❌ **OOM** |

**Naive full-model load does not fit.** A 7B FP16 model is ~14 GB; loading it
whole on a 16 GB M5 aborts with a Metal out-of-memory error (MLX-lm's FP16
conversion crashes; HF `from_pretrained` to MPS hits the same wall). This is the
exact problem SWLP and AirLLM exist to solve — only the two streaming runtimes
produce a result at all.

## Quantized reference tier (NOT equal quality — 4-bit)

| System | Backend | Quant | tok/s | TTFT | Generate (32 tok) | Peak RAM |
|---|---|---|---|---|---|---|
| Ollama | llama.cpp full-model | Q4_K_M | 28.25 | 61 ms (warm) | 1.16 s | — |
| MLX-lm | naive full-model | 4-bit | 29.96 | 1.11 s | 2.20 s | 4.36 GB |

4-bit quantization shrinks Mistral-7B to ~4 GB, so it loads whole and runs
~70× faster than FP16 streaming — but at a quality cost SWLP explicitly refuses
to pay. Ollama's cold TTFT (first request, model not resident) was 15.9 s; the
61 ms figure is the warm steady state.

---

## Headline result — SWLP vs AirLLM (FP16 tier)

The paper's central claim is *"SWLP beats AirLLM on speed at equal quality."*
Both run identical FP16 Mistral-7B weights — quality is identical by construction.

| Metric | AirLLM | SWLP (W=2) | SWLP advantage |
|---|---|---|---|
| Throughput | 0.208 tok/s | 0.422 tok/s | **2.03× faster** |
| Generate time (32 tok) | 153.8 s | 75.9 s | **50.7% faster** |
| TTFT | 6.54 s | 13.6 ms | **481× lower** |

Speedup vs AirLLM: `(153.8 − 75.9) / 153.8 = 50.7%`. **SWLP shows a positive
speedup over AirLLM at equal (FP16, lossless) quality — Phase 3 goal met.**

**Why SWLP wins:**
- *Throughput* — SWLP overlaps the next layers' SSD→RAM transfer with current-layer
  compute via background prefetch (Phase 1: 36% overlap gain). AirLLM loads each
  layer, computes, evicts — strictly sequential, no overlap.
- *TTFT* — SWLP keeps embeddings + `lm_head` + norms permanently resident and
  prefetches the window, so the first token emerges in milliseconds. AirLLM pays
  a full 32-layer disk sweep before it can emit token one.

---

## Quality check

All FP16-tier completions are coherent and on-topic; SWLP and AirLLM produce
semantically equivalent answers (both describe processor, storage, battery,
weight). SWLP's KV path is lossless (Phase 2: zlib is bit-exact), so SWLP
introduces **zero quality loss** relative to a full-model FP16 run.

---

## SWLP window sweep (Mistral-7B)

End-to-end SWLP on Mistral-7B, M5, by sliding-window depth W:

| Window W | tok/s | TTFT | RAM peak | Source |
|---|---|---|---|---|
| **W=2** | **0.422** | 13.6 ms | 1.13 GB | Phase 3 (`benchmarks/phase3.json`) |
| W=4 | 0.29 | 14 ms | 1.03 GB | Phase 1 (`docs/hardware_baseline.md`) |
| W=6 | 0.25 | 22 ms | 0.93 GB | Phase 1 (`docs/hardware_baseline.md`) |

**W=2 is the best end-to-end window on M5** — a small window minimises
unified-memory pressure between CPU prefetch and GPU compute. (The synthetic
ThreadedPipeline POC favoured larger W for hiding pure SSD latency; the real
run inverts this. See Phase 1 notes.)

### `swlp suite` structured-JSON artifact

`swlp suite` always runs an HF *full-model* baseline per prompt before the SWLP
cases. Mistral-7B FP16 cannot full-load on 16 GB (hard OOM aborts the process),
so the suite cannot run on Mistral here. The suite tooling is therefore
validated on **tiny-gpt2** (a full-loadable model) to produce the structured
`SuiteResult` JSON: `benchmarks/suite-20260520T044433Z.json`
(config: `configs/suite_phase3.toml` + `configs/baseline.toml`).

| | Baseline (HF) | SWLP (best, W=4) |
|---|---|---|
| Throughput | 40.7 tok/s | 325.0 tok/s |
| Quality overlap vs baseline | — | 1.00 (identical) |

The Mistral-7B paper numbers come from the dedicated `scripts/research/phase3_baselines.py`
harness, which does not require a full-model baseline.

---

---

## Phase 4 — Adaptive residency (M5 16 GB finding)

**Goal:** Reduce per-token disk I/O by caching layers in RAM.

**Finding:** On M5 16 GB with Mistral-7B FP16 (13.96 GB), adaptive residency does
**not improve throughput** due to insufficient RAM headroom.

| Configuration | tok/s | TTFT | RAM peak | Notes |
|---|---|---|---|---|
| SWLP W=2 (Phase 3, all streaming) | 0.502 | 13.6 ms | 1.13 GB | Reference |
| Phase 4: MPS-resident (17 layers locked on Metal) | 0.041 | — | ~9 GB | Metal allocator fragmentation |
| Phase 4: CPU-RAM resident (17 layers in Python heap) | 0.080 | 108 ms | ~9 GB | macOS memory compressor triggered |
| Phase 4: fixed `auto` (full-model-fit guard → 0 resident) | **0.505** | 10.6 ms | 1.47 GB | ✅ Phase 3 speed restored |

**Root cause of both residency failures:** 17 resident layers × 436 MB = 7.4 GB
locked in memory starves the OS page cache for the 15 streaming shards (6.5 GB).
macOS triggers its memory compressor, adding massive latency to every memory
access across the process.

**Fix:** `plan_residency()` now includes a **full-model-fit guard** — residency is
only enabled when `total_model_bytes ≤ usable_budget`. On M5 16 GB:
usable = (16−4−2) × 0.75 = 7.5 GB < 13.96 GB → 0 resident layers → all streaming.

**Condition for residency to help:** the full model must fit within 75% of
`(total_ram − 6 GB)`. For 7B FP16 this requires ≥ 32 GB unified memory.
For smaller models (GPT-2, 1B, 3B), all layers become resident on 16 GB.

---

## Phase 5 — Speculative decoding (prompt-lookup)

**Goal:** Verify multiple tokens per disk sweep so throughput is no longer
capped at one token per 32-layer SSD read.

**Approach:** prompt-lookup (n-gram) speculative decoding — no draft model. An
n-gram drafter proposes up to K=8 continuation tokens by matching the trailing
3-gram against earlier context; the streamed Mistral-7B verifies all K in a
single disk sweep. Output is **bit-identical to greedy SWLP** (lossless by
construction). See `docs/phase5_design_decisions.md` for the design rationale.

All runs on M5 (Apple Silicon, 16 GB), Mistral-7B FP16, greedy decoding.

| Workload | Drafts proposed / accepted | Acceptance | tokens / sweep | tok/s | vs 0.505 baseline |
|---|---|---|---|---|---|
| Novel text (standard prompt, 32 tok) | 0 / 0 | — (no n-gram recurs) | 1.03 | 0.471 | 0.93× (drafting overhead) |
| Mildly repetitive ("repeat" prompt, 32 tok) | 6 / 6 | 100% | 1.28 | 0.578 | 1.15× |
| Repetition-heavy (pattern continuation, 48 tok) | 35 / 35 | 100% | **4.00** | **1.66** | **3.29×** |

**Key results.**

- **Lossless confirmed.** The standard-prompt completion is byte-for-byte
  identical to the Phase 3/4 greedy SWLP output ("A MacBook Air is an excellent
  choice for development due to its powerful processor, large storage capacity,
  long battery life, lightweight design, and compatibility"). Speculation
  changes throughput only, never the tokens.
- **When the drafter fires, acceptance is 100%.** Prompt-lookup proposes exact
  spans of prior context; whenever the model is genuinely continuing a repeated
  span, the target verifies every proposed token. The variable is *how often* a
  matching n-gram exists, not whether proposals are accepted.
- **Speedup scales with output repetitiveness.** Novel free-form text has no
  recurring n-grams → 0 drafts → speculative decoding degrades gracefully to
  baseline minus a small (~7%) drafting overhead. Repetition-heavy output
  (long-context QA that quotes the source, code, structured/list output) reaches
  **3.29× at 4.0 tokens/sweep**. The ceiling with K=8 is ~9× (9 tokens/sweep).
- **Zero extra RAM.** Prompt-lookup needs no draft model — critical on the
  memory-bound 16 GB M5 (peak RAM stayed ~1.0–1.2 GB, same as plain SWLP).

**Honest framing.** Speculative decoding is not a universal speedup; it is a
*workload-dependent* one. It is free (lossless, ~zero memory) and never
materially slower than baseline, and it is dramatically faster exactly on the
workloads SWLP targets — long-context inference where the answer echoes the
context.

---

## Phase 6 — Model-ladder climb: 14B rung (Qwen2.5-14B-Instruct)

**Goal:** Prove SWLP can stream a 14B FP16 model on a 16 GB M5 that cannot
full-load such a model at all.

**Model:** `Qwen/Qwen2.5-14B-Instruct` — 48 transformer layers, 5120 hidden dim,
`float16`. Sharded via stream-shard rewrite (no full-model RAM load during sharding).

| Property | Value |
|---|---|
| Total sharded weight | 26.4 GB (48 × 550.5 MB/layer) |
| Shard format | 48 × `layer_NNN.pt` + `embed.pt` + `lm_head.pt` |
| Sharding RAM peak | < 2 GB (stream-shard: no full-model load) |
| Integrity check | `verify_shards()` → ✅ all 50 files present, ZIP magic valid |

### Run results (M5, 16 GB, SWLP W=2, greedy, 32 new tokens)

Prompt: `"Explain in one sentence what makes a Macbook Air good for development."`

Completion:
> "The MacBook Air's combination of portability, long battery life, and powerful
> performance makes it an excellent choice for developers who need to work
> efficiently on the go."

| Metric | Value |
|---|---|
| **Throughput** | **0.194 tok/s** |
| **TTFT** | **24.8 ms** |
| Generate time (32 tok) | 165.3 s |
| RAM peak | **1.66 GB** (on a 16 GB machine) |
| Load time | 5.1 s |
| Config | `configs/swlp_qwen_mps.toml` |

### Model-ladder summary (M5, 16 GB, SWLP W=2, FP16)

| Model | Params | Layers | Layer size | Total | tok/s | TTFT | RAM peak | Fits? |
|---|---|---|---|---|---|---|---|---|
| Mistral-7B | ~7B | 32 | 436 MB | 13.96 GB | 0.422 | 13.6 ms | 1.13 GB | ✅ SWLP |
| Qwen2.5-14B | ~14B | 48 | 551 MB | 26.4 GB | **0.194** | **24.8 ms** | **1.66 GB** | ✅ SWLP |
| (20B rung) | ~20B | — | — | ~40 GB | — | — | — | deferred |
| (30B rung) | ~30B | — | — | ~60 GB | — | — | — | deferred |

**Key finding.** The 14B FP16 model (26.4 GB) fits on a 16 GB M5 with only
1.66 GB RAM peak — SWLP's sliding window keeps just 2 layers resident at a time.
Naive full-model load would require > 26 GB RAM and hard-OOM on this machine.

**Throughput scales as expected with model size:**
- 7B → 14B: 0.422 → 0.194 tok/s (0.46× ratio)
- Theoretical from layer count + size: `(32 × 436) / (48 × 551) = 0.53×`
- Measured ratio 0.46× is slightly below theoretical due to increased per-layer
  compute cost at 14B hidden dim (5120 vs 4096).

**RAM stays flat** — the W=2 window uses `2 × layer_size` regardless of model
depth. 14B: 1.66 GB vs 7B: 1.13 GB — the difference is the larger layer size
(550 MB vs 436 MB) plus the larger embed/lm_head tensors.

---

## Phase 7 — FP8 weight storage: a measured negative result

**Hypothesis:** storing layer shards as FP8 (half the bytes) would halve disk
traffic and let the model fit the residency budget — projected 5–15 tok/s.

**A+B spike:** built the FP8 shard format (`model/quant.py`, per-output-channel
scaled `float8_e4m3`, FP16 compute), re-quantized the existing shards, measured.

| Run | tok/s | TTFT | RAM peak | vs FP16 baseline |
|---|---|---|---|---|
| Mistral-7B FP16 (Phase 4) | 0.505 | 13.6 ms | 1.47 GB | — |
| **Mistral-7B FP8** (residency engaged, 32/32 layers cached) | **0.436** | 5.6 ms | 7.34 GB | **0.86× — slower** |
| Qwen2.5-14B FP16 (Phase 6) | 0.194 | 24.8 ms | 1.66 GB | — |
| **Qwen2.5-14B FP8** (streaming, 0 resident) | **0.102** | 12.8 ms | 3.60 GB | **0.53× — ~2× slower** |

FP8 layer sizes: Mistral 218 MB (was 436), Qwen 275 MB (was 551) — disk bytes
genuinely halved. **Throughput still got worse.**

**Quality (the one part of the thesis that held):** the FP8-7B completion is
*byte-identical* to the FP16 completion — "A MacBook Air is an excellent choice
for development due to its powerful processor, large storage capacity, long
battery life, lightweight design, and compatibility". Per-channel-scaled FP8
weight quantization is near-lossless, exactly as the literature predicts.

**Root cause — the bottleneck was never disk bandwidth alone.** SWLP
re-materializes every layer onto the device on *every token*
(`to_empty` → transfer → `load_state_dict` → compute → `evict`). FP8 keeps
compute in FP16, so each layer is dequantized FP8→FP16 on the CPU every token.
That CPU dequant costs as much as the disk read it replaces — so halving disk
bytes bought nothing, and the dequant overhead made it net slower. This
re-confirms the Phase 4 finding (removing the disk read via residency did not
help) from the precision angle.

**Consequence:** the "store weights smaller, dequant in the streaming window"
strategy is a dead end under the current per-token-materialization architecture
— INT4 would fail worse still. The measured path to interactive speed is
*native quantized compute* (MLX 4-bit hit ~30 tok/s in the Phase 3 table) or
speculative decoding (Phase 5), not weight-streaming precision tricks. SWLP's
proven, defensible niche remains **lossless FP16 feasibility of models that do
not fit RAM** — not raw throughput.

---

## Phase 8 — MLX interactive backend: the throughput breakthrough

Phase 7 proved weight-streaming precision tricks cannot beat the disk wall on M5.
Phase 8 takes the measured lesson — "store smaller" only helps if it becomes
"compute faster", and on Apple Silicon only **MLX** has native quantized matmul —
and adds an `MlxRunner`, interchangeable via `build_runner()` (`backend="mlx"`).

All runs on M5 (Apple Silicon, 16 GB), greedy, 32 new tokens, same prompt.

| Model | Backend | tok/s | TTFT | vs SWLP FP16 | Completion vs FP16 |
|---|---|---|---|---|---|
| Mistral-7B | SWLP FP16 streaming | 0.50 | 13.6 ms | 1× | reference |
| **Mistral-7B** | **MLX int8** | **16.0** | 1.09 s | **32× faster** | **byte-identical (lossless)** |
| Mistral-7B | MLX int4 | 27.9 | 2.96 s | 56× faster | minor wording drift |
| Qwen2.5-14B | SWLP FP16 streaming | 0.19 | 24.8 ms | 1× | reference |
| Qwen2.5-14B | MLX int8 | — | — | — | **OOM** — 14 GB model > 16 GB |
| **Qwen2.5-14B** | **MLX int4** | **13.8** | (see note) | **71× faster** | minor wording drift |

**Key results.**

- **The interactive-speed goal is met.** MLX int8 on Mistral-7B runs at **16 tok/s
  and its completion is byte-identical to the FP16 baseline** — int8 weight
  quantization is genuinely lossless here. This is the near-lossless default
  tier: top-notch quality *and* interactive speed.
- **The quality dial is real.** int4 trades a small, visible wording drift for
  ~1.7× more speed (Mistral-7B 28 tok/s). int8 = lossless default; int4 = fast
  tier, clearly labelled — the honest two-tier strategy.
- **Quant doubles as a memory dial.** 14B int8 (~14 GB) OOMs on the 16 GB M5;
  14B int4 (~7 GB) fits and runs at 13.8 tok/s. On 16 GB, int4 is the 14B path.
- The 14B int4 TTFT is inflated by first-run kernel compilation; steady-state
  throughput (MLX's own `generation_tps`) is the reliable figure.
- `ram_peak_bytes` for MLX is measured via psutil RSS, which under-reports
  MLX's memory-mapped / wired GPU memory — treat MLX RAM figures as a floor.

**Positioning.** SWLP's streaming runners remain the **lossless FP16
big-model-feasibility** tool (run a 26 GB model in 1.7 GB RAM). `MlxRunner` is
the **interactive-speed** tool. Two runners, one factory — the user picks the
point on the quality/speed/feasibility surface that fits the job.

---

## Phase 10 — Batched streaming (column-wise execution)

SWLP streams every layer from disk once per decode step. The disk read costs the
same whether 1 or N sequences pass through that layer, so processing a **batch**
in lockstep amortizes the read across all N — FlexGen's "column-wise execution".

**Measured (M5, SmolLM2-360M FP16 shards, window=2, decode-sweep wall time):**

| Batch | Sweep time | Aggregate throughput |
|------:|-----------:|---------------------:|
|     1 |    0.282 s |        3.54 tok/s    |
|     2 |    0.326 s |        6.13 tok/s    |
|     4 |    0.264 s |       15.13 tok/s    |
|     8 |    0.337 s |       23.73 tok/s    |
|    16 |    0.243 s |       65.75 tok/s    |

**Key result.** Decode-sweep wall time is essentially **flat** (~0.24–0.34 s)
across batch 1→16 — the per-sweep disk cost is batch-independent. Aggregate
throughput therefore scales ~linearly: **batch 16 is ~18.5× the batch-1 rate**,
fully lossless FP16.

**Lossless confirmed.** Each sequence in a batch produces output **bit-identical**
to a batch-1 run (greedy decode is row-independent): batched row 0 and `run()`
both emit `"Paris.\n\nParis is the capital"` on the same prompt. Left-padded
sequences in the same batch are also correct.

**Bug fixed en route.** Batched streaming initially produced garbage — root
cause: `DynamicCache(config=…)` pre-structures the per-layer cache for the
config layout and silently corrupts batched (N>1) K/V writes. Batch-1 always
worked, so it was latent through Phases 1–8. `LlamaLikeAdapter.init_past_state`
now uses plain `DynamicCache()`, which grows dynamically and handles any batch.

> Numbers are on SmolLM2-360M (small layers — absolute rates are modest); the
> headline 7B/14B batched measurement is pending re-download of the Mistral /
> Qwen shards. The *scaling* (flat sweep time → linear aggregate throughput) is
> architecture-independent and the conclusion the paper rests on.

---

## Pending — NVIDIA column (MX230)

To be measured on Pop!_OS + MX230 (2 GB VRAM) and added as a second hardware
column: SWLP RAM→VRAM async-PCIe streaming vs the same baselines.

---

## Phase 20 — Hot-path copy elimination (measured improvement)

**Diagnosis.** The Phase 19 warm-cache median (~0.21 tok/s) sat at only 44% of
the cold-SSD ceiling (0.496 tok/s) even though warm reads are far faster than
the SSD — proof the bottleneck was CPU-side overhead, not disk. Per layer per
token the old path performed three full-shard copies on the compute thread
(chunk-list join → safetensors deserialize → host→device cast) plus a wasted
`to_empty(device)` allocation that `assign=True` immediately replaced.

**Fix (`core/shard_io.py` + `core/streaming.py` rewrite):** shards are read
once via `readinto()` into a reusable per-worker buffer, tensors are zero-copy
views into that buffer, and the single host→device copy runs on a persistent
worker pool — `ensure()` on the compute thread is reduced to a
pointer-assigning `load_state_dict(assign=True)`.

**Measured (M5 16 GB, same prompt/harness, direct I/O = F_NOCACHE on, W=2,
greedy; 2026-06-11):**

| Model | Metric | Before | After | Change |
|---|---|---|---|---|
| Mistral-7B FP16 (8 tok) | tok/s | 0.218 | **0.372** (median of 3: 0.360/0.372/0.379) | **+71%** (75% of the 0.496 cold ceiling, up from 44%) |
| Mistral-7B FP16 | prefill | 5.86 s | **2.30 s** | 2.5× faster |
| Mistral-7B FP16 | RAM peak | 1.18 GB | 1.17 GB | flat |
| Qwen2.5-0.5B FP16 (16 tok) | tok/s | 5.47 | **8.39** | +53% |
| Qwen2.5-0.5B FP16 | TTFT | 1.38 s | 0.44 s | 3.1× faster |

Completions are **byte-identical** before/after on both models — the change is
pure I/O-path engineering, zero quality impact.

**Direct-I/O policy (`SWLP_DIRECT_IO=auto|on|off`).** F_NOCACHE is no longer
hardwired: `auto` bypasses the page cache only when the model exceeds ~60% of
available RAM (cyclic access through a too-small LRU cache gets ~0 hits and
only evicts useful pages). Models that fit get page-cache residency for free —
unlike Phase 4 heap residency it is reclaimable under pressure, so it cannot
trigger the memory-compressor collapse. Qwen2.5-0.5B with `auto` (cached):
**9.85 tok/s, TTFT 0.195 s**. Benchmark harnesses pin `swlp_direct_io="on"`
so controlled runs can never be silently warmed.

Numbers above are single runs (n=1) recorded during development; the
paper-grade re-measurement (≥5 runs, median ± IQR via
`bench_common.summarize_runs`, `--cold` purge) is queued in ROADMAP open work.

## Phase 21 — Draft-model speculative decoding (lossless, acceptance-bound)

A resident Qwen2.5-0.5B-Instruct (~1 GB FP16) drafts up to 8 tokens per step;
the streamed Qwen2.5-14B target verifies them all in **one** 48-layer disk
sweep. Unlike the Phase 5 n-gram drafter (~0% acceptance on novel text), the
draft model proposes on every step. Draft length adapts to acceptance (AIMD:
double on full accept, halve on poor accept, floor 1), which caps the
worst-case drafting overhead on low-agreement text.

**Measured (M5 16 GB, Qwen2.5-14B-Instruct FP16 streamed from `shards/qwen-14b`
~28 GB, W=2, direct I/O on, greedy, 32 new tokens; single runs 2026-06-11):**

| Workload | Baseline swlp | Draft-spec | Speedup | Acceptance | tokens/sweep |
|---|---|---|---|---|---|
| Open-ended sentence | 0.187 tok/s | **0.545 tok/s** | **2.9×** | 47.7% | 3.2 |
| Constrained list | 0.197 tok/s | **1.158 tok/s** | **5.9×** | 90.0% | 8.0 |

Completions are **byte-identical** to plain greedy SWLP on both prompts —
every drafted token is greedily verified by the target; speculation changes
throughput only. The first interactive-class number (>1 tok/s) on a 28 GB FP16
model from a 16 GB machine, with zero quality compromise.

Two negative results worth recording:

- **Fixed draft length regresses.** K=8 without adaptation scored 0.151 tok/s
  on the open-ended prompt (−19% vs baseline) — greedy 0.5B/14B chains diverge
  at the first token of a free continuation, and every sweep then pays 8
  wasted drafter forwards. Adaptive K turned the same workload into 2.9×.
- **A 3× larger drafter bought nothing.** Qwen2.5-1.5B matched the 0.5B's
  acceptance exactly (5/34 on the hard prompt) at triple the residency;
  disagreement on open-ended text is about the target's specific phrasing,
  not drafter capacity. 0.5B remains the default
  (`configs/swlp_qwen_draft_mps.toml`).

The speedup is acceptance-bound and acceptance is workload-bound: quote the
2.9×–5.9× range, not a point estimate. Mistral-7B has no same-tokenizer small
draft model, so draft-spec currently applies to the Qwen column only.
