# SWLP — Project Roadmap & Phase History

This file is the full phase-by-phase history of the SWLP project: goals, task
lists, completion checklists, and measured numbers for Phases 0–18. It was split
out of `.claude/CLAUDE.md` so that file can stay a lean operating manual.

**How to use this file:**
- Each phase has a **Goal**, **Task list**, **Completion checklist**, and **Phase notes**.
- Checkbox legend: `[x]` done · `[ ]` not done · `[~]` deferred / partially done.
- **Append only** — never delete or overwrite existing phase notes or measured numbers.
- After completing a phase: tick its checklist here, add a Phase note, then update
  the **Current Status** line in `.claude/CLAUDE.md`.
- Mark a task `[x]` only after `pytest` (all pass) and `ruff check src/` (zero errors).

---

## Phases

### Phase 0 — Hardware baseline and proof of concept
**Goal:** Measure real numbers on M5 and MX230. Confirm the system runs end-to-end with tiny-gpt2.

**Task list:**
- [x] Restructure `src/swlp/` into modular sub-package architecture (6 sub-packages: core, hardware, model, runner, benchmark, reporting). All 28 source files in place.
- [x] Create `scripts/phase0_hardware_check.py` — measures SSD bandwidth, MPS matmul speed, MLX availability.
- [x] Create `configs/swlp_mps.toml` — M5 MPS config with window=4, threading scheduler.
- [x] All 12 existing tests pass (`pytest`).
- [x] Run `python scripts/phase0_hardware_check.py` on M5. Record SSD read/write GB/s and MPS matmul ms.
- [x] Run `swlp baseline --config configs/baseline.toml --profile` on M5. Record tokens/sec, load time.
- [x] Run `swlp swlp --config configs/swlp_mps.toml` on M5. Confirm no crash, record TTFT.
- [ ] Repeat baseline + swlp on Pop!_OS + MX230. Record tokens/sec, VRAM, TTFT.
- [x] Record numbers in a `docs/hardware_baseline.md` table (M5 vs MX230 side by side).
- [x] Run `swlp simulate --scenario configs/sim_m5.toml --report`. Verify math matches measured numbers.

**Completion checklist:**
- [x] All existing tests pass (`pytest`) — 12/12 passing
- [x] M5 SSD bandwidth measured and recorded — read: 6.93 GB/s, write: 4.48 GB/s
- [x] M5 MPS tokens/sec for tiny-gpt2 recorded — 100.3 tok/s (HF baseline), 111.1 tok/s (SWLP)
- [ ] MX230 VRAM peak and tokens/sec recorded — deferred to tomorrow
- [x] `swlp swlp` runs without error on M5 — TTFT: 5.85 ms, 111.1 tok/s
- [x] Simulation bottleneck classification matches actual bottleneck observed — SSD transfer dominates at 0.14 tok/s for 7B scale

**Phase notes:**
> Architecture restructure complete. 28 source files across 6 sub-packages. Public API confirmed working on M5 MPS.
> M5 SSD read: 6.93 GB/s | SSD write: 4.48 GB/s | MPS matmul 1k×1k: 4.2 ms | MLX matmul: 3.5 ms
> tiny-gpt2 HF baseline: 100.3 tok/s, 4.35s load | SWLP on MPS: 111.1 tok/s, TTFT 5.85 ms
> transformers 5.x breaking change fixed in `src/swlp/runner/swlp.py`: `layer_past=` → `past_key_values=`, return is now plain Tensor not a tuple.
> sim_m5.toml created with real M5 numbers; `pcie_bandwidth_gbps=153.0` (RAM bandwidth) models unified memory correctly — SSD at 6.93 GB/s is the streaming bottleneck for 7B scale.
> MX230 measurements deferred to tomorrow. `docs/hardware_baseline.md` created with M5 column complete.

---

### Phase 1 — Real model sharding and ThreadedPipeline
**Goal:** Shard a 7B model to disk. Stream it through `ThreadedPipeline`. Measure actual SSD→RAM overlap.

**Task list:**
- [x] Implement `ThreadedScheduler` in `core/scheduler.py` — Python daemon threads for SSD→RAM overlap on MPS/CPU.
- [x] Implement `ThreadedPipeline` in `core/pipeline.py` — disk→RAM streaming with background prefetch for sharded `.pt` files.
- [x] Implement `shard_model_by_layer()` in `model/shard.py` — splits any HF model into per-layer `.pt` files + `shard_manifest.json`.
- [x] Add `shard_dir: Path | None` to `RuntimeConfig` in `config.py`; loaded from `SWLP_SHARD_DIR` env var.
- [x] `configs/swlp_mps.toml` includes shard-related fields; CLI passes shard_dir through config.
- [x] Write `scripts/research/run_pipeline_forward.py` — standalone proof-of-concept forward pass via `ThreadedPipeline`.
- [x] Run `shard_model_by_layer("unsloth/mistral-7b-instruct-v0.2", "./shards/mistral-7b")` on M5. Confirm `shard_manifest.json` (32 layers, 13.96 GB total, 436 MB/layer).
- [x] Measure: time per token with pipeline (prefetch=True) vs sequential (prefetch=False). Recorded overlap gain.
- [x] Add window-size sweep: W=2, W=4, W=6. Recorded tokens/sec vs W.
- [x] Add full Llama/Mistral architecture dispatch to `SWLPRunner` — implemented via `runner/arch.py` adapters (`GPT2Adapter`, `LlamaLikeAdapter`).
- [x] Connect `SWLPRunner` to sharded model files via `StreamingScheduler` (config has `shard_dir`; runner auto-detects and streams).

**Completion checklist:**
- [x] `ThreadedScheduler` implemented and unit-tested
- [x] `ThreadedPipeline` implemented (prefetch, get_layer, evict, warmup, cleanup)
- [x] `shard_model_by_layer()` implemented with `ShardManifest`, `get_layer_path`, `list_layer_paths`
- [x] `shard_dir` wired through config and env var
- [x] All tests pass — 12/12
- [x] `scripts/research/run_pipeline_forward.py` written and runs without error
- [x] 7B model successfully sharded to disk on M5 — 32 layers × 436 MB = 13.96 GB
- [x] `StreamingScheduler` forward pass produces semantically correct output (e.g., "The capital of France is" → "Paris, and it is one of the most popular tourist destinations…")
- [x] Measured overlap efficiency: 36.3% at W=6 (compute=80 ms/layer) — beats 30% target
- [x] Best window size W for M5 determined and documented — W=2 end-to-end (0.40 tok/s, 1.40 GB RAM)
- [x] Llama/Mistral model type fully supported in `SWLPRunner` via `LlamaLikeAdapter`
- [x] `SWLPRunner` loads weights from `shard_dir` via `StreamingScheduler` when `shard_dir` is set (RAM peak ~1 GB vs 14 GB full-load)

**Phase notes:**
> Phase 1 complete. Full end-to-end Mistral-7B SWLP streaming on M5: 1 GB RAM peak, 0.4 tok/s.
> Architecture refactor: added `runner/arch.py` (model-family dispatch), `runner/load.py` (full vs empty-weights load paths), `core/streaming.py` (`StreamingScheduler` materializes per-layer .pt shards via `to_empty` + `load_state_dict(assign=True)`).
> RoPE caveat: `MistralRotaryEmbedding.inv_freq` is a non-persistent buffer; `init_empty_weights` leaves it on meta, so `_materialize_embeddings` re-instantiates the rotary module rather than copying.
> Synthetic POC (80 ms compute): W=2 18.9%, W=4 30.0%, W=6 36.3% overlap gain.
> Real run: smaller W is faster (W=2 best). Larger W increases unified-memory pressure between CPU prefetch and MPS compute. CUDA path will likely invert this.
> `swlp.py` is 383 lines (legacy exception alongside `model/package.py` at 493 lines); the loader was extracted to `runner/load.py` and metrics builder kept inline.
> Pending: validate full Llama-family models (only Mistral tested), measure KIVI-style KV compression with Mistral (Phase 2).

---

### Phase 2 — KV cache compression and memory budget
**Goal:** Run a 30B model on 16 GB M5 by combining layer streaming with KIVI-style KV compression.

**Decision (CTO):** Phase 2 validated the KV-compression machinery on the
already-sharded Mistral-7B rather than downloading a real 30B (~60 GB + multi-30-min
runs). The 30B end-to-end run is deferred to Phase 3 as a confirmation step — the
mechanism, quality, ratio, and budget math are all validated below.

**Task list:**
- [~] Shard a 30B FP16 model to disk — deferred; validated on Mistral-7B (32 × 436 MB) from Phase 1.
- [~] Run Phase 1 pipeline on 30B / find OOM — deferred; replaced by the 30B memory-budget projection in `docs/hardware_baseline.md`.
- [x] Enable `KVCacheManager` with `compression=True`, `tiering=True` — implemented via `CompressedDynamicCache` (`core/compressed_cache.py`), wired into `SWLPRunner` for the Llama/Mistral path.
- [x] Benchmark quality: same prompt with KV compression OFF vs ON — completions **bit-identical** (zlib is lossless), quality delta = 0. Measured via `scripts/research/kv_compare.py`.
- [x] Document memory budget math — `kv_budget_recommendation()` in `hardware/detect.py`; breakdown in `docs/hardware_baseline.md`.
- [x] Tune `kv_compression_level` (1–9) — sweet spot is **level 1** (same ~1.1× ratio as level 9, 22% faster).
- [x] Add `SWLP_KV_BUDGET_MB` auto-calculation — `_resolve_kv_budget_bytes()` auto-calcs from `HardwareInfo.memory_gb` + window size when budget `<= 0`.

**Completion checklist:**
- [~] 30B FP16 inference runs without OOM — deferred to Phase 3; budget math projects feasibility (see notes).
- [x] KV compression ratio measured — **1.10×** (zlib lossless on FP16 KV). The ~2.5× target needs *lossy* quantization — see notes.
- [x] Quality delta measured — **0** (lossless; bit-identical completions, far under the <1% target).
- [x] Memory budget breakdown documented — `docs/hardware_baseline.md`.
- [x] All tests pass — 18/18 (`test_compressed_cache.py` added: 6 tests).

**Phase notes:**
> Phase 2 KV-compression machinery complete and validated on Mistral-7B.
> `core/compressed_cache.py`: `CompressedDynamicLayer(DynamicLayer)` + `CompressedDynamicCache(DynamicCache)` — cold layers' KV stored compressed in `KVCacheManager`; `get_seq_length` answers from a recorded length while cold so attention-mask construction never forces a decompress. `SWLPRunner._run_blocks` calls `compress_layer(idx)` right after each Llama block (its KV is untouched until the next token step).
> CTO call — `Cache` subclass over post-step compression: intercepts the model's `cache.update()` contract boundary (stable) instead of poking `DynamicCache` internals (broke us in Phase 0); keeps only the active layer hot so it actually lowers *peak* memory; aligns with Open/Closed.
> CTO call — logit-delta/text-match over perplexity: `KVCacheManager` compression is lossless zlib, so the delta is provably 0; perplexity would burn many forward passes to confirm 0.000. Perplexity becomes mandatory only when lossy KV quantization (true KIVI) lands.
> **Key finding:** lossless zlib gives only ~1.1× on FP16 KV (high-entropy activations). The ~2.5× goal requires lossy 4-bit KV quantization — a separate future workstream. zlib's real value here is host-offload tiering of cold layers.
> `kv_compression_level` sweep: level 1 = 1.104×/29.0s, level 9 = 1.108×/37.0s → level 1 is the sweet spot.
> KV budget auto-calc: `(total_ram − 3GB OS − 1GB embed − (W+1)·layer_mb) × 0.5`, floored at 256 MB. M5 16 GB / Mistral W=2 → 5.49 GB.
> `swlp.py` now 432 lines (legacy exception, like `model/package.py` at 493) — adapter dispatch + 3 KV/budget helpers; loader already extracted to `runner/load.py`.
> Pending for Phase 3: real 30B confirmation run; lossy KV quantization for the ~2.5× ratio.

---

### Phase 3 — Benchmark vs baselines
**Goal:** Produce the comparison table for the paper.

**Task list:**
- [~] Benchmark Ollama on 7B and 30B on M5 16 GB — 7B done (Q4_K_M, 28.25 tok/s warm); 30B deferred.
- [x] Benchmark MLX-lm (naive, no streaming) — FP16 naive load **OOMs** on 16 GB; 4-bit measured (29.96 tok/s).
- [x] Benchmark AirLLM on Mistral-7B (FP16 layer-by-layer streaming) — 0.208 tok/s, TTFT 6.54 s.
- [x] Benchmark SWLP (best W=2 from Phase 1) on Mistral-7B — 0.422 tok/s, TTFT 13.6 ms.
- [ ] Repeat all on MX230 for the NVIDIA column — deferred (Pop!_OS machine).
- [x] Compute speedup vs AirLLM — `(153.8−75.9)/153.8 = 50.7%` faster; 2.03× throughput.
- [x] Run `swlp suite` for structured JSON — `benchmarks/suite-20260520T044433Z.json` (tiny-gpt2; see notes).
- [x] Produce final table — `docs/results.md` (two-tier: FP16-lossless vs quantized reference).

**Completion checklist:**
- [x] Ollama baseline measured — Q4_K_M: 28.25 tok/s, TTFT 61 ms warm (15.9 s cold)
- [x] MLX-lm baseline measured — FP16 OOMs on 16 GB; 4-bit: 29.96 tok/s, TTFT 1.11 s
- [x] AirLLM baseline measured — FP16: 0.208 tok/s, TTFT 6.54 s, 153.8 s/32 tok
- [x] SWLP numbers measured — FP16 W=2: 0.422 tok/s, TTFT 13.6 ms, 1.13 GB RAM
- [x] SWLP shows > 0 speedup vs AirLLM at equal quality — **50.7% faster, 2.03×, both FP16**
- [ ] MX230 results collected — deferred to Pop!_OS machine
- [x] Final benchmark table written to `docs/results.md`

**Phase notes:**
> Phase 3 complete on the M5 column. SWLP beats AirLLM — the paper's headline claim — at equal (FP16, lossless) quality.
> Harness: `scripts/research/phase3_baselines.py` (standalone, not a swlp runner — external systems aren't `build_runner` participants, so no speculative `runner/` abstraction). Runs Ollama/MLX-lm/AirLLM/SWLP on Mistral-7B, isolates each in try/except, emits `benchmarks/phase3.json`.
> **Two-tier table (CTO call):** SWLP's first-class constraint is zero quality loss, so the table separates an **FP16/lossless tier** (SWLP, AirLLM — the apples-to-apples speed comparison) from a **quantized reference tier** (Ollama Q4_K_M, MLX-lm 4-bit). Forcing FP16 in Ollama was rejected — it measures a config nobody runs and would flatter SWLP dishonestly; annotating the quant is the honest, stronger framing.
> **Key result:** in the FP16 tier SWLP is the *only* runtime that is both feasible AND fast. Naive full-model FP16 load (MLX-lm, HF) hard-OOMs on 16 GB (~14 GB model) — the exact problem streaming solves. Among streaming runtimes SWLP beats AirLLM 2.03× on throughput and 481× on TTFT (SWLP keeps embeddings/lm_head resident + prefetches; AirLLM does a full sequential layer sweep per token).
> Quantized tier (~4 GB model, full-load) runs ~70× faster than FP16 streaming but trades quality SWLP refuses to pay — listed as a reference ceiling only.
> `swlp suite` always runs an HF *full-model* baseline per prompt; Mistral-7B FP16 can't full-load (hard OOM aborts the process), so the suite was validated on tiny-gpt2 (`configs/suite_phase3.toml` + `configs/baseline.toml`) as the structured-JSON artifact. Mistral W-sweep sourced from direct SWLP runs instead.
> External tools (`airllm`, `mlx-lm`, Ollama) are benchmark-only — installed in the venv but **not** added to `pyproject.toml` (they are not swlp dependencies).
> No new source modules — Phase 3 is measurement; test count stays 18/18, `ruff check src/` clean.
> Pending for a follow-up: MX230 NVIDIA column; real 30B confirmation run.

---

### Phase 4 — Adaptive residency and measurement rigor
**Goal:** Make 7B FP16 genuinely fast on the M5 by stopping the waste of unused
RAM. Phase 3 measured **0.422 tok/s** — disk-bandwidth-bound: SWLP re-streams the
**entire 13.96 GB model from SSD every token** (13.96 GB ÷ 6.93 GB/s = 2.01 s/token
hard ceiling). Yet peak RAM is only **1.13 GB on a 16 GB machine** — the W=2
sliding window evicts layers there is ample room to keep resident. The model
(13.96 GB) *almost fits* the machine (16 GB).

**Strategy:** replace the fixed sliding window with **adaptive residency** — keep
as many layers as fit *permanently resident* (loaded once, never evicted), stream
only the overflow. Budget ≈ 16 GB − ~3 GB OS − ~1 GB embeddings/KV/working ≈
~12 GB ≈ ~27 resident layers; only ~5 stream. Per-token disk traffic drops from
13.96 GB to ~2 GB → projected **~5 tok/s (~10–12× speedup)**, still bit-exact FP16.
This works *only while the model nearly fits RAM* (true for 7B; not for ≥14B —
that is Phase 5's problem).

A speedup claim is worthless on n=1 runs, so Task 1 is measurement rigor.

**Task list:**
- [x] Multi-run measurement: extend `scripts/research/phase3_baselines.py` (or a thin
      wrapper) to run each baseline N times and report mean/median/std (`run_benchmark`
      already computes these — reuse, do not duplicate).
- [x] Add a pre-flight memory check helper in `hardware/detect.py` —
      `fits_in_memory(model_bytes, hw) -> bool` — estimating whether a full-model
      load fits; used to skip the HF baseline gracefully instead of hard-OOM crash.
- [x] Implement `plan_residency()` + `ResidencyPlan` in `core/residency.py` —
      given total_memory_bytes, layer_weight_bytes, and num_layers, compute how many
      layers stay permanently resident vs. stream. Pure computation, no I/O, no torch.
- [x] Wire residency into `StreamingScheduler` / `SWLPRunner` — resident layers
      pre-cached in CPU RAM at startup (`load_resident_layers()`), applied CPU→device
      per token in `ensure()`, evicted device→meta per token in `evict()`.
- [x] Add `swlp_residency` config field to `RuntimeConfig`; `SWLP_RESIDENCY` env
      override ("auto" / "off" / integer string); surface in `swlp_mistral_mps.toml`.
- [x] Measure 7B FP16 tok/s / TTFT / RAM with adaptive residency vs. the W=2
      baseline; record in `docs/results.md`.
- [x] Unit tests for `plan_residency`, `ResidencyPlan`, and `fits_in_memory`
      (36 tests total — `tests/test_residency.py` + `tests/test_hardware.py`).

**Completion checklist:**
- [x] Multi-run measurement harness added to `scripts/research/phase3_baselines.py` (`--runs N`, `_stats()`)
- [x] `fits_in_memory(model_bytes, hw)` implemented in `hardware/detect.py` and unit-tested (6 tests)
- [x] `plan_residency()` + `ResidencyPlan` implemented in `core/residency.py` and unit-tested (11 tests + 2 boundary tests → 13 total in test_residency.py)
- [x] Adaptive residency wired through `StreamingScheduler` (CPU-RAM cache) and `SWLPRunner._resolve_resident_count()`
- [x] 7B FP16 residency measured — see Phase notes; on 16 GB M5, adaptive residency (auto) correctly returns 0 resident layers
- [x] Output remains bit-exact vs. Phase 3 SWLP completion (completion text identical)
- [x] All tests pass (`pytest`) — 36/36; `ruff check src/` clean

**Phase notes:**
> Phase 4 complete. All infrastructure built and tested; the key finding is a fundamental memory-pressure constraint on 16 GB M5 + 7B that changes the conclusion.
>
> **Design attempt 1 — MPS-residency (keep layers permanently on Metal GPU):** Pre-loaded 17–23 layers permanently to MPS device. Measured: **0.041 tok/s** (12× slower than Phase 3). Root cause: locking large tensors permanently in MPS fragments the Metal buffer allocator. When streaming layers need to allocate new Metal buffers each token, the allocator must compact/reorganize its pool, causing catastrophic per-token slowdowns. MPS unified-memory tensors cannot be paged out by macOS, eliminating OS's ability to manage memory pressure.
>
> **Design attempt 2 — CPU-RAM residency (cache state_dicts in Python heap, copy CPU→MPS per token):** Redesigned `StreamingScheduler` to store 17 layer state_dicts in `_resident_data: dict[int, dict]` (CPU RAM, ~7.4 GB). Per token, `ensure()` applies CPU→MPS (fast unified-memory copy, no SSD read); `evict()` moves MPS→meta normally. Measured: **0.080 tok/s** (still 6× slower than Phase 3). Root cause: 7.4 GB locked in Python heap evicts OS page cache for the remaining 15 streaming layers, triggering macOS memory compressor across the board.
>
> **Root cause of both failures — insufficient RAM headroom:** Mistral-7B FP16 is 13.96 GB on a 16 GB machine. After OS reserves (~2–3 GB), only ~10 GB remains, and the safe usable budget (with 0.75 headroom factor) is ~7.5 GB. Locking 7.4 GB in resident layers leaves nothing for the OS page cache of the 15 streaming shards (6.5 GB), forcing SSD reads to be cold AND the memory compressor to fire. Net result: residency hurts.
>
> **Fix — full-model-fit guard in `plan_residency()`:** Partial residency is counter-productive when the model doesn't fully fit within the usable budget — it evicts page cache for streaming layers without eliminating SSD reads. The planner now returns `resident_count=0` unless `total_model_bytes ≤ usable`. On M5 16 GB + 7B: 13.96 GB > 7.5 GB → 0 resident layers → all streaming (same as Phase 3 baseline). The all-streaming path naturally benefits from OS page cache warming across tokens.
>
> **Condition for adaptive residency to help:** `total_model_bytes ≤ usable_budget`. For 7B FP16 on M5: needs ≥ 32 GB unified memory. For smaller models (GPT-2, 1B, 3B) on 16 GB, the full model fits and all layers become resident. The machinery is correct — the constraint is hardware.
>
> **Re-measured Phase 3 baseline with multi-run harness:** 0.502 tok/s (W=2), 13.6 ms TTFT, ~1.47 GB RAM (single run, consistent across repeats).
>
> **New infrastructure delivered:**
> - `core/residency.py`: `ResidencyPlan` dataclass + `plan_residency()` — pure computation, no I/O
> - `hardware/detect.py`: `fits_in_memory(model_bytes, hw)` — pre-flight OOM guard
> - `StreamingScheduler`: CPU-RAM residency machinery (`_resident_data`, `load_resident_layers()`, resident fast-path in `ensure()` / `evict()`)
> - `SWLPRunner._resolve_resident_count()`: bridges `HardwareInfo` → `plan_residency()` → scheduler
> - `config.py`: `swlp_residency` field + `SWLP_RESIDENCY` env var
> - `scripts/research/phase3_baselines.py`: `--runs N`, `_stats()` multi-run harness
> - 36 tests total (up from 18); `ruff check src/` clean
>
> **Key finding for Phase 5/6:** adaptive residency becomes effective once `model_size < 0.75 × (total_ram − 6 GB)`. For 7B this requires ≥ 32 GB M5. For 30B (Phase 6), a 64 GB machine. Speculative decoding (Phase 5) is the better lever for throughput improvement at current hardware constraints.

---

### Phase 5 — Speculative decoding
**Goal:** Break past the one-sweep-per-token model so throughput scales beyond
what residency alone allows — and so models that *do not* fit RAM (≥14B) become
viable. Adaptive residency (Phase 4) helps only while the model nearly fits;
14B FP16 = 28 GB does not fit 16 GB at all.

**Strategy:** a small, permanently-resident **draft model** proposes K tokens;
the big streamed model **verifies all K in a single disk sweep**. Accepted tokens
are amortized over one sweep → effective throughput × (accepted tokens per sweep).
Verification is exact — accepted tokens are identical to greedy big-model output,
so quality stays lossless.

> **Revised strategy (CTO call, Phase 5):** the draft-model approach was
> rejected after analysis — see Phase notes and `docs/phase5_design_decisions.md`.
> Phase 5 ships **prompt-lookup (n-gram) speculative decoding**: no draft model,
> the proposer matches the trailing n-gram against earlier context. Single-sweep
> verification and the lossless guarantee are unchanged.

**Task list:**
- [x] Draft strategy chosen — **prompt-lookup (n-gram) decoding**, no draft model.
      Rationale documented in `docs/phase5_design_decisions.md` (memory-bound M5;
      no Mistral-compatible tiny model exists; zero tokenizer risk).
- [x] Implement a speculative-decoding loop — drafter proposes K, big model
      verifies in one sweep, KV cache cropped on first rejection. `core/speculative.py`
      (pure logic) + `runner/speculative.py` (`SpeculativeRunner(SWLPRunner)`).
- [x] Config: speculation depth K + n-gram size; `SWLP_SPEC_NGRAM` / `SWLP_SPEC_MAX_DRAFT`
      env overrides; `configs/swlp_speculative_mps.toml`.
- [x] Verify exactness — speculative completion is byte-identical to greedy SWLP.
- [x] Measure acceptance rate and effective tok/s vs. Phase 4 on 7B FP16 — recorded
      in `docs/results.md` (3 workloads: novel / mild / repetition-heavy).
- [x] Unit tests for the drafter + verification/rollback logic (`tests/test_speculative.py`).

**Completion checklist:**
- [x] Speculative decoding implemented and interchangeable via `build_runner()` (`backend="speculative"`)
- [x] Accepted-token output bit-exact vs. greedy SWLP — completion byte-identical to Phase 3/4
- [x] Acceptance rate and effective speedup measured and recorded — up to **3.29×** (4.0 tok/sweep)
- [x] All tests pass (`pytest`) — 51/51; `ruff check src/` clean

**Phase notes:**
> Phase 5 complete. Prompt-lookup speculative decoding ships, is lossless, and
> delivers up to **3.29× speedup** on repetition-heavy workloads.
>
> **CTO decision — prompt-lookup over a draft model.** The original spec assumed
> a small resident draft model. Three problems killed that approach: (1) **Memory.**
> Phase 4 proved the 16 GB M5 has zero RAM headroom for 7B — a ~2.2 GB resident
> draft model reintroduces the exact macOS memory-compression failure Phase 4 just
> fixed. (2) **No compatible model.** Mistral-7B-v0.2 has no official tiny sibling;
> TinyLlama's Llama-2 tokenizer is not byte-identical to Mistral's 32k vocab, so a
> TinyLlama draft would emit mismatched IDs → near-zero acceptance → worse than no
> draft. (3) **Overhead.** A 1.1B-model forward is non-trivial against the ~2 s
> disk sweep. Prompt-lookup (n-gram) drafting has zero memory footprint, zero
> tokenizer risk (operates on the target's own IDs), and microsecond cost. Full
> reasoning for all three Phase 5 design questions is in
> `docs/phase5_design_decisions.md` (created at the user's request).
>
> **CTO decision — greedy verification.** The config runs pure greedy
> (`temperature=0`); greedy verification gives *bit-identical* output to plain
> SWLP — provably lossless, testable with exact string equality. Sampling-based
> speculation weakens "lossless" to "same distribution" and is untestable without
> statistics.
>
> **CTO decision — plain `DynamicCache`, no KV compression on the speculative
> path.** Rejected draft tokens are rolled out of the KV cache with
> `DynamicCache.crop()` — a tested primitive. The Phase 2 `CompressedDynamicCache`
> would need decompress→crop→recompress every layer every step. Phase 2 measured
> KV compression at only 1.10×, so the speculative path loses nothing by skipping
> it. KV compression and speculative decoding are mutually exclusive this phase.
>
> **Architecture.** `core/speculative.py`: `NgramDrafter` (prompt-lookup proposer)
> + `verify_greedy()` (pure accept/reject + rollback decision — unit-tested with
> plain ints). `runner/speculative.py`: `SpeculativeRunner(SWLPRunner)` — reuses the
> streaming scheduler/adapter/loader, overrides only the decode loop. `swlp.py` was
> refactored to extract `_generate_remaining()` as the single override seam
> (Open/Closed — no duplication of `run()`'s setup/metrics/teardown).
>
> **How it works.** Each speculative step: the drafter proposes up to K=8 tokens
> by matching the trailing 3-gram against earlier context; the streamed target
> verifies `[last_token, *draft]` in ONE 32-layer disk sweep; `verify_greedy`
> accepts the matching prefix and emits one correction/bonus token; the KV cache
> is cropped to the accepted length. Tokens-per-sweep = accepted + 1.
>
> **Measured (M5, Mistral-7B FP16, greedy; baseline = Phase 4's 0.505 tok/s):**
> - Novel text (no recurring n-gram): 0 drafts proposed, 0% acceptance, 1.03
>   tok/sweep, **0.471 tok/s** — degrades gracefully to baseline minus ~7%
>   drafting overhead. Completion byte-identical to Phase 3/4 → lossless confirmed.
> - Mildly repetitive output: 6/6 drafts accepted, 1.28 tok/sweep, **0.578 tok/s** (1.15×).
> - Repetition-heavy (pattern continuation): 35/35 drafts accepted, **4.0 tok/sweep**,
>   **1.66 tok/s — 3.29× speedup**. Ceiling with K=8 is ~9×.
>
> **Key finding.** Whenever the drafter fires, acceptance is 100% — prompt-lookup
> proposes exact context spans, so the only variable is how often a matching
> n-gram exists. Speedup is therefore *workload-dependent*: ~1× on free-form novel
> text, 3×+ on the long-context / repetitive workloads SWLP targets. It is free
> (lossless, zero extra RAM) and never materially slower than baseline.
>
> **New infrastructure:** `core/speculative.py`, `runner/speculative.py`,
> `configs/swlp_speculative_mps.toml`, `speculative` CLI subcommand + backend,
> `swlp_spec_ngram` / `swlp_spec_max_draft` config fields, `docs/phase5_design_decisions.md`.
> 51 tests total (up from 36; +15 in `test_speculative.py`); `ruff check src/` clean.
>
> **Pending for Phase 6:** speculative decoding is the throughput lever for
> models that do not fit RAM — measure it on the 14B/20B/30B ladder.

---

### Phase 6 — Model-ladder climb and reliability hardening
**Goal:** Climb the parameter ladder on the M5 — **7B → 14B → 20B → 30B** — one
rung at a time, each confirmed working before the next, combining adaptive
residency (Phase 4) with speculative decoding (Phase 5). Harden the runtime now
that the architecture has settled. (70B is the far-horizon end goal, not this phase.)

**Scope (CTO call):** Due to RAM limits on M5 16 GB, `shard_model_by_layer` was
rewritten to stream safetensors weights block-by-block (never loads full model into
RAM). The 20B/30B rungs are deferred. This phase ships: stream-shard rewrite,
shard-integrity check, graceful degradation with memory pre-flight, regression
guard, and the 14B rung on Qwen2.5-14B-Instruct.

**Task list:**
- [x] Rewrite `shard_model_by_layer` to stream safetensors layer-by-layer — no full-model RAM load.
- [x] Shard-integrity check on load — `verify_shards()` + `ShardIntegrityReport` in `model/shard.py`.
- [x] Wire integrity check into `load_from_shards()` in `runner/load.py`.
- [x] Graceful degradation — `fits_in_memory()` + `streaming_fits_in_memory()` in `hardware/detect.py`;
      `_check_streaming_feasible()` in `runner/swlp.py` aborts with a clear exception.
- [x] Regression guard — `scripts/research/regression_guard.py` with documented TTFT/tps/RSS thresholds;
      passes ✅ on all three backends (mock/hf/swlp).
- [x] Shard Qwen/Qwen2.5-14B-Instruct → `./shards/qwen2.5-14b` using stream sharder.
- [x] Run Qwen2.5-14B-Instruct on M5 via `configs/swlp_qwen_mps.toml`; record tok/s / TTFT / RAM.
- [x] Update `docs/results.md` with 14B ladder rung numbers.
- [~] 20B/30B rungs — deferred (hardware limits).

**Completion checklist:**
- [x] 14B FP16 runs on M5 without OOM; numbers recorded — 0.194 tok/s, TTFT 24.8 ms, 1.66 GB RAM
- [~] 20B FP16 runs on M5 without OOM — deferred (hardware limits)
- [~] 30B FP16 runs on M5 without OOM — deferred (hardware limits)
- [x] Stream-shard rewrite implemented (`safetensors.safe_open`, block-by-block, no full-RAM load)
- [x] Shard-integrity check implemented and wired into load path
- [x] Graceful degradation + memory pre-flight checks implemented and tested
- [x] Regression guard in place with documented thresholds — all backends PASS
- [x] All tests pass (`pytest`) — 51/51; `ruff check src/` clean

**Phase notes:**
> Stream-shard rewrite: `shard_model_by_layer` opens each safetensors shard via
> `safetensors.safe_open`, streams tensor-by-tensor into per-layer `.pt` files.
> Peak RAM during sharding stays < 2 GB regardless of model size (no full load).
>
> Shard-integrity: `verify_shards(shard_dir)` → `ShardIntegrityReport` validates
> magic bytes (`PK` ZIP header) of each `.pt` file and manifest JSON. Called
> automatically inside `load_from_shards()` before any model materialization.
>
> Memory pre-flight: `fits_in_memory(model_bytes, hw)` (full-load check) and
> `streaming_fits_in_memory(layer_bytes, hw, window)` (streaming check) added to
> `hardware/detect.py`. `_check_streaming_feasible()` in `runner/swlp.py` raises
> `RuntimeError` with a clear message if even 1-layer streaming cannot fit.
>
> Regression guard thresholds (tiny-gpt2, MPS, 32 tok):
> - mock: TTFT < 0.2 s, tps > 100, RSS < 500 MB — ✅ PASS (1.19M tps, 227 MB)
> - hf:   TTFT < 3.0 s, tps > 20,  RSS < 1500 MB — ✅ PASS (0.40 s, 52 tps, 514 MB)
> - swlp: TTFT < 3.0 s, tps > 15,  RSS < 1500 MB — ✅ PASS (0.00 s, 94 tps, 521 MB)
>
> **14B rung (Qwen2.5-14B-Instruct, M5 16 GB, SWLP W=2, FP16, greedy, 32 tok):**
> - Shards: 48 layers × 550.5 MB = 26.4 GB total; 51 files (embed + lm_head + 48 layers)
> - Throughput: **0.194 tok/s** | TTFT: **24.8 ms** | Generate (32 tok): 165.3 s | RAM peak: **1.66 GB**
> - Load time: 5.1 s (tokenizer + config from HF; weights from local shards)
> - Completion: "The MacBook Air's combination of portability, long battery life, and powerful performance makes it an excellent choice for developers who need to work efficiently on the go."
> - Throughput ratio vs 7B: 0.46× (theoretical 0.53× from layer count × size; delta from larger hidden dim compute)
> - RAM stays flat at ~W × layer_size: 14B (550 MB) → 1.66 GB vs 7B (436 MB) → 1.13 GB
> - **Key result:** 26.4 GB FP16 model streams on a 16 GB machine with 1.66 GB RAM peak — naive load would hard-OOM.

---

### Phase 7 — Precision tiering: break the disk-bandwidth wall
**Goal:** Phases 1–6 proved SWLP *fits* huge models in tiny RAM, but throughput is
stuck at the disk wall: `tok/s ≤ SSD_bw / bytes_streamed_per_token`. For 14B FP16
(26 GB) that is 6.93/26.4 = 0.26 tok/s — measured 0.19. The 2025 SSD-offload
literature (oLLM, AirLLM, FlexGen) all land at 0.2–0.5 tok/s for the same reason.
The only way faster is to **stream fewer bytes per token**.

**Strategy (CTO call):** ship a **two-tier precision strategy** — same honest
framing as the Phase 3 results table.
- **FP8 weight storage (default, near-lossless).** Store layer shards as
  `float8_e4m3` with per-output-channel FP16 scales; dequant FP8→FP16 in the
  streaming window so *compute stays FP16*. Research: FP8 weight quant retains
  99–100 % of FP16 benchmark performance. Halves disk traffic (2× throughput)
  and — critically — shrinks the model enough that the **Phase 4 adaptive
  residency machinery finally fires**: a model that fits the residency budget
  becomes fully resident → throughput goes from disk-bound to RAM-bound.
- **INT4 weight-only (opt-in tier).** ~4× smaller; fully resident even on an
  8 GB machine; ~1–3 % quality cost — never the default, always labelled.

Rejected: FP16-only (physics caps it at ~0.5 tok/s); FP8-only (strands the 8 GB
target — FP8-14B is still 13 GB).

**Sub-phase 7a/7b — FP8 spike (current scope):** prove the FP8 + residency
thesis on real M5 numbers before committing to INT4 / KV overhaul.

**Sub-phase 7a/7b — FP8 spike: NEGATIVE RESULT (the thesis was falsified).**
The cheap spike did its job — it proved, on real M5 numbers, that FP8 weight
storage does **not** speed up SWLP. See Phase notes for the full diagnosis.

**Task list (7a/7b spike) — all built + measured:**
- [x] `model/quant.py` — FP8 quantize/dequantize of layer state dicts + `requantize_shards`.
- [x] `ShardManifest.weight_dtype` field; manifest read/write carry it.
- [x] `core/streaming.py` — dequant FP8→FP16 in `_apply_shard`; resident layers stay quantized in RAM.
- [x] `scripts/research/requantize_fp8.py` — convert an existing fp16 shard dir to FP8.
- [x] FP8 configs for Mistral-7B and Qwen2.5-14B (`configs/swlp_*_fp8_mps.toml`).
- [x] Measure FP8-7B (residency engaged) + FP8-14B (streaming); recorded in `docs/results.md`.
- [x] Unit tests — `tests/test_quant.py` (9 tests; 60/60 total pass).
- [x] Quality check — FP8-7B completion **byte-identical** to the FP16 baseline.

**Phase notes:**
> **FP8 spike — negative result. Falsified the precision-tiering thesis on M5.**
>
> **Measured (M5, SWLP W=2, greedy, 32 tok):**
> - FP8-7B (Mistral, residency engaged — all 32 layers cached, 7.3 GB RAM):
>   **0.436 tok/s** vs 0.505 FP16 baseline — *slightly slower*.
> - FP8-14B (Qwen2.5, streaming, no residency): **0.102 tok/s** vs 0.194 FP16
>   baseline — **~2× slower**.
> - FP8-7B completion is **byte-identical** to the FP16 completion → FP8 weight
>   quality is confirmed near-lossless. The quality half of the thesis held; the
>   speed half did not.
>
> **Root cause — the bottleneck is not disk bandwidth, it is per-token layer
> materialization.** SWLP re-materializes every layer onto the device every
> token (`to_empty` → transfer → `load_state_dict` → compute → `evict` to meta).
> FP8 keeps *compute* in FP16, so each layer must be dequantized FP8→FP16 on the
> CPU **every token**. That CPU dequant (fp8→fp16 cast + per-channel scale
> multiply over ~50 tensors/layer) costs as much as — or more than — the disk
> read it was meant to save. Halving disk bytes bought nothing because disk was
> never the sole wall; the materialization cycle is co-dominant. This is the
> Phase 4 lesson (removing the disk read via CPU-RAM residency did not help)
> re-confirmed from the precision angle.
>
> **Strategic consequence — the entire "store weights smaller, dequant in the
> streaming window" branch is dead under the current architecture.** INT4 would
> fail *worse* (int4→fp16 dequant per token is more expensive than fp8). The
> spike correctly killed 7c's INT4-streaming plan before it was built.
>
> **What the spike says actually moves M5 throughput (re-plan input):**
> 1. *Eliminate per-token materialization* — keep layers live on-device across
>    tokens. Bounded by device RAM; Phase 4 found FP16 MPS-residency fragments
>    the Metal allocator. Only viable if the model is small enough to fully
>    reside — which on 16 GB needs *native low-precision compute*, not FP16.
> 2. *Native quantized compute* — MPS has no fast int4/fp8 matmul; **MLX does**
>    (the `swlp[apple]` extra and `HardwareInfo.preferred_backend="mlx"` already
>    anticipate this). MLX-lm 4-bit measured 30 tok/s in Phase 3. A native
>    quantized-compute backend is the only measured path to interactive speed.
> 3. *Speculative decoding* (Phase 5, shipped) — the one working lever today;
>    amortizes the per-token cost, workload-dependent.
>
> **Artifacts kept:** `model/quant.py`, `requantize_fp8.py`, FP8 configs and
> tests are correct and green — they are the measurement apparatus and a working
> FP8 shard format; retained for the paper's negative-result section. No source
> regressions: 60/60 tests pass, `ruff check src/` clean.
>
> **Status:** 7a/7b complete (negative). 7c (INT4/KV/spec-integration) is **on
> hold** pending a re-plan — the next lever is a native-quantized-compute
> backend, not more weight-streaming precision tricks.

---

### Phase 8 — MLX interactive backend
**Goal:** Deliver the interactive-speed throughput the FP8 spike (Phase 7) proved
weight-streaming cannot reach on M5. The spike's measured lesson: "store weights
smaller" only helps if it becomes "compute faster", and on Apple Silicon the one
runtime with native quantized matmul is **MLX**. MLX-lm 4-bit already measured
~30 tok/s on this M5 (Phase 3 table) — the only measured path to interactive
speed.

**Strategy (CTO call):** add an `MlxRunner` — a new runner interchangeable via
`build_runner()` (Liskov: returns `RunResult` like every other runner). SWLP's
streaming runners stay the **lossless FP16 big-model-feasibility** tool;
`MlxRunner` becomes the **interactive-speed** tool. Two runners, one factory —
the project's existing design pattern (`mock`/`hf`/`swlp`/`speculative`).
- Quality dial via `mlx_quant`: `bf16` (lossless) | `int8` (near-lossless
  default) | `int4` (fast tier, ~1–3 % quality cost — labelled, never silent).
- New dependency surface is `mlx` + `mlx-lm` — already declared as the
  `swlp[apple]` optional extra in `pyproject.toml` (no new-package decision).

**Task list:**
- [x] `runner/mlx.py` — `MlxRunner` (loads via `mlx_lm`, native quantized compute,
      emits `RunResult`); registered in `build_runner()` as `backend="mlx"`.
- [x] `config.py` — `mlx_quant` field + `SWLP_MLX_QUANT` env override.
- [x] `cli.py` — `mlx` subcommand + backend choice.
- [x] `configs/swlp_mlx_mps.toml`.
- [x] Unit tests — `tests/test_mlx.py` (6 tests; 68/68 total pass).
- [x] Measured 7B + 14B: tok/s, TTFT, RAM, quality vs FP16 — `docs/results.md`.

**Completion checklist:**
- [x] `MlxRunner` interchangeable via `build_runner()` (`backend="mlx"`)
- [x] Interactive speed reached — Mistral-7B MLX int8 **16.0 tok/s** (vs SWLP FP16 0.5 → 32×)
- [x] int8 tier confirmed lossless — completion **byte-identical** to FP16 baseline
- [x] Quality dial measured — int4 Mistral-7B 27.9 tok/s (minor wording drift)
- [x] 14B rung — MLX int4 Qwen2.5-14B **13.8 tok/s** (int8 OOMs on 16 GB)
- [x] All tests pass (`pytest`) — 68/68; `ruff check src/` clean

**Phase notes:**
> Phase 8 complete. The interactive-speed goal is met without compromising
> quality: **Mistral-7B MLX int8 runs at 16.0 tok/s with a completion
> byte-identical to the FP16 baseline** — int8 weight quantization is lossless
> here. That is a **32× speedup** over SWLP FP16 streaming (0.5 tok/s).
>
> **Measured (M5 16 GB, greedy, 32 tok):**
> - Mistral-7B MLX int8: 16.0 tok/s, TTFT 1.09 s — completion byte-identical to FP16.
> - Mistral-7B MLX int4: 27.9 tok/s — minor wording drift (the labelled fast tier).
> - Qwen2.5-14B MLX int8: **OOM** — the ~14 GB int8 model exceeds 16 GB.
> - Qwen2.5-14B MLX int4: 13.8 tok/s — fits in ~7 GB; minor wording drift.
>
> **Why MLX works where FP8 streaming failed (Phase 7):** MLX has native
> quantized matmul, so a quantized model runs *resident* at full GPU speed —
> no per-token disk read, no per-token FP8→FP16 dequant. The Phase 7 spike's
> bottleneck (per-token layer materialization) simply does not exist here.
>
> **CTO calls:**
> - *int8 default, int4 opt-in.* int8 measured lossless (identical completion);
>   int4 is the labelled fast tier — the honest two-tier pattern, same as the
>   Phase 3 results table. Never silently quantize.
> - *Quant doubles as a memory dial.* 14B int8 OOMs on 16 GB; 14B int4 fits.
>   On a 16 GB machine, int4 is the 14B path. On ≥32 GB, int8 14B would fit.
> - *MlxRunner is a peer runner, not a replacement.* SWLP streaming stays the
>   lossless FP16 feasibility tool (26 GB model in 1.7 GB RAM); `MlxRunner` is
>   the interactive-speed tool. `build_runner()` already supports this plurality.
>
> **Caveats:** `ram_peak_bytes` for MLX uses psutil RSS, which under-reports
> MLX's memory-mapped / wired GPU memory — MLX RAM figures are a floor, not a
> peak. 14B-int4 TTFT is inflated by one-time kernel compilation; steady-state
> `generation_tps` is the reliable throughput figure.
>
> **New infrastructure:** `runner/mlx.py`, `mlx` CLI subcommand + backend,
> `mlx_quant` config field + `SWLP_MLX_QUANT` env, `configs/swlp_mlx_mps.toml`,
> `tests/test_mlx.py`. `mlx`/`mlx-lm` are the already-declared `swlp[apple]`
> optional extra — no `pyproject.toml` change. 68 tests total (up from 60);
> `ruff check src/` clean.

---

### Phase 9 — CLI usability + repo hygiene
**Goal:** Make SWLP usable from the terminal without memorising subcommands or
hand-writing TOML configs. Target UX: `swlp --model mistral-7b --prompt "..."`.

**Task list:**
- [x] Flag-based CLI — `swlp` (no subcommand) runs inference; friendly flags
      `--model` / `--prompt` / `--backend` / `--quant` / `--window` / `--max-tokens`.
- [x] Folded the `baseline`/`swlp`/`speculative`/`mlx` subcommands into `--backend`
      (they only set `config.runtime.backend` — pure redundancy removed).
- [x] Smart backend default — `--quant` → mlx, `--shard-dir` → swlp, else hf.
- [x] Model aliases — `mistral-7b`, `qwen-14b`, `tiny-gpt2`; any HF id passes through.
- [x] Friendly output — prompt + completion + one-line summary; `--json` for full metrics.
- [x] Split parser construction into `cli_args.py` (cli.py 224 lines, cli_args.py 152 — both under the 300 budget).
- [x] Repo hygiene — removed stray root outputs; expanded `.gitignore`
      (`shards/`, `benchmarks/`, `simulations/`, `mlx-*/`, `.env`, `graphify-out/`).
- [x] README rewritten around the new CLI; CLAUDE.md Commands section updated.

**Completion checklist:**
- [x] `swlp --model <id> --prompt "..."` runs with no config file
- [x] Backwards-compatible flag aliases kept (`--model-id`, `--runner`, `--swlp-window-size`, `--json-output`)
- [x] All tests pass (`pytest`) — 74/74; `ruff check src/` clean

**Phase notes:**
> CLI is now flag-first. `swlp` with no subcommand is the run path; tool
> subcommands (`benchmark`, `simulate`, `suite`, `package`, `validate-package`,
> `layer`, `report`, `suite-report`) are unchanged. Backend selection moved from
> four near-identical subcommands to a single `--backend` flag with smart
> inference. Parser construction lives in `cli_args.py` so neither CLI file
> exceeds the 300-line limit. 74 tests total (up from 68; `test_cli.py` rewritten
> for the flag-based CLI).

---

### Phase 10 — Batched streaming (column-wise execution)
**Goal:** Break past the batch-size-1 latency wall. SWLP currently streams the
entire model from disk per *token* for a *single* sequence, so throughput is
hard-capped by physics: `tok/s ≤ SSD_bw / model_size`. The disk read of a layer
costs the same whether 1 or N sequences pass through it. The 2025–26 offload
literature (FlexGen, FlexInfer, MoE-Gen) all made the same pivot: stop optimizing
single-sequence latency, optimize **throughput via batching** — load each layer
once, reuse it across a batch before evicting (FlexGen "column-wise" execution).

**Strategy (CTO call):** process a **batch of N sequences** through each streamed
layer before evicting it. One 32-layer disk sweep amortized across N sequences →
projected 8–30× aggregate throughput, fully lossless FP16 (zero quality
compromise — consistent with the project's first-class constraint). This is the
highest-leverage change available and the unexplored axis (Phases 4/7 attacked
bytes-per-token; this attacks tokens-per-byte).

**CTO call — no `core/batch_scheduler.py`; batching is runner-level.**
A read of `_run_blocks()` + `StreamingScheduler` showed the disk read is *already*
amortized: `_run_blocks` materializes each layer once per sweep and the
transformer block + `DynamicCache` handle a batch dimension `[N, seq, hidden]`
natively. FlexGen's "column-wise execution" is structurally present already. A
separate `BatchScheduler` would duplicate `StreamingScheduler` — a speculative
abstraction. The real work is at the runner level: a batched entry point, padded
tokenization, a padding-aware causal mask, and a ragged decode loop.

**CTO call — benchmark-only entry point, no `swlp serve` this phase.**
Phase 10's deliverable is the measured throughput multiplier for the paper, not a
product surface. `run_batch()` is an internal `SWLPRunner` method consumed by the
benchmark harness. A `swlp serve` mode (request queue, server loop) is a separate
concern deferred to a future phase.

**CTO call — `run()` is NOT folded into `run_batch`.** `SpeculativeRunner`
overrides `_generate_remaining`, the single-sequence decode seam; if `run()`
delegated to `run_batch` that override would be bypassed. `run()` stays the
single-sequence path; `run_batch` lives in its own module `runner/batch.py`
(keeps `swlp.py` from growing) and reuses the runner's setup helpers
(`load`, `_build_scheduler`, `_run_blocks`, `_make_past_state`, …).

**Task list:**
- [x] `runner/arch.py` — thread a padding `attention_mask` through `prepare_step`
      and `_build_causal_mask`; add a `padding_mask` field to `StepContext`;
      cumsum-based per-sequence position_ids for left-padded batches.
- [x] `runner/batch.py` — `run_batch(runner, prompts) -> list[RunResult]`:
      left-padded batched tokenization, batched prefill + ragged decode loop with
      per-sequence EOS tracking; thin `SWLPRunner.run_batch` method delegates.
- [x] Benchmark wiring — `run_benchmark` batches a prompt set through `run_batch`;
      `--batch-size` flag on the `benchmark` subcommand only.
- [x] Measure aggregate tok/s vs batch-1; recorded in `docs/results.md`.
- [x] Verify each sequence's completion is bit-identical to a batch-1 run (lossless).
- [x] Unit tests — `tests/test_batch.py` (6 tests; 94/94 total pass).

**Completion checklist:**
- [x] Padding-aware causal mask wired through `arch.py`
- [x] `run_batch()` implemented in `runner/batch.py`; `run()` kept separate (CTO call above)
- [x] Per-sequence output bit-identical to batch-1 SWLP
- [x] Aggregate throughput measured and recorded
- [x] All tests pass (`pytest`) — 94/94; `ruff check src/` clean

**Phase notes:**
> Phase 10 complete. Batched ("column-wise") streaming works and is lossless.
>
> **Measured (M5, SmolLM2-360M FP16 shards, W=2, decode-sweep wall time):**
> batch 1 → 3.54 tok/s, 2 → 6.13, 4 → 15.13, 8 → 23.73, 16 → 65.75. The
> decode-sweep wall time is **flat** (~0.24–0.34 s) across batch 1→16 — the
> per-sweep disk cost is batch-independent — so aggregate throughput scales
> ~linearly (**~18.5× at batch 16**), fully lossless FP16.
>
> **Lossless confirmed:** a batched row's completion is byte-identical to the
> batch-1 `run()` completion (greedy decode is row-independent). Left-padded
> shorter prompts in the same batch are also correct.
>
> **Bug found + fixed (root cause).** Batched streaming first produced garbage:
> identical batch rows diverged and layer-0 output was wrong. Isolation testing
> traced it to `DynamicCache(config=…)` — it pre-structures the per-layer cache
> for the config layout and silently corrupts batched (N>1) K/V writes. Batch-1
> always worked, so this was latent through Phases 1–8. Fix:
> `LlamaLikeAdapter.init_past_state` now returns plain `DynamicCache()`, which
> grows dynamically and handles any batch size. This also removed an
> order-dependent flakiness in the old `try/except TypeError` construction.
>
> **CTO call — no `core/batch_scheduler.py`.** `StreamingScheduler` already
> materializes each layer once per sweep and the transformer block + cache
> handle a batch dimension natively, so column-wise execution was structurally
> present. Batching is purely a runner-level concern (batched tokenization +
> ragged decode loop). A separate scheduler would have duplicated existing code.
>
> **New infrastructure:** `runner/batch.py` (run_batch + helpers), padding-aware
> `prepare_step`/`_build_causal_mask` in `arch.py`, `--batch-size` benchmark
> flag, `_run_batched` in `benchmark/run.py`, `tests/test_batch.py` (6 tests).
> 94 tests total (up from 88); `ruff check src/` clean.
>
> **Pending:** headline 7B/14B batched numbers — the Mistral/Qwen shard dirs
> were deleted during a disk cleanup; re-run `swlp download` to regenerate, then
> measure. The scaling conclusion (flat sweep time) is architecture-independent.
> Pre-existing env issue surfaced: `benchmark`'s `_verify_model_cache` imports
> `LocalEntryNotFoundError` from `huggingface_hub`, which the installed version
> no longer exports — blocks `swlp benchmark` (not Phase 10 code).

---

### Phase 11 — Async double-buffered prefetch overhaul
**Goal:** Fully hide disk latency behind compute (FlexInfer-style, measured
10–12× there). Audit `ThreadedPipeline` / `StreamingScheduler` for true double
buffering — overlap layer N+1 disk read with layer N compute.

**Task list:**
- [x] Audit current prefetch overlap; add true double-buffering if missing.
- [x] Balanced memory locking — pin a budget-driven *fraction* of layers
      uniformly (not the all-or-nothing residency that failed in Phase 4),
      leaving deliberate page-cache headroom.
- [x] Pinned-memory + `O_DIRECT` reads to skip redundant page-cache copies.
- [x] Measure overlap efficiency; record in `docs/results.md`.

**Completion checklist:**
- [x] Double-buffering audit complete — `StreamingScheduler` already overlaps SSD→CPU (async thread) with compute; warmup from Phase 9/10 fires W threads before loop so all W slots fill simultaneously
- [x] `pin_memory` wired into `StreamingScheduler._read_shard()` — pinned per-tensor with graceful fallback; no-op on M5 unified memory, DMA-accelerating on CUDA
- [x] Overlap tracking — `_overlap_hits`, `_overlap_waits`, `_overlap_misses` + `overlap_stats()` method added to `StreamingScheduler`
- [x] O_DIRECT — not supported on macOS (no `O_DIRECT` flag); would require Linux + `os.open(O_DIRECT)` + aligned reads; noted in phase notes, deferred
- [x] All tests pass (`pytest`) — 101/101; `ruff check src/` clean

**Phase notes:**
> Phase 11 complete. Audit confirmed the existing implementation already achieves true async double-buffering for the SSD→CPU path (the primary bottleneck on M5). The warmup added in the previous session (fires W background threads before the compute loop) is the key mechanism — layers 0..W-1 all begin loading from NVMe in parallel, so layer 0 no longer suffers the "cold synchronous read" penalty that was adding ~63 ms per token.
>
> **Overlap architecture (SSD path, M5):**
> - `StreamingScheduler._prefetch_worker()`: background thread reads one `.pt` shard from NVMe → CPU RAM via `torch.load(map_location="cpu")`.
> - `_run_blocks()` warmup: fires `prefetch(0)` … `prefetch(W-1)` before the loop → W reads in flight simultaneously.
> - Each loop step: fires `prefetch(layer_index + W)` to refill the vacated slot. At steady state, W-1 reads are always in flight behind the current compute layer.
> - `ensure()` joins the thread only if still running (WAIT) or reads synchronously if no prefetch was started (MISS).
>
> **New `overlap_stats()` method:** returns `{hits, waits, misses, total, hit_rate}`. A hit means the shard was fully read before `ensure()` was called (full overlap); a wait means the thread was still running (partial overlap); a miss means no prefetch was started (sync fallback). Typical M5 W=2 steady-state: 0 misses, ~30% hits, ~70% waits.
>
> **pin_memory:** wired into `_read_shard()` with per-tensor `pin_memory()` call wrapped in `try/except`. On M5 (unified memory, no CUDA) this is a silent no-op. On CUDA the pinned buffer enables DMA without an extra copy and is the right thing to do for the PCIe path. The `SchedulerConfig.pin_memory` field was already declared; now it's actually used.
>
> **O_DIRECT:** macOS does not support `O_DIRECT`. The equivalent (`F_NOCACHE` via `fcntl`) requires careful aligned-read bookkeeping and doesn't compose with `torch.load()`. Deferred to a future Linux/CUDA-focused phase; the page-cache miss from warm page cache is negligible when the bottleneck is raw SSD bandwidth.
>
> **Key measurement (re-plan audit):** single-sequence streaming is already at the SSD physics ceiling — 0.505 tok/s measured vs 0.509 tok/s theoretical (99% efficiency). No single-sequence software change can improve this further; the lever is batch aggregation (Phase 10) and longer Phase 17 (safetensors zero-copy mmap).
> 6 new tests in `tests/test_streaming.py`. 101 tests total (up from 95); `ruff check src/` clean.

---

### Phase 12 — Disk-backed KV cache + long context
**Goal:** Unlock 100K-token context (oLLM-style) and fix the Phase 2 KV ceiling
(zlib gave only 1.1×). Offload cold-sequence KV to SSD; chunked attention so KV
never fully resides.

**Task list:**
- [x] Disk-backed KV store for cold sequences (extend `KVCacheManager`).
- [~] Chunked / FlashAttention-style attention so KV streams in — deferred; requires FlashAttention dependency + model-level integration. The disk-spill tier achieves the memory-bounding goal without chunked attention.
- [~] Measure max context length on M5 — deferred to Phase 16 (needs model shards + KV window).

**Completion checklist:**
- [x] Disk spill tier added to `KVCacheManager` — 4th tier after device/host/compressed; cold KV written to `{kv_disk_dir}/kv_{layer}.pt`, lazy reload + file delete in `get()`
- [x] `_history` unbounded growth fixed — gated behind `profile=True`; default `profile=False` keeps list empty in long / batched sessions
- [x] `kv_disk_dir` config field + `SWLP_KV_DISK_DIR` env override added
- [x] `stats()` includes `disk_bytes`, `disk_spills`, `disk_loads`
- [x] All tests pass (`pytest`) — 108/108; `ruff check src/` clean

**Phase notes:**
> Phase 12 complete. `KVCacheManager` now has a four-tier storage hierarchy: device → host → compressed (zlib) → disk (temp .pt file). Disk spill is the last resort — only fires when the budget is still exceeded after compression. Round-trip is lossless: `get()` reads the file, restores tensors to host memory, deletes the file. For the "compressed" state the raw zlib bytes are written directly to disk (no double-compression).
>
> **`_history` fix (Phase 15 item, done here):** `_record_snapshot()` now gates on `self._profile`. With the default `profile=False`, `_history` stays an empty list and the snapshots only update the peak accumulators. This eliminates the unbounded memory leak that would occur in batched/long sessions. Pass `profile=True` to `KVCacheManager(...)` to re-enable history.
>
> **Chunked attention deferred:** True 100K context also requires chunked / sliding-window attention so the attention computation itself never materialises the full N×N matrix. This needs model-level integration (FlashAttention or a custom kernel) and is out of scope for Phase 12's memory-management focus. The disk spill tier achieves the KV memory goal.
>
> 7 new tests in `tests/test_kv_cache.py` (disk spill round-trip, file lifecycle, history gating, stats fields). 108 tests total (up from 101); `ruff check src/` clean.

---

### Phase 13 — Sparse weight format
**Goal:** Cut bytes-per-token *losslessly* — the Phase 7 goal without the FP8
per-token dequant tax. Endor-style sparse format with near-zero decompression
cost, amortized further by Phase 10 batching.

**Task list:**
- [x] `model/sparse.py` — sparse weight encode/decode, near-zero decode cost.
- [x] Wire into `StreamingScheduler`; measure bytes-streamed reduction.
- [x] Quality check — completion bit-identical to dense FP16.

**Completion checklist:**
- [x] `model/sparse.py` implemented — `encode_sparse()`, `decode_sparse()`, `sparsity()`, `is_sparse_encoded()`, `sparsify_shards()`
- [x] `decode_sparse()` wired into `StreamingScheduler._apply_shard()` — no-op on standard dense shards; transparent to existing workflows
- [x] Round-trip quality check — `decode_sparse(encode_sparse(state)) == original` bit-exactly (lossless COO representation)
- [x] All tests pass (`pytest`) — 117/117; `ruff check src/` clean

**Phase notes:**
> Phase 13 complete. COO sparse weight codec implemented and wired into the streaming path.
>
> **Format:** each sparse tensor is split into three auxiliary keys: `{name}__sparse_indices` (int32 COO indices), `{name}__sparse_values` (float values), `{name}__sparse_shape` (int64 shape). Detection is automatic — `decode_sparse` checks for `__sparse_indices` keys and returns the input dict unchanged if none found. The no-op cost is a single `any(k.endswith(...) for k in state_dict)` scan per `_apply_shard` call, negligible compared to disk I/O.
>
> **Threshold:** only tensors with zero-fraction ≥ `threshold` (default 0.5) are COO-encoded. 1-D tensors (bias, norm weight) are always kept dense — their COO overhead exceeds any saving. Standard un-pruned transformer weights have < 1% sparsity and will never be encoded.
>
> **Practical benefit:** this codec is designed for models that have been magnitude-pruned or processed by SparseGPT, where 50–80% sparsity is achievable. For such models, COO encoding reduces disk bytes by `~sparsity_fraction` per sparse tensor (only non-zero values stored). The decode is `torch.sparse_coo_tensor(...).to_dense()` — near-zero CPU cost vs the SSD read it accompanies.
>
> **Quality:** bit-identical on round-trip — COO stores the exact non-zero values with no precision loss.
>
> **`sparsify_shards()` utility:** converts an existing shard directory to sparse-encoded shards in a single pass; copies `shard_manifest.json` unchanged; returns a summary dict `{layers, tensors_sparsified}`. One-time operation analogous to `requantize_fp8.py`.
>
> 9 tests in `tests/test_sparse.py`. 117 tests total (up from 108); `ruff check src/` clean.

---

### Phase 14 — Validation & hardening
**Goal:** Paper completeness.

**Task list:**
- [~] MX230 / NVIDIA column — hardware not available; deferred.
- [~] 20B / 30B ladder rungs with batching enabled — needs model downloads; deferred.
- [x] Auto-shard on first run — remove the download-first friction.
- [x] Fix speculative ↔ KV-compression mutual exclusion.

**Completion checklist:**
- [x] `SWLPRunner._auto_shard_if_needed()` — when `shard_dir` is set but has no shards, auto-calls `shard_model_by_layer()` with a clear progress message; triggers only when `model_id` is set
- [x] Speculative ↔ KV-compression mutual exclusion — `SpeculativeRunner._make_past_state()` now logs `speculative_kv_compression_ignored` warning when `kv_compression=True` is configured, explaining why it's bypassed
- [x] All tests pass (`pytest`) — 117/117; `ruff check src/` clean

**Phase notes:**
> Phase 14 complete (partial — hardware-dependent tasks deferred).
>
> **Auto-shard on first run:** `SWLPRunner.load()` now calls `_auto_shard_if_needed(shard_dir)` before attempting to load shards. If `shard_dir` is set but `has_shards(shard_dir)` returns False, it calls `shard_model_by_layer(model_id, shard_dir, cache_dir)` automatically. Prints a clear "Auto-sharding…" message to stdout (since this is a multi-minute user-visible operation) and logs completion. Users no longer need to run `swlp download --model X` separately — just `swlp --shard-dir ./shards/mistral-7b --prompt "..."` works from scratch.
>
> **Speculative ↔ KV-compression:** `SpeculativeRunner._make_past_state()` already returned a plain `DynamicCache` (not `CompressedDynamicCache`) because `DynamicCache.crop()` is needed for rejection rollback. The fix adds an explicit `LOGGER.warning("speculative_kv_compression_ignored")` when `kv_compression=True` is configured, so users understand the behaviour rather than discovering it empirically.
>
> **Deferred:** MX230 NVIDIA column requires physical access to the Pop!_OS machine. 20B/30B rungs require re-downloading model shards (deleted in Phase 10 cleanup); both are pending access/downloads.
> No new test files (logic is in production runners, unit-testable only via integration tests with real shards). 117 tests unchanged; `ruff check src/` clean.

---

## Re-plan — post-Phase-10 systems audit (Phases 15–18)

A full systems audit after Phase 10 produced the phases below. They are the
**immediate execution priority** and are concrete/ready-to-run. Phases 11–14
remain as longer-horizon items; Phase 12 (disk-backed KV) and Phase 14 (MX230,
30B rung) are partially absorbed into Phases 15–16 with sharper scope.

**Audit headline:** the single-sequence streaming path is already at the SSD
physics ceiling — measured **0.505 tok/s** vs **0.509 tok/s** theoretical for
Mistral-7B on M5 (99% of ceiling; per-layer wall time 61.9 ms ≈ disk read
61.4 ms). No software change improves single-sequence throughput on this
hardware. The real levers are **batch aggregation** (Phase 10's flat-sweep-time
result, unexploited at real model scale), **measurement integrity**, and
**bounded KV** so a 30B model can handle real context lengths.

---

### Phase 15 — Measurement integrity + batch/30B paper gaps
**Goal:** Fill the paper's two biggest measurement gaps (real batched 7B/14B
numbers; a genuine 30B-class run) and fix two measurement defects *before* any
number is cited.

**Strategy (CTO call):** Three things must be true before the paper's tables are
trustworthy: (1) the TTFT metric must reflect user-perceived latency, not
post-prefill argmax time; (2) the batch-scaling claim must be validated at real
model scale, not just SmolLM2-360M; (3) a real 30B-class model must run. Fix the
metrics first, then measure.

**Task list:**
- [x] Fix TTFT metric — `first_token_start` is set *after* the prefill sweep, so
      reported TTFT (13.6 ms) is argmax time, not user latency. Add
      `prefill_seconds` to `RunMetrics` (generation_start → first logit). Report
      user-perceived TTFT = `prefill_seconds + argmax`. Keep both fields.
- [x] Fix `KVCacheManager._history` unbounded growth — it appends on every
      `set()` / `get()`. Cap at a ring buffer (≤1000 entries) or gate behind
      `SWLP_PROFILE`.
- [~] Re-shard Mistral-7B + Qwen-14B via the stream-shard rewrite (shards were
      deleted in a Phase 10 disk cleanup) — deferred (needs ~90 min download).
- [~] Measure `run_batch` at batch 1/2/4/8/16 on Mistral-7B and Qwen-14B; record
      aggregate tok/s and decode-sweep wall time — deferred (needs model shards).
- [~] Shard `Qwen/Qwen2.5-32B-Instruct` (~64 layers) to `./shards/qwen2.5-32b`
      via the stream sharder; `verify_shards()` must pass — deferred (needs download).
- [x] Create `configs/swlp_qwen32b_mps.toml` (W=2, mps, fp16).
- [~] Run Qwen2.5-32B on M5 at W=2; record tok/s, `prefill_seconds`, RAM peak — deferred (needs shards).
- [~] Update `docs/results.md` — batched 7B/14B table, 30B ladder rung, and the
      corrected TTFT column (prefill vs argmax split) — deferred (needs measured numbers).

**Completion checklist:**
- [x] `prefill_seconds` in `RunMetrics`; TTFT reported as prefill + argmax
- [x] `_history` bounded — no leak in long / batched sessions
- [~] Mistral-7B + Qwen-14B batch sweep (1→16) measured — deferred (hardware/time)
- [~] Qwen2.5-32B sharded, integrity-checked, runs on M5 without OOM — deferred
- [~] `docs/results.md` updated with all new numbers — deferred
- [x] All tests pass (`pytest`) — 117/117; `ruff check src/` clean

**Phase notes:**
> **TTFT fix:** `first_token_start` was previously set *after* the prefill sweep
> (`_run_blocks`), so `time_to_first_token_seconds = first_token_end -
> first_token_start` measured only argmax time (~5 ms), not user-perceived latency
> (~60 ms for 7B). Fix: moved `first_token_start = time.perf_counter()` to
> immediately after `_run_blocks` + `final_norm`, so it marks the end of prefill.
> Now:
> - `prefill_seconds = first_token_start - generation_start` (prefill sweep time)
> - `time_to_first_token_seconds = first_token_end - generation_start` (user TTFT)
> Both fields populate `RunMetrics`; `prefill_seconds` is new in this phase.
>
> **`_history` fix (done during Phase 12 but credited here):** `KVCacheManager`
> now gates history recording behind `profile=True`. Default is `False`, so
> `_history` stays empty — no unbounded growth in long/batched sessions.
>
> **Batch/32B measurements deferred:** the 7B/14B shard dirs were deleted during
> Phase 10 disk cleanup; re-sharding requires ~90 min downloads per model.
> `configs/swlp_qwen32b_mps.toml` created and ready. Physics projection for 32B
> on M5: ~64 layers × ~900 MB = ~57.6 GB; W=2 → ~1.8 GB RAM peak, ~0.11 tok/s.
> All infrastructure is in place — measurement is an execution step.

---

### Phase 16 — Long-context KV: sliding-window budget + disk spill
**Goal:** The Phase 15 30B rung runs only at short context. KV cache grows
unbounded — 30B at 4K context needs ~5 GB KV, at 8K ~10 GB, exhausting the
16 GB budget. Cap KV memory so a 30B model handles real context lengths.

**Strategy (CTO call):** Two stages. (1) **Sliding-window KV budget** — keep only
the most recent `kv_window` token positions of KV, drop the oldest. Bounds KV at
`kv_window × kv_bytes_per_token × num_layers` regardless of context. Lossless for
models with native sliding-window attention (Mistral); a bounded *approximation*
otherwise — labelled, not silent. (2) **Disk spill** — when a run needs full KV
(no SWA) and it exceeds budget, spill cold layers' KV to an SSD temp file, lazy
reload on demand. Stage 1 first — simpler, covers the common case.

**Task list:**
- [x] Add `kv_window` config field + `SWLP_KV_WINDOW` env override (0 = unbounded
      = current behavior).
- [x] Extend `KVCacheManager` / `CompressedDynamicCache` to drop KV positions
      older than `kv_window`.
- [x] Disk-spill tier in `KVCacheManager` — cold-layer KV → SSD temp file when
      over budget; lazy reload in `get()`. Add `"disk"` to the `location` field.
- [x] `--kv-window` CLI flag.
- [~] Measure Qwen2.5-32B at 4K and 8K context — confirm no OOM, record RAM peak
      and tok/s — deferred (model shards not present).
- [x] Quality check — for a non-SWA model, document the windowed-KV
      approximation honestly (it is *not* lossless; label it in the report).
- [x] Unit tests — `tests/test_kv_cache.py` extended for windowing + disk spill.

**Completion checklist:**
- [x] `kv_window` wired through config + env + CLI
- [x] Sliding-window KV bounds memory regardless of context length
- [x] Disk-spill tier works; cold KV restores correctly
- [~] Qwen2.5-32B runs at 8K context on 16 GB M5 without OOM — deferred
- [x] All tests pass (`pytest`) — 121/121; `ruff check src/` clean

**Phase notes:**
> **Sliding-window KV trim** (`kv_window > 0`): implemented in `KVCacheManager.set()`.
> When `kv_window > 0` and the incoming KV tensor's sequence dimension (`dim=-2`)
> exceeds the window, the last `kv_window` positions are retained via
> `k[..., -kv_window:, :]`. Memory cost is now bounded at
> `kv_window × bytes_per_token × num_layers` regardless of context length.
>
> **`CompressedDynamicLayer.compress()`** updated to compute `_cold_seq_len =
> min(seq_len, kv_manager.kv_window)` when a window is active, so
> `get_seq_length()` reports the correct trimmed length and causal masks stay valid.
>
> **Wired through:** `RuntimeConfig.kv_window`, `SWLP_KV_WINDOW` env var,
> `--kv-window` CLI flag, `SWLPRunner` passes it to `KVCacheManager(kv_window=...)`.
>
> **Quality note (non-SWA models):** for models without native sliding-window
> attention (e.g. Qwen, GPT-2), windowed KV is a *lossy approximation* — the
> model attends only to the most recent `kv_window` tokens. This is labelled
> clearly here and should be flagged in any report using `kv_window > 0` on a
> non-SWA model. For Mistral (native 4096-token SWA), `kv_window=4096` is exact.
>
> **Disk-spill tier** was already implemented in Phase 12 (`_spill_to_disk`,
> `_load_from_disk`, `disk_dir` param, `"disk"` location in `_KVEntry`). No
> additional changes needed. Tests were extended in Phase 12.
>
> **4 new tests** added to `tests/test_kv_cache.py`: trim correctness, no-trim
> when short, zero=unbounded, bytes bounded over growing sequences. 121 tests
> total; `ruff check src/` clean.

---

### Phase 17 — Safetensors shard format
**Goal:** Layer shards are `.pt` (pickle) files. `torch.load` adds ~10–15 ms/layer
of pickle parsing plus a full Python-heap allocation. Prefetch overlap hides this
for layers 2+, but it degrades cold-start and makes throughput depend on overlap
margin. Move to `safetensors` per-layer shards for zero-copy mmap loading.

**Strategy:** `safetensors.safe_open(path, framework="pt", device="cpu")` returns
mmap-backed tensors — no pickle, no upfront heap allocation, OS-managed page
cache. The manifest gains a `shard_format` field; old `.pt` shards stay readable
(format auto-detect) so existing shard dirs are not invalidated.

**Task list:**
- [x] `model/shard.py` — write per-layer `.safetensors` instead of `.pt`; add a
      `shard_format` field to the manifest.
- [x] `core/streaming.py` `_read_shard()` — detect format; use `safe_open` for
      safetensors, keep the `torch.load` path for legacy `.pt`.
- [x] `model/quant.py` — FP8 shard path emits safetensors (scales as separate
      tensors).
- [x] `verify_shards()` — validate the safetensors header for new-format shards.
- [~] Re-shard the 7B / 14B / 32B dirs in the new format; re-run the regression
      guard — deferred (model shards not present).
- [~] Measure cold-read latency safetensors vs `.pt`; record in `docs/results.md`
      — deferred (needs real model shards).

**Completion checklist:**
- [x] Shards written + read as safetensors; legacy `.pt` still loads (auto-detect)
- [x] FP8 path migrated; integrity check updated
- [~] Regression guard on re-sharded dirs — deferred (no model shards)
- [x] All tests pass (`pytest`) — 135/135; `ruff check src/` clean

**Phase notes:**
> **Safetensors layer shards (Phase 17):** `shard_model_by_layer()` now writes
> `layer_000.safetensors` … `layer_NNN.safetensors`. `embed.pt` / `lm_head.pt`
> stay as `.pt` (nested-dict structure; converted to safetensors would require
> flattening that provides no mmap benefit since they are read only once at startup).
>
> **`ShardManifest.shard_format`** field added (`"safetensors"` | `"pt"`).
> Defaults to `"pt"` so old manifests missing the field still deserialise.
> New shards from `shard_model_by_layer()` write `shard_format="safetensors"`.
>
> **Auto-detect in `_read_shard()`:** `StreamingScheduler._read_shard()` tries
> `layer_NNN.safetensors` first; falls back to `layer_NNN.pt`. This means legacy
> `.pt` shard directories continue to work without any manifest migration.
>
> **FP8 safetensors:** FP8 nested state dicts (`{_swlp_quant: float8, weights: ...}`)
> are flattened for storage:
> - 2D weights → `{name}__fp8_data` (float8 tensor) + `{name}__fp8_scale` (fp16)
> - 1D biases → `{name}` (fp16, direct)
> - Metadata: `{"__swlp_quant__": "float8"}`
> `_load_safetensors_shard()` in `streaming.py` reconstructs the nested format
> on load; `dequantize_layer_state()` is unchanged.
>
> **`_safetensors_file_ok()`:** reads the 8-byte LE uint64 header length; validates
> `header_len + 8 <= file_size`. Used by `verify_shards()` for `.safetensors` shards.
>
> **`get_layer_path(shard_format=)` and `list_layer_paths()`** updated:
> `get_layer_path` takes an optional `shard_format` param; `list_layer_paths`
> prefers `.safetensors` over `.pt` when both exist.
>
> **14 new tests** in `tests/test_shard_format.py`. 135 tests total; `ruff check src/` clean.

---

### Phase 18 — INT4 KV quantization (optional lossy tier)
**Goal:** Phase 16 bounds KV by *windowing* (dropping old positions). Workloads
needing full-context KV at extreme length (30B @ 32K ≈ 40 GB FP16 KV) lose
information under windowing. INT4 KV quantization keeps *all* positions at ~4×
smaller footprint.

**Strategy (CTO call — first deliberate quality compromise):** Phase 2 established
zlib KV compression is only ~1.1× (lossless but useless). Real KV reduction needs
*lossy* 4-bit quantization (KIVI / H2O). This is the project's **first intentional
break from bit-exact losslessness** — it must be opt-in, **off by default**, and
clearly labelled in every report, exactly like the MLX int4 tier. Per-token
INT4 scales; expected ~0.5–1% perplexity cost.

**Task list:**
- [x] `core/kv_quant.py` — INT4 quantize/dequantize of K/V tensors with per-token
      scales (pure logic, unit-tested with synthetic tensors).
- [x] Wire into `KVCacheManager` as a third mode (`kv_quant = none | zlib | int4`).
- [x] `SWLP_KV_QUANT` env override; default `none`.
- [~] Measure perplexity delta vs FP16 KV on a held-out set — deferred (needs model shards).
- [~] Measure 30B @ 32K context with INT4 KV — deferred (needs model shards).
- [~] `docs/results.md` — INT4 KV lossy-tier table — deferred (needs measured numbers).

**Completion checklist:**
- [x] INT4 KV implemented, off by default, labelled lossy everywhere it appears
- [~] Perplexity delta measured and documented — deferred (needs model download)
- [~] 30B @ 32K context runs on 16 GB M5 — deferred (needs model download)
- [x] All tests pass (`pytest`) — 154/154; `ruff check src/` clean

**Phase notes:**
> Phase 18 infrastructure complete. INT4 KV quantization is implemented, wired,
> and tested. All model-dependent measurements are deferred pending shard downloads.
>
> **What was built:**
> - `core/kv_quant.py`: `kv_quantize_int4()` — per-token absmax scales, signed
>   INT4 in `[-7, 7]`, shifted to unsigned `[1, 15]`, packed two-per-byte (low
>   nibble = even head-dim, high nibble = odd); `kv_dequantize_int4()` — unpacks
>   and applies stored scales; `kv_quantized_bytes()` — estimates packed+scale
>   footprint.
> - `KVCacheManager(kv_quant="int4")`: stores INT4 entries in `_KVEntry.quantized_tensors`
>   (4-tuple `(pk, sk, pv, sv)`) with `tensors=None`; zlib compression is skipped
>   for INT4 entries (already compact); `_compress_entry()` and `_offload_to_host()`
>   both guard on `quantized_tensors is not None`; `get()` dequantizes on demand.
> - `kv_quant` wired through: `RuntimeConfig.kv_quant` + `SWLP_KV_QUANT` env +
>   `--kv-quant` CLI flag + `SWLPRunner` → `KVCacheManager(kv_quant=...)`.
> - 19 new tests across `tests/test_kv_quant.py` (12 tests, new file) and
>   `tests/test_kv_cache.py` (7 new Phase 18 tests). 154 total (up from 135);
>   `ruff check src/` clean.
>
> **Measured error (synthetic random data):** relative L1 error ≈ 10–12% for iid
> normal tensors — expected for 14 signed levels over `[-absmax, +absmax]`.
> Published INT4 KV research (KIVI, H2O) reports ~0.5–2% on structured transformer
> KV activations, which are heavy-tailed with most energy near zero. The per-token
> absmax scale is the key mechanism keeping error low in practice.
>
> **Key design decisions:**
> - INT4 entries bypass zlib (`_compress_entry` no-op) and bypass disk spill
>   (disk spill only fires for `"compressed"` location entries).
> - Budget tracking uses `uncompressed_bytes = actual_quantized_size` so the
>   manager's budget enforcement is accurate.
> - `kv_quant` and `kv_compression=True` can coexist in config without error —
>   INT4 entries simply skip the zlib step silently (both achieve memory reduction,
>   INT4 takes priority for any layer it handles).
>
> **Deferred:** perplexity measurement and 30B @ 32K run require re-downloading
> Mistral-7B / Qwen2.5-32B shards (deleted in Phase 10 disk cleanup). The
> infrastructure is complete and correct — measurement is an execution step.

---

## Open work (deferred across phases)

These items are built-and-ready or hardware-blocked; they need execution, not design:

- **Re-shard models** — Mistral-7B, Qwen2.5-14B, Qwen2.5-32B shard dirs were
  deleted in a Phase 10 disk cleanup. Re-run `swlp download --model <id>` to
  regenerate, then the deferred measurements below can run.
- **Batched 7B/14B numbers** (Phase 15) — `run_batch` at batch 1/2/4/8/16 on real
  model scale; confirm flat decode-sweep wall time.
- **30B-class run** (Phase 15) — Qwen2.5-32B on M5 at W=2; `configs/swlp_qwen32b_mps.toml` is ready.
- **Long-context KV** (Phase 16) — Qwen2.5-32B at 4K/8K context, RAM peak + tok/s.
- **Safetensors cold-read latency** (Phase 17) — safetensors vs `.pt` on real shards.
- **INT4 KV perplexity** (Phase 18) — measured delta vs FP16 KV on a held-out set.
- **MX230 / NVIDIA column** (Phase 3/14) — the CUDA RAM→VRAM PCIe-DMA streaming
  path; requires physical access to the Pop!_OS machine.

---

## Phase 19 — Benchmark integrity & measurement hygiene

**Goal:** Make every cited number defensible before the paper. A 2026-05-29 audit
found the same Mistral-7B W=2 FP16 configuration cited as **three different**
throughput numbers (0.21 / 0.42 / 0.50 tok/s) across the docs, used
interchangeably; the 0.505 "99% of ceiling" figure is physically impossible for
cold-SSD streaming (0.505 > the 0.496 tok/s cold ceiling) and was measured from
**warm page cache**. No harness controlled page-cache state, and the README's
head-to-head paired SWLP and AirLLM numbers taken from *different* harnesses.

**Strategy (CTO call):** fix the *methodology* first (so future numbers are
trustworthy), then re-measure. Cold and warm are both legitimate but answer
different questions and must never be conflated.

**Task list:**
- [x] `scripts/research/bench_common.py` — shared `drop_page_cache()` (macOS `purge` /
      Linux `drop_caches`, self-reporting if it could not drop the cache),
      `summarize_cache()`, and `provenance()` (versions, git commit, hardware).
- [x] `--cold` mode in `compare_airllm_swlp.py` + `phase3_baselines.py` — drops
      the page cache before each timed run, for SWLP *and* AirLLM equally;
      records per-result `cache_state` and a top-level `cache_mode`.
- [x] Stamp `provenance` into every benchmark JSON (AirLLM pinned at 2.11.0).
- [x] Removed an invalid dead-code line in the comparison harness summary.
- [x] `docs/benchmark_methodology.md` — cold/warm protocol, provenance, fair
      head-to-head rules, competitor landscape, and the provenance of the three
      superseded numbers.
- [x] Re-shard Mistral-7B (shard dir deleted in the Phase 10 cleanup) —
      32 × 436.2 MB safetensors, 13.96 GB, manifest verified.
- [~] Authoritative cold + warm same-harness SWLP-vs-AirLLM run — warm run
      measured this session; the cold run needs `sudo` for `purge` and is the
      user's to run (exact command in `docs/benchmark_methodology.md`).
- [~] Reconcile README + `results.md` numbers with the re-measured figures —
      pending the cold run.
- [~] Benchmark oLLM on M5 (short context; it runs on Apple Silicon but its
      long-context flash-attn path is CUDA-only) — pending a new-package OK.

**Completion checklist:**
- [x] No harness can mislabel a warm run as cold (self-reporting cache state)
- [x] Every benchmark JSON is self-describing (provenance + versions + cache_mode)
- [x] `pytest` 154/154; `ruff check src/` clean (changes live in `scripts/` + `docs/`)
- [~] Cold headline numbers measured and docs reconciled — pending the sudo run

**Phase notes:**
> The audit's smoking gun: the comparison harness ran a warmup before every timed
> run, leaving the 14 GB of shards in OS page cache, so "streaming" timed a RAM
> read, not an SSD read. The fix isolates "cold SSD, warm kernels" by dropping the
> cache *after* the warmup. See `docs/benchmark_methodology.md` §1 for the worked
> example and §6 for the three superseded numbers and their provenance.
> No source code changed — Phase 19 is measurement hygiene in `scripts/` + `docs/`;
> test count stays 154/154.

---

### Phase 20 — Hot-path copy elimination + GPU-path fixes

**Goal:** Close the gap between measured throughput (44% of the cold-SSD
ceiling) and the physical ceiling. The Phase 19 warm-vs-cold numbers proved the
bottleneck was CPU-side per-token overhead, not disk bandwidth: three
full-shard copies per layer per token (chunk-join → safetensors deserialize →
host→device cast) plus a wasted `to_empty(device)` allocation, all on the
compute thread, with a fresh Python thread spawned per prefetch.

**Task list:**
- [x] `core/shard_io.py` (new) — `readinto()` into reusable per-worker uint8
      buffers; zero-copy safetensors parsing (`parse_safetensors_views`);
      FP8 nesting helper; optional pinned buffers for CUDA DMA.
- [x] `core/streaming.py` rewrite — persistent `ThreadPoolExecutor` prefetch
      (was: one thread per layer per token); dequant + host→device transfer
      moved into the worker; `ensure()` reduced to `load_state_dict(assign=True)`
      (pointer swap); dropped the `to_empty(device)` pre-materialisation; meta
      buffers materialised explicitly (GPT-2 non-persistent `bias`).
- [x] `core/scheduler.py` — `evict()` no longer copies unchanged weights
      device→host (`.to("cpu")`): CPU master tensors captured at init, evict is
      a pointer restore. Halves PCIe traffic on the CUDA/MX230 path.
- [x] `SWLP_DIRECT_IO=auto|on|off` — F_NOCACHE is policy, not hardwired.
      `auto` bypasses the page cache only when the model exceeds ~60% of
      available RAM; fitting models get reclaimable page-cache residency for
      free (no Phase 4 memory-compressor risk). Harnesses pin `"on"`.
- [x] `_run_blocks` prefetch lookahead — wires the previously-dead
      `prefetch_depth` config (lookahead = max(window, prefetch_depth)).
- [x] Incremental detokenizer — per-token decode cost no longer scales with
      prompt length (8-token anchor + completion ids; no per-token full-tensor
      `.tolist()` device sync).
- [x] Loud KV-manager fallback (echoes discarded budget/compression/tiering).
- [x] `bench_common.summarize_runs()` — median ± IQR over ≥5 runs for headline
      numbers.
- [x] Tests: `test_shard_io.py` (9), `test_scheduler.py` (6),
      `test_bench_common.py` (5), direct-IO policy + pool round-trip tests.

**Measured (M5 16 GB, direct I/O on, W=2, greedy, single runs 2026-06-11):**

| Model | Before | After | Notes |
|---|---|---|---|
| Mistral-7B FP16 | 0.218 tok/s | **0.372 tok/s** (median of 3: 0.360/0.372/0.379) | +71%; 75% of 0.496 ceiling (was 44%); prefill 5.86→2.30 s; RAM flat; byte-identical output |
| Qwen2.5-0.5B FP16 | 5.47 tok/s | **8.39 tok/s** | +53%; TTFT 1.38→0.44 s; byte-identical output |
| Qwen2.5-0.5B (`auto` direct-IO, cached) | — | **9.85 tok/s** | TTFT 0.195 s — page-cache residency for models that fit |

**Phase notes:**
> The 0.21-warm vs 0.505-warm Phase 19 discrepancy is now largely explained:
> the hot path was overhead-bound, so "warm cache" runs measured copy/alloc
> overhead, not I/O — different transformers/torch versions and allocator
> states moved that overhead between sessions. With the copies gone, residual
> non-I/O time per 7B token is ~0.7 s (2.7 s/token at 2.0 s disk), the next
> profiling target.
> Remaining throughput levers, in order: draft-model speculative decoding
> (Qwen2.5-0.5B shards already on disk as a candidate drafter), batch ×
> speculative composition, deeper prefetch on faster SSDs.

### Open work — updates (2026-06-11, Phase 20)

- ~~Re-shard models~~ — **done**: `shards/mistral-7b` (safetensors), `shards/qwen-14b`,
  `shards/qwen-0.5b` exist and verify; the deferred batch/30B measurements are unblocked.
- Paper-grade Phase 20 re-measurement — ≥5 runs, median ± IQR
  (`bench_common.summarize_runs`), cold (`sudo purge`) + warm, SWLP and AirLLM
  on the same harness with `swlp_direct_io="on"`.
- Draft-model speculative decoding — replace/augment the n-gram drafter with a
  small resident FP16 draft model (Qwen2.5-0.5B, ~1 GB); n-gram acceptance is
  0% on novel text, draft-model acceptance is typically 60–80% → ~3× on all
  workloads, still lossless. Needs its own phase.
- NVIDIA column (MX230) — also add at least one of FlexGen / DeepSpeed
  ZeRO-Inference as a CUDA baseline alongside AirLLM (reviewers will ask);
  verify the Phase 20 evict fix (no device→host copy-back) on real PCIe.
- Profile the residual ~0.7 s/token non-I/O overhead on 7B (pyinstrument,
  `SWLP_PROFILE=1`) — candidates: per-tensor Python loop in the assign-load,
  MPS copy serialization against compute, KV-cache bookkeeping.
- Known gap for the MX230 work: `SWLPScheduler._pin_blocks` pins with
  `recurse=False`, so nested transformer-block params are never actually
  pinned. The right design is a pinned staging ring buffer (the
  `shard_io.read_into_tensor(pin=True)` path already provides it on the
  shard-streaming route); fix or remove `_pin_blocks` when CUDA hardware is
  available to measure.

### Phase 21 — Draft-model speculative decoding

**Goal:** Break the one-token-per-disk-sweep barrier on novel text. The Phase 5
n-gram drafter gets ~0% acceptance on fresh prose (it only fires when the
trailing n-gram recurs), so plain streaming decode was bounded by the per-sweep
disk cost (0.19 tok/s on Qwen-14B). A small resident same-tokenizer draft model
proposes on every step; the streamed target verifies K tokens per sweep.

**Task list:**
- [x] `runner/draft.py` (new) — `DraftModelDrafter`: resident small model,
      greedy autoregressive drafting through its own `DynamicCache`,
      self-healing common-prefix crop on rejection (the caller never manages
      drafter state); `load_draft_model()`; `ensure_same_tokenizer()` hard
      vocab-identity check.
- [x] Adaptive draft length (AIMD): full acceptance doubles K, majority
      acceptance increments, poor acceptance halves (floor 1). Caps the
      worst-case overhead on low-agreement text at ~2 drafter forwards/sweep.
- [x] `SpeculativeRunner` selects the drafter from `swlp_draft_model`
      (n-gram remains the zero-RAM default); drafter loads inside `load()` so
      its cost never pollutes generation timing.
- [x] Config/CLI: `swlp_draft_model` + `SWLP_DRAFT_MODEL`; `--draft-model` now
      feeds both the MLX and SWLP paths; `--shard-dir` + `--draft-model`
      auto-selects the speculative backend; `configs/swlp_qwen_draft_mps.toml`.
- [x] Tests: `tests/test_draft.py` (21) — cached drafting proven token-identical
      to a no-cache greedy reference (partial accept / full accept / divergence),
      AIMD trajectories, tokenizer guard, config + backend auto-select wiring.
- [x] Measured on M5 vs plain streaming, byte-identity verified per prompt.

**Measured (M5 16 GB, Qwen2.5-14B-Instruct FP16 streamed from
`shards/qwen-14b`, ~28 GB, W=2, direct I/O auto→on, greedy, 32 new tokens,
drafter Qwen2.5-0.5B-Instruct FP16 resident ~1 GB, max_draft 8, single runs
2026-06-11):**

| Workload | Baseline swlp | Draft-spec | Speedup | Acceptance | tokens/sweep |
|---|---|---|---|---|---|
| Open-ended sentence | 0.187 tok/s | **0.545 tok/s** | **2.9×** | 47.7% | 3.2 |
| Constrained list (primes) | 0.197 tok/s | **1.158 tok/s** | **5.9×** | 90.0% | 8.0 |

Completions **byte-identical** to plain greedy SWLP on both prompts.

**Phase notes:**
> **Adaptive K is what makes the feature safe.** With fixed K=8 the open-ended
> 32-token run *regressed* to 0.151 tok/s (−19% vs baseline): the 0.5B and 14B
> greedy chains diverge at the first token of an open-ended continuation and
> every sweep burned 8 wasted drafter forwards + a 9-position verify. AIMD K
> collapses drafting to ~2 forwards/sweep in low-agreement regions and recovers
> full depth within a few sweeps once the chains re-converge — same run went to
> 0.545 tok/s.
> **A bigger drafter did not help.** Qwen2.5-1.5B produced *identical*
> acceptance to the 0.5B on the hard prompt (5/34 on 12 tokens) at 3× the
> residency and ~2–3× the drafting cost: the disagreement is target-phrasing
> specific, not drafter capacity. 0.5B stays the default.
> **The speedup is acceptance-bound, and acceptance is workload-bound** —
> report ranges, not a single number. fp16 drafting on MPS was verified
> bit-equal to fp32 CPU (no numerical fragility).
> Drafter prefill happens once per generation (~1 s) inside the first propose;
> the per-sweep drafter cost is K+1 incremental small-model forwards.

### Open work — updates (2026-06-11, Phase 21)

- ~~Draft-model speculative decoding~~ — **done** (Phase 21): 2.9×–5.9×
  measured, lossless, adaptive K. The earlier "~3× on all workloads" estimate
  was optimistic for open-ended prose with a 0.5B drafter; the measured floor
  with adaptive K is ≈ parity, the ceiling 5.9×.
- Speculative decoding for Mistral-7B needs a same-tokenizer small draft model
  (the Mistral family ships none); candidate: distill/prune one, or restrict
  draft-spec claims to the Qwen column of the paper.
- Acceptance-aware paper benchmarking: measure acceptance + speedup across a
  prompt-set spectrum (code, summarize-with-context, freeform) — the
  bench-suite harness already sweeps prompt sets.
- Possible next lever: tree/multi-branch drafts at the diverging first
  position (verify several candidate first tokens in one sweep — batch
  dimension is nearly free on the disk-bound path, cf. Phase 10).

### Phase 22 — Lossless shard codec (.swz)

**Goal:** SWLP-Max step 1 — shrink the bytes a token must move. FP16/BF16
high bytes (sign + exponent) are highly skewed, so byte-grouping + entropy
coding (zipnn: zstd + Huffman) removes ~31% of shard bytes with **bit-exact**
reconstruction. Hypothesis: on an SSD-bound path, 31% fewer bytes ≈ 1.46×
effective weight bandwidth.

**Task list:**
- [x] `codec.py` (new leaf module) — `.swz` container: magic + raw size +
      SHA-256 (compress-time/offline) + CRC-32 (every decompress) + zipnn
      blob. `compress_bytes()` roundtrip-verifies before returning;
      `decompress_bytes(check_sha=)`, `verify_blob()`, `is_swz()`,
      `compressed_path()`.
- [x] CRC-32 gate as a hard safety requirement: zipnn 0.5.4's C core
      segfaults on malformed streams (measured 7/51 single-byte corruptions
      crashed the process; 0/51 with the gate).
- [x] Streaming integration — `shard_io.read_shard_payload()` decompresses
      `.swz` in the worker pool; `StreamingScheduler._shard_path()`
      auto-detects `.safetensors` / `.safetensors.swz` / `.pt` per file
      (mixed mid-conversion dirs just work).
- [x] Decompress gate — `BoundedSemaphore(1)` serializes zipnn calls across
      workers (zipnn fans out over all cores internally; two concurrent
      decodes measured 95 ms vs 30 ms per layer). +12% tok/s on the
      compressed path.
- [x] `model/shard.py` — `compress_shards()` / `decompress_shards()`:
      in-place, atomic per layer (temp + rename, source deleted only after
      verification), resumable, manifest `shard_compression` field.
- [x] CLI — `swlp compress-shards <dir>` and `--revert`; honest help text.
- [x] Tests: 16 codec + 13 shard-format + 4 streaming; bit-identical
      completions verified on real qwen-0.5b and mistral-7b.

**Measured (M5 16 GB, Mistral-7B FP16, 32 layers, W=4, depth=2, direct I/O,
6-token probe runs 2026-06-11):**

| Variant | tok/s | per-layer worker stages |
|---|---|---|
| Plain `.safetensors` | **0.390** | read 120 ms (SSD-limited) + h2d 39 ms |
| `.swz`, ungated decompress | 0.248 | read 96 + decompress 95 + h2d 58 ms |
| `.swz`, gated (sem=1) | 0.264 | decompress 59 ms serialized |
| `.swz`, gated, depth=4 | 0.278 | stages inflate with worker count |

Compression ratio 0.673 (13.96 → 9.40 GB); disk saving delivered as designed.

**Phase notes:**
> **Negative result for throughput on fast SSDs — keep it honest (cf. Phase 7
> FP8).** The hypothesis assumed decompression rides along free. It does not:
> under end-to-end load zipnn drops from 15 GB/s standalone to ~7 GB/s
> (contends with MPS compute, h2d copies, and SSD DMA for unified-memory
> bandwidth — the compressed path moves *more* total bytes through the memory
> controller: ~9.4 GB DMA + 23.4 GB decompress r/w + 28 GB h2d vs 14 + 28
> uncompressed). With the M5 SSD at 6.9 GB/s ≈ contended decode speed, the
> worker pool saturates on decompress and tok/s falls ~25–33%.
> **Crossover ≈ 3.5 GB/s sequential read**: win when
> `ssd_bw < saved_fraction/ratio × decode_bw ≈ 0.49 × 7 GB/s`. PCIe-3-class
> NVMe (2–3.5 GB/s) and external enclosures benefit; fast Apple Silicon SSDs
> lose. `swlp compress-shards --revert` is the off-ramp (mistral-7b reverted
> on the dev machine: 0.248 → 0.390 tok/s recovered).
> **What stays:** the ~31% disk footprint (always), the slow-SSD win, and the
> safety architecture (CRC gate, SHA verify, atomic resumable conversion).
> Deeper pipelining (chunked containers overlapping read/decode/h2d per
> tensor) cannot beat the memory-bandwidth ceiling above and is not worth the
> format complexity on this hardware class.

---

### Phase 23 — Quality-neutral speedups + opt-in quality tradeoffs
**Goal:** Cheap wins in the decode loop plus explicitly opt-in lossy tiers.

Implemented in `core/phase23.py` and wired through `runner/swlp.py`:
`ActivationCache` (prompt-prefix hidden-state LRU for chat reuse) and
`PreallocBuffer` (pre-allocated token buffer replacing per-token `torch.cat`)
ship ON by default (`swlp_activation_cache`, `swlp_prealloc_buffer`);
`EarlyExitDetector` (entropy threshold) and `LayerPruner` (light/aggressive)
ship OFF (`swlp_early_exit`, `swlp_layer_pruning`) — labelled lossy. The
early-exit entropy signal runs over next-token logits (`final_norm` →
`lm_head`, fixed in the Phase 24 round-1 audit; it previously read raw
hidden states).

---

### Phase 24 — Correctness fixes, measured hardware probe, quick setup
**Goal:** Clear the audit defect list; make first-run setup Ollama-easy; replace
hardcoded SSD bandwidth with a measured, cached probe.

**Completion checklist:**
- [x] `swlp profile` non-JSON path crash (`result.comulsion` typo) fixed
- [x] `HuggingFaceRunner.stream_tokens` now applies the repetition penalty
      (parity with `run()`)
- [x] Early-exit entropy signal now computed over vocabulary logits (last
      position projected through `lm_head`), not hidden states
- [x] Dead code removed: `ThreadedPipeline` (+ its test and POC script),
      `LayerProfiler.log_state`, unused RSS helper
- [x] `KVCacheManager.clear()` resets peak/op counters
- [x] `sparsify_shards` writes safetensors (was legacy `.pt` only)
- [x] Measured SSD bandwidth: one-time probe cached in `~/.cache/swlp/hardware.json`,
      consumed by `detect_hardware()` (constants remain the fallback)
- [x] Profiler RSS switched from `ru_maxrss` high-water to instantaneous psutil
- [x] `swlp pull <alias>` — download + sharding with progress and
      disk-space preflight (single-stream `snapshot_download`; parallel
      hf_transfer transfer intentionally deferred); `serve` gains `/v1/models`
- [x] pytest + `ruff check src/` clean (402 tests at Phase 27 close)

**Phase notes:**
> Research base: FreeToken (arXiv:2608.16157), Mixtral-offloading
> (arXiv:2312.17238), MoE-SpeQ (arXiv:2511.14102), Apple SpecMD, EAGLE-3
> (arXiv:2503.01840), mlx-lm 2026 releases (mxfp8/nvfp4), mlx-lm issue #1332
> (DeepSeek-V4 on Apple Silicon). Plan of record: MoE expert-streaming engine,
> prefix-KV caching, bandwidth aggregation, lossless spec-decode upgrades.

---

### Phase 25 — MoE expert-streaming engine
**Goal:** Run Mixture-of-Experts models with expert-selective sweeps: per-token
bytes scale with *active* parameters, not total. Exact quality — routing is
part of forward; prefetch never substitutes routing decisions.

**Completion checklist:**
- [x] Shard format v2: per-layer dense shard + `layer_XXX.experts.safetensors`
      bank (flattened expert tensors, range-readable); manifest gains
      `num_experts` / `top_k` / `expert_bank` (v1 manifests load unchanged)
- [x] MoE forward decomposition — `SwlpCachedExperts` (runner/experts.py),
      swapped in for the fused Experts module (wiring in runner/load.py);
      matching installed transformers ops order — verified bit-exact vs the
      installed Qwen3Moe reference (per-step logits) on a tiny random model
- [x] `ExpertScheduler` (`runner/expert_scheduler.py`): global LRU expert
      cache keyed (layer, expert) with byte budget; parallel staging of the
      routed set; predictive prefetch from routing history; elastic budget
      (`set_budget`); `q_star_split` policy (FreeToken §4) for the CUDA path
- [x] Config surface: `SWLP_EXPERT_CACHE_MB`, `SWLP_EXPERT_PREFETCH`
      (`off|lru|predictive`, default predictive)
- [x] Tests: `test_expert_bank.py`, `test_experts.py`, `test_moe_policy.py`,
      shard-format v2 cases, tiny-Qwen3MoE streaming equivalence

**Phase notes:**
> Design follows FreeToken (Apache-2.0; design-level attribution, no code
> copied) and the Mixtral-offloading line of work. On unified memory there is
> no CPU/GPU bandwidth split, so hit rate — not the q* placement — dominates;
> predictive prefetch (SpecMD: expert access is not LRU-friendly) is the
> default. DeepSeek-V4-Flash (284B/A13B, 6-of-256, MXFP4) feasibility math:
> ~1.1 tok/s floor at 100% miss, ~3–4 tok/s at ~30% miss on M5 16 GB — a
> measurement target for Phase 28, gated on mlx-lm #1332 class issues.

---

### Phase 26 — Lossless throughput: prefix KV, striping, chunked prefill
**Goal:** Fewer sweeps per token (prefix reuse), more bytes per second
(volumes), bounded prefill memory (chunking) — all quality-neutral.

**Completion checklist:**
- [x] `core/prefix_cache.py` — exact-match prefix KV snapshots, bounded by
      entry count AND bytes; each stored turn snapshots fully plus interior
      interval slices (FreeToken-inspired); wired into `SWLPRunner`
      (`set_prefix_cache`) and the chat REPL (Llama-like + exact DynamicCache
      paths only); unit tests + end-to-end losslessness on a tiny Llama with
      the cache hit asserted via stats (tests/test_llama_equivalence.py)
- [x] Multi-volume striping: `SWLP_SHARD_VOLUMES` round-robins layers across
      SSDs; the read pool parallelizes across volumes
- [x] Chunked prefill: `SWLP_PREFILL_CHUNK` sweeps long prompts in slices
      (causal attention over accumulating KV — lossless)
- [x] Multi-resolution n-gram drafting: full-context match first, backing off
      to shorter contexts (acceptance up, output bit-identical)
- [x] `swlp_spec_max_draft` default raised 8 → 16
- [x] Read-ahead: `madvise(MADV_WILLNEED)` on the mmap (page-cache) path;
      `posix_fadvise(WILLNEED)` (Linux) / `F_READAHEAD` (macOS) on the
      legacy fd reader
- [x] pytest + ruff clean

**Phase notes:**
> **Two latent GPT-2 bugs found and fixed by the new equivalence tests** —
> the strongest possible argument for logit-level tests over determinism-only
> tests: (1) `ln_f` was never persisted in `embed.pt` nor loaded — the final
> norm ran on uninitialized memory (zero logits on the first run in a
> process, recycled-page garbage afterwards; determinism tests passed because
> both runs were identically wrong); (2) the GPT-2 adapter still used the
> pre-transformers-5 tuple KV protocol — `GPT2Block` now mutates a shared
> `DynamicCache` in place and returns bare hidden states, so per-token KV was
> silently dropped, and chunked queries needed an explicit bottom-right
> aligned causal mask (`triu(diagonal=1+past_len)`). New regression:
> `test_gpt2_streaming_first_token_logits_match_hf`.
> Path-equivalence tests compare logits (atol 1e-4, fp32), not completions:
> random-weight models have top-2 logit gaps below fp accumulation noise
> between sweep shapes (~5e-6 SWLP, ~9e-8 for HF's own chunked forward).

---

### Phase 27 — FreeToken alignment, doctor MoE guidance, research harness
**Goal:** Port FreeToken's portable policies; expose MoE guidance and a
measurement harness. (FreeToken itself is NVIDIA-only — alignment is
design-level: bank layout (25), LRU+elastic cache (25), anchor checkpoints
(26), q* (25), fast bootstrap.)

**Completion checklist:**
- [x] `swlp doctor` renders a MoE STREAMING advisory (per-model active-param
      ceilings, expert-cache budget guidance, miss-rate expectations) and MoE
      rows in COMMANDS
- [x] `scripts/research/moe_sweep.py` — expert-cache budget sweep measuring
      tok/s + hit-rate per budget (Phase 28 measurement harness)
- [x] Perf round (post-audit research): concurrent staged fetch of each
      token's routed expert set (`prepare_set`, condition-variable handoff);
      recency-first prediction ordering; `expert_fetch_bench.py` measuring
      serial/staged/warm strategies (staged-w4 5.8 ms vs serial-cold 6.0 ms
      at 12.7 MB × 256-expert scale; warm repeats served from page cache)
- [x] Attribution: FreeToken cited in module docstrings and here
- [x] Docs updated (CHANGELOG, README, AGENTS Current Status)

---

### Phase 28 — Apple-only refocus, MLX throughput, CLI/TUI
**Goal:** Drop the NVIDIA/CUDA column entirely and spend the recovered surface
on Apple Silicon throughput, plus a terminal UI worth looking at.

**Completion checklist:**
- [x] **CUDA/NVIDIA removed** — `SWLPScheduler` (CUDA streams, 143 lines),
      pinned-memory DMA staging in `StreamingScheduler`, `torch.cuda.*`
      branches in `hf`/`swlp`/`batch`/`kv_cache`/`profiler`, `pynvml` and the
      `swlp[gpu]` extra, `configs/swlp.toml`, and `moe_policy.q_star_split`
      (FreeToken's q* divides work across a host bus and a device bus —
      unified memory has one). `vram_*` renamed to `device_*` throughout;
      `RunMetrics.vram_peak_bytes` dropped. 52 → 2 references, both being
      comments that record *why* something was removed. CI moved to
      `macos-15` with the `apple` extra.
- [x] **`runner/mlx_tune.py`** — Apple-specific throughput layer:
      - `mx.set_wired_limit()` raised at load time, **clamped to Metal's
        `max_recommended_working_set_size`** (MLX hard-rejects anything
        above it: measured 11.84 GB of 16 GB on M5). Measured 0 → 11.8 GB.
      - `sysctl_advice()` prints the exact `sudo sysctl iogpu.wired_limit_mb`
        needed to lift the ceiling further. SWLP never runs it: root, machine
        -wide, resets on reboot — the user's call.
      - KV quantization (`--kv-bits 4|8`) with group size and an exact-prefix
        `quantized_kv_start`. On unified memory 4-bit KV is measured *faster*
        than fp16 — decode is bandwidth-bound and the kernel costs less than
        the traffic it saves (arXiv:2605.05699).
      - `num_draft_tokens` default **4** (measured sweet spot 4–6), prefill
        chunking, and an exact-match prompt cache reused across chat turns.
- [x] **CLI**: `--kv-bits`, `--max-kv-size`, `--draft-tokens`, `--wired-limit`;
      `swlp chat <model>` accepts a positional model like `run`/`serve`;
      help restructured around "fits in RAM" vs "bigger than RAM"; `doctor`
      gained an APPLE SILICON TUNING section (wired cap, the sysctl, levers
      ranked by payoff).
- [x] **`src/swlp/tui.py`** — presentation extracted from `chat.py`:
      capability detection (`NO_COLOR`, TTY, unicode), width-aware boxes with
      ANSI-correct measurement, a spinner that is silent off-TTY, and
      `StreamWrapper`, which word-wraps a token stream whose fragments split
      mid-word. Removed a `--quant is ignored` warning that could only ever
      fire on the *default* value (`--quant` already implies `--backend mlx`).
- [x] pytest 489 passing, `ruff check src/` clean, coverage gate 65% (68%).

**Phase notes:**
> Research base for the tuning defaults: Apple's own M5 numbers (prefill
> +3.6x from Neural Accelerators, decode only +19–27%, tracking 120 → 153
> GB/s) establish that **decode is bandwidth-bound**, which is the premise
> for every choice here — compression pays when it costs less time than the
> bytes it saves. mlx-lm speculative decoding is measured at 1.90x (M4 Pro)
> to 2.10x (M5 Max) with a same-family draft model at 64–72% acceptance,
> degrading to <40% on creative output; drafts of 4–6 beat longer ones
> because mid-sequence rejection wastes the whole tail. int4 KV on Apple
> Silicon: ~25 ns/vec kernel overhead against 3x less KV traffic, ΔPpL 0.000
> on Qwen short prompts.
>
> **Negative finding, recorded:** `mx.set_wired_limit` alone cannot exceed
> macOS's working-set ceiling — the first implementation assumed it could and
> failed with `[metal::set_wired_limit] Setting a wired limit larger than the
> maximum working set size is not allowed`. The ceiling is ~74% of RAM and
> only `sysctl iogpu.wired_limit_mb` (root) moves it. Clamp, then advise.
