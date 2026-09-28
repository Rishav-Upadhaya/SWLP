# Benchmark Methodology & Integrity

This document defines how SWLP's benchmarks are run so the numbers are
reproducible and defensible. It exists because an audit (2026-05-29) found three
different "canonical" throughput numbers for the *same* configuration cited
interchangeably across the docs, and at least one number that is physically
impossible for cold-SSD streaming (it was measured from warm page cache). The
rules below prevent both classes of error.

---

## 1. Why page-cache state is the dominant confound

SWLP streams weights from SSD. The hard throughput ceiling for any layer-streaming
runtime is:

```
max_tok/s  =  SSD_bandwidth / model_bytes_streamed_per_token
```

For Mistral-7B FP16 on the M5 (6.93 GB/s measured, 13.96 GB model) that is
**0.496 tok/s**. A *real* cold run must fall **below** this once per-layer compute
is added — compute and the per-token layer materialization are not free.

The trap: if a model's shards are already resident in the **OS page cache** (left
there by a warmup run, a previous benchmark, or simply a prior invocation), then a
"streaming" benchmark reads from RAM, not SSD. It can then **meet or exceed the
cold-SSD ceiling**, because the SSD is no longer in the loop.

> **Worked example of the bug this caused.** An earlier harness reported
> **0.505 tok/s** for Mistral-7B and the docs claimed "99% of the 0.509 tok/s
> theoretical ceiling." 0.505 > 0.496 is impossible for cold streaming on a
> 6.93 GB/s SSD — so that run was reading page-cached shards from RAM, not
> streaming from disk. Meanwhile the rigorous 3-prompt median was **0.21 tok/s**.
> Both came from warmup-contaminated harnesses; neither controlled cache state.

**Cold and warm are both legitimate numbers — they answer different questions:**

| Number | Question it answers | Use in the paper |
|---|---|---|
| **Cold** (page cache dropped) | True SSD-streaming throughput on a fresh run | The headline streaming number; the apples-to-apples comparison vs AirLLM/oLLM |
| **Warm** (shards page-cached) | Best case when the OS happens to have cached the model | A clearly-labelled upper bound; never the headline |

They must never be conflated or silently mixed.

---

## 2. The cold protocol

Both benchmark harnesses (`scripts/research/compare_airllm_swlp.py`,
`scripts/research/phase3_baselines.py`) take a `--cold` flag. In cold mode the OS page
cache is dropped immediately before **each timed run**, after any warmup (so MPS
kernels and the tokenizer stay warm — only the *weights* are cold):

```bash
# True cold-SSD streaming (needs elevated privileges to drop the cache):
sudo python scripts/research/compare_airllm_swlp.py --models 7b --cold
sudo python scripts/research/phase3_baselines.py --baseline all --cold

# Warm (default) — fast iteration, but an upper bound, not the headline:
python scripts/research/compare_airllm_swlp.py --models 7b
```

Cache-drop mechanism (`scripts/research/bench_common.py::drop_page_cache`):
- **macOS:** `purge` (requires `sudo`).
- **Linux:** `sync` + write `3` to `/proc/sys/vm/drop_caches` (requires root).

**The harness never lies about cache state.** If the cache could not actually be
dropped (e.g. `--cold` without sudo), `drop_page_cache()` returns
`"warm (purge needs sudo …)"` and that exact string is recorded as the run's
`cache_state`. A warm run is therefore never mislabelled as cold. Every result
carries a `cache_state` field; every report carries a top-level `cache_mode`.

---

## 3. Provenance stamping

Every benchmark JSON now embeds a `provenance` block
(`scripts/research/bench_common.py::provenance`) so a result is self-describing and
reproducible:

```json
"provenance": {
  "timestamp_utc": "...", "git_commit": "...", "platform": "...",
  "python": "...",
  "hardware": {"chip": "Apple M5", "memory_gb": 16.0, "ssd_bandwidth_gbps": 6.93, ...},
  "versions": {"airllm": "2.11.0", "mlx": "0.31.2", "torch": "2.12.0", ...}
}
```

Pinning library versions matters: AirLLM has changed across releases (recent
versions add next-layer prefetch), so a comparison is only meaningful with the
version recorded. SWLP-vs-AirLLM numbers in this repo were taken against
**airllm 2.11.0**.

---

## 4. Fair head-to-head rules

1. **Same harness, same run.** Never compare SWLP from harness A against a
   competitor from harness B. The earlier README table paired SWLP=0.50
   (`phase3_baselines.py`) against AirLLM=0.21 (a cross-doc number) — not a valid
   head-to-head, and the AirLLM figure contradicted the dedicated comparison's
   0.085. Both systems must be measured in the same process invocation, on the
   same prompts, with the same `cache_mode`.
2. **Same cache state for both systems.** `--cold` drops the cache before *both*
   the SWLP and the AirLLM timed runs.
3. **Median of ≥2 timed runs**, 1 warmup discarded. Report the prompt set.
4. **Same precision tier.** FP16-vs-FP16 only for the headline; quantized systems
   (Ollama Q4_K_M, MLX 4-bit) are a separate, labelled reference tier.

---

## 5. Competitor landscape (what to compare against)

The paper compares SWLP against the right systems, not just AirLLM:

| System | Approach | Lossless? | Runs on M5 (MPS)? | Status in this repo |
|---|---|---|---|---|
| **AirLLM** | Layer load→compute→discard (now with some prefetch) | ✅ FP16 | ✅ (via MLX) | Benchmarked (airllm 2.11.0) |
| **oLLM** | Layer + KV SSD streaming, no quant, FlashAttention-2 | ✅ FP16/BF16 | ✅ — but long-context (flash-attn) is CUDA-only | **Not yet benchmarked** — see note |
| FlexGen | Throughput-via-batching offload | ❌ 4-bit | CUDA | Related work |
| LLM in a flash (Apple) | Neuron-sparsity flash loading | ⚠️ needs ReLU sparsity | — | Related work |
| PowerInfer | Hot/cold neuron GPU+CPU split | ⚠️ needs sparsity | CUDA | Related work |
| DeepSpeed ZeRO-Inference | NVMe weight offload + prefetch | ✅ FP16 | CUDA | Related work |

> **oLLM is the most direct competitor and is benchmarkable on this M5.** It
> shares SWLP's exact pitch — lossless FP16/BF16 SSD streaming, no quantization —
> and is newer (Sept 2025) and more feature-complete (FlashAttention-2 + online
> softmax + chunked MLP + disk-backed KV). On Apple Silicon it runs but loses the
> long-context path (flash-attn is CUDA-only). A SWLP-vs-oLLM head-to-head on the
> M5 (short context, where both are comparable) would materially strengthen the
> paper's novelty argument; installing it is a benchmark-only dependency
> (like airllm — not added to `pyproject.toml`).

---

## 6. The numbers being corrected

For the record, the three pre-audit Mistral-7B W=2 FP16 throughput figures and
their provenance:

| tok/s | Source | Conditions | Verdict |
|---|---|---|---|
| 0.422 | `benchmarks/phase3.json` | 1 run, 1 prompt, cache state uncontrolled | superseded |
| 0.502 / 0.505 | `phase3_baselines.py` multi-run | warmup → **warm cache** | upper bound, not cold |
| 0.210 | `compare_airllm_swlp.py` | 3-prompt median, warmup → partially warm | closest to honest, still not cold |

The authoritative replacement is a single same-harness, provenance-stamped run in
both `cache_mode=cold` and `cache_mode=warm`, reported side by side. See
`docs/results.md` once re-measured.
