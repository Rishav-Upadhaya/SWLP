# Hardware Baseline — Phase 0 Measurements

All numbers collected via `python scripts/phase0_hardware_check.py` and `swlp baseline/swlp` CLI.

---

## M5 (Apple Silicon, 16 GB unified memory)

| Metric | Value | Notes |
|---|---|---|
| SSD read bandwidth | 6.93 GB/s | `phase0_hardware_check.py` sequential read |
| SSD write bandwidth | 4.48 GB/s | `phase0_hardware_check.py` sequential write |
| MPS matmul 1k×1k | 4.2 ms | `torch.matmul` on MPS device |
| MLX matmul 1k×1k | 3.5 ms | `mlx.core.matmul`, no warmup overhead |
| RAM bandwidth | 153 GB/s | LPDDR5X spec (M5) |
| Unified RAM | 16 GB | No PCIe; SSD is the streaming bottleneck |

### tiny-gpt2 (HF baseline, MPS)

| Metric | Value |
|---|---|
| Load time | 4.35 s |
| Throughput | 100.3 tok/s |
| Backend | `hf` (full model in unified RAM) |
| Config | `configs/baseline.toml --runner mock` → `configs/baseline.toml` |

### tiny-gpt2 (SWLP, MPS, window=4)

| Metric | Value |
|---|---|
| TTFT | 5.85 ms |
| Throughput | 111.1 tok/s |
| Backend | `swlp`, `ThreadedScheduler` |
| Config | `configs/swlp_mps.toml` |
| Note | tiny-gpt2 fits in RAM; SSD streaming not exercised at this scale |

### Mistral-7B simulation (sim_m5.toml)

Modeled via `swlp simulate --scenario configs/sim_m5.toml --report`.
Parameters: 32 layers, 175 MB/layer, 6.93 GB/s SSD, 20 ms compute/layer.

| Window W | Per-token | Throughput | Bottleneck |
|---|---|---|---|
| 2 | 6.905 s | 0.14 tok/s | SSD transfer |
| 4 | 6.905 s | 0.14 tok/s | SSD transfer |
| 6 | 6.905 s | 0.14 tok/s | SSD transfer |

Transfer dominates: 32 layers × ~197 ms/layer (SSD) vs 32 × 20 ms (compute) = 11× slower.
Overlap with W layers has minimal gain at this ratio. Real speedup path: NVMe throughput or pre-sharding strategy.

---

## MX230 (NVIDIA discrete, 2 GB VRAM) — **Pending**

> Run on Pop!_OS. Execute `python scripts/phase0_hardware_check.py` and `swlp baseline/swlp` on MX230.

| Metric | Value | Notes |
|---|---|---|
| PCIe bandwidth | — | To be measured |
| VRAM | 2 GB | MX230 spec |
| RAM | — | To be measured |
| tiny-gpt2 TTFT | — | — |
| tiny-gpt2 tok/s | — | — |
| SWLP TTFT | — | — |
| SWLP tok/s | — | — |

---

### Mistral-7B SWLP — real disk streaming (Phase 1)

Model: `unsloth/mistral-7b-instruct-v0.2` sharded to `./shards/mistral-7b/`
(32 layers × 436 MB each, 13.96 GB total).
Runner: `SWLPRunner` + `StreamingScheduler` (loads `layer_NNN.pt` on demand).
Prompt: "The capital of France is" → "Paris, and it is one of the most popular tourist destinations in the world".

**ThreadedPipeline POC** (`scripts/research/run_pipeline_forward.py`, 80 ms simulated compute/layer):

| Window | Sequential | Pipelined | Overlap gain |
|---|---|---|---|
| W=2 | 4.62 s | 3.75 s | 18.9% |
| W=4 | 4.62 s | 3.23 s | 30.0% |
| W=6 | 4.62 s | 2.95 s | **36.3%** ✓ |

**End-to-end SWLPRunner** (Mistral-7B, MPS, 16 new tokens):

| Window | Throughput | TTFT | Generate | RAM peak |
|---|---|---|---|---|
| W=2 | 0.40 tok/s | 6 ms | 40.1 s | 1.40 GB |
| W=4 | 0.29 tok/s | 14 ms | 55.9 s | 1.03 GB |
| W=6 | 0.25 tok/s | 22 ms | 65.0 s | 0.93 GB |

Best end-to-end W on M5 = **W=2** — small window minimizes unified-memory pressure
between CPU prefetch and GPU compute, even though synthetic POC showed bigger W
better at hiding pure SSD latency. The RAM peak of ~1 GB vs 14 GB full-load confirms
true sliding-window streaming.

### KV cache compression (Phase 2)

`CompressedDynamicCache` routes Llama/Mistral KV through `KVCacheManager` —
cold layers are zlib-compressed in host RAM, decompressed on the next token
step. Measured on Mistral-7B, M5, `scripts/research/kv_compare.py`:

**Quality (compression OFF vs ON, same prompt):**

| Check | Result |
|---|---|
| Completion text match | **identical** (lossless) |
| Quality delta | **0** — zlib is bit-exact on the tensor bytes |
| Generate-time overhead | ~3% (compress/decompress cost) |

**Compression-level sweep (zlib 1–9, Mistral-7B):**

| Level | Ratio | Compressed | Generate |
|---|---|---|---|
| 1 | 1.104× | 2665 KB | 29.0 s |
| 3 | 1.104× | 2666 KB | 32.6 s |
| 6 | 1.108× | 2658 KB | 34.5 s |
| 9 | 1.108× | 2658 KB | 37.0 s |

**Sweet spot = level 1** — same ratio as level 9 but 22% faster. FP16 KV
activations are high-entropy, so higher zlib effort buys almost nothing.

**Key finding:** lossless zlib yields only ~1.1× on FP16 KV. The CLAUDE.md
"~2.5×" target assumes *lossy* KV quantization (KIVI-style 4-bit) — that is a
separate, future workstream. zlib's value here is host-offload tiering of cold
layers, not raw ratio.

### KV memory budget math

`kv_budget_recommendation()` (`hardware/detect.py`) sizes the KV budget:

```
window_footprint = (window_size + 1) × layer_weight_mb
headroom         = total_ram − OS_reserve(3 GB) − embed_reserve(1 GB) − window_footprint
kv_budget        = max(headroom × 0.5, 256 MB)
```

M5 16 GB, Mistral-7B (436 MB/layer):

| Window | Footprint | Headroom | KV budget |
|---|---|---|---|
| W=2 | 1.31 GB | 10.98 GB | 5.49 GB |
| W=4 | 2.18 GB | 10.11 GB | 5.05 GB |

Triggered when `kv_memory_budget_mb <= 0` (or `SWLP_KV_BUDGET_MB=0`).

**30B projection:** a 30B model (~60 layers, GQA) has KV ≈ 240 KB/token →
~7.5 GB at 32K context. zlib (1.1×) trims that to ~6.8 GB — still large.
Fitting 30B + long context on 16 GB needs lossy KV quantization, confirming
the Phase 2 / future split.

---

## Key observations (Phase 0)

- **SSD is the M5 bottleneck** at 6.93 GB/s. For a 7B model (5.6 GB weights), streaming all layers takes ~0.8s cold — every token at full-model scale pays this.
- **Tiny models are RAM-bound**, not SSD-bound. SWLP on tiny-gpt2 is faster than HF (111 vs 100 tok/s) due to lower peak RAM pressure, not streaming gains.
- **transformers 5.x breaking change**: `GPT2Block.forward()` renamed `layer_past=` → `past_key_values=` and now returns a plain `Tensor` (not a tuple). Fixed in `src/swlp/runner/swlp.py`.
- **Phase 1 target**: shard Mistral-7B, run real SSD→RAM streaming via `ThreadedPipeline`, measure actual overlap efficiency.
