# SWLP System Architecture

## Full Feedback Loop

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SWLP SYSTEM ARCHITECTURE                             │
│                    Hardware-Aware Scheduling Framework                       │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐
  │   HARDWARE   │  M1/M2/M3/M4/M5, 8-64 GB RAM, NVMe/SSD
  │   PROBE      │  chip_name, memory_gb, ssd_bandwidth_gbps
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │   PIPELINE   │  estimate_pipeline_ratio()
  │   MODEL      │  = (SSD Read + Deserialize + Upload) / Block Compute
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐     ┌──────────────┐
  │   RESIDENT   │────▶│  CONFIDENCE  │  explainable score [0,1]
  │   POLICY     │     │  ESTIMATOR   │  + factor breakdown
  └──────┬───────┘     └──────────────┘
         │
         ▼
  ┌──────────────┐
  │   RESIDENCY  │  plan_residency()
  │   PLANNER    │  memory safety clamp + reasoning chain
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  STREAMING   │  sliding-window layer scheduling
  │  SCHEDULER   │  prefetch, evict, double-buffer
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  EXECUTION   │  per-layer: SSD→Deserialize→Upload→Compute→Evict
  │  (INFERENCE) │  tok/s, TTFT, RAM peak, GB/token
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  PROFILER    │  per-layer timing: read, deser, upload, compute, evict
  │  + ANALYZER  │  GPU idle %, overlap hit rate, pipeline stall count
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  LEARNING    │  compare predicted vs best-observed
  │  LOOP        │  update calibration grid, confidence model
  └──────┬───────┘
         │
         └──────────────────────────────────────┐
                                                │
         ┌──────────────────────────────────────┘
         │
         ▼
  ┌──────────────┐
  │  FEEDBACK    │  cross-machine validation dataset
  │  DATASET     │  M1-M5, 8-64GB, different SSDs, models, quants
  └──────────────┘
```

## Runtime Overhead Decomposition

```
End-to-end tok/s
    │
    ├── Pipeline (simulator models this)
    │   ├── SSD Read
    │   ├── Deserialize
    │   ├── Upload (host→device)
    │   ├── Block Compute
    │   └── Evict
    │
    ├── Runtime Overhead (simulator does NOT model this)
    │   ├── Tokenizer (BPE encoding)
    │   ├── Attention kernels
    │   ├── KV cache management
    │   ├── Python runtime + GIL
    │   ├── Metal/CUDA kernel launch
    │   ├── Synchronization barriers
    │   ├── Memory allocator (Metal/UVM)
    │   └── OS scheduling
    │
    └── Gap ≈ 5-15× on Apple Silicon
        (measured: 2.49 sim vs 0.37 real for Mistral-7B)
```

## Pipeline Ratio

```
Pipeline Ratio = (SSD Read + Deserialize + Upload) / Block Compute

  Ratio > 1  →  I/O-bound  →  streaming scheduler helps
  Ratio < 1  →  compute-bound  →  streaming has limited benefit
  Ratio ≈ 1  →  balanced  →  overlap is critical

Example (Mistral-7B FP16 on M5):
  SSD Read:     ~90 ms
  Deserialize:  ~42 ms
  Upload:       ~11 ms
  Compute:      ~40 ms
  ─────────────────────
  Pipeline Ratio = (90 + 42 + 11) / 40 = 3.58
  → I/O-bound, streaming helps significantly
```
