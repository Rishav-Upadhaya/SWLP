# CLAUDE.md

Operating manual for Claude Code working in this repository. Read it before
writing any code. These instructions override default behaviour — follow them
exactly.

The phase-by-phase project history (goals, task lists, measured numbers for
Phases 0–18) lives in **`docs/ROADMAP.md`**, not here.

---

## Project Overview

**SWLP — Sliding Window Layer Pipeline.** A research library proving that large
LLMs run on consumer hardware **without quantization** by streaming model
weights through a sliding window. A transformer computes one layer at a time, so
only a window of W layers ever needs to be resident in memory.

**Apple Silicon only.** The CUDA/NVIDIA path was removed (2026-09-20) — SWLP
targets Apple Silicon exclusively. On an M-series Mac layers stream NVMe SSD →
unified RAM via background threads; the SSD (~6.9 GB/s) is the bottleneck, and
decode is memory-bandwidth-bound (M5: 153 GB/s unified). There is no PCIe bus,
no separate VRAM, and no host/device bandwidth split — several optimisations
that make sense on discrete GPUs are actively wrong here, which is why they
were removed rather than left dormant.

**Not quantization, not pruning, not distillation. Zero quality compromise is a
first-class constraint.** The one deliberate exception is explicitly opt-in,
off-by-default, labelled lossy tiers (MLX int4, INT4 KV).

| End-goal target | Model | Hardware | Goal |
|---|---|---|---|
| Apple Silicon | 30B FP16 | M5 16 GB unified | Feasibility (impossible elsewhere) + interactive *aggregate* throughput via batching/speculation |

Deliverable: a `pip install swlp` library + a paper showing SWLP beats AirLLM on
speed at equal quality.

---

## Current Status

Phases 0–27 are implemented; the test suite is **489 tests, all passing**.
The architecture has settled (streaming runners + the MLX interactive backend).
Phase 20 (2026-06-11) eliminated the per-token copy overhead in the streaming
hot path (`core/shard_io.py` + rewritten `core/streaming.py`): Mistral-7B
direct-I/O streaming went 0.218 → 0.372 tok/s (75% of the cold-SSD ceiling),
byte-identical output. `SWLP_DIRECT_IO=auto` now gives fitting models
page-cache residency for free. Phase 21 (2026-06-11) added draft-model
speculative decoding (`runner/draft.py`, `--draft-model`): a resident
Qwen2.5-0.5B drafts for the streamed Qwen2.5-14B with adaptive (AIMD) draft
length — measured 0.19 → **0.55–1.16 tok/s** (2.9×–5.9×, acceptance-bound),
byte-identical output. Phase 22 (2026-06-11) added the lossless `.swz` shard
codec (`codec.py`, `swlp compress-shards` / `--revert`): bit-exact ~31% disk
shrink, CRC-gated against zipnn segfaults — a **negative result for tok/s on
fast SSDs** (decompression contends for unified-memory bandwidth; crossover
≈ 3.5 GB/s SSD read, M5 loses ~25%), so plain shards stay the default on the
dev machine.

Phases 24–27 (2026-08-22) landed the research-driven upgrade: the **MoE
expert-streaming engine** (shard format v2 expert banks + decomposed MoE
forward, bit-exact vs HF; `ExpertScheduler` with global LRU expert cache,
predictive routing prefetch, elastic budgets),
**prefix-KV caching** with turn anchors (lossless re-prefill skipping in
chat/serve), **multi-volume SSD striping**, **chunked prefill**,
multi-resolution n-gram drafting, measured-SSD-bandwidth probing, `swlp pull`,
`/v1/models`, doctor MoE advisory, and the `moe_sweep.py` harness. Two latent
GPT-2 bugs were found by the new HF-reference logits tests and fixed: `ln_f`
was never persisted/loaded (ran on uninitialized memory), and the adapter
still used the legacy tuple-KV protocol (removed in transformers ≥5) (per-token KV was
silently dropped). Design attribution: FreeToken (arXiv:2608.16157),
Mixtral-offloading (arXiv:2312.17238), MoE-SpeQ, SpecMD, EAGLE-3.

Remaining work is **execution, not design** — see the "Open work" section of
`docs/ROADMAP.md`: the paper-grade re-measurement (≥5 runs, median ± IQR,
cold+warm, now including the draft-spec prompt spectrum), the Phase 28 MoE
hardware measurements (Qwen3-30B-A3B first, DeepSeek-V4-Flash feasibility),
and the deferred batched / 30B / long-context / perplexity measurements.
(The MX230 NVIDIA column is dropped: the project is Apple-only.)

**When a phase completes:** update its checklist and Phase notes in
`docs/ROADMAP.md`, then update this Current Status section. New phases are
appended to `docs/ROADMAP.md`.

---

## Working Rules

**Process:**
1. **Plan before you act.** State what you intend to change and why. Do not
   write code until the plan is agreed. Never take unrequested actions.
2. **One task at a time.** Finish a task, run the relevant test/command, then
   move on.
3. **If blocked or uncertain: STOP and ask.** Do not guess and proceed.
4. **Consult `docs/ROADMAP.md`** for the current phase's scope. Do not jump
   ahead of the planned work.

**Before writing code:**
5. **Never write code anonymously** — before creating a function or class, run
   `grep -r "function_name" src/` to confirm it does not already exist.
6. **Never duplicate logic** — if it exists, extend it; if it belongs to another
   module, import it. If logic appears twice, extract it to a shared module.
7. **Never create abstractions speculatively.** Build only what the current task
   needs.

**Must never do:**
8. **Never violate the folder structure** — new runners go in `runner/`, new
   reporters in `reporting/`, algorithm changes in `core/`. No files outside the
   established layout.
9. **Never hardcode** model paths, device strings, or magic numbers — use
   `AppConfig`, TOML configs, and `SWLP_*` env vars.
10. **Never use `print()`** for runtime output — use `configure_logging()` from
    `logging.py`. (Direct user-facing CLI/chat output is the only exception.)
11. **Never touch `metrics.py` structure** without explicit instruction —
    `RunMetrics` and `RunResult` are the canonical output contract.
12. **Never install new packages** without listing alternatives and getting
    explicit confirmation.
13. **Never delete phase history.** Phase notes and measured numbers in
    `docs/ROADMAP.md` are append-only.
14. **Never mark a task `[x]`** in `docs/ROADMAP.md` without first running
    `pytest` (all pass) and `ruff check src/` (zero errors).
15. **Remove dead code** — if a file, class, or function is unused, delete it.
16. **Wire it or delete it** — a config field that reaches `AppConfig` and no
    consumer is worse than no field. `tests/test_architecture.py` enforces
    this, plus the import table below. Guard lazily-built attributes with
    `is None`, never `not hasattr` — `__init__` declares them, so a hasattr
    guard is always False and the feature silently never initializes.
17. **Degrade loudly** — hot-path `except` blocks call `self.degrade(reason, exc)`,
    never a bare `LOGGER.exception`. It records to `RunMetrics.degradations`
    and re-raises under `SWLP_STRICT=1`.

---

## Commands

The CLI is flag-based: `swlp` with no subcommand runs inference. The backend is
auto-selected (`--quant` → mlx, `--shard-dir` → swlp, else hf) or set explicitly
with `--backend` (`mock` | `hf` | `swlp` | `speculative` | `mlx`). A `--config`
TOML is optional.

```bash
# Inference
swlp --backend mock --prompt "Hey"                               # offline, no model
swlp --model mistral-7b --prompt "Explain transformers."         # HF inference
swlp --model mistral-7b --backend mlx --quant int8 --prompt "Hi" # native MLX (Apple Silicon)
swlp --shard-dir ./shards/mistral-7b --window 2 --prompt "Hi"    # SWLP layer streaming
swlp --config configs/swlp_speculative_mps.toml --prompt "Hi"    # speculative decoding
swlp                                                             # no args → prints help

# Interactive chat (history kept across turns; streams tokens)
swlp chat --model mistral-7b --backend mlx --quant int8
swlp chat --shard-dir ./shards/mistral-7b --window 2

# Download + shard a model to disk (one-time; required for the swlp backend)
swlp download --model mistral-7b                                 # → ./shards/<name>

# Benchmark / report / suite
swlp benchmark --prompt-set short --runs 5 --warmup-runs 1 --report --config configs/baseline.toml
swlp report benchmarks/baseline-<timestamp>.json
swlp suite --suite configs/bench_suite.toml --report
swlp suite-report benchmarks/suite-<timestamp>.json

# Simulation (pure math, no model needed)
swlp simulate --scenario configs/sim_baseline.toml --report

# Model packaging / inspection
swlp package /path/to/checkpoint /path/to/output --model-name demo-model
swlp validate-package /path/to/output
swlp layer /path/to/output model.layers.0

# Hardware check + per-model recommendation (run first on every new machine)
swlp doctor                     # chip / RAM / MLX + best command per model
swlp models                     # list aliases with sizes and HF ids
python scripts/phase0_hardware_check.py   # measured SSD bandwidth baseline

# Test and lint — both must be clean before any task is marked done
pytest                          # all tests
pytest tests/test_simulator.py  # single file
pytest -k test_config           # single test by name
ruff check src/                 # lint
ruff check --fix src/           # lint + autofix
```

Model aliases: `mistral-7b`, `qwen-14b`, `tiny-gpt2`, `qwen3-30b-a3b`,
`mixtral-8x7b`, `deepseek-v4-flash` — any HuggingFace id also works. Default output is a friendly summary; `--json` prints full metrics.

**Environment variables** (override config without editing TOML):

```
SWLP_MODEL_ID        model to load                SWLP_DEVICE        mps | cpu
SWLP_BACKEND         hf | swlp | mock | ...        SWLP_WINDOW_SIZE   sliding window depth
SWLP_PREFETCH_DEPTH  layers ahead to prefetch      SWLP_SHARD_DIR     path to pre-sharded layers
SWLP_KV_BUDGET_MB    KV cache RAM budget           SWLP_KV_COMPRESSION enable zlib KV compression
SWLP_KV_TIERING      host offload for KV           SWLP_KV_DISK_DIR   disk-spill dir for cold KV
SWLP_KV_QUANT        none (default) | int4 (lossy) SWLP_KV_WINDOW     keep last N KV positions (0=∞)
SWLP_RESIDENCY       auto | off | <integer>        SWLP_MLX_QUANT     bf16 | int8 | int4
SWLP_DIRECT_IO       auto | on | off (bypass page cache only when model > ~60% of free RAM)
SWLP_SPEC_NGRAM      n-gram match size             SWLP_SPEC_MAX_DRAFT max draft tokens per sweep
SWLP_SHARD_VOLUMES   extra shard dirs (striping)   SWLP_PREFILL_CHUNK tokens per prefill slice
SWLP_EXPERT_CACHE_MB MoE expert-cache budget       SWLP_EXPERT_PREFETCH off|lru|predictive
SWLP_DRAFT_MODEL     resident draft model for speculative decoding (must share target tokenizer)
SWLP_STRICT          1 = re-raise hot-path failures instead of degrading (clean benchmarks)
SWLP_MLX_WIRED_LIMIT auto | off | <MB>          SWLP_MLX_KV_BITS     0 | 4 | 8
SWLP_MLX_KV_GROUP_SIZE  KV quant group (64)     SWLP_MLX_QUANTIZED_KV_START exact-prefix tokens
SWLP_MLX_NUM_DRAFT_TOKENS  drafts/step (4)      SWLP_MLX_PREFILL_STEP  prefill chunk (2048)
SWLP_MLX_PROMPT_CACHE  reuse prefix KV in chat
SWLP_LOG_LEVEL       DEBUG | INFO | WARNING        SWLP_PROFILE       1 to collect detailed timings
```

---

## Code Style & Conventions

- `snake_case` for variables, functions, files, modules. `PascalCase` for classes.
- **One concern per module, one responsibility per class.** Follow the
  sub-package layout exactly.
- **SOLID:**
  - *Single Responsibility* — `KVCacheManager` manages KV memory only;
    `HardwareInfo` is a read-only data container only.
  - *Open/Closed* — extend via new classes (add `XyzRunner` in `runner/`,
    register in `build_runner()`); never modify existing runners.
  - *Liskov* — all runners are interchangeable via `build_runner()` and return
    `RunResult` (`MockRunner`, `HuggingFaceRunner`, `SWLPRunner`,
    `SpeculativeRunner`, `MlxRunner`).
  - *Interface Segregation* — `runner/__init__.py` exports only `build_runner`;
    internals stay internal.
  - *Dependency Inversion* — depend on abstractions; callers see one interface
    regardless of which scheduler/runner is chosen.
- All functions have full type annotations (arguments + return). No `Any` types,
  no module-level globals, no magic values — use `AppConfig`, dataclasses, named
  constants.
- **No new file exceeds 300 lines** — split by responsibility. Known legacy
  exceptions (do not let others follow): `runner/swlp.py`, `model/package.py`.
- Configs are TOML in `configs/` — never hardcode runtime values in source.

---

## Folder Structure

```
swlp/                            ← project root
├── src/swlp/                    ← installable package
│   ├── __init__.py              public API: HardwareInfo, ShardManifest, build_runner, RunMetrics, RunResult
│   ├── __main__.py              enables `python -m swlp`
│   ├── cli.py                   CLI dispatch + run path
│   ├── cli_args.py              argparse parser construction (keeps cli.py under budget)
│   ├── cli_help.py              CLI help text
│   ├── cli_doctor.py            `swlp doctor` / `swlp models` — hardware probe + MoE advisory + aliases
│   ├── chat.py                  interactive chat REPL (run_chat); history across turns
│   ├── tui.py                   terminal presentation: colour, boxes, spinner, stream wrap
│   ├── serve.py                 `swlp serve` — stdlib OpenAI-compatible HTTP server (+SSE)
│   ├── codec.py                 lossless `.swz` shard codec (zipnn byte-grouping); recommend_compression()
│   ├── config.py                AppConfig + load_config(); env-var overlay — imported everywhere
│   ├── logging.py               configure_logging() — always use this, never print()
│   ├── metrics.py               RunMetrics + RunResult — canonical output contract; never moved
│   │
│   ├── core/                    SWLP algorithm engine — pure computation, no I/O
│   │   ├── scheduler.py         ThreadedScheduler — CPU RAM <-> MPS block swapping
│   │   ├── streaming.py         StreamingScheduler — materializes per-layer shards; CPU-RAM residency
│   │   ├── shard_io.py          zero-copy shard reads (mmap / readinto) + safetensors views (Phase 20)
│   │   ├── residency.py         plan_residency() + ResidencyPlan; calibration + explainable decision
│   │   ├── resident_policy.py   (pipeline_ratio, free_RAM) → target resident count, bilinear anchors
│   │   ├── pipeline_model.py    pipeline-ratio estimation from hardware / from measured metrics
│   │   ├── prefix_cache.py      PrefixKVCache — exact-match prefix KV reuse across chat turns
│   │   ├── phase23.py           ActivationCache, PreallocBuffer (on) + EarlyExit, LayerPruner (opt-in lossy)
│   │   ├── speculative.py       NgramDrafter (prompt-lookup, multi-resolution) + verify_greedy()
│   │   ├── moe_policy.py        q_star_split() (FreeToken) + RoutingHistory — pure MoE policy math
│   │   ├── kv_cache.py          KVCacheManager — device/host/compressed/disk tiers
│   │   ├── compressed_cache.py  CompressedDynamicCache/Layer — transformers Cache over KVCacheManager
│   │   ├── kv_quant.py          INT4 KV quantize/dequantize (opt-in lossy tier)
│   │   ├── profiler.py          LayerProfiler — per-stage absolute timestamps; PipelineMetrics
│   │   ├── simulator.py         discrete-event pipeline simulator with resource contention
│   │   ├── sweep.py             run_sweep() — parameter sweeps over the simulator
│   │   ├── evaluator.py         evaluate_policies() — side-by-side scheduling-policy comparison
│   │   ├── analyzer.py          analyze_traces() — observation→diagnosis→recommendation report
│   │   └── confidence.py        estimate_policy_confidence() — decision confidence + reasoning
│   │
│   ├── hardware/                Hardware detection — read-only, no side effects
│   │   └── detect.py            detect_hardware() → HardwareInfo; measured-bandwidth cache; fits_in_memory()
│   │
│   ├── model/                   Model file I/O — disk read/write only, no compute
│   │   ├── package.py           SWLP package format: package_checkpoint(), validate_package(), load_layer()
│   │   ├── shard.py             shard_model_by_layer(); compress/decompress_shards(); verify_shards()
│   │   ├── expert_bank.py       shard-format-v2 MoE expert banks — ranged reads of single experts
│   │   ├── quant.py             FP8 weight quantize/dequantize of layer shards (Phase 7, negative)
│   │   └── sparse.py            COO sparse weight encode/decode
│   │
│   ├── runner/                  Inference runners — all interchangeable via build_runner()
│   │   ├── base.py              build_runner() factory + execute_baseline() + check_hf_oom()
│   │   ├── mock.py              MockRunner — deterministic offline responses; use in unit tests
│   │   ├── hf.py                HuggingFaceRunner — standard full-model HF inference
│   │   ├── swlp.py              SWLPRunner — sliding-window streaming inference (hot path)
│   │   ├── swlp_setup.py        SWLPSetupMixin — residency/direct-IO resolution + RunMetrics
│   │   ├── speculative.py       SpeculativeRunner(SWLPRunner) — verify K drafts in one disk sweep
│   │   ├── draft.py             DraftModelDrafter — resident small model proposer (Phase 21)
│   │   ├── mlx.py               MlxRunner — native MLX quantized compute (Apple Silicon)
│   │   ├── mlx_tune.py          wired-memory ceiling, KV-quant + spec-decode kwargs
│   │   ├── batch.py             run_batch() — batched ("column-wise") streaming
│   │   ├── experts.py           SwlpCachedExperts — slot-cached fused-Experts replacement (MoE)
│   │   ├── expert_scheduler.py  ExpertScheduler — global LRU expert cache + predictive prefetch
│   │   ├── arch.py              ArchAdapter — GPT2 vs Llama/Mistral dispatch
│   │   └── load.py              load_full_model() + load_from_shards()
│   │
│   ├── benchmark/               Benchmarking & simulation — no side effects on runners
│   │   ├── run.py               run_benchmark() — N timed runs, mean/median/std
│   │   ├── suite.py             run_suite() — sweep across prompt sets and window configs
│   │   ├── simulator.py         simulate_scenario() — pure-Python bottleneck math, no model
│   │   └── event_simulator.py   EventSimulator + scheduling strategies — full layer-lifecycle model
│   │
│   └── reporting/               Output formatting — read-only; terminal/JSON/CSV
│       ├── run_report.py        print_report() — benchmark result table
│       ├── sim_report.py        print_simulation_report()
│       ├── suite_report.py      print_suite_report() — baseline vs SWLP table
│       └── policy_report.py     print_policy_report() — scheduling-policy comparison table
│
├── tests/                       pytest suite — mirrors src/swlp/ layout (see Testing)
├── configs/                     TOML config profiles — never hardcode values in source
├── scripts/                     user-facing utilities: bootstrap.sh, phase0_hardware_check.py,
│   │                            generate_figures.py — not part of the package
│   └── research/                phase one-offs & paper benchmarks (bench_common.py, phase3_baselines.py,
│                                compare_airllm_swlp.py, moe_sweep.py, …) — root-finding uses parents[2]
├── docs/                        documentation, incl. ROADMAP.md (phase history) and results.md
├── experiments/ research/       raw experiment data and the paper source
└── pyproject.toml               package metadata, deps, ruff (line-length=100, py311), pytest config
```

**Import rules — which sub-packages may import from which:**

| Module | May import from | Must NOT import from |
|---|---|---|
| `config`, `metrics`, `logging` | stdlib only | anything in `src/swlp/` |
| `core/` | `config`, `metrics`, `logging` | `runner/`, `benchmark/`, `reporting/` |
| `hardware/` | `config`, `metrics`, `logging` | `core/`, `runner/`, `benchmark/`, `reporting/` |
| `model/` | `config`, `metrics`, `logging` | `core/`, `runner/`, `benchmark/`, `reporting/` |
| `runner/` | `config`, `metrics`, `logging`, `core/`, `hardware/`, `model/` | `benchmark/`, `reporting/` |
| `benchmark/` | `config`, `metrics`, `logging`, `runner/` | `reporting/` |
| `reporting/` | `config`, `metrics`, `logging`, `benchmark/` | `runner/` |
| `cli.py`, `cli_args.py`, `chat.py` | any sub-package | — |

---

## Architecture

**Key invariants:**
- `SWLPRunner` keeps **embedding, lm_head, and layer-norm always on device** —
  only transformer blocks stream in/out.
- `SWLPRunner._build_scheduler()` auto-picks: `StreamingScheduler` when shards
  exist, else `ThreadedScheduler` (Python threads) on MPS/CPU.
- `load()` is **idempotent** on `HuggingFaceRunner`/`SWLPRunner` — it returns
  early if the model is already in memory, so `run()` / `stream_tokens()` /
  `run_chat()` never double-load (a double full-model load OOMs a 16 GB machine).
- Config flow: TOML → `AppConfig.to_dict()` → `SWLP_*` env-var overlay →
  `AppConfig`. No magic, no globals.
- `RunResult` lives in `metrics.py` (not `runner/`) to prevent circular imports.

**Config profiles** (`configs/`) — pick the one matching the target, never
hardcode values:

| File | Device / Backend | When to use |
|---|---|---|
| `baseline.toml` | auto / hf | Standard HF baseline, any hardware |
| `swlp_mps.toml` / `swlp_mistral_mps.toml` | mps / swlp | M5 streaming (generic / Mistral-7B) |
| `swlp_qwen_mps.toml` / `swlp_qwen32b_mps.toml` | mps / swlp | M5 streaming Qwen2.5-14B / -32B |
| `swlp_speculative_mps.toml` | mps / speculative | M5 prompt-lookup speculative decoding |
| `swlp_mlx_mps.toml` | mps / mlx | M5 native MLX quantized backend |
| `swlp_mistral_fp8_mps.toml` / `swlp_qwen_fp8_mps.toml` | mps / swlp | FP8 shard tier (Phase 7, negative result) |
| `sim_baseline.toml` / `sim_m5.toml` / `sim_large.toml` | — | Simulator scenarios |
| `bench_suite.toml` / `suite_phase3.toml` | — | Multi-prompt suite runs |

---

## Tech Stack & Versions

| Layer | Package | Pinned (pyproject.toml) | Installed |
|---|---|---|---|
| Language | Python | `>=3.11` | 3.13.13 |
| ML framework | torch | `>=2.2` | 2.12.0 |
| Model loading | transformers | `>=4.41` | 5.8.1 |
| Device dispatch | accelerate | `>=0.33` | 1.13.0 |
| Model hub | huggingface_hub | `>=0.24` | 1.15.0 |
| Weight serialization | safetensors | `>=0.4` | 0.7.0 |
| Structured logging | python-json-logger | `>=2.0` | 4.1.0 |
| Process/memory stats | psutil | `>=5.9` | 7.2.2 |
| CPU profiler | pyinstrument | `>=4.6` | 5.1.2 |
| Linter/formatter | ruff | `>=0.5` | 0.15.13 |
| Test runner | pytest | `>=8.2` | 9.0.3 |

**Optional extras:** `swlp[apple]`
(mlx + mlx-lm — required for the MLX backend) · `swlp[codec]` (zipnn — the
off-by-default `.swz` codec) · `swlp[dev]` (pytest, pytest-cov, ruff).

**Serialization formats:** `.toml` runtime configs · `.safetensors` per-layer
shards (Phase 17; `embed.pt` / `lm_head.pt` stay `.pt`) and model packages ·
`.pt` legacy shards (still auto-detected) · `.json` benchmark outputs / manifests
/ structured logs · `.csv` optional benchmark export.

---

## Testing

Use the lightest tier that genuinely verifies the change.

**Unit tests (pytest)** — every new function in `src/swlp/` needs a test in
`tests/`. Use `MockRunner` for runner tests (never call real models). Use
`simulate_scenario()` for pipeline math (no hardware needed).

**Integration tests (real execution):**
- CLI change → `swlp --backend mock --prompt "hi"`
- config-field change → validate via `load_config()` or `swlp validate-package`
- hardware-detection change → `python scripts/phase0_hardware_check.py`

**End-to-end (real model, real hardware):** sharding change → run
`shard_model_by_layer(...)` and confirm `shard_manifest.json`; real-model change
→ run on M5 and record TTFT / tok/s. Never mock what can be tested for real.

**Before marking any task done:** (1) `pytest` all pass, (2) `ruff check src/`
zero errors, (3) run the relevant CLI command and confirm output, (4) only then
tick the checklist in `docs/ROADMAP.md`.

Tests live in `tests/` and mirror the `src/swlp/` layout — each `test_*.py` maps
to one module (e.g. `test_kv_cache.py` → `core/kv_cache.py`, `test_mlx.py` →
`runner/mlx.py`). Current suite: **489 tests**. Run a single test with
`pytest tests/test_simulator.py::test_simulate_scenario_overlap`.

---

## Environment Setup

```bash
bash scripts/bootstrap.sh        # creates .venv, installs swlp[dev]
source .venv/bin/activate
pip install swlp[apple]          # optional — Apple Silicon MLX support
```
