# CLAUDE.md

Operating manual for Claude Code working in this repository. Read it before
writing any code. These instructions override default behaviour — follow them
exactly.

The phase-by-phase project history (goals, task lists, measured numbers for
Phases 0–32) lives in **`docs/ROADMAP.md`**, not here.

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

Phases 0–27 and 29–33 are implemented (one known-flaky codec test:
`test_codec.py::test_roundtrip_odd_length`, zipnn nondeterminism — the codec's
own roundtrip gate refuses the bad write). The architecture has settled:
streaming runners (`swlp`, `speculative`) + the MLX backends (`mlx`, `mlx-moe`).
Measured numbers live in `docs/results.md`; the per-phase story in `docs/ROADMAP.md`.

2026-09-30 pre-release cleanup (v0.1.0, PyPI-ready, not published): removed
the FP8 shard tier, sparse shards, lossy EarlyExit/LayerPruner, the dead
`pin_memory`/`double_buffer` knobs and `--adaptive-precision`; moved the
scheduling simulators/analyzer (`sim`/`analyze`/`sweep`/`evaluate`/
`policy-report`) out of the package to `scripts/research/simtools/`
(`python -m scripts.research.simtools <cmd>`); docs consolidated to six pages.
Releases: bump `swlp.__version__`, update CHANGELOG, tag `vX.Y.Z` →
`.github/workflows/release.yml` (PyPI Trusted Publishing).

Remaining work is **execution, not design** — see "Open work" in
`docs/ROADMAP.md` (paper-grade re-measurement, Phase 28 MoE hardware runs,
deferred batched / 30B / long-context / perplexity measurements).

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
8. **Never violate the folder structure** — new runners go in `runner/`,
   terminal output goes through `ui.py`, algorithm changes in `core/`. No files
   outside the established layout.
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
    `pytest` (all pass) and `ruff check src tests scripts` (zero errors).
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

Eight commands; users name a model and `cli_resolve.py` picks the backend
(MoE shards/MLX MoE → `mlx-moe`; dense shards → `swlp`, or `speculative` when an
MTP head exists; `-q` → resident `mlx`; else `swlp pull` first). `--backend`
(`mlx` | `mlx-moe` | `swlp` | `speculative` | `hf` | `mock`) overrides it.
Presentation is `ui.py` (rich); tuning is `SWLP_*` env vars or `--config` TOML.

```bash
swlp chat MODEL [-q int4|int8|bf16] [-n N]   # interactive (/help /clear /think /stats /exit)
swlp chat MODEL -d                            # plan: backend + alternatives + settings (no load)
swlp run MODEL "prompt" [--json]              # one answer; "-" reads stdin
swlp serve MODEL [--host H --port P]          # OpenAI-compatible HTTP API
swlp pull MODEL [--output-dir DIR]            # download + shard (MLX repos: download only)
swlp models                                   # installed models + aliases
swlp rm MODEL [-y]                            # delete shards / HF downloads / MLX copies
swlp doctor [MODEL]                           # machine + what it can run + tuning
swlp bench MODEL [--runs N] [--json]          # tok/s, first token, peak RAM (medians)
swlp run anything "hi" --backend mock         # offline smoke test, no model

# Test and lint — both must be clean before any task is marked done
pytest                          # all tests
pytest tests/test_cli.py        # single file
ruff check src/                 # lint
```

Model aliases: `swlp models` (e.g. `gemma4-26b`, `qwen3.6-35b`, `olmoe-7b`,
`mistral-7b`, `qwen-14b`); any HuggingFace id or local directory also works.

**Environment variables:** every `RuntimeConfig` field `x` is overridable as
`SWLP_X` (see `config.py`); the user-facing reference is `docs/configuration.md`.
`SWLP_STRICT=1` re-raises hot-path failures instead of degrading (clean benchmarks).

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
│   ├── cli.py                   CLI dispatch — 8 commands (chat run serve pull models rm doctor bench)
│   ├── cli_args.py              argparse parser + model aliases
│   ├── cli_resolve.py           model → backend choice (Target)
│   ├── cli_help.py              the `swlp` / `swlp -h` screen
│   ├── cli_doctor.py            `swlp doctor` — machine, what it can run, tuning
│   ├── cli_models.py            `swlp models` / `swlp rm` — installed models, aliases, on-disk artifacts
│   ├── chat.py                  chat REPL + streamed answers (shared with `swlp run`)
│   ├── ui.py                    rich presentation: panels, tables, streamed markdown, status line
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
│   │   ├── decode_cache.py      ActivationCache + PreallocBuffer — byte-identical decode speedups
│   │   ├── speculative.py       NgramDrafter (prompt-lookup, multi-resolution) + verify_greedy()
│   │   ├── moe_policy.py        RoutingHistory — expert-routing history for predictive prefetch
│   │   ├── kv_cache.py          KVCacheManager — device/host/compressed/disk tiers
│   │   ├── compressed_cache.py  CompressedDynamicCache/Layer — transformers Cache over KVCacheManager
│   │   ├── kv_quant.py          INT4 KV quantize/dequantize (opt-in lossy tier)
│   │   ├── profiler.py          LayerProfiler — per-stage absolute timestamps; PipelineMetrics
│   │   └── confidence.py        estimate_policy_confidence() — decision confidence + reasoning
│   │
│   ├── hardware/                Hardware detection — read-only, no side effects
│   │   └── detect.py            detect_hardware() → HardwareInfo; measured-bandwidth cache; fits_in_memory()
│   │
│   ├── model/                   Model file I/O — disk read/write only, no compute
│   │   ├── package.py           SWLP package format: package_checkpoint(), validate_package(), load_layer()
│   │   ├── shard.py             shard_model_by_layer(); compress/decompress_shards(); verify_shards()
│   │   ├── expert_bank.py       shard-format-v2 MoE expert banks — ranged reads of single experts
│   │   └── mlx_expert_index.py  expert byte ranges inside MLX-format (quantized) checkpoints
│   │
│   ├── runner/                  Inference runners — all interchangeable via build_runner()
│   │   ├── base.py              build_runner() factory + execute_baseline() + check_hf_oom()
│   │   ├── mock.py              MockRunner — deterministic offline responses; use in unit tests
│   │   ├── hf.py                HuggingFaceRunner — standard full-model HF inference
│   │   ├── swlp.py              SWLPRunner — sliding-window streaming inference (hot path)
│   │   ├── swlp_setup.py        SWLPSetupMixin — residency/direct-IO resolution + RunMetrics
│   │   ├── speculative.py       SpeculativeRunner(SWLPRunner) — verify K drafts in one disk sweep
│   │   ├── draft.py             DraftModelDrafter — resident small model proposer
│   │   ├── mlx.py               MlxRunner — native MLX quantized compute (Apple Silicon)
│   │   ├── mlx_tune.py          wired-memory ceiling, KV-quant + spec-decode kwargs
│   │   ├── mlx_moe.py           MlxMoeRunner — MoE expert streaming on MLX (resident dense, cached experts)
│   │   ├── mlx_expert_cache.py  MlxExpertCache — per-expert MLX arrays, heap LFU, parallel pread
│   │   ├── mlx_switch.py        CachedSwitchGLU — mlx_lm SwitchGLU drop-in over MlxExpertCache
│   │   ├── mtp.py               MTP-head drafter — the checkpoint's own multi-token-prediction head
│   │   ├── hybrid_rollback.py   state rollback for hybrid (linear-attention) models under speculation
│   │   ├── batch.py             run_batch() — batched ("column-wise") streaming
│   │   ├── experts.py           SwlpCachedExperts — slot-cached fused-Experts replacement (MoE)
│   │   ├── expert_scheduler.py  ExpertScheduler — global LRU expert cache + predictive prefetch
│   │   ├── arch.py              ArchAdapter — GPT2 vs Llama/Mistral dispatch
│   │   └── load.py              load_full_model() + load_from_shards()
│   │
│   └── benchmark/               Benchmarking & simulation — no side effects on runners
│       ├── run.py               run_benchmark() — `swlp bench`; N timed runs, mean/median/std
│       ├── suite.py             run_suite() — prompt-set × window sweeps (tests, scripts/research)
│       └── simulator.py         simulate_scenario() — pure-Python bottleneck math, no model
│
├── tests/                       pytest suite — mirrors src/swlp/ layout (see Testing)
├── configs/                     TOML config profiles — never hardcode values in source
├── scripts/                     user-facing utilities: bootstrap.sh, phase0_hardware_check.py,
│   │                            generate_figures.py — not part of the package
│   └── research/                phase one-offs & paper benchmarks (bench_common.py, phase3_baselines.py,
│       │                        compare_airllm_swlp.py, moe_sweep.py, …) — root-finding uses parents[2]
│       └── simtools/            scheduling simulators + trace analyzer (moved out of the package)
├── docs/                        configuration, architecture, formats, benchmarking, results, ROADMAP
├── experiments/ research/       local-only (gitignored): raw experiment data, paper source, notes
└── pyproject.toml               package metadata, deps, ruff (line-length=100, py311), pytest config
```

**Import rules — which sub-packages may import from which:**

| Module | May import from | Must NOT import from |
|---|---|---|
| `config`, `metrics`, `logging` | stdlib only | anything in `src/swlp/` |
| `core/` | `config`, `metrics`, `logging` | `runner/`, `benchmark/` |
| `hardware/` | `config`, `metrics`, `logging` | `core/`, `runner/`, `benchmark/` |
| `model/` | `config`, `metrics`, `logging` | `core/`, `runner/`, `benchmark/` |
| `runner/` | `config`, `metrics`, `logging`, `core/`, `hardware/`, `model/` | `benchmark/` |
| `benchmark/` | `config`, `metrics`, `logging`, `runner/` | — |
| `cli*.py`, `chat.py`, `ui.py` | any sub-package | — |

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

**Before marking any task done:** (1) `pytest` all pass, (2) `ruff check src tests scripts`
zero errors, (3) run the relevant CLI command and confirm output, (4) only then
tick the checklist in `docs/ROADMAP.md`.

Tests live in `tests/` and mirror the `src/swlp/` layout — each `test_*.py` maps
to one module (e.g. `test_kv_cache.py` → `core/kv_cache.py`, `test_mlx.py` →
`runner/mlx.py`). Run a single test with
`pytest tests/test_simulator.py::test_simulate_scenario_overlap`.

---

## Environment Setup

```bash
bash scripts/bootstrap.sh        # creates .venv, installs swlp[dev]
source .venv/bin/activate
pip install swlp[apple]          # optional — Apple Silicon MLX support
```
