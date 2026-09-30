# Configuration

Every runtime setting lives in one `AppConfig` (`src/swlp/config.py`). This page lists each
setting, its TOML key, its `SWLP_*` environment variable, and its default.

## Backends

You normally never choose a backend: name a model and SWLP picks one. `swlp chat MODEL -d`
shows the choice, which other backends can run that model, and the chosen backend's settings.

| Backend | Runner | Use it for |
|---|---|---|
| `swlp` | `SWLPRunner` | FP16/BF16 layer streaming from a shard directory. Lossless; works for models larger than RAM. |
| `speculative` | `SpeculativeRunner` | `swlp` plus speculative decoding: n-gram prompt lookup, a resident draft model (`SWLP_DRAFT_MODEL`), or the checkpoint's MTP head (used automatically when present). Output is identical to greedy `swlp`. |
| `mlx` | `MlxRunner` | Native MLX compute for models that fit in RAM: `-q bf16` (lossless), `int8`, or `int4` (lossy). Needs `swlp[apple]`. |
| `mlx-moe` | `MlxMoeRunner` | Mixture-of-Experts on MLX: dense weights resident, experts streamed through an LFU cache. Runs pulled MoE shards (`-q int4|int8` quantizes on load) or MLX-format checkpoints. Needs `swlp[apple]`. |
| `hf` | `HuggingFaceRunner` | Full-model load with transformers. Only for models that fit in RAM. |
| `mock` | `MockRunner` | Deterministic offline output with no model. Use it for tests and smoke checks. |

The automatic choice (`src/swlp/cli_resolve.py`), first match wins:

1. `--backend NAME` is given: that backend.
2. The model is a pulled shard directory (`shards/<name>` or a path): `mlx-moe` if it has expert
   banks, `speculative` if it has an MTP head, otherwise `swlp`.
3. The model is a local or Hub MLX-format checkpoint: `mlx-moe` for MoE models, otherwise `mlx`.
4. `-q` is given: `mlx` at that precision (refused up front if it cannot fit the GPU memory).
5. Anything else must be pulled first: `swlp pull MODEL`.

## Precedence

Each layer overrides the one before it:

```
built-in defaults  →  TOML file  →  SWLP_* env vars  →  CLI flags
```

- **TOML file.** SWLP reads the `--config` path if you pass one, then `$SWLP_CONFIG`, then
  `configs/default.toml` relative to the working directory. A missing file is not an error: the
  built-in defaults apply.
- **Environment.** A field's variable is `SWLP_` plus the field name in upper case, with any
  leading `swlp_` removed. For example, `swlp_window_size` becomes `SWLP_WINDOW_SIZE`. The
  exceptions are marked with † in the tables below.
- **Empty values.** An empty variable counts as unset, except for string and boolean fields,
  where `""` is a real value.
- **Booleans.** `1`, `true`, `yes` and `on` (any case) mean true. Any other value means false.
- **Paths** are `~`-expanded and resolved to absolute paths.

The `configs/*.toml` profiles ship in the source repository. They are **not** included in the pip
wheel. After `pip install swlp`, pass your own file with `--config` or use environment
variables.

```bash
SWLP_WINDOW_SIZE=1 SWLP_DIRECT_IO=on swlp run mistral-7b "Hi"
swlp chat mistral-7b --config my.toml        # flags still override the file
```

CLI flags cover the common settings: `-q/--quant`, `-n/--max-tokens`, `--resident`,
`--window`, `--backend` and `--config`. Everything else is a `SWLP_*` variable or a TOML key.

## TOML layout

```toml
[model]        # what to load
[cache]        # Hugging Face download cache
[generation]   # decoding parameters
[runtime]      # backend, streaming, KV, MLX, speculative
```

### `[model]`

| Key | Env var | Default | Meaning |
|---|---|---|---|
| `model_id` | `SWLP_MODEL_ID` | `sshleifer/tiny-gpt2` | Hugging Face id or alias. For a pulled shard directory SWLP reads it from the shard manifest. |
| `local_model_path` | `SWLP_MODEL_PATH` † | unset | Load from this local directory instead of `model_id`. |
| `trust_remote_code` | `SWLP_TRUST_REMOTE_CODE` | `false` | Passed to transformers. |
| `revision` | `SWLP_REVISION` | unset | Hub revision (branch, tag, or commit). |

### `[cache]`

| Key | Env var | Default | Meaning |
|---|---|---|---|
| `cache_dir` | `SWLP_CACHE_DIR` | `.cache/hf` | Hugging Face download cache. |
| `offline` | `SWLP_CACHE_OFFLINE` † | `false` | Never contact the Hub. |

### `[generation]`

| Key | Env var | Default | Meaning |
|---|---|---|---|
| `max_new_tokens` | `SWLP_MAX_NEW_TOKENS` | `32` | Library default. The CLI instead runs until the model finishes its answer unless `-n/--max-tokens` caps it. |
| `temperature` | `SWLP_TEMPERATURE` | `0.0` | Sampling temperature. |
| `top_p` | `SWLP_TOP_P` | `1.0` | Nucleus sampling. |
| `do_sample` | `SWLP_DO_SAMPLE` | `false` | `false` means greedy decoding. |
| `repetition_penalty` | `SWLP_REPETITION_PENALTY` | `1.0` | 1.0 disables the penalty. |
| `seed` | `SWLP_SEED` | `42` | RNG seed. |
| `prompt` | `SWLP_PROMPT` | a welcome-message prompt | Default prompt for library use (`runner.run()` with no prompt). |

### `[runtime]`: general

| Key | Env var | Default | Meaning |
|---|---|---|---|
| `backend` | `SWLP_BACKEND` | `hf` | See [Backends](#backends). |
| `device` | `SWLP_DEVICE` | `auto` | `auto`, `mps` or `cpu`. |
| `dtype` | `SWLP_DTYPE` | `auto` | Compute dtype. With `auto`, the streaming path follows the manifest's `weight_dtype`. |
| `allow_mock_fallback` | `SWLP_ALLOW_MOCK_FALLBACK` | `true` | If the `hf` runner fails, return mock output instead of raising. |
| `log_level` | `SWLP_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO` or `WARNING`. The repo's `configs/default.toml` sets `WARNING`. |
| `json_logs` | `SWLP_JSON_LOGS` | `true` | Structured JSON logs. `configs/default.toml` sets `false`. |
| `profile` | `SWLP_PROFILE` | `false` | Collect per-layer timings (`--profile`). |
| `swlp_strict` | `SWLP_STRICT` | `false` | Re-raise hot-path failures instead of degrading. Turn this on for clean benchmarks. |
| `swlp_fallback_to_baseline` | `SWLP_FALLBACK_BASELINE` † | `true` | If a streaming run fails, retry with the `hf` runner. |

### `[runtime]`: layer streaming (`swlp`, `speculative`)

| Key | Env var | Default | Meaning |
|---|---|---|---|
| `shard_dir` | `SWLP_SHARD_DIR` | unset | Shard directory. The CLI sets it for pulled models; selects streaming. |
| `swlp_window_size` | `SWLP_WINDOW_SIZE` | `2` | Layers materialized at once (`--window`). |
| `swlp_prefetch_depth` | `SWLP_PREFETCH_DEPTH` | `2` | Prefetch lookahead. The effective lookahead is `max(window, depth)`. |
| `swlp_prefetch` | `SWLP_PREFETCH` | `true` | Background prefetch on or off. |
| `swlp_residency` | `SWLP_RESIDENCY` | `auto` | The first N layers kept in RAM (`--resident`): `auto` (planner decides), `off`, or an integer. The CLI refuses a count that free RAM cannot hold. |
| `swlp_direct_io` | `SWLP_DIRECT_IO` | `auto` | `on` bypasses the page cache (`F_NOCACHE`). `off` uses it. `auto` bypasses only when the model is larger than 60% of available RAM. |
| `swlp_shard_volumes` | `SWLP_SHARD_VOLUMES` | `""` | Comma-separated extra shard directories. Layers are striped round-robin across them. |
| `swlp_prefill_chunk` | `SWLP_PREFILL_CHUNK` | `0` | Prefill the prompt in chunks of N tokens. 0 means one sweep. Lossless. |
| `swlp_activation_cache` | `SWLP_ACTIVATION_CACHE` | `true` | Reuse the hidden states of a shared prompt prefix (`--no-activation-cache` turns it off). |
| `swlp_activation_cache_max_entries` | `SWLP_ACTIVATION_CACHE_MAX` † | `16` | Maximum entries in that cache. |
| `swlp_prealloc_buffer` | `SWLP_PREALLOC_BUFFER` | `true` | Use a pre-allocated token buffer (`--no-prealloc-buffer` turns it off). |

### `[runtime]`: speculative decoding

| Key | Env var | Default | Meaning |
|---|---|---|---|
| `swlp_spec_ngram` | `SWLP_SPEC_NGRAM` | `3` | N-gram size for prompt-lookup drafting. |
| `swlp_spec_max_draft` | `SWLP_SPEC_MAX_DRAFT` | `16` | Maximum draft tokens per sweep. Draft length adapts below this. |
| `swlp_draft_model` | `SWLP_DRAFT_MODEL` | `""` | Resident draft model (alias or HF id) for the streaming path. It must use the target's tokenizer. |
| `swlp_mtp` | `SWLP_MTP` | `false` | Draft with `<shard_dir>/mtp.safetensors`. Cannot be combined with a draft model. |

### `[runtime]`: KV cache (torch backends)

| Key | Env var | Default | Meaning |
|---|---|---|---|
| `kv_memory_budget_mb` | `SWLP_KV_BUDGET_MB` † | `512` | KV RAM budget (`--kv-budget-mb`). |
| `kv_compression` | `SWLP_KV_COMPRESSION` | `false` | Lossless zlib compression of cold-layer KV. |
| `kv_compression_level` | `SWLP_KV_COMPRESSION_LEVEL` | `0` | zlib level. |
| `kv_tiering` | `SWLP_KV_TIERING` | `false` | Offload KV to host RAM. |
| `kv_disk_dir` | `SWLP_KV_DISK_DIR` | unset | Spill directory for KV overflow. Unset means no disk spill. |
| `kv_window` | `SWLP_KV_WINDOW` | `0` | Keep only the last N token positions. 0 means unbounded. Lossy when it truncates. |
| `kv_quant` | `SWLP_KV_QUANT` | `none` | `none` (lossless) or `int4` (lossy, about 4× smaller). |

### `[runtime]`: MLX (`mlx`, `mlx-moe`)

| Key | Env var | Default | Meaning |
|---|---|---|---|
| `mlx_quant` | `SWLP_MLX_QUANT` | `int8` | `bf16`, `int8` or `int4` (`-q`). |
| `mlx_wired_limit` | `SWLP_MLX_WIRED_LIMIT` | `auto` | Metal wired-memory ceiling: `auto`, `off`, or a size in MB. Clamped to Metal's recommended working set. |
| `mlx_kv_bits` | `SWLP_MLX_KV_BITS` | `0` | KV quantization: `0` (off), `4` or `8`. |
| `mlx_kv_group_size` | `SWLP_MLX_KV_GROUP_SIZE` | `64` | KV quantization group size. |
| `mlx_quantized_kv_start` | `SWLP_MLX_QUANTIZED_KV_START` | `512` | Keep the first N tokens of KV exact. |
| `mlx_num_draft_tokens` | `SWLP_MLX_NUM_DRAFT_TOKENS` | `4` | Draft tokens per speculative step (`--draft-tokens`). |
| `mlx_prefill_step_size` | `SWLP_MLX_PREFILL_STEP` † | `2048` | Prefill chunk size. |
| `mlx_prompt_cache` | `SWLP_MLX_PROMPT_CACHE` | `true` | Reuse prefix KV across chat turns. |
| `mlx_draft_model` | `SWLP_MLX_DRAFT_MODEL` | `""` | MLX draft model (same tokenizer as the target). |

### `[runtime]`: MoE expert streaming

| Key | Env var | Default | Meaning |
|---|---|---|---|
| `swlp_expert_cache_mb` | `SWLP_EXPERT_CACHE_MB` | `0` | Expert-cache budget. With `0`, torch uses a quarter of available RAM (capped at 4 GB) and `mlx-moe` uses free RAM clamped to the Metal working set. |
| `swlp_expert_prefetch` | `SWLP_EXPERT_PREFETCH` | `lru` | `lru`, `predictive` or `off`. Predictive measured slower on M5. |
| `swlp_moe_quant` | `SWLP_MOE_QUANT` | `none` | MoE shards only: `int8` or `int4` quantizes on load (dense once, experts as they are cached; lossy). `-q` sets it. |

† The variable name doesn't follow the `SWLP_<FIELD>` rule.

## Other environment variables

These are read directly and are not part of `AppConfig`.

| Variable | Default | Meaning |
|---|---|---|
| `SWLP_CONFIG` | `configs/default.toml` | TOML file used when you don't pass `--config`. |
| `SWLP_SSD_BW_GBPS` | unset | Overrides the SSD read bandwidth used by the planner and `swlp doctor`. |
| `SWLP_HW_CACHE` | `~/.cache/swlp/hardware.json` | Where the measured SSD bandwidth is stored (written by `scripts/phase0_hardware_check.py`). |
| `SWLP_RETRIES` | `2` | Attempts for Hub model loads. |
| `SWLP_TUNING_FILE` | `swlp_tuning.json` | Optional per-device tuning profile the streaming runner reads at startup. |
| `NO_COLOR` | unset | Turns off terminal colour. |
