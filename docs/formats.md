# On-disk formats

SWLP uses two on-disk layouts. **Shard directories** are what the streaming backends run from.
**Layer packages** are a checksummed format for inspecting and validating checkpoints.

| Format | Written by | Read by | Marker file |
|---|---|---|---|
| Shard directory | `swlp download`, `swlp pull`, `swlp run` | `swlp`, `speculative` and `mlx-moe` backends | `shard_manifest.json` |
| Layer package | `swlp package` | `swlp validate-package`, `swlp layer` | `manifest.json` + `layer-index.json` |

## Shard directory

`model/shard.py::shard_model_by_layer` reads the checkpoint's safetensors one tensor at a time,
so sharding never loads the whole model into RAM. Weights keep their native half-precision
dtype: bf16 checkpoints stay bf16.

```bash
swlp download --model mistral-7b                  # → ./shards/mistral-7b
swlp pull --model qwen-14b --output-dir /Volumes/ssd/qwen-14b
```

Before downloading, `swlp download` checks that there is enough free disk for the model's FP16
size.

### Files

| File | Contents |
|---|---|
| `shard_manifest.json` | Model metadata (see below). |
| `layer_NNN.safetensors` | One transformer block (dense weights), zero-padded index (`layer_000`, `layer_001`, …). |
| `layer_NNN.safetensors.swz` | The same block after `swlp compress-shards`. |
| `layer_NNN.pt` | Legacy torch-pickle block. Still detected and loaded. |
| `layer_NNN.experts.safetensors` | MoE only: that layer's expert bank (see [MoE expert banks](#moe-expert-banks-shard-format-v2)). |
| `expert_index.json` | MoE only: byte ranges of every expert in every bank. |
| `embed.pt` | Token embeddings plus final norm (`wte`/`wpe` and `ln_f` for GPT-2). Loaded once. |
| `lm_head.pt` | LM head. Tied models reuse the input embedding. Loaded once. |
| `mtp.safetensors` | Optional multi-token-prediction head, written when the checkpoint has `mtp.*` tensors. Used by `--mtp`. |

The reader looks for each layer as `.safetensors`, then `.safetensors.swz`, then `.pt`. A
directory that mixes formats, such as one where compression was interrupted, still loads. At load
time `verify_shards()` checks that every file exists and has a valid header: the safetensors
header length, the `SWZ1` magic, or the ZIP magic for `.pt` files.

### `shard_manifest.json`

```json
{
  "model_id": "allenai/OLMoE-1B-7B-0125-Instruct",
  "num_layers": 16,
  "layer_weight_mb": 33.83,
  "total_weight_mb": 13426.57,
  "embed_file": "embed.pt",
  "lm_head_file": "lm_head.pt",
  "model_type": "olmoe",
  "weight_dtype": "bfloat16",
  "shard_format": "safetensors",
  "shard_compression": "none",
  "num_experts": 64,
  "top_k": 8,
  "expert_bank": true
}
```

| Field | Meaning |
|---|---|
| `model_id` | Source Hugging Face id. `--shard-dir` uses it when you don't pass `--model`. |
| `num_layers` | Number of transformer blocks. |
| `layer_weight_mb` | Average **dense** bytes per layer (MB). Expert banks are excluded. |
| `total_weight_mb` | All weights, including experts (MB). |
| `embed_file`, `lm_head_file` | Always `embed.pt` and `lm_head.pt`. |
| `model_type` | transformers `model_type`. It selects the architecture adapter. |
| `weight_dtype` | `float16` or `bfloat16`. With `dtype=auto`, the runner computes in this dtype. |
| `shard_format` | `safetensors`, or `pt` for legacy. `pt` is assumed when the field is missing. |
| `shard_compression` | `none` or `swz`. |
| `num_experts`, `top_k`, `expert_bank` | MoE routing. `0`, `0` and `false` for dense models. |

Older manifests that don't have the newer fields still load with the defaults above. A manifest
with `weight_dtype: "float8"` fails to load with an error, because FP8 shards are no longer
supported. Re-shard with `swlp download`.

### Compressed shards (`.swz`)

`swlp compress-shards` converts each `layer_NNN.safetensors` into `layer_NNN.safetensors.swz`,
one layer at a time and in place. The codec (`src/swlp/codec.py`) groups the FP16/BF16 bytes
and entropy-codes them with zipnn (zstd + Huffman). Reconstruction is **bit-exact** and the
files are about 31% smaller. `embed.pt` and `lm_head.pt` stay uncompressed.

```
magic   4s   b"SWZ1"
raw     u64  decompressed size
sha256  32s  SHA-256 of the raw payload (checked at compress time and on --revert)
crc     u32  CRC-32 of the compressed blob (checked on every read)
blob    ...  zipnn payload
```

Each layer's roundtrip is verified before the original is deleted, and each layer is swapped
with an atomic rename, so an interrupted run can be resumed. The CRC check is mandatory: zipnn's
C core can crash on corrupt input, so only CRC-verified bytes reach it.

```bash
pip install 'swlp[codec]'
swlp compress-shards ./shards/mistral-7b          # ~31% smaller
swlp compress-shards --revert ./shards/mistral-7b # back to plain .safetensors
```

Compression helps throughput only when the SSD is the bottleneck. The crossover is about
3.5 GB/s sequential read. On faster SSDs, such as the internal SSDs on Apple Silicon,
decompression competes for unified-memory bandwidth and costs about 25% tok/s. `swlp doctor`
prints a recommendation for your machine. See
[results.md](results.md#lossless-shard-codec-swz).

### MoE expert banks (shard format v2)

For Mixture-of-Experts models, each layer is split in two:

- `layer_NNN.safetensors` holds the dense part: attention, norms and router.
- `layer_NNN.experts.safetensors` holds the expert weights.

The streaming window only carries the dense stream. `model/expert_bank.py` reads single experts
out of the bank by byte range, using the safetensors header's `data_offsets`:

| Bank layout | Tensor names | Expert *j* |
|---|---|---|
| Stacked (transformers ≥ 5 fused Experts) | `…experts.gate_up_proj` `[E, 2I, H]`, `…experts.down_proj` `[E, H, I]`, or `…experts.w1/w2/w3` | Row range `[j, j+1)` of each tensor |
| Per-expert (original HF checkpoints) | `…experts.{j}.gate_proj/up_proj/down_proj` (or `w1/w2/w3`) | One whole tensor per slot |

`expert_index.json` maps each layer to `bank_file`, `num_experts`, `dtype`, `hidden`,
`intermediate` and a `slices` list. Each slice has `slot` (`gate`, `up` or `down`), `offset`,
`nbytes`, `dtype`, `rows` and `cols`. A v1 directory, where the experts are stored inline in
`layer_NNN.safetensors`, is indexed in place. It works, but the dense stream then re-reads the
expert bytes.

Only `.safetensors` banks support ranged reads. Legacy `.pt` MoE shards stream whole layers:
the output is still exact, but there's no expert cache. Quantized banks are rejected.

`mlx-moe` can also stream from an MLX-format checkpoint with no shard directory
(`--model <mlx repo>`). In that case `model/mlx_expert_index.py` indexes the expert byte ranges
in place, and packed 4-bit experts are supported.

## Layer package

A layer package stores one `.safetensors` file per layer group, with a SHA-256 for every file.
You can validate it and inspect single layers without loading the model. The streaming backends
don't read this format.

```bash
swlp package /path/to/checkpoint ./pkg/my-model --model-name my-model
swlp validate-package ./pkg/my-model
swlp layer ./pkg/my-model 0                 # or: model.layers.0
```

Layout:

```
manifest.json      package summary
layer-index.json   per-tensor detail
layers/            one .safetensors per layer group
```

`manifest.json` has `schema_version = 1` and `package_format = "swlp.layer-package"`:

| Field | Meaning |
|---|---|
| `model_name` | Human-readable label. |
| `source_format` | Source layout: `safetensors`, `torch`, or a sharded variant. |
| `source_checksum`, `source_files` | Deterministic checksum and sorted names of the source weight files. |
| `layer_count`, `total_size_bytes` | Number of layer groups and total tensor bytes. |
| `layers[]` | Per layer: `index`, `name`, `file`, `tensor_count`, `total_size_bytes`, `transfer_cost_bytes`, `compute_estimate_flops`, `sha256`. |

`layer-index.json` adds a `tensors` list for each layer, with `name`, `shape`, `dtype` and
`size_bytes` for each tensor. `validate-package` checks the schema version, layer count, file
presence, SHA-256, and each tensor's name, shape, dtype and size. Conversion is deterministic:
tensors are grouped and sorted, and the JSON is written with sorted keys.
