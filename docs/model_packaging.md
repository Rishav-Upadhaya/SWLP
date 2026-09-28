# SWLP Model Packaging

SWLP Phase 3 stores model weights as deterministic layer packages instead of a single monolithic checkpoint.

## Layout

Each packaged model directory contains:

- `manifest.json` for the high-level model description
- `layer-index.json` for the detailed layer index
- `layers/` containing one `.safetensors` file per layer group

## Manifest fields

The manifest is versioned and currently uses `schema_version = 1`.

Important fields:

- `package_format`: stable format identifier
- `model_name`: human-readable model label
- `source_format`: original checkpoint layout such as `safetensors`, `torch`, or sharded variants
- `source_checksum`: deterministic checksum of the source weight files used for conversion
- `source_files`: sorted source weight file names used to build the package
- `layer_count`: number of packaged layer groups
- `total_size_bytes`: total tensor storage bytes across all layers
- `layers`: summary metadata for each packaged layer

## Layer metadata

Each layer entry records:

- `index`: stable ordering
- `name`: logical layer name such as `model.layers.0`
- `file`: relative path to the layer file
- `tensor_count`: number of tensors in the layer file
- `total_size_bytes`: total tensor storage bytes for the layer
- `transfer_cost_bytes`: the layer transfer cost estimate
- `compute_estimate_flops`: a simple compute estimate derived from tensor size
- `sha256`: integrity hash for the stored layer file
- `tensors`: per-tensor `name`, `shape`, `dtype`, and `size_bytes` entries in the layer index

## Commands

Convert a checkpoint into SWLP packaging:

```bash
swlp package /path/to/checkpoint /path/to/output --model-name my-model
```

Validate a package:

```bash
swlp validate-package /path/to/output
```

Inspect a single packaged layer without loading the full model:

```bash
swlp layer /path/to/output model.layers.0
```

## Integrity checks

Validation checks the manifest version, layer count, file presence, file checksums, tensor names, tensor shapes, tensor dtypes, and tensor storage sizes.

## Conversion behavior

Conversion is deterministic for a given source checkpoint layout because tensor names are grouped and sorted consistently, layer files are named in index order, and the manifest/index JSON is written with sorted keys.
