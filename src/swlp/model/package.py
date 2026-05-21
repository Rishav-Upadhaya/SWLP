from __future__ import annotations

import hashlib
import json
from collections import OrderedDict, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file

SCHEMA_VERSION = 1
PACKAGE_FORMAT = "swlp.layer-package"
MANIFEST_FILENAME = "manifest.json"
LAYER_INDEX_FILENAME = "layer-index.json"
LAYER_DIRNAME = "layers"


@dataclass(slots=True)
class TensorMetadata:
    name: str
    shape: list[int]
    dtype: str
    size_bytes: int


@dataclass(slots=True)
class LayerSummary:
    index: int
    name: str
    file: str
    tensor_count: int
    total_size_bytes: int
    transfer_cost_bytes: int
    compute_estimate_flops: int
    sha256: str


@dataclass(slots=True)
class LayerRecord(LayerSummary):
    tensors: list[TensorMetadata] = field(default_factory=list)


@dataclass(slots=True)
class PackageManifest:
    schema_version: int = SCHEMA_VERSION
    package_format: str = PACKAGE_FORMAT
    model_name: str = "unknown"
    source_format: str = "checkpoint"
    source_checksum: str = ""
    source_files: list[str] = field(default_factory=list)
    tensor_format: str = "safetensors"
    layer_index_file: str = LAYER_INDEX_FILENAME
    layer_dir: str = LAYER_DIRNAME
    layer_count: int = 0
    total_size_bytes: int = 0
    layers: list[LayerSummary] = field(default_factory=list)


@dataclass(slots=True)
class LayerIndex:
    schema_version: int = SCHEMA_VERSION
    layers: list[LayerRecord] = field(default_factory=list)


@dataclass(slots=True)
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_bytes(data: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_size_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.element_size() * tensor.numel())


def _tensor_dtype_name(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).replace("torch.", "")


def _natural_key(value: str) -> tuple[tuple[int, int | str], ...]:
    key: list[tuple[int, int | str]] = []
    for part in value.split("."):
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part))
    return tuple(key)


def _layer_name_from_parameter(parameter_name: str) -> str:
    parts = parameter_name.split(".")
    for index, part in enumerate(parts):
        if part.isdigit():
            return ".".join(parts[: index + 1])
    if len(parts) > 1 and parts[-1] in {"weight", "bias"}:
        return ".".join(parts[:-1])
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return parts[0] if parts else "root"


def _sanitize_layer_filename(layer_name: str) -> str:
    return layer_name.replace("/", "_").replace(":", "_")


def _load_weight_file(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        return dict(load_safetensors_file(str(path), device="cpu"))

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        isinstance(payload, dict)
        and "state_dict" in payload
        and isinstance(payload["state_dict"], dict)
    ):
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint payload in {path}")

    tensors: dict[str, torch.Tensor] = {}
    for name, value in payload.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Checkpoint entry {name!r} in {path} is not a tensor")
        tensors[str(name)] = value
    return tensors


def _discover_checkpoint_sources(checkpoint_path: Path) -> tuple[list[Path], str]:
    if checkpoint_path.is_file():
        if checkpoint_path.suffix == ".safetensors":
            return [checkpoint_path], "safetensors"
        if checkpoint_path.suffix in {".bin", ".pt", ".pth"}:
            return [checkpoint_path], "torch"
        raise ValueError(f"Unsupported checkpoint file type: {checkpoint_path.suffix}")

    if not checkpoint_path.is_dir():
        raise FileNotFoundError(checkpoint_path)

    index_candidates = sorted(checkpoint_path.glob("*.index.json"), key=lambda path: path.name)
    for index_path in index_candidates:
        payload = _json_load(index_path)
        weight_map = payload.get("weight_map")
        if isinstance(weight_map, dict) and weight_map:
            shard_names = sorted({str(name) for name in weight_map.values()})
            shard_paths = [checkpoint_path / shard_name for shard_name in shard_names]
            if not all(path.exists() for path in shard_paths):
                missing = [str(path) for path in shard_paths if not path.exists()]
                raise FileNotFoundError(f"Missing checkpoint shards: {missing}")
            source_format = (
                "safetensors-sharded"
                if index_path.name.endswith(".safetensors.index.json")
                else "torch-sharded"
            )
            return shard_paths, source_format

    candidates = [
        path
        for path in sorted(checkpoint_path.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.suffix in {".safetensors", ".bin", ".pt", ".pth"}
    ]
    if len(candidates) == 1:
        source_format = "safetensors" if candidates[0].suffix == ".safetensors" else "torch"
        return candidates, source_format

    raise ValueError(
        "Could not infer checkpoint sources."
        " Provide a single weight file or a directory with an index file."
    )


def _source_files_relative(checkpoint_path: Path, source_files: Iterable[Path]) -> list[str]:
    if checkpoint_path.is_file():
        return [checkpoint_path.name]
    relative_files = [path.relative_to(checkpoint_path).as_posix() for path in source_files]
    return sorted(relative_files)


def _source_checksum(checkpoint_path: Path, source_files: list[Path]) -> str:
    digest = hashlib.sha256()
    for source_file in sorted(source_files, key=lambda path: path.as_posix()):
        if checkpoint_path.is_file():
            relative_name = source_file.name
        else:
            relative_name = source_file.relative_to(checkpoint_path).as_posix()
        digest.update(relative_name.encode("utf-8"))
        digest.update(b"\0")
        with source_file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _group_tensors(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, OrderedDict[str, torch.Tensor]]:
    grouped: dict[str, OrderedDict[str, torch.Tensor]] = defaultdict(OrderedDict)
    for name in sorted(state_dict):
        value = state_dict[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"State dict entry {name!r} is not a tensor")
        layer_name = _layer_name_from_parameter(name)
        grouped[layer_name][name] = value.detach().cpu().contiguous()
    return dict(grouped)


def _tensor_metadata(name: str, tensor: torch.Tensor) -> TensorMetadata:
    return TensorMetadata(
        name=name,
        shape=[int(dimension) for dimension in tensor.shape],
        dtype=_tensor_dtype_name(tensor),
        size_bytes=_tensor_size_bytes(tensor),
    )


def _layer_summary(
    index: int,
    layer_name: str,
    file_name: str,
    layer_file: Path,
    tensor_metadata: list[TensorMetadata],
) -> LayerSummary:
    total_size_bytes = sum(tensor.size_bytes for tensor in tensor_metadata)
    return LayerSummary(
        index=index,
        name=layer_name,
        file=file_name,
        tensor_count=len(tensor_metadata),
        total_size_bytes=total_size_bytes,
        transfer_cost_bytes=total_size_bytes,
        compute_estimate_flops=total_size_bytes * 2,
        sha256=_sha256_file(layer_file),
    )


def _load_index_record(data: dict[str, Any]) -> LayerRecord:
    tensors = [
        TensorMetadata(
            name=str(tensor["name"]),
            shape=[int(dimension) for dimension in tensor["shape"]],
            dtype=str(tensor["dtype"]),
            size_bytes=int(tensor["size_bytes"]),
        )
        for tensor in data.get("tensors", [])
    ]
    return LayerRecord(
        index=int(data["index"]),
        name=str(data["name"]),
        file=str(data["file"]),
        tensor_count=int(data["tensor_count"]),
        total_size_bytes=int(data["total_size_bytes"]),
        transfer_cost_bytes=int(data["transfer_cost_bytes"]),
        compute_estimate_flops=int(data["compute_estimate_flops"]),
        sha256=str(data["sha256"]),
        tensors=tensors,
    )


def _load_summary(data: dict[str, Any]) -> LayerSummary:
    return LayerSummary(
        index=int(data["index"]),
        name=str(data["name"]),
        file=str(data["file"]),
        tensor_count=int(data["tensor_count"]),
        total_size_bytes=int(data["total_size_bytes"]),
        transfer_cost_bytes=int(data["transfer_cost_bytes"]),
        compute_estimate_flops=int(data["compute_estimate_flops"]),
        sha256=str(data["sha256"]),
    )


def package_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    model_name: str | None = None,
) -> PackageManifest:
    source_files, source_format = _discover_checkpoint_sources(checkpoint_path)
    state_dict: dict[str, torch.Tensor] = {}
    for source_file in source_files:
        state_dict.update(_load_weight_file(source_file))

    grouped_layers = _group_tensors(state_dict)

    package_dir = output_dir.expanduser().resolve()
    layer_dir = package_dir / LAYER_DIRNAME
    layer_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[LayerSummary] = []
    records: list[LayerRecord] = []
    total_size_bytes = 0

    for index, layer_name in enumerate(sorted(grouped_layers, key=_natural_key)):
        tensors = grouped_layers[layer_name]
        tensor_metadata = [_tensor_metadata(name, tensor) for name, tensor in tensors.items()]
        file_name = f"{index:04d}-{_sanitize_layer_filename(layer_name)}.safetensors"
        file_path = layer_dir / file_name
        save_safetensors_file(tensors, str(file_path))
        summary = _layer_summary(
            index, layer_name, f"{LAYER_DIRNAME}/{file_name}", file_path, tensor_metadata
        )
        record = LayerRecord(**asdict(summary), tensors=tensor_metadata)
        summaries.append(summary)
        records.append(record)
        total_size_bytes += summary.total_size_bytes

    manifest = PackageManifest(
        model_name=model_name or checkpoint_path.stem,
        source_format=source_format,
        source_checksum=_source_checksum(checkpoint_path, source_files),
        source_files=_source_files_relative(checkpoint_path, source_files),
        layer_index_file=LAYER_INDEX_FILENAME,
        layer_dir=LAYER_DIRNAME,
        layer_count=len(summaries),
        total_size_bytes=total_size_bytes,
        layers=summaries,
    )
    layer_index = LayerIndex(layers=records)

    _json_dump(package_dir / MANIFEST_FILENAME, asdict(manifest))
    _json_dump(package_dir / LAYER_INDEX_FILENAME, asdict(layer_index))
    return manifest


def load_manifest(package_dir: Path) -> PackageManifest:
    payload = _json_load(package_dir / MANIFEST_FILENAME)
    if int(payload.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported manifest schema version: {payload.get('schema_version')}")
    layers = [_load_summary(layer) for layer in payload.get("layers", [])]
    return PackageManifest(
        schema_version=int(payload["schema_version"]),
        package_format=str(payload["package_format"]),
        model_name=str(payload["model_name"]),
        source_format=str(payload["source_format"]),
        source_checksum=str(payload["source_checksum"]),
        source_files=[str(item) for item in payload.get("source_files", [])],
        tensor_format=str(payload["tensor_format"]),
        layer_index_file=str(payload["layer_index_file"]),
        layer_dir=str(payload["layer_dir"]),
        layer_count=int(payload["layer_count"]),
        total_size_bytes=int(payload["total_size_bytes"]),
        layers=layers,
    )


def load_layer_index(package_dir: Path) -> LayerIndex:
    payload = _json_load(package_dir / LAYER_INDEX_FILENAME)
    if int(payload.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported layer index schema version: {payload.get('schema_version')}")
    layers = [_load_index_record(layer) for layer in payload.get("layers", [])]
    return LayerIndex(schema_version=int(payload["schema_version"]), layers=layers)


def _resolve_layer_identifier(layer_index: LayerIndex, layer_identifier: int | str) -> LayerRecord:
    if isinstance(layer_identifier, int):
        try:
            return layer_index.layers[layer_identifier]
        except IndexError as exc:
            raise IndexError(f"Layer index out of range: {layer_identifier}") from exc

    for record in layer_index.layers:
        if record.name == layer_identifier or record.file == layer_identifier:
            return record
    raise KeyError(f"Layer not found: {layer_identifier}")


def load_layer(
    package_dir: Path,
    layer_identifier: int | str,
    map_location: str | torch.device = "cpu",
) -> tuple[LayerRecord, OrderedDict[str, torch.Tensor]]:
    layer_index = load_layer_index(package_dir)
    record = _resolve_layer_identifier(layer_index, layer_identifier)
    layer_path = package_dir / record.file
    if not layer_path.exists():
        raise FileNotFoundError(layer_path)
    if _sha256_file(layer_path) != record.sha256:
        raise ValueError(f"Integrity check failed for layer {record.name}")

    tensors = load_safetensors_file(str(layer_path), device="cpu")
    ordered = OrderedDict()
    for name in sorted(tensors):
        tensor = tensors[name]
        if map_location != "cpu":
            tensor = tensor.to(map_location)
        ordered[name] = tensor
    return record, ordered


def validate_package(package_dir: Path) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    try:
        manifest = load_manifest(package_dir)
    except Exception as exc:
        return ValidationResult(valid=False, errors=[f"manifest: {exc}"])

    try:
        layer_index = load_layer_index(package_dir)
    except Exception as exc:
        errors.append(f"layer-index: {exc}")
        return ValidationResult(valid=False, errors=errors)

    if manifest.layer_count != len(manifest.layers):
        errors.append("manifest layer_count does not match manifest layers length")
    if len(manifest.layers) != len(layer_index.layers):
        errors.append("manifest layers do not match layer index length")

    manifest_layers = manifest.layers
    index_layers = layer_index.layers
    for manifest_layer, index_layer in zip(manifest_layers, index_layers, strict=False):
        expected_summary = LayerSummary(
            index=index_layer.index,
            name=index_layer.name,
            file=index_layer.file,
            tensor_count=index_layer.tensor_count,
            total_size_bytes=index_layer.total_size_bytes,
            transfer_cost_bytes=index_layer.transfer_cost_bytes,
            compute_estimate_flops=index_layer.compute_estimate_flops,
            sha256=index_layer.sha256,
        )
        if manifest_layer != expected_summary:
            errors.append(f"layer summary mismatch for {index_layer.name}")

        layer_path = package_dir / index_layer.file
        if not layer_path.exists():
            errors.append(f"missing layer file: {index_layer.file}")
            continue

        if _sha256_file(layer_path) != index_layer.sha256:
            errors.append(f"checksum mismatch for layer: {index_layer.name}")
            continue

        tensors = load_safetensors_file(str(layer_path), device="cpu")
        tensor_names = sorted(tensors)
        expected_names = sorted(tensor.name for tensor in index_layer.tensors)
        if tensor_names != expected_names:
            errors.append(f"tensor names mismatch in layer: {index_layer.name}")
            continue

        for tensor_name, tensor_metadata in zip(tensor_names, index_layer.tensors, strict=False):
            tensor = tensors[tensor_name]
            if [int(dimension) for dimension in tensor.shape] != tensor_metadata.shape:
                errors.append(
                    f"shape mismatch for tensor {tensor_name} in layer {index_layer.name}"
                )
            if _tensor_dtype_name(tensor) != tensor_metadata.dtype:
                errors.append(
                    f"dtype mismatch for tensor {tensor_name} in layer {index_layer.name}"
                )
            if _tensor_size_bytes(tensor) != tensor_metadata.size_bytes:
                errors.append(
                    f"size mismatch for tensor {tensor_name} in layer {index_layer.name}"
                )

        if sum(tensor.size_bytes for tensor in index_layer.tensors) != index_layer.total_size_bytes:
            errors.append(f"layer size mismatch for {index_layer.name}")

    if manifest.total_size_bytes != sum(layer.total_size_bytes for layer in manifest.layers):
        errors.append("manifest total_size_bytes does not match layer totals")

    return ValidationResult(valid=not errors, errors=errors, warnings=warnings)


def describe_manifest(package_dir: Path) -> dict[str, Any]:
    manifest = load_manifest(package_dir)
    return asdict(manifest)
