from collections import OrderedDict

import torch
from safetensors.torch import save_file

from swlp.model.package import load_layer, load_manifest, package_checkpoint, validate_package


def _write_checkpoint(path):
    tensors = OrderedDict(
        [
            ("model.embed_tokens.weight", torch.arange(12, dtype=torch.float32).reshape(3, 4)),
            ("model.layers.0.self_attn.weight", torch.ones(2, 2, dtype=torch.float16)),
            ("model.layers.0.self_attn.bias", torch.zeros(2, dtype=torch.float16)),
            ("model.layers.1.mlp.weight", torch.full((2, 3), 2.0, dtype=torch.float32)),
            ("lm_head.weight", torch.eye(3, dtype=torch.float32)),
        ]
    )
    save_file(tensors, str(path))


def test_package_checkpoint_splits_layers(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.safetensors"
    output_dir = tmp_path / "package"
    _write_checkpoint(checkpoint_path)

    manifest = package_checkpoint(checkpoint_path, output_dir, model_name="demo-model")

    loaded_manifest = load_manifest(output_dir)
    assert loaded_manifest.model_name == "demo-model"
    assert loaded_manifest.layer_count == 4
    assert loaded_manifest.source_files == ["checkpoint.safetensors"]
    assert manifest.total_size_bytes == loaded_manifest.total_size_bytes

    result = validate_package(output_dir)
    assert result.valid is True
    assert result.errors == []


def test_load_layer_independently(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.safetensors"
    output_dir = tmp_path / "package"
    _write_checkpoint(checkpoint_path)
    package_checkpoint(checkpoint_path, output_dir, model_name="demo-model")

    record, tensors = load_layer(output_dir, "model.layers.0")

    assert record.name == "model.layers.0"
    assert list(tensors) == ["model.layers.0.self_attn.bias", "model.layers.0.self_attn.weight"]
    assert tensors["model.layers.0.self_attn.weight"].shape == torch.Size([2, 2])
    assert tensors["model.layers.0.self_attn.weight"].dtype == torch.float16


def test_package_validation_detects_integrity_errors(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.safetensors"
    output_dir = tmp_path / "package"
    _write_checkpoint(checkpoint_path)
    package_checkpoint(checkpoint_path, output_dir, model_name="demo-model")

    layer_file = next((output_dir / "layers").glob("*.safetensors"))
    layer_file.write_bytes(layer_file.read_bytes()[:-1] + b"0")

    result = validate_package(output_dir)
    assert result.valid is False
    assert any("checksum mismatch" in error for error in result.errors)
