"""Tests for swlp.codec — the lossless ``.swz`` shard container.

Covers:
- bit-exact roundtrip (even/odd sizes, FP16-like payloads)
- decompress returns a writable bytearray usable with torch.frombuffer
- header parsing: decompressed_size, is_swz, compressed_path
- error paths: bad magic, truncation, empty payload, corrupt payload
- verify_blob full offline check
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from swlp import codec

# zipnn is an optional extra (`pip install swlp[codec]`): the .swz codec is
# off by default and a measured throughput LOSS on fast SSDs. Skip rather
# than fail when it is not installed.
pytest.importorskip("zipnn", reason="needs the optional swlp[codec] extra")


def _fp16_payload(n_values: int = 4096) -> bytes:
    torch.manual_seed(7)
    return torch.randn(n_values, dtype=torch.float16).numpy().tobytes()


# ── roundtrip ─────────────────────────────────────────────────────────────────


def test_roundtrip_bit_exact() -> None:
    data = _fp16_payload()
    blob = codec.compress_bytes(data)
    assert bytes(codec.decompress_bytes(blob)) == data


def test_roundtrip_odd_length() -> None:
    # safetensors files have a JSON header, so payloads are not 2-byte aligned.
    data = b"\x07" + _fp16_payload(1023)
    blob = codec.compress_bytes(data)
    assert bytes(codec.decompress_bytes(blob)) == data


def test_roundtrip_small_payload() -> None:
    data = b"\x01\x02"
    blob = codec.compress_bytes(data)
    assert bytes(codec.decompress_bytes(blob)) == data


def test_compress_accepts_memoryview() -> None:
    data = _fp16_payload()
    blob = codec.compress_bytes(memoryview(data))
    assert bytes(codec.decompress_bytes(memoryview(blob))) == data


def test_decompress_output_wraps_with_frombuffer() -> None:
    # The streaming hot path wraps the decompressed bytearray zero-copy.
    data = _fp16_payload()
    out = codec.decompress_bytes(codec.compress_bytes(data))
    assert isinstance(out, bytearray)
    tensor = torch.frombuffer(out, dtype=torch.uint8)
    assert tensor.numel() == len(data)
    assert bytes(tensor.numpy().tobytes()) == data


# ── header parsing ────────────────────────────────────────────────────────────


def test_decompressed_size_matches_payload() -> None:
    data = _fp16_payload(100)
    assert codec.decompressed_size(codec.compress_bytes(data)) == len(data)


def test_is_swz_detects_container() -> None:
    blob = codec.compress_bytes(_fp16_payload(8))
    assert codec.is_swz(blob)
    assert not codec.is_swz(b"\x00" * 16)


def test_compressed_path_appends_suffix() -> None:
    path = Path("/x/layer_000.safetensors")
    assert codec.compressed_path(path) == Path("/x/layer_000.safetensors.swz")


# ── error paths ───────────────────────────────────────────────────────────────


def test_bad_magic_raises() -> None:
    blob = codec.compress_bytes(_fp16_payload(8))
    with pytest.raises(codec.CodecError, match="magic"):
        codec.decompress_bytes(b"XXXX" + blob[4:])


def test_truncated_header_raises() -> None:
    with pytest.raises(codec.CodecError, match="truncated"):
        codec.decompress_bytes(codec.MAGIC + b"\x00" * 4)


def test_empty_payload_refused() -> None:
    with pytest.raises(codec.CodecError, match="empty"):
        codec.compress_bytes(b"")


def test_corrupt_payload_rejected_by_crc_gate() -> None:
    # Must fail at the CRC check — corrupt bytes never reach zipnn's C core,
    # which can segfault on malformed streams (see codec module docstring).
    blob = bytearray(codec.compress_bytes(_fp16_payload()))
    blob[codec.HEADER_SIZE + 8] ^= 0xFF  # flip a byte inside the zipnn stream
    with pytest.raises(codec.CodecError, match="crc"):
        codec.decompress_bytes(bytes(blob))


def test_check_sha_accepts_good_blob() -> None:
    data = _fp16_payload()
    assert bytes(codec.decompress_bytes(codec.compress_bytes(data), check_sha=True)) == data


def test_check_sha_rejects_tampered_digest() -> None:
    # Corrupt only the stored SHA-256: the CRC (over the compressed payload)
    # still passes, so only check_sha=True catches it.
    blob = bytearray(codec.compress_bytes(_fp16_payload()))
    blob[len(codec.MAGIC) + 8] ^= 0xFF  # first byte of the sha256 field
    codec.decompress_bytes(bytes(blob))  # hot path does not check the digest
    with pytest.raises(codec.CodecError, match="sha256"):
        codec.decompress_bytes(bytes(blob), check_sha=True)


# ── offline verification ──────────────────────────────────────────────────────


def test_verify_blob_accepts_good_blob() -> None:
    assert codec.verify_blob(codec.compress_bytes(_fp16_payload()))


def test_verify_blob_rejects_corruption() -> None:
    blob = bytearray(codec.compress_bytes(_fp16_payload()))
    blob[codec.HEADER_SIZE + 8] ^= 0xFF
    assert not codec.verify_blob(bytes(blob))


# ── throughput recommendation (Phase 24) ─────────────────────────────────────

def test_recommend_compression_below_crossover():
    from swlp.codec import recommend_compression
    # PCIe-3-class NVMe: compressed reads win.
    assert recommend_compression(2.0) is True
    assert recommend_compression(3.4) is True


def test_recommend_compression_above_crossover():
    from swlp.codec import recommend_compression
    # Fast Apple SSDs: decompression contends for memory bandwidth.
    assert recommend_compression(6.9) is False
    assert recommend_compression(3.6) is False


def test_recommend_compression_unknown_bandwidth():
    from swlp.codec import recommend_compression
    assert recommend_compression(0.0) is False
