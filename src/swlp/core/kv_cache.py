"""KV cache manager with tiered storage: device → host → compressed → disk.

Phase 12 adds a fourth tier: when a session exceeds the memory budget even after
zlib compression, cold-layer KV is serialised to a temporary file on disk. The
file is lazily reloaded (and deleted) when that layer is needed again.  This
allows arbitrarily long contexts at the cost of SSD latency on cold re-reads.

Phase 12 also fixes the _history unbounded-growth bug from Phase 2: history
snapshots are now only recorded when the manager is constructed with
``profile=True``.  Long runs and batched sessions no longer leak memory through
the history list.
"""
from __future__ import annotations

import io
import logging
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import torch

from .kv_quant import kv_dequantize_int4, kv_quantize_int4

LOGGER = logging.getLogger(__name__)


@dataclass
class _KVEntry:
    tensors: tuple[torch.Tensor, torch.Tensor] | None
    compressed: bytes | None
    uncompressed_bytes: int
    compressed_bytes: int
    last_access: float
    location: str  # "device" | "host" | "compressed" | "disk"
    disk_path: str | None = None
    disk_bytes: int = 0
    # Phase 18: packed INT4 KV storage (packed_k, scales_k, packed_v, scales_v).
    # When set, ``tensors`` is None and quantized bytes are counted instead.
    quantized_tensors: tuple[torch.Tensor, ...] | None = None


class KVCacheManager:
    """Conservative KV cache manager with four storage tiers.

    Tiers (coldest evicted first when over budget):
    1. *device*     — active, on compute device (GPU / MPS).
    2. *host*       — CPU RAM (detached tensors).
    3. *compressed* — zlib-compressed bytes in Python heap.
    4. *disk*       — serialised to a temp ``.pt`` file on SSD (Phase 12).

    Set ``profile=True`` to enable per-step history snapshots (``_history``
    list).  With ``profile=False`` (default) the list stays empty, avoiding
    unbounded growth in long / batched sessions.
    """

    def __init__(
        self,
        budget_bytes: int | None = None,
        compression: bool = True,
        compression_level: int = 6,
        tiering: bool = True,
        device: torch.device | None = None,
        device_budget_bytes: int | None = None,
        profile: bool = False,
        disk_dir: Path | str | None = None,
        kv_window: int = 0,
        kv_quant: str = "none",
    ) -> None:
        self.budget_bytes = budget_bytes or (512 * 1024 * 1024)
        self.compression = compression
        self.compression_level = compression_level
        self.tiering = tiering
        self.device = device
        self.device_budget_bytes = device_budget_bytes or self.budget_bytes
        self._profile = profile
        # Phase 16: sliding-window KV budget.  When > 0, KV tensors are trimmed
        # to the most recent kv_window token positions on every set() call.  This
        # bounds KV memory to kv_window × bytes_per_token × num_layers regardless
        # of context length.  0 = unbounded (current behaviour).
        self.kv_window = kv_window
        # Phase 18: INT4 KV quantization mode — "none" | "int4".
        # "none" = lossless (default).  "int4" = lossy 4× compression; must be
        # explicitly opt-in.  Never silently activate — always label in reports.
        if kv_quant not in ("none", "int4"):
            raise ValueError(f"kv_quant must be 'none' or 'int4'; got {kv_quant!r}")
        self.kv_quant = kv_quant
        self.disk_dir = Path(disk_dir) if disk_dir else None
        if self.disk_dir is not None:
            self.disk_dir.mkdir(parents=True, exist_ok=True)
        self._entries: dict[int, _KVEntry] = {}
        self._device_bytes = 0
        self._host_bytes = 0
        self._compressed_bytes = 0
        self._disk_bytes = 0
        self._peak_device_bytes = 0
        self._peak_host_bytes = 0
        self._peak_total_bytes = 0
        self._compressions = 0
        self._decompressions = 0
        self._offloads = 0
        self._moves_to_device = 0
        self._budget_violations = 0
        self._disk_spills = 0
        self._disk_loads = 0
        self._history: list[dict] = []

    def _estimate_size(self, tensors: tuple[torch.Tensor, ...] | None) -> int:
        if tensors is None:
            return 0
        size = 0
        for t in tensors:
            if t is None:
                continue
            size += t.element_size() * t.numel()
        return int(size)

    def _record_snapshot(self) -> None:
        total = self._device_bytes + self._host_bytes + self._compressed_bytes + self._disk_bytes
        self._peak_device_bytes = max(self._peak_device_bytes, self._device_bytes)
        self._peak_host_bytes = max(self._peak_host_bytes, self._host_bytes)
        self._peak_total_bytes = max(self._peak_total_bytes, total)
        # Only append when profiling — prevents unbounded list growth in long sessions.
        if self._profile:
            self._history.append(
                {
                    "timestamp": time.time(),
                    "device_bytes": self._device_bytes,
                    "host_bytes": self._host_bytes,
                    "compressed_bytes": self._compressed_bytes,
                    "disk_bytes": self._disk_bytes,
                    "total_bytes": total,
                }
            )

    def _detach_to_cpu(
        self, tensors: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if tensors is None:
            return None
        return tuple(t.detach().cpu() for t in tensors)

    def _update_usage_for_entry(self, entry: _KVEntry, remove: bool) -> None:
        factor = -1 if remove else 1
        if entry.location == "device":
            self._device_bytes += factor * entry.uncompressed_bytes
        elif entry.location == "host":
            self._host_bytes += factor * entry.uncompressed_bytes
        elif entry.location == "compressed":
            self._compressed_bytes += factor * entry.compressed_bytes
        elif entry.location == "disk":
            self._disk_bytes += factor * entry.disk_bytes

    def set(self, layer_index: int, tensors: tuple[torch.Tensor, torch.Tensor] | None) -> None:
        """Store or update KV tensors for a given layer.

        When ``kv_window > 0`` the KV tensors are trimmed to the most recent
        ``kv_window`` token positions before storing (sliding-window KV budget).

        When ``kv_quant == "int4"`` the tensors are quantized to INT4 before
        storage (Phase 18 lossy tier).  The quantized entry stays on the host
        (CPU RAM) and is exempt from further zlib compression.
        """
        # Phase 16: sliding-window trim — keep only the last kv_window positions.
        if self.kv_window > 0 and tensors is not None:
            k, v = tensors
            seq_len = k.shape[-2]
            if seq_len > self.kv_window:
                k = k[..., -self.kv_window :, :]
                v = v[..., -self.kv_window :, :]
                tensors = (k, v)
        now = time.time()

        prev = self._entries.get(layer_index)
        if prev:
            # Clean up any disk file from the previous entry.
            if prev.location == "disk" and prev.disk_path:
                try:
                    Path(prev.disk_path).unlink(missing_ok=True)
                except Exception:
                    pass
            self._update_usage_for_entry(prev, remove=True)

        # Phase 18: INT4 quantization — store packed 4-tuple on host, exempt
        # from zlib.  uncompressed_bytes is set to the actual quantized size so
        # budget enforcement accounts for the real footprint.
        if self.kv_quant == "int4" and tensors is not None:
            k, v = tensors
            k_cpu = k.detach().cpu()
            v_cpu = v.detach().cpu()
            pk, sk = kv_quantize_int4(k_cpu)
            pv, sv = kv_quantize_int4(v_cpu)
            quant_bytes = self._estimate_size((pk, sk, pv, sv))
            entry = _KVEntry(
                tensors=None,
                compressed=None,
                uncompressed_bytes=quant_bytes,
                compressed_bytes=0,
                last_access=now,
                location="host",
                quantized_tensors=(pk, sk, pv, sv),
            )
            self._entries[layer_index] = entry
            self._update_usage_for_entry(entry, remove=False)
            self._record_snapshot()
            self._enforce_budget()
            return

        uncompressed = self._estimate_size(tensors)
        location = "host"
        if tensors is not None and tensors[0].is_cuda:
            location = "device"

        entry = _KVEntry(
            tensors=tensors,
            compressed=None,
            uncompressed_bytes=uncompressed,
            compressed_bytes=0,
            last_access=now,
            location=location,
        )
        self._entries[layer_index] = entry
        self._update_usage_for_entry(entry, remove=False)

        self._record_snapshot()
        self._enforce_device_budget()
        self._enforce_budget()

    def get(
        self, layer_index: int, target_device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return tensors for the layer, reloading from any tier as needed."""
        entry = self._entries.get(layer_index)
        if entry is None:
            return None
        entry.last_access = time.time()

        # Phase 18: dequantize INT4 on demand and return.
        if entry.quantized_tensors is not None:
            pk, sk, pv, sv = entry.quantized_tensors
            desired = target_device or self.device
            if desired is not None:
                pk, sk, pv, sv = (t.to(desired) for t in (pk, sk, pv, sv))
            k = kv_dequantize_int4(pk, sk)
            v = kv_dequantize_int4(pv, sv)
            return (k, v)

        # Tier 4 → 3: load from disk back into host memory.
        if entry.location == "disk":
            self._load_from_disk(layer_index, entry)

        # Tier 3 → 2: decompress.
        if entry.location == "compressed" and entry.compressed is not None:
            buf = zlib.decompress(entry.compressed)
            bio = io.BytesIO(buf)
            data = torch.load(bio, weights_only=False)
            entry.tensors = (data.get("k"), data.get("v"))
            entry.compressed = None
            self._update_usage_for_entry(entry, remove=True)
            entry.compressed_bytes = 0
            entry.location = "host"
            self._update_usage_for_entry(entry, remove=False)
            self._decompressions += 1

        if entry.tensors is not None:
            desired = target_device or self.device
            if desired is not None and entry.location != "device":
                self._update_usage_for_entry(entry, remove=True)
                entry.tensors = tuple(t.to(desired) for t in entry.tensors)
                entry.location = "device"
                self._update_usage_for_entry(entry, remove=False)
                self._moves_to_device += 1
                self._record_snapshot()
                self._enforce_device_budget()
                self._enforce_budget()
        return entry.tensors

    # ── disk spill tier (Phase 12) ───────────────────────────────────────────

    def _spill_to_disk(self, layer_index: int, entry: _KVEntry) -> None:
        """Serialise KV entry to a temp file; free host / compressed memory."""
        if self.disk_dir is None:
            return
        disk_path = self.disk_dir / f"kv_{layer_index}.pt"
        try:
            self._update_usage_for_entry(entry, remove=True)
            if entry.location == "compressed" and entry.compressed is not None:
                # Save the already-compressed bytes directly — no extra compression pass.
                disk_path.write_bytes(entry.compressed)
                entry.compressed = None
                entry.compressed_bytes = 0
            else:
                cpu_tensors = self._detach_to_cpu(entry.tensors)
                bio = io.BytesIO()
                torch.save({"k": cpu_tensors[0], "v": cpu_tensors[1]}, bio)
                disk_path.write_bytes(bio.getvalue())
                entry.tensors = None
            entry.disk_path = str(disk_path)
            entry.disk_bytes = disk_path.stat().st_size
            entry.location = "disk"
            self._update_usage_for_entry(entry, remove=False)
            self._disk_spills += 1
            LOGGER.debug(
                "kv_spilled_to_disk",
                extra={"layer": layer_index, "bytes": entry.disk_bytes},
            )
        except Exception:
            LOGGER.exception("kv_disk_spill_failed", extra={"layer": layer_index})

    def _load_from_disk(self, layer_index: int, entry: _KVEntry) -> None:
        """Reload KV from disk into host memory; delete the spill file."""
        if entry.disk_path is None or entry.location != "disk":
            return
        disk_path = Path(entry.disk_path)
        try:
            raw = disk_path.read_bytes()
            disk_path.unlink(missing_ok=True)
            self._update_usage_for_entry(entry, remove=True)
            entry.disk_path = None
            entry.disk_bytes = 0
            # Try to restore: first as a torch save dict (plain), then as zlib.
            try:
                bio = io.BytesIO(raw)
                data = torch.load(bio, weights_only=False)
                entry.tensors = (data.get("k"), data.get("v"))
                entry.compressed = None
                entry.compressed_bytes = 0
                entry.uncompressed_bytes = self._estimate_size(entry.tensors)
                entry.location = "host"
            except Exception:
                # Assume raw bytes are zlib-compressed (spilled from "compressed" state).
                entry.compressed = raw
                entry.compressed_bytes = len(raw)
                entry.location = "compressed"
            self._update_usage_for_entry(entry, remove=False)
            self._disk_loads += 1
            LOGGER.debug("kv_loaded_from_disk", extra={"layer": layer_index})
        except Exception:
            LOGGER.exception("kv_disk_load_failed", extra={"layer": layer_index})

    # ── budget enforcement ───────────────────────────────────────────────────

    def _offload_to_host(self, entry: _KVEntry) -> None:
        if entry.tensors is None or entry.location != "device":
            return
        self._update_usage_for_entry(entry, remove=True)
        entry.tensors = self._detach_to_cpu(entry.tensors)
        entry.location = "host"
        self._update_usage_for_entry(entry, remove=False)
        self._offloads += 1

    def _compress_entry(self, entry: _KVEntry) -> None:
        # INT4-quantized entries are already compact — skip zlib compression.
        if entry.quantized_tensors is not None:
            return
        if entry.tensors is None:
            return
        cpu_tensors = self._detach_to_cpu(entry.tensors)
        bio = io.BytesIO()
        torch.save({"k": cpu_tensors[0], "v": cpu_tensors[1]}, bio)
        raw = bio.getvalue()
        comp = zlib.compress(raw, level=self.compression_level)
        self._update_usage_for_entry(entry, remove=True)
        entry.tensors = None
        entry.compressed = comp
        entry.compressed_bytes = len(comp)
        entry.location = "compressed"
        self._update_usage_for_entry(entry, remove=False)
        self._compressions += 1

    def maybe_evict(self, layer_index: int) -> None:
        """Attempt to free GPU/CPU memory for a given layer by compressing it."""
        entry = self._entries.get(layer_index)
        if entry is None:
            return
        if entry.tensors is None and entry.location != "compressed":
            return
        if self.tiering and entry.location == "device":
            self._offload_to_host(entry)
        if self.compression and entry.location != "compressed":
            self._compress_entry(entry)
        self._record_snapshot()

    def _enforce_budget(self) -> None:
        if self.budget_bytes is None:
            return
        usage = self._device_bytes + self._host_bytes + self._compressed_bytes + self._disk_bytes
        if usage <= self.budget_bytes:
            return
        items = sorted(self._entries.items(), key=lambda kv: kv[1].last_access)
        for layer_index, entry in items:
            if usage <= self.budget_bytes:
                break
            if entry.location == "disk":
                continue
            if self.tiering and entry.location == "device":
                self._offload_to_host(entry)
            if self.compression and entry.location not in ("compressed", "disk"):
                self._compress_entry(entry)
            # Tier 4: disk spill as last resort.
            if self.disk_dir is not None and entry.location == "compressed":
                self._spill_to_disk(layer_index, entry)
            usage = (
                self._device_bytes + self._host_bytes
                + self._compressed_bytes + self._disk_bytes
            )
        if usage > self.budget_bytes:
            self._budget_violations += 1
        self._record_snapshot()

    def _enforce_device_budget(self) -> None:
        if self.device_budget_bytes is None:
            return
        if self._device_bytes <= self.device_budget_bytes:
            return
        if not self.tiering:
            return
        items = sorted(self._entries.items(), key=lambda kv: kv[1].last_access)
        for _, entry in items:
            if self._device_bytes <= self.device_budget_bytes:
                break
            if entry.location == "device":
                self._offload_to_host(entry)
        self._record_snapshot()

    def stats(self) -> dict:
        total = (
            self._device_bytes + self._host_bytes
            + self._compressed_bytes + self._disk_bytes
        )
        return {
            "entries": len(self._entries),
            "device_bytes": int(self._device_bytes),
            "host_bytes": int(self._host_bytes),
            "compressed_bytes": int(self._compressed_bytes),
            "disk_bytes": int(self._disk_bytes),
            "total_bytes": int(total),
            "peak_device_bytes": int(self._peak_device_bytes),
            "peak_host_bytes": int(self._peak_host_bytes),
            "peak_total_bytes": int(self._peak_total_bytes),
            "compressions": self._compressions,
            "decompressions": self._decompressions,
            "offloads": self._offloads,
            "moves_to_device": self._moves_to_device,
            "disk_spills": self._disk_spills,
            "disk_loads": self._disk_loads,
            "budget_violations": self._budget_violations,
            "budget_bytes": int(self.budget_bytes),
            "device_budget_bytes": int(self.device_budget_bytes),
        }

    def clear(self) -> None:
        """Clear all cached entries; delete any spill files."""
        try:
            # Clean up disk spill files.
            for entry in self._entries.values():
                if entry.disk_path:
                    try:
                        Path(entry.disk_path).unlink(missing_ok=True)
                    except Exception:
                        pass
            self._entries.clear()
            self._device_bytes = 0
            self._host_bytes = 0
            self._compressed_bytes = 0
            self._disk_bytes = 0
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception:
            return
