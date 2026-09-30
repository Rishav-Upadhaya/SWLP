"""Native multi-token-prediction (MTP) self-drafting for Qwen3.5 checkpoints.

Qwen3.5 / Qwen3.8 dense checkpoints ship a 1-layer MTP head (``mtp.*``, which
HF transformers drops on load). Given the target's hidden state ``h_t`` at
position t and the token ``x_{t+1}``, it predicts ``x_{t+2}``:

    e   = pre_fc_norm_embedding(embed(x_{t+1}))
    h   = pre_fc_norm_hidden(h_t)
    z   = fc(cat([e, h], -1))              # embedding first, then hidden
    z   = decoder_layer(z)                 # full-attention layer, own KV cache
    out = norm(z);  logits = lm_head(out)  # lm_head shared with the target

Sources: vLLM ``qwen3_next_mtp.py`` (``Qwen3NextMultiTokenPredictor.forward``:
same order, GemmaRMSNorm i.e. ``(1 + w)`` norms, full-attention layer) and
mlx-lm PR #990 (``MTPModule``). ``h_t`` here is the target's **post-final-norm**
hidden, which is what vLLM feeds (its model forward returns normed states);
mlx-lm #990 feeds the pre-norm state instead — see the open question in the
PR report. Chained drafts (K > 1) feed the head's own normed output back as
the next ``h``, exactly as vLLM does for ``num_speculative_tokens > 1``.

MTP KV cache invariant: between calls it holds one entry per *confirmed*
position t, built from the true ``(h_t, x_{t+1})``. The runner hands every
newly confirmed pair to ``extend()``; ``propose()`` commits them in its first
forward, chains K-1 speculative entries, then crops those back off.
Positions count committed entries from 0: RoPE is relative, so a constant
offset from the target's absolute positions changes nothing.
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import PretrainedConfig
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
)

from ..model.shard import MTP_FILE
from .arch import _build_causal_mask, rope_embeddings
from .draft import AdaptiveDraftLength

_MTP_PREFIX = "mtp."


class MtpHead(nn.Module):
    """The checkpoint's ``mtp.*`` module (names match with the prefix stripped)."""

    def __init__(self, text_config: PretrainedConfig) -> None:
        super().__init__()
        cfg = copy.deepcopy(text_config)
        cfg.layer_types = ["full_attention"]
        cfg.num_hidden_layers = 1
        self.config = cfg
        hidden, eps = cfg.hidden_size, cfg.rms_norm_eps
        self.fc = nn.Linear(hidden * 2, hidden, bias=False)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(cfg, 0)])
        self.norm = Qwen3_5RMSNorm(hidden, eps=eps)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(hidden, eps=eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(hidden, eps=eps)

    def forward(
        self,
        embeds: torch.Tensor,
        hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        cache: DynamicCache,
    ) -> torch.Tensor:
        fused = self.fc(
            torch.cat([self.pre_fc_norm_embedding(embeds), self.pre_fc_norm_hidden(hidden)], -1)
        )
        fused = self.layers[0](
            fused,
            position_embeddings=position_embeddings,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )
        return self.norm(fused)


def load_mtp_head(
    shard_dir: Path, text_config: PretrainedConfig, dtype: torch.dtype, device: torch.device
) -> MtpHead:
    """Load ``<shard_dir>/mtp.safetensors`` fully resident on ``device``."""
    path = Path(shard_dir) / MTP_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found — this shard dir has no MTP head. Re-shard a "
            "checkpoint that ships mtp.* weights (e.g. Qwen3.8-27B) or drop --mtp."
        )
    state = {k.removeprefix(_MTP_PREFIX): v for k, v in load_file(str(path)).items()}
    head = MtpHead(text_config)
    head.load_state_dict(state, strict=True)
    return head.to(device=device, dtype=dtype).eval()


class MtpDrafter(AdaptiveDraftLength):
    """Drafts up to K tokens per sweep with the target's own MTP head.

    ``model`` is the target (for its embeddings, rotary module and lm_head —
    all shared with the MTP head). Same ``propose`` / ``observe`` shape as
    ``DraftModelDrafter``; additionally the runner must ``extend()`` it with
    the target hidden states of every confirmed position.
    """

    def __init__(
        self, head: MtpHead, model: nn.Module, device: torch.device, max_draft: int
    ) -> None:
        super().__init__(max_draft)
        self._head = head
        self._model = model
        self._device = device
        self._cache = DynamicCache()
        self._pending_hidden: list[torch.Tensor] = []
        self._pending_ids: list[int] = []

    def extend(self, hidden: torch.Tensor, next_ids: list[int]) -> None:
        """Queue confirmed pairs: ``hidden[:, j]`` is h_t, ``next_ids[j]`` is x_{t+1}."""
        if int(hidden.shape[1]) != len(next_ids):
            raise ValueError(f"{hidden.shape[1]} hidden states vs {len(next_ids)} next ids")
        self._pending_hidden.append(hidden)
        self._pending_ids.extend(next_ids)

    def propose(self, tokens: list[int]) -> list[int]:
        """Return up to K draft tokens continuing the confirmed sequence.

        ``tokens`` is accepted for interface parity; the MTP head drafts from
        the hidden states fed through ``extend()``, not from token history.
        """
        del tokens
        if self._k == 0 or not self._pending_ids:
            return []
        hidden = torch.cat(self._pending_hidden, dim=1)
        ids = self._pending_ids
        self._pending_hidden, self._pending_ids = [], []
        committed = self._cache.get_seq_length() + len(ids)
        drafted: list[int] = []
        with torch.no_grad():
            for _ in range(self._k):
                out = self._forward(hidden, ids)
                logits = self._model.lm_head(out[:, -1:, :])
                drafted.append(int(logits[0, -1].argmax().item()))
                hidden, ids = out[:, -1:, :], drafted[-1:]
        # Drop the speculative chain entries; keep only confirmed positions.
        extra = self._cache.get_seq_length() - committed
        if extra > 0:
            self._cache.crop(-extra)
        return drafted

    def _forward(self, hidden: torch.Tensor, ids: list[int]) -> torch.Tensor:
        embed = self._model.get_input_embeddings()
        id_tensor = torch.tensor([ids], dtype=torch.long, device=embed.weight.device)
        embeds = embed(id_tensor).to(self._device)
        start = self._cache.get_seq_length()
        position_ids = torch.arange(
            start, start + len(ids), device=self._device, dtype=torch.long
        ).unsqueeze(0)
        mask = _build_causal_mask(self._head.config, embeds, self._cache, position_ids)
        return self._head(
            embeds,
            hidden.to(device=self._device, dtype=embeds.dtype),
            rope_embeddings(self._model.model, embeds, position_ids),
            mask,
            position_ids,
            self._cache,
        )
