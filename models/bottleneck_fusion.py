"""Model 3 — Bottleneck Fusion (DBA / XMBT-style).

Each modality is projected to a shared hidden dim and processed by a
self-attention encoder. A small bank of learnable bottleneck tokens
mediates ALL cross-modal exchange (no direct token-to-token attention
between modalities). Modalities can attend back to the bottleneck.

Optional domain separation:
    Each modality is split into an INVARIANT branch and a PRIVATE branch.
    The bottleneck operates on the private branch (modality-specific bits
    that benefit from controlled exchange). The invariant branch is aligned
    across modalities.

    Auxiliary losses (computed in training/losses.py):
        - separation: invariant ⊥ private (cosine -> 0) per modality
        - invariant : MSE alignment between modality invariants
        - reconstruction: [inv | priv] -> original pooled feature

Returns logits [B, 6]. No sigmoid inside.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.common_layers import (
    BottleneckBlock,
    SelfAttentionEncoder,
    make_mask_from_features,
    masked_mean,
)


class _Branch(nn.Module):
    """Project + self-attention for one (modality, branch)."""

    def __init__(self, in_dim: int, hidden_dim: int, num_heads: int, num_layers: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.encoder = SelfAttentionEncoder(
            hidden_dim, num_layers=num_layers, num_heads=num_heads, dropout=dropout
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.norm(self.proj(x))
        return self.encoder(x, mask=mask)


class BottleneckFusion(nn.Module):
    def __init__(
        self,
        num_classes: int = 6,
        text_dim: int = 300,
        audio_dim: int = 74,
        vision_dim: int = 35,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        num_bottleneck_tokens: int = 16,
        dropout: float = 0.3,
        use_domain_sep: bool = False,
        **_unused,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_bottleneck_tokens = num_bottleneck_tokens
        self.use_domain_sep = use_domain_sep
        self.modalities = ("text", "audio", "vision")
        in_dims = {"text": text_dim, "audio": audio_dim, "vision": vision_dim}

        # Private branches (always present — used by bottleneck)
        self.private = nn.ModuleDict({
            m: _Branch(in_dims[m], hidden_dim, num_heads, num_layers, dropout)
            for m in self.modalities
        })

        # Optional invariant branches
        if use_domain_sep:
            self.invariant = nn.ModuleDict({
                m: _Branch(in_dims[m], hidden_dim, num_heads, num_layers, dropout)
                for m in self.modalities
            })
            self.recon_heads = nn.ModuleDict({
                m: nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim * 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 2, in_dims[m]),
                )
                for m in self.modalities
            })

        # Bottleneck tokens + bottleneck blocks
        self.bottleneck_tokens = nn.Parameter(
            torch.randn(num_bottleneck_tokens, hidden_dim) * 0.02
        )
        self.blocks = nn.ModuleList([
            BottleneckBlock(hidden_dim, num_heads=num_heads,
                            dropout=dropout, modalities=self.modalities)
            for _ in range(num_layers)
        ])

        # Classifier head
        per_mod = hidden_dim * 2 if use_domain_sep else hidden_dim
        fusion_dim = per_mod * len(self.modalities) + hidden_dim  # + pooled bottleneck
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def _build_inputs(self, batch: dict):
        seqs_raw = {
            "text": batch["text"],
            "audio": batch["audio"],
            "vision": batch["vision"],
        }
        masks = {m: make_mask_from_features(seqs_raw[m]) for m in self.modalities}
        return seqs_raw, masks

    def forward(
        self,
        batch: dict,
        return_domains: bool = False,
    ):
        seqs_raw, masks = self._build_inputs(batch)
        B = seqs_raw["text"].size(0)

        # Private encoders
        priv_seqs = {m: self.private[m](seqs_raw[m], masks[m]) for m in self.modalities}

        # Bottleneck stack works on private branches
        bn = self.bottleneck_tokens.unsqueeze(0).expand(B, -1, -1).contiguous()
        seqs = priv_seqs
        for block in self.blocks:
            seqs, bn = block(seqs, masks, bn)

        priv_pooled = {m: masked_mean(seqs[m], masks[m]) for m in self.modalities}
        bn_vec = bn.mean(dim=1)  # bottleneck has no padding

        if not self.use_domain_sep:
            fused = torch.cat(
                [priv_pooled["text"], priv_pooled["audio"], priv_pooled["vision"], bn_vec],
                dim=-1,
            )
            logits = self.classifier(fused)
            if return_domains:
                return logits, {}, {}
            return logits

        # Domain separation: independent invariant branches
        inv_seqs = {m: self.invariant[m](seqs_raw[m], masks[m]) for m in self.modalities}
        inv_pooled = {m: masked_mean(inv_seqs[m], masks[m]) for m in self.modalities}

        fused = torch.cat(
            [
                torch.cat([inv_pooled[m], priv_pooled[m]], dim=-1)
                for m in self.modalities
            ] + [bn_vec],
            dim=-1,
        )
        logits = self.classifier(fused)

        domain_data: Dict[str, torch.Tensor] = {}
        recon_data: Dict[str, torch.Tensor] = {}
        for m in self.modalities:
            domain_data[f"{m}_inv_pool"] = inv_pooled[m]
            domain_data[f"{m}_priv_pool"] = priv_pooled[m]

            joint = torch.cat([inv_pooled[m], priv_pooled[m]], dim=-1)
            recon_data[f"{m}_recon"] = self.recon_heads[m](joint)
            recon_data[f"{m}_original"] = masked_mean(seqs_raw[m], masks[m])

        if return_domains:
            return logits, domain_data, recon_data
        return logits
