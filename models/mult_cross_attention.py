"""Model 2 — MulT-like Cross-Modal Attention.

Each modality is projected to a shared hidden dim, gets self-attention,
then six pairwise cross-attention streams are computed:
    T<-A, T<-V, A<-T, A<-V, V<-T, V<-A.

For each modality we sum the two cross-attended views with the self
representation, mean-pool, concat all three, and classify.

No sigmoid inside. Loss: BCEWithLogitsLoss.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from models.common_layers import (
    CrossAttentionBlock,
    SelfAttentionEncoder,
    make_mask_from_features,
    masked_mean,
)


class _ModalityStream(nn.Module):
    """Project + self-attention encoder for one modality."""

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


class _CrossStack(nn.Module):
    """Stacked cross-attention from query modality to a key/value modality."""

    def __init__(self, hidden_dim: int, num_heads: int, num_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossAttentionBlock(hidden_dim, num_heads=num_heads, dropout=dropout)
             for _ in range(num_layers)]
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor, kv_mask: Optional[torch.Tensor]) -> torch.Tensor:
        for layer in self.layers:
            q = layer(q, kv, kv_mask=kv_mask)
        return q


class MulTCrossAttention(nn.Module):
    def __init__(
        self,
        num_classes: int = 6,
        text_dim: int = 300,
        audio_dim: int = 74,
        vision_dim: int = 35,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        **_unused,
    ):
        super().__init__()
        self.text_stream = _ModalityStream(text_dim, hidden_dim, num_heads, num_layers, dropout)
        self.audio_stream = _ModalityStream(audio_dim, hidden_dim, num_heads, num_layers, dropout)
        self.vision_stream = _ModalityStream(vision_dim, hidden_dim, num_heads, num_layers, dropout)

        # Six cross-attention streams (query <- key/value)
        def cross():
            return _CrossStack(hidden_dim, num_heads, num_layers, dropout)

        self.t_from_a = cross()
        self.t_from_v = cross()
        self.a_from_t = cross()
        self.a_from_v = cross()
        self.v_from_t = cross()
        self.v_from_a = cross()

        concat_dim = hidden_dim * 3
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        text = batch["text"]
        audio = batch["audio"]
        vision = batch["vision"]

        m_t = make_mask_from_features(text)
        m_a = make_mask_from_features(audio)
        m_v = make_mask_from_features(vision)

        t = self.text_stream(text, m_t)
        a = self.audio_stream(audio, m_a)
        v = self.vision_stream(vision, m_v)

        # Cross views: each modality fused with the other two
        t_view = t + self.t_from_a(t, a, kv_mask=m_a) + self.t_from_v(t, v, kv_mask=m_v)
        a_view = a + self.a_from_t(a, t, kv_mask=m_t) + self.a_from_v(a, v, kv_mask=m_v)
        v_view = v + self.v_from_t(v, t, kv_mask=m_t) + self.v_from_a(v, a, kv_mask=m_a)

        t_vec = masked_mean(t_view, m_t)
        a_vec = masked_mean(a_view, m_a)
        v_vec = masked_mean(v_view, m_v)

        fused = torch.cat([t_vec, a_vec, v_vec], dim=-1)
        return self.classifier(fused)
