"""Shared encoder / pooling / attention blocks for all three models.

Conventions:
    Inputs are (B, T, D) tensors. Modality masks are (B, T) bool, True = valid.
    Outputs are (B, T, D) sequence tensors or (B, D) pooled vectors.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def make_mask_from_features(x: torch.Tensor) -> torch.Tensor:
    """Default mask: True where the feature row is not all-zero (padding)."""
    return x.abs().sum(dim=-1) > 0


def masked_mean(seq: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean over time, ignoring padded steps.

    seq: (B, T, D); mask: (B, T) bool. Returns (B, D).
    """
    if mask is None:
        return seq.mean(dim=1)
    m = mask.unsqueeze(-1).to(seq.dtype)
    summed = (seq * m).sum(dim=1)
    denom = m.sum(dim=1).clamp(min=1.0)
    return summed / denom


class MeanPooling(nn.Module):
    def forward(self, seq: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return masked_mean(seq, mask)


class AttentionPooling(nn.Module):
    """Additive attention pooling: returns weighted sum over T."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, seq: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = self.score(seq).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B, T, 1)
        return (seq * weights).sum(dim=1)


class ProjectionLayer(nn.Module):
    """Linear projection with LayerNorm + dropout."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.norm(self.proj(x)))


class TextBiLSTMEncoder(nn.Module):
    """BiLSTM over GloVe-like text features. (B, T, in_dim) -> (B, T, hidden_dim*2)."""

    def __init__(self, in_dim: int = 300, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_dim * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.dropout(out)


class VisionBiLSTMEncoder(nn.Module):
    """BiLSTM over Facet visual features. (B, T, in_dim) -> (B, T, hidden_dim*2)."""

    def __init__(self, in_dim: int = 35, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_dim * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.dropout(out)


class AudioCNNEncoder(nn.Module):
    """1D-CNN over COVAREP audio features. (B, T, in_dim) -> (B, T, out_dim)."""

    def __init__(self, in_dim: int = 74, out_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, D) -> (B, D, T) -> conv -> (B, T, D)
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention with residual + feed-forward."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        kv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # MultiheadAttention key_padding_mask: True = ignore
        kpm = ~kv_mask if kv_mask is not None else None
        q = self.norm_q(query)
        k = self.norm_kv(kv)
        attn_out, _ = self.attn(q, k, k, key_padding_mask=kpm, need_weights=False)
        x = query + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.norm_ff(x)))
        return x


class SelfAttentionEncoder(nn.Module):
    """Standard pre-norm Transformer encoder stack (self-attention)."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        ff_mult: int = 2,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        kpm = ~mask if mask is not None else None
        return self.encoder(x, src_key_padding_mask=kpm)


class BottleneckBlock(nn.Module):
    """One layer of bottleneck fusion.

    Each modality has self-attention. A shared bank of bottleneck tokens
    is updated by attending to each modality, then each modality attends
    back to the (updated) bottleneck. This restricts cross-modal flow to
    a small set of tokens.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        modalities: tuple[str, ...] = ("text", "audio", "vision"),
    ):
        super().__init__()
        self.modalities = modalities
        # Self-attention per modality
        self.self_attn = nn.ModuleDict(
            {m: SelfAttentionEncoder(hidden_dim, num_layers=1,
                                     num_heads=num_heads, dropout=dropout)
             for m in modalities}
        )
        # bottleneck <- modality (collect)
        self.bn_from_mod = nn.ModuleDict(
            {m: CrossAttentionBlock(hidden_dim, num_heads=num_heads, dropout=dropout)
             for m in modalities}
        )
        # modality <- bottleneck (broadcast)
        self.mod_from_bn = nn.ModuleDict(
            {m: CrossAttentionBlock(hidden_dim, num_heads=num_heads, dropout=dropout)
             for m in modalities}
        )

    def forward(
        self,
        seqs: dict[str, torch.Tensor],
        masks: dict[str, Optional[torch.Tensor]],
        bottleneck: torch.Tensor,
    ):
        # 1) self-attention inside each modality
        seqs = {
            m: self.self_attn[m](seqs[m], masks.get(m))
            for m in self.modalities
        }
        # 2) bottleneck collects info from each modality (sequentially)
        bn = bottleneck
        for m in self.modalities:
            bn = self.bn_from_mod[m](bn, seqs[m], kv_mask=masks.get(m))
        # 3) each modality reads from updated bottleneck
        seqs = {
            m: self.mod_from_bn[m](seqs[m], bn)  # bottleneck has no mask
            for m in self.modalities
        }
        return seqs, bn
