"""
simple_fusion.py — простой concat-fusion без bottleneck.

Архитектура:
    Text:   BERT frozen → CLS → Linear(768→128)
    Audio:  masked mean pool → Linear(74→128)
    Visual: masked mean pool → Linear(713→128)
    Fusion: cat[text, audio, visual] → Linear(384→256) → ReLU → Dropout → Linear(256→6)

Это уровень 2 в сравнительной таблице:
"можно ли добиться того же результата просто склеив модальности без bottleneck?"
"""

import torch
import torch.nn as nn
from transformers import BertModel


class SimpleFusion(nn.Module):
    def __init__(self, num_classes: int = 6, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()

        # ── Text: BERT (заморожен) + проекция CLS ─────────────────────────
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for p in self.bert.parameters():
            p.requires_grad = False

        self.text_proj = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # ── Audio ─────────────────────────────────────────────────────────
        self.audio_proj = nn.Sequential(
            nn.Linear(74, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # ── Visual ────────────────────────────────────────────────────────
        self.visual_proj = nn.Sequential(
            nn.Linear(713, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # ── Classifier: cat(3×128) = 384 → 6 ─────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    @staticmethod
    def _masked_mean(seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """seq:(B,T,D)  mask:(B,T) 1=real → (B,D)"""
        m = mask.unsqueeze(-1).float()                  # (B,T,1)
        return (seq * m).sum(1) / m.sum(1).clamp(min=1)

    def forward(
        self,
        input_ids: torch.Tensor,        # (B, Lt)
        attention_mask: torch.Tensor,   # (B, Lt)
        audio: torch.Tensor,            # (B, Ta, 74)
        audio_mask: torch.Tensor,       # (B, Ta)
        visual: torch.Tensor,           # (B, Tv, 713)
        visual_mask: torch.Tensor,      # (B, Tv)
    ) -> torch.Tensor:                  # (B, 6)

        # Text: CLS токен
        bert_out  = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls       = bert_out.last_hidden_state[:, 0, :]   # (B, 768)
        text_vec  = self.text_proj(cls)                   # (B, 128)

        # Audio: masked mean
        audio_emb  = self.audio_proj(audio)               # (B, Ta, 128)
        audio_vec  = self._masked_mean(audio_emb, audio_mask)  # (B, 128)

        # Visual: masked mean
        visual_emb = self.visual_proj(visual)             # (B, Tv, 128)
        visual_vec = self._masked_mean(visual_emb, visual_mask)  # (B, 128)

        # Concat + classify
        fused  = torch.cat([text_vec, audio_vec, visual_vec], dim=1)  # (B, 384)
        return self.classifier(fused)                                   # (B, 6)
