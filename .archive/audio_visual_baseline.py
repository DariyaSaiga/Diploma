import torch
import torch.nn as nn


class AudioVisualBaseline(nn.Module):
    def __init__(self, num_classes=6, hidden_dim=128, dropout=0.3):
        super().__init__()

        # Аудио ветка: (B, 50, 74) → (B, 50, 128) → (B, 128)
        self.audio_proj = nn.Sequential(
            nn.Linear(74, hidden_dim),
            nn.ReLU(),
        )

        # Визуальная ветка: (B, 50, 713) → (B, 50, 128) → (B, 128)
        self.visual_proj = nn.Sequential(
            nn.Linear(713, hidden_dim),
            nn.ReLU(),
        )

        # Классификатор: два вектора по 128 склеиваем → 256 → 6
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    @staticmethod
    def _masked_mean(seq: torch.Tensor, mask=None) -> torch.Tensor:
        """seq:(B,T,D)  mask:(B,T) 1=real, None=global mean → (B,D)"""
        if mask is None:
            return seq.mean(dim=1)
        m = mask.unsqueeze(-1).float()
        return (seq * m).sum(1) / m.sum(1).clamp(min=1)

    def forward(self, audio, visual, audio_mask=None, visual_mask=None):
        # audio:  (B, T, 74)
        # visual: (B, T, 713)

        audio_emb  = self.audio_proj(audio)    # (B, T, 128)
        visual_emb = self.visual_proj(visual)  # (B, T, 128)

        audio_emb  = self._masked_mean(audio_emb,  audio_mask)   # (B, 128)
        visual_emb = self._masked_mean(visual_emb, visual_mask)  # (B, 128)

        fused  = torch.cat([audio_emb, visual_emb], dim=1)  # (B, 256)
        logits = self.classifier(fused)                      # (B, 6)
        return logits