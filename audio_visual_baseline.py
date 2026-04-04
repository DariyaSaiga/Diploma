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

    def forward(self, audio, visual):
        # audio:  (B, 50, 74)
        # visual: (B, 50, 713)

        # Проецируем каждый временной шаг
        audio_emb  = self.audio_proj(audio)    # (B, 50, 128)
        visual_emb = self.visual_proj(visual)  # (B, 50, 128)

        # Mean pooling по временной оси (dim=1)
        audio_emb  = audio_emb.mean(dim=1)   # (B, 128)
        visual_emb = visual_emb.mean(dim=1)  # (B, 128)

        # Склеиваем и классифицируем
        fused  = torch.cat([audio_emb, visual_emb], dim=1)  # (B, 256)
        logits = self.classifier(fused)                      # (B, 6)

        return logits