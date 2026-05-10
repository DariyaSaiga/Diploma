"""Model 1 — Late Fusion CNN-BiLSTM baseline.

Architecture:
    Text   (B,60,300) -> BiLSTM(128, bidir) -> masked mean -> (B,256)
    Audio  (B,60, 74) -> Conv1d->Conv1d     -> masked mean -> (B,128)
    Vision (B,60, 35) -> BiLSTM(128, bidir) -> masked mean -> (B,256)
    concat (B,640) -> FC(640,256) -> ReLU -> Dropout -> FC(256,6) -> logits

No sigmoid inside the model. Loss: BCEWithLogitsLoss.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.common_layers import (
    AudioCNNEncoder,
    TextBiLSTMEncoder,
    VisionBiLSTMEncoder,
    make_mask_from_features,
    masked_mean,
)


class LateFusionCNNBiLSTM(nn.Module):
    def __init__(
        self,
        num_classes: int = 6,
        text_dim: int = 300,
        audio_dim: int = 74,
        vision_dim: int = 35,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        **_unused,
    ):
        super().__init__()
        self.text_encoder = TextBiLSTMEncoder(text_dim, hidden_dim, dropout)
        self.audio_encoder = AudioCNNEncoder(audio_dim, hidden_dim, dropout)
        self.vision_encoder = VisionBiLSTMEncoder(vision_dim, hidden_dim, dropout)

        fusion_in = self.text_encoder.out_dim + self.audio_encoder.out_dim + self.vision_encoder.out_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        text = batch["text"]
        audio = batch["audio"]
        vision = batch["vision"]

        text_mask = make_mask_from_features(text)
        audio_mask = make_mask_from_features(audio)
        vision_mask = make_mask_from_features(vision)

        text_seq = self.text_encoder(text)        # (B, T, 256)
        audio_seq = self.audio_encoder(audio)     # (B, T, 128)
        vision_seq = self.vision_encoder(vision)  # (B, T, 256)

        text_vec = masked_mean(text_seq, text_mask)
        audio_vec = masked_mean(audio_seq, audio_mask)
        vision_vec = masked_mean(vision_seq, vision_mask)

        fused = torch.cat([text_vec, audio_vec, vision_vec], dim=-1)
        return self.classifier(fused)
