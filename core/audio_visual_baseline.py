import torch
import torch.nn as nn
from core.encoders import AudioCNNEncoder, VideoBiLSTMEncoder


class AudioVisualBaseline(nn.Module):
    """Audio + Visual baseline using 1D-CNN (audio) and BiLSTM (video).

    Architecture:
        audio  → AudioCNNEncoder  → masked mean → audio_vec  (B, 128)
        visual → VideoBiLSTMEncoder → masked mean → visual_vec (B, 256)
        concat(audio_vec, visual_vec) → classifier → logits
    """

    def __init__(self, num_classes=6, hidden_dim=128, dropout=0.3,
                 audio_input_dim=74, visual_input_dim=713):
        super().__init__()
        lstm_hidden = hidden_dim  # BiLSTM output = hidden_dim * 2 = 256
        self.audio_encoder = AudioCNNEncoder(audio_input_dim, hidden_dim, dropout)
        self.visual_encoder = VideoBiLSTMEncoder(visual_input_dim, lstm_hidden, dropout)

        # audio: hidden_dim=128, visual: lstm_hidden*2=256 → concat=384
        fusion_dim = hidden_dim + lstm_hidden * 2
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    @staticmethod
    def _masked_mean(seq, mask=None):
        if mask is None:
            return seq.mean(dim=1)
        m = mask.unsqueeze(-1).float()
        return (seq * m).sum(1) / m.sum(1).clamp(min=1)

    def forward(self, batch, return_domains=False):
        audio = batch["audio"]
        audio_mask = batch.get("audio_mask")
        visual = batch["visual"]
        visual_mask = batch.get("visual_mask")

        audio_seq = self.audio_encoder(audio)      # (B, T, 128)
        visual_seq = self.visual_encoder(visual)   # (B, T, 256)

        audio_vec = self._masked_mean(audio_seq, audio_mask)    # (B, 128)
        visual_vec = self._masked_mean(visual_seq, visual_mask)  # (B, 256)

        fused = torch.cat([audio_vec, visual_vec], dim=-1)  # (B, 384)
        logits = self.classifier(fused)                      # (B, num_classes)

        if return_domains:
            return logits, {}
        return logits
