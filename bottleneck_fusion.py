import torch
import torch.nn as nn
from transformers import BertModel


class BottleneckModel(nn.Module):
    def __init__(self, num_classes=6, hidden_dim=128, bottleneck_dim=64, dropout=0.3):
        super().__init__()

        # 🔹 TEXT (BERT)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.text_proj = nn.Linear(768, hidden_dim)

        # 🔹 AUDIO
        self.audio_proj = nn.Sequential(
            nn.Linear(74, hidden_dim),
            nn.ReLU(),
        )

        # 🔹 VISUAL
        self.visual_proj = nn.Sequential(
            nn.Linear(713, hidden_dim),
            nn.ReLU(),
        )

        # 🔥 BOTTLENECK (сжатие)
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim * 3, bottleneck_dim),
            nn.ReLU(),
        )

        # 🔹 CLASSIFIER
        self.classifier = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids, attention_mask, audio, visual):
        # ---------------- TEXT ----------------
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS токен
        text_emb = bert_out.last_hidden_state[:, 0, :]  # (B, 768)
        text_emb = self.text_proj(text_emb)             # (B, hidden_dim)

        # ---------------- AUDIO ----------------
        audio_emb = self.audio_proj(audio)              # (B, T, hidden_dim)
        audio_emb = audio_emb.mean(dim=1)               # (B, hidden_dim)

        # ---------------- VISUAL ----------------
        visual_emb = self.visual_proj(visual)           # (B, T, hidden_dim)
        visual_emb = visual_emb.mean(dim=1)             # (B, hidden_dim)

        # ---------------- FUSION ----------------
        fused = torch.cat([text_emb, audio_emb, visual_emb], dim=1)  # (B, hidden_dim*3)

        bottleneck = self.bottleneck(fused)  # (B, bottleneck_dim)

        logits = self.classifier(bottleneck)

        return logits