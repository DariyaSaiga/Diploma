"""
Baseline 1: Late Fusion — BERT (frozen) + 1D-CNN (audio) + BiLSTM (vision)

Архитектура:
  - Text:   BERT frozen, avg последних 4 слоёв → Linear(768→128) → CLS токен
  - Audio:  COVAREP [B,60,74] → Conv1D → BN → ReLU → mean pool
  - Vision: OpenFace [B,60,35] → BiLSTM(hidden=64, bidir=True) → mean pool
  - Fusion: cat([text, audio, vision]) → FF classifier (без cross-modal attention)

Такой же BERT и training protocol что у Proposed → разница только в fusion.
"""

import torch
import torch.nn as nn
import math
from transformers import BertModel

HIDDEN_DIM   = 128
DROPOUT      = 0.1
N_EMOTIONS   = 6
BERT_MODEL   = "bert-base-uncased"


class ModalityProjection(nn.Module):
    """Conv1D проекция одной модальности в HIDDEN_DIM (то же что в Proposed)."""
    def __init__(self, input_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(input_dim, HIDDEN_DIM, kernel_size=3, padding=1),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.transpose(1, 2)   # [B, D, T]
        x = self.proj(x)         # [B, HIDDEN_DIM, T]
        x = x.transpose(1, 2)   # [B, T, HIDDEN_DIM]
        return x


class LateFusionModel(nn.Module):
    """
    Baseline 1: независимые энкодеры + конкатенация без cross-modal attention.
    Интерфейс совместим с BottleneckFusionModel — возвращает 4 логита.
    """

    def __init__(self):
        super().__init__()

        # ── Text: BERT frozen + linear projection (то же что Proposed) ────────
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        for param in self.bert.parameters():
            param.requires_grad = False

        self.text_proj = nn.Linear(768, HIDDEN_DIM)

        # ── Audio: 1D-CNN (Conv1D + BN + ReLU) → same as Proposed ────────────
        self.audio_encoder = ModalityProjection(input_dim=74)

        # ── Vision: BiLSTM — ключевое отличие от Proposed ────────────────────
        # hidden_size=64, bidirectional=True → output dim = 64*2 = 128 = HIDDEN_DIM
        self.vision_lstm = nn.LSTM(
            input_size=35,
            hidden_size=HIDDEN_DIM // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT,
        )

        self.dropout = nn.Dropout(DROPOUT)

        # ── Classifier: cat(text+audio+vision) → 6 логитов ───────────────────
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 3, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, N_EMOTIONS),
        )

        # ── Auxiliary heads (по одной на модальность) ─────────────────────────
        # Те же что в Proposed — для fair comparison и лучшей сходимости
        self.head_text   = nn.Linear(HIDDEN_DIM, N_EMOTIONS)
        self.head_audio  = nn.Linear(HIDDEN_DIM, N_EMOTIONS)
        self.head_vision = nn.Linear(HIDDEN_DIM, N_EMOTIONS)

    def forward(self, input_ids, attention_mask, audio, vision,
                audio_mask, vision_mask):
        B = input_ids.size(0)

        # ── Text: BERT avg 4 слоёв → linear → CLS токен ──────────────────────
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        text = torch.stack(bert_out.hidden_states[-4:], dim=0).mean(dim=0)  # [B,50,768]
        text = self.dropout(self.text_proj(text))                            # [B,50,128]
        text_cls = text[:, 0, :]                                             # [B,128]

        # ── Audio: 1D-CNN → mean pool ─────────────────────────────────────────
        audio_enc  = self.audio_encoder(audio)   # [B,60,128]
        audio_pool = audio_enc.mean(dim=1)       # [B,128]

        # ── Vision: BiLSTM → mean pool всех скрытых состояний ─────────────────
        vision_out, _ = self.vision_lstm(vision)  # [B,60,128]
        vision_pool   = vision_out.mean(dim=1)    # [B,128]

        # ── Late fusion: простая конкатенация → классификатор ────────────────
        fused = torch.cat([text_cls, audio_pool, vision_pool], dim=-1)  # [B,384]
        fused = self.dropout(fused)

        logits_fuse   = self.classifier(fused)
        logits_text   = self.head_text(text_cls)
        logits_audio  = self.head_audio(audio_pool)
        logits_vision = self.head_vision(vision_pool)

        return logits_fuse, logits_text, logits_audio, logits_vision


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
    from dataset import get_dataloaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loaders, pos_weight = get_dataloaders()
    model = LateFusionModel().to(device)

    total   = sum(p.numel() for p in model.parameters())
    bert_p  = sum(p.numel() for p in model.bert.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Параметры всего      : {total:,}")
    print(f"  BERT (frozen)      : {bert_p:,}")
    print(f"  Обучаемые          : {train_p:,}")

    batch = next(iter(loaders["train"]))
    with torch.no_grad():
        out = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["audio"].to(device),
            batch["vision"].to(device),
            batch["audio_mask"].to(device),
            batch["vision_mask"].to(device),
        )
    print(f"logits_fuse  : {out[0].shape}")
    print(f"logits_text  : {out[1].shape}")
    print(f"logits_audio : {out[2].shape}")
    print(f"logits_vision: {out[3].shape}")
    print("✅ baseline1_model.py работает")
