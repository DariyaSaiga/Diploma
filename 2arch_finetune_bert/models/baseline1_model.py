"""
Baseline 1: Late Fusion — BERT (last layer fine-tuned) + 1D-CNN (audio) + BiLSTM (vision)

Архитектура:
  - Text:   BERT, заморожен кроме последнего слоя (encoder.layer[-1])
            CLS токен последнего слоя → Linear(768→128)
  - Audio:  COVAREP [B,60,74] → Conv1D → BN → ReLU → mean pool → [B,128]
  - Vision: OpenFace [B,60,35] → BiLSTM(hidden=64, bidir) → mean pool → [B,128]
  - Fusion: cat([text, audio, vision]) → Linear(384→6) — одна линейная проекция

Ключевые отличия от Proposed:
  1. BERT: fine-tune только последнего слоя (Proposed — fine-tune N слоёв bottleneck)
  2. Vision: BiLSTM вместо Conv1D (разная модальная архитектура)
  3. Fusion: простая конкатенация без cross-modal attention
"""

import torch
import torch.nn as nn
from transformers import BertModel

HIDDEN_DIM = 128
DROPOUT    = 0.1
N_EMOTIONS = 6
BERT_MODEL = "bert-base-uncased"


class ModalityProjection(nn.Module):
    """Conv1D проекция аудио-признаков → HIDDEN_DIM."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(input_dim, HIDDEN_DIM, kernel_size=3, padding=1),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] → [B, T, HIDDEN_DIM]
        return self.proj(x.transpose(1, 2)).transpose(1, 2)


class LateFusionModel(nn.Module):
    """
    Baseline 1: независимые энкодеры + простая конкатенация.

    BERT: заморожен, кроме последнего трансформер-блока (encoder.layer[-1]).
    Text-фича = CLS токен последнего слоя BERT → Linear(768→128).

    Fusion = cat(text_cls, audio_pool, vision_pool) → Linear(384→6).
    Нет cross-modal attention — ключевое отличие от Baseline 2 и Proposed.
    """

    def __init__(self):
        super().__init__()

        # ── BERT: freeze all → unfreeze last transformer layer ────────────────
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        for param in self.bert.parameters():
            param.requires_grad = False
        # Разморозить только последний блок (слой 11 у bert-base)
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True

        self.text_proj = nn.Linear(768, HIDDEN_DIM)

        # ── Audio: 1D-CNN → mean pool ─────────────────────────────────────────
        self.audio_encoder = ModalityProjection(input_dim=74)

        # ── Vision: BiLSTM → mean pool (отличие от Proposed) ─────────────────
        # hidden=64, bidir=True → output=128=HIDDEN_DIM
        self.vision_lstm = nn.LSTM(
            input_size=35,
            hidden_size=HIDDEN_DIM // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT,
        )

        self.dropout = nn.Dropout(DROPOUT)

        # ── Fusion: concat → одна линейная проекция (без ReLU/скрытого слоя) ─
        self.classifier = nn.Linear(HIDDEN_DIM * 3, N_EMOTIONS)

        # ── Auxiliary heads (для auxiliary losses в train_baseline1.py) ───────
        self.head_text   = nn.Linear(HIDDEN_DIM, N_EMOTIONS)
        self.head_audio  = nn.Linear(HIDDEN_DIM, N_EMOTIONS)
        self.head_vision = nn.Linear(HIDDEN_DIM, N_EMOTIONS)

    def forward(
        self,
        input_ids:      torch.Tensor,   # [B, 50]
        attention_mask: torch.Tensor,   # [B, 50]
        audio:          torch.Tensor,   # [B, 60, 74]
        vision:         torch.Tensor,   # [B, 60, 35]
        audio_mask:     torch.Tensor,   # [B, 60]  (не используется, но нужен для совместимости)
        vision_mask:    torch.Tensor,   # [B, 60]  (не используется, но нужен для совместимости)
    ):
        # ── Text: CLS токен последнего слоя BERT ─────────────────────────────
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_state: [B, 50, 768]
        text_cls = bert_out.last_hidden_state[:, 0, :]       # [B, 768]
        text_cls = self.dropout(self.text_proj(text_cls))    # [B, 128]

        # ── Audio: Conv1D → mean pool ─────────────────────────────────────────
        audio_pool = self.audio_encoder(audio).mean(dim=1)   # [B, 128]

        # ── Vision: BiLSTM → mean pool ────────────────────────────────────────
        vision_out, _ = self.vision_lstm(vision)              # [B, 60, 128]
        vision_pool   = vision_out.mean(dim=1)                # [B, 128]

        # ── Late fusion: concat → единственный линейный слой ─────────────────
        fused = self.dropout(
            torch.cat([text_cls, audio_pool, vision_pool], dim=-1)  # [B, 384]
        )

        return (
            self.classifier(fused),          # logits_fuse   [B, 6]
            self.head_text(text_cls),         # logits_text   [B, 6]
            self.head_audio(audio_pool),      # logits_audio  [B, 6]
            self.head_vision(vision_pool),    # logits_vision [B, 6]
        )


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
    frozen  = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Параметры всего           : {total:,}")
    print(f"  BERT всего              : {bert_p:,}")
    print(f"  BERT frozen             : {frozen - (total - bert_p - train_p):,}")
    print(f"  Обучаемые (всего)       : {train_p:,}")

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
