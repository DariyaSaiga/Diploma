"""
Baseline 2: MulT-style Cross-Modal Attention Fusion

Архитектура (по Tsai et al., 2019 — Multimodal Transformer):
  - Text:   BERT frozen, avg 4 слоёв → Linear(768→128) + pos enc
  - Audio:  COVAREP [B,60,74] → Conv1D(74→128) + pos enc
  - Vision: OpenFace [B,60,35] → Conv1D(35→128) + pos enc
  - Fusion: каждая модальность крест-внимательно посещает две другие
            T' = T + CrossAttn(Q=T, KV=A) + CrossAttn(Q=T, KV=V)
            A' = A + CrossAttn(Q=A, KV=T) + CrossAttn(Q=A, KV=V)
            V' = V + CrossAttn(Q=V, KV=T) + CrossAttn(Q=V, KV=A)
  - Pool: CLS (text), mean (audio, vision) → cat → FF classifier

Ключевое отличие от Proposed: полный pairwise cross-attention (O(T²))
вместо bottleneck токенов. Ключевое отличие от Baseline 1: есть
cross-modal interaction до классификации.
"""

import torch
import torch.nn as nn
import math
from transformers import BertModel

HIDDEN_DIM   = 128
N_HEADS      = 8
N_LAYERS     = 2       # два слоя cross-modal attention (аналог N_LAYERS в Proposed)
DROPOUT      = 0.1
N_EMOTIONS   = 6
BERT_MODEL   = "bert-base-uncased"


class ModalityProjection(nn.Module):
    """Conv1D проекция → HIDDEN_DIM (идентично Proposed)."""
    def __init__(self, input_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(input_dim, HIDDEN_DIM, kernel_size=3, padding=1),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = x.transpose(1, 2)
        return x


class PositionalEncoding(nn.Module):
    """Sine-cosine позиционное кодирование (идентично Proposed)."""
    def __init__(self, max_len=128):
        super().__init__()
        pe  = torch.zeros(max_len, HIDDEN_DIM)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, HIDDEN_DIM, 2).float() * (-math.log(10000.0) / HIDDEN_DIM)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class CrossModalLayer(nn.Module):
    """
    Один слой MulT-style cross-modal attention.

    Каждая модальность посещает обе другие через cross-attention,
    затем FFN с residual connection — аналог одного слоя трансформера.
    """

    def __init__(self):
        super().__init__()

        # Cross-attention: T←A, T←V
        self.ca_t_from_a = nn.MultiheadAttention(HIDDEN_DIM, N_HEADS, DROPOUT, batch_first=True)
        self.ca_t_from_v = nn.MultiheadAttention(HIDDEN_DIM, N_HEADS, DROPOUT, batch_first=True)

        # Cross-attention: A←T, A←V
        self.ca_a_from_t = nn.MultiheadAttention(HIDDEN_DIM, N_HEADS, DROPOUT, batch_first=True)
        self.ca_a_from_v = nn.MultiheadAttention(HIDDEN_DIM, N_HEADS, DROPOUT, batch_first=True)

        # Cross-attention: V←T, V←A
        self.ca_v_from_t = nn.MultiheadAttention(HIDDEN_DIM, N_HEADS, DROPOUT, batch_first=True)
        self.ca_v_from_a = nn.MultiheadAttention(HIDDEN_DIM, N_HEADS, DROPOUT, batch_first=True)

        # FFN для каждой модальности
        self.ffn_t = self._ffn()
        self.ffn_a = self._ffn()
        self.ffn_v = self._ffn()

        # LayerNorm
        self.norm_t1 = nn.LayerNorm(HIDDEN_DIM)
        self.norm_a1 = nn.LayerNorm(HIDDEN_DIM)
        self.norm_v1 = nn.LayerNorm(HIDDEN_DIM)
        self.norm_t2 = nn.LayerNorm(HIDDEN_DIM)
        self.norm_a2 = nn.LayerNorm(HIDDEN_DIM)
        self.norm_v2 = nn.LayerNorm(HIDDEN_DIM)

    def _ffn(self):
        return nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 4),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM * 4, HIDDEN_DIM),
            nn.Dropout(DROPOUT),
        )

    def forward(self, text, audio, vision, text_mask=None):
        t_pad = (text_mask == 0) if text_mask is not None else None

        # ── Text посещает Audio и Vision ──────────────────────────────────────
        t_a, _ = self.ca_t_from_a(text, audio,  audio)
        t_v, _ = self.ca_t_from_v(text, vision, vision)
        text   = self.norm_t1(text + t_a + t_v)
        text   = self.norm_t2(text + self.ffn_t(text))

        # ── Audio посещает Text и Vision ──────────────────────────────────────
        a_t, _ = self.ca_a_from_t(audio, text,   text,   key_padding_mask=t_pad)
        a_v, _ = self.ca_a_from_v(audio, vision, vision)
        audio  = self.norm_a1(audio + a_t + a_v)
        audio  = self.norm_a2(audio + self.ffn_a(audio))

        # ── Vision посещает Text и Audio ──────────────────────────────────────
        v_t, _ = self.ca_v_from_t(vision, text,  text,  key_padding_mask=t_pad)
        v_a, _ = self.ca_v_from_a(vision, audio, audio)
        vision = self.norm_v1(vision + v_t + v_a)
        vision = self.norm_v2(vision + self.ffn_v(vision))

        return text, audio, vision


class CrossModalFusionModel(nn.Module):
    """
    Baseline 2: MulT-style полный pairwise cross-modal attention.
    Интерфейс совместим с BottleneckFusionModel.
    """

    def __init__(self):
        super().__init__()

        # ── Text: BERT frozen + linear (то же что Proposed) ──────────────────
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        for param in self.bert.parameters():
            param.requires_grad = False

        self.text_proj = nn.Linear(768, HIDDEN_DIM)

        # ── Audio и Vision: Conv1D + positional encoding (то же что Proposed) ─
        self.audio_proj  = ModalityProjection(input_dim=74)
        self.vision_proj = ModalityProjection(input_dim=35)
        self.pos_enc_audio  = PositionalEncoding(max_len=60)
        self.pos_enc_vision = PositionalEncoding(max_len=60)

        # ── N_LAYERS слоёв cross-modal attention ──────────────────────────────
        self.layers = nn.ModuleList([CrossModalLayer() for _ in range(N_LAYERS)])

        self.dropout = nn.Dropout(DROPOUT)

        # ── Classifier (то же что Proposed) ──────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 3, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, N_EMOTIONS),
        )

        # ── Auxiliary heads ───────────────────────────────────────────────────
        self.head_text   = nn.Linear(HIDDEN_DIM, N_EMOTIONS)
        self.head_audio  = nn.Linear(HIDDEN_DIM, N_EMOTIONS)
        self.head_vision = nn.Linear(HIDDEN_DIM, N_EMOTIONS)

    def forward(self, input_ids, attention_mask, audio, vision,
                audio_mask, vision_mask):

        # ── Text: BERT avg 4 слоёв → linear ──────────────────────────────────
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        text = torch.stack(bert_out.hidden_states[-4:], dim=0).mean(dim=0)  # [B,50,768]
        text = self.dropout(self.text_proj(text))                            # [B,50,128]

        # ── Audio и Vision: Conv1D + positional encoding ──────────────────────
        audio  = self.pos_enc_audio(self.audio_proj(audio))    # [B,60,128]
        vision = self.pos_enc_vision(self.vision_proj(vision))  # [B,60,128]

        # ── N_LAYERS слоёв cross-modal attention ──────────────────────────────
        for layer in self.layers:
            text, audio, vision = layer(text, audio, vision, text_mask=attention_mask)

        # ── Aggregation: CLS (text) + mean pool (audio, vision) ──────────────
        text_cls    = text[:, 0, :]        # [B,128]
        audio_pool  = audio.mean(dim=1)    # [B,128]
        vision_pool = vision.mean(dim=1)   # [B,128]

        fused = self.dropout(
            torch.cat([text_cls, audio_pool, vision_pool], dim=-1)  # [B,384]
        )

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
    model = CrossModalFusionModel().to(device)

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
    print("✅ baseline2_model.py работает")
