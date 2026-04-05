import torch
import torch.nn as nn
from transformers import BertModel


class BottleneckFusion(nn.Module):
    """
    Sequence-level multimodal fusion через learnable bottleneck tokens.

    Архитектура:
        1. Text branch:   BERT last_hidden_state → Linear(768→128) + LN + Dropout  → (B, Lt, 128)
        2. Audio branch:  Linear(74→128) + ReLU + LN + Dropout                     → (B, Ta, 128)
        3. Visual branch: Linear(713→128) + ReLU + LN + Dropout                    → (B, Tv, 128)
        4. Concat all sequences                                                     → (B, Lt+Ta+Tv, 128)
        5. Learnable bottleneck tokens attend к fused sequence (cross-attention)    → (B, n_bn, 128)
        6. Mean pooling по bottleneck tokens → Linear(128→6)                       → (B, 6)

    Masks:
        attention_mask : (B, Lt) — 1=real, 0=pad  (из BERT tokenizer)
        audio_mask     : (B, Ta) — 1=real, 0=pad
        visual_mask    : (B, Tv) — 1=real, 0=pad
    """

    def __init__(
        self,
        num_classes: int = 6,
        hidden_dim: int = 128,
        num_bottleneck_tokens: int = 8,
        num_heads: int = 4,
        dropout: float = 0.3,
        freeze_bert: bool = True,
    ):
        super().__init__()

        # ── Text branch ──────────────────────────────────────────────────
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.text_proj = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # ── Audio branch ─────────────────────────────────────────────────
        self.audio_proj = nn.Sequential(
            nn.Linear(74, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # ── Visual branch ────────────────────────────────────────────────
        self.visual_proj = nn.Sequential(
            nn.Linear(713, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # ── Bottleneck tokens ────────────────────────────────────────────
        self.bottleneck_tokens = nn.Parameter(
            torch.randn(1, num_bottleneck_tokens, hidden_dim)
        )

        # ── Cross-attention: bottleneck → fused sequence ─────────────────
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # ── Classifier ───────────────────────────────────────────────────
        self.classifier = nn.Linear(hidden_dim, num_classes)

    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,       # (B, Lt)
        attention_mask: torch.Tensor,  # (B, Lt)  1=real, 0=pad
        audio: torch.Tensor,           # (B, Ta, 74)
        audio_mask: torch.Tensor,      # (B, Ta)  1=real, 0=pad
        visual: torch.Tensor,          # (B, Tv, 713)
        visual_mask: torch.Tensor,     # (B, Tv)  1=real, 0=pad
    ) -> torch.Tensor:                 # (B, num_classes)

        B = input_ids.size(0)

        # ── Text: (B, Lt, 768) → (B, Lt, 128) ───────────────────────────
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_seq = self.text_proj(bert_out.last_hidden_state)  # (B, Lt, 128)

        # ── Audio: (B, Ta, 74) → (B, Ta, 128) ───────────────────────────
        audio_seq = self.audio_proj(audio)    # (B, Ta, 128)

        # ── Visual: (B, Tv, 713) → (B, Tv, 128) ─────────────────────────
        visual_seq = self.visual_proj(visual)  # (B, Tv, 128)

        # ── Concat sequences и маски ──────────────────────────────────────
        fused_seq = torch.cat([text_seq, audio_seq, visual_seq], dim=1)
        # (B, Lt+Ta+Tv, 128)

        # key_padding_mask для MultiheadAttention: True = позиция-паддинг
        text_pad   = (attention_mask == 0)  # (B, Lt)
        audio_pad  = (audio_mask == 0)      # (B, Ta)
        visual_pad = (visual_mask == 0)     # (B, Tv)
        fused_pad_mask = torch.cat([text_pad, audio_pad, visual_pad], dim=1)
        # (B, Lt+Ta+Tv)  True=pad

        # ── Bottleneck cross-attention ─────────────────────────────────
        bn_tokens = self.bottleneck_tokens.expand(B, -1, -1)  # (B, n_bn, 128)

        attn_out, _ = self.cross_attn(
            query=bn_tokens,
            key=fused_seq,
            value=fused_seq,
            key_padding_mask=fused_pad_mask,
        )  # (B, n_bn, 128)
        attn_out = self.attn_norm(attn_out + bn_tokens)  # residual + LN

        # ── Mean pooling по bottleneck tokens ─────────────────────────
        pooled = attn_out.mean(dim=1)   # (B, 128)

        # ── Classifier ─────────────────────────────────────────────────
        logits = self.classifier(pooled)  # (B, 6)
        return logits
