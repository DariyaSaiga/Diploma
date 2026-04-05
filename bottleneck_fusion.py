import torch
import torch.nn as nn
from transformers import BertModel


class BottleneckLayer(nn.Module):
    """
    Один слой MBT-fusion:
      - Каждая модальность делает self-attention внутри себя
      - Ботлнек-токены собирают информацию из каждой модальности (cross-attn)
      - Ботлнек усредняется между модальностями
      - FFN + residual + LN
    """

    def __init__(self, hidden_dim: int, num_heads: int, num_bn: int, dropout: float):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.self_norm = nn.LayerNorm(hidden_dim)

        self.cross_attn_text   = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_audio  = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_visual = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm   = nn.LayerNorm(hidden_dim)
        self.drop       = nn.Dropout(dropout)

    def forward(self, text, audio, visual, bn, text_pad, audio_pad, visual_pad):
        # 1) Unimodal self-attention
        def self_att(x, pad):
            out, _ = self.self_attn(x, x, x, key_padding_mask=pad)
            return self.self_norm(x + self.drop(out))

        text   = self_att(text,   text_pad)
        audio  = self_att(audio,  audio_pad)
        visual = self_att(visual, visual_pad)

        # 2) Cross-attention: ботлнек → каждая модальность отдельно
        bn_t, _ = self.cross_attn_text(bn,   text,   text,   key_padding_mask=text_pad)
        bn_a, _ = self.cross_attn_audio(bn,  audio,  audio,  key_padding_mask=audio_pad)
        bn_v, _ = self.cross_attn_visual(bn, visual, visual, key_padding_mask=visual_pad)

        # 3) Усредняем обновлённые ботлнеки по модальностям (как в MBT)
        bn_avg = (bn_t + bn_a + bn_v) / 3.0

        # 4) Residual + LN + FFN
        bn = self.cross_norm(bn + self.drop(bn_avg))
        bn = self.ffn_norm(bn + self.drop(self.ffn(bn)))

        return text, audio, visual, bn


class BottleneckFusion(nn.Module):
    """
    Multi-layer Multimodal Bottleneck Transformer.

    Архитектура:
        1. Text branch:   BERT last_hidden_state → Linear(768→128) + LN + Dropout  → (B, Lt, 128)
        2. Audio branch:  Linear(74→128) + ReLU + LN + Dropout                     → (B, Ta, 128)
        3. Visual branch: Linear(713→128) + ReLU + LN + Dropout                    → (B, Tv, 128)
        4. N слоёв BottleneckLayer (unimodal self-attn + cross-attn + FFN)
        5. Mean pooling по bottleneck tokens → Linear(128→6)

    Masks:
        attention_mask : (B, Lt) — 1=real, 0=pad
        audio_mask     : (B, Ta) — 1=real, 0=pad
        visual_mask    : (B, Tv) — 1=real, 0=pad

    Ablation flags:
        use_audio  : если False — audio branch исключается из fusion
        use_visual : если False — visual branch исключается из fusion
    """

    def __init__(
        self,
        num_classes: int = 6,
        hidden_dim: int = 256,
        num_bottleneck_tokens: int = 16,
        num_bottleneck_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        freeze_bert: bool = True,
        use_audio: bool = True,
        use_visual: bool = True,
    ):
        super().__init__()

        self.use_audio  = use_audio
        self.use_visual = use_visual

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

        # ── Learnable bottleneck tokens ───────────────────────────────────
        self.bottleneck_tokens = nn.Parameter(
            torch.randn(1, num_bottleneck_tokens, hidden_dim) * 0.02
        )

        # ── Стек MBT-слоёв ────────────────────────────────────────────────
        self.layers = nn.ModuleList([
            BottleneckLayer(hidden_dim, num_heads, num_bottleneck_tokens, dropout)
            for _ in range(num_bottleneck_layers)
        ])

        # ── Multimodal classifier (главный, из bottleneck токенов) ──────────
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # ── Unimodal classifiers для DRA loss ────────────────────────────
        self.classifier_text   = nn.Linear(hidden_dim, num_classes)
        self.classifier_audio  = nn.Linear(hidden_dim, num_classes)
        self.classifier_visual = nn.Linear(hidden_dim, num_classes)

    # ── Вспомогательный: masked mean pooling по реальным токенам ─────────────
    @staticmethod
    def _masked_mean(seq: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        seq     : (B, T, D)
        pad_mask: (B, T)  True=pad, False=real
        returns : (B, D)
        """
        real = (~pad_mask).unsqueeze(-1).float()          # (B, T, 1)
        return (seq * real).sum(dim=1) / real.sum(dim=1).clamp(min=1.0)

    # ── Общий внутренний проход через все слои ────────────────────────────────
    def _encode(self, input_ids, attention_mask, audio, audio_mask, visual, visual_mask):
        """Возвращает все промежуточные тензоры после всех BottleneckLayer."""
        B = input_ids.size(0)

        bert_out   = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_seq   = self.text_proj(bert_out.last_hidden_state)   # (B, Lt, D)
        audio_seq  = self.audio_proj(audio)                        # (B, Ta, D)
        visual_seq = self.visual_proj(visual)                      # (B, Tv, D)

        text_pad   = (attention_mask == 0)
        audio_pad  = (audio_mask == 0)
        visual_pad = (visual_mask == 0)

        bn = self.bottleneck_tokens.expand(B, -1, -1).clone()

        for layer in self.layers:
            text_seq, audio_seq, visual_seq, bn = layer(
                text_seq, audio_seq, visual_seq, bn,
                text_pad, audio_pad, visual_pad
            )

        return text_seq, audio_seq, visual_seq, bn, text_pad, audio_pad, visual_pad

    # ── Стандартный forward (один выход — для evaluate и baseline) ───────────
    def forward(
        self,
        input_ids: torch.Tensor,       # (B, Lt)
        attention_mask: torch.Tensor,  # (B, Lt)  1=real, 0=pad
        audio: torch.Tensor,           # (B, Ta, 74)
        audio_mask: torch.Tensor,      # (B, Ta)  1=real, 0=pad
        visual: torch.Tensor,          # (B, Tv, 713)
        visual_mask: torch.Tensor,     # (B, Tv)  1=real, 0=pad
    ) -> torch.Tensor:                 # (B, num_classes)

        text_seq, audio_seq, visual_seq, bn, *_ = self._encode(
            input_ids, attention_mask, audio, audio_mask, visual, visual_mask
        )
        return self.classifier(bn.mean(dim=1))   # (B, 6)

    # ── DRA forward (четыре выхода — для DRA loss) ────────────────────────────
    def forward_dra(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio: torch.Tensor,
        audio_mask: torch.Tensor,
        visual: torch.Tensor,
        visual_mask: torch.Tensor,
    ) -> tuple:    # (logits_fused, logits_text, logits_audio, logits_visual)
        """
        Для DRA loss: возвращает логиты от каждой ветки.
        logits_fused  — из bottleneck токенов (главный, для предсказаний)
        logits_text   — из text_seq masked mean (aux loss)
        logits_audio  — из audio_seq masked mean (aux loss)
        logits_visual — из visual_seq masked mean (aux loss)
        """
        text_seq, audio_seq, visual_seq, bn, \
            text_pad, audio_pad, visual_pad = self._encode(
                input_ids, attention_mask, audio, audio_mask, visual, visual_mask
            )

        logits_fused  = self.classifier(bn.mean(dim=1))
        logits_text   = self.classifier_text(self._masked_mean(text_seq,   text_pad))
        logits_audio  = self.classifier_audio(self._masked_mean(audio_seq,  audio_pad))
        logits_visual = self.classifier_visual(self._masked_mean(visual_seq, visual_pad))

        return logits_fused, logits_text, logits_audio, logits_visual