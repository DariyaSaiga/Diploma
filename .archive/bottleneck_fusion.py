import torch
import torch.nn as nn
from transformers import BertModel


class BottleneckFusion(nn.Module):
    """
    Domain-Separated Bottleneck Architecture для multimodal emotion recognition.

    Разделяет каждую модальность на:
    - Invariant domain: эмоция (shared across модальностей)
    - Private domain: специфика модальности (только для одной модальности)

    Bottleneck tokens (B=16) служат "информационными воротами" между доменами.
    """

    def __init__(self, num_classes=6, hidden_dim=128, num_bottleneck_tokens=16,
                 dropout=0.1, freeze_bert=True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_bottleneck_tokens = num_bottleneck_tokens
        self.num_classes = num_classes

        # =============================================
        # 1️⃣ INPUT ENCODERS (что у тебя есть)
        # =============================================

        # TEXT: BERT
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.text_proj = nn.Linear(768, hidden_dim)  # 768 -> 128

        # AUDIO: Linear projection
        self.audio_proj = nn.Linear(74, hidden_dim)  # 74 -> 128

        # VISUAL: Linear projection
        self.visual_proj = nn.Linear(713, hidden_dim)  # 713 -> 128

        # =============================================
        # 2️⃣ DOMAIN SEPARATION (Invariant + Private)
        # =============================================

        # Для каждой модальности создаём две ветки
        # Invariant: эмоциональный сигнал (shared)
        # Private: специфика модальности (не shared)

        # TEXT domain separation
        self.text_invariant = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.text_private = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # AUDIO domain separation
        self.audio_invariant = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.audio_private = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # VISUAL domain separation
        self.visual_invariant = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.visual_private = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # =============================================
        # 3️⃣ BOTTLENECK TOKENS (информационные врата)
        # =============================================

        # Learnable bottleneck tokens: (B, hidden_dim)
        # B=16 означает что информация сжимается через 16 токенов
        self.bottleneck_tokens = nn.Parameter(
            torch.randn(num_bottleneck_tokens, hidden_dim) * 0.02
        )

        # =============================================
        # 4️⃣ CROSS-ATTENTION (информационный обмен)
        # =============================================

        # Для каждой модальности: attention от private domain к bottleneck tokens
        num_heads = 8

        # TEXT cross-attention
        self.text_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # AUDIO cross-attention
        self.audio_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # VISUAL cross-attention
        self.visual_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # =============================================
        # 5️⃣ RECONSTRUCTION HEADS (сохранение информации)
        # =============================================

        # Восстанавливаем исходные размеры из [invariant || private]
        self.text_reconstruct = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 768),  # обратно в BERT размер
        )

        self.audio_reconstruct = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 74),  # обратно в audio размер
        )

        self.visual_reconstruct = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 713),  # обратно в visual размер
        )

        # =============================================
        # 6️⃣ FINAL FUSION & CLASSIFICATION
        # =============================================

        # Fusion: concatenate [inv_text, priv_text, inv_audio, priv_audio, inv_visual, priv_visual]
        fusion_dim = hidden_dim * 6  # 3 модальности * 2 домена

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    @staticmethod
    def _masked_mean(seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет masked mean.
        seq:  (B, T, D)
        mask: (B, T)  1=real, 0=pad
        returns: (B, D)
        """
        m = mask.unsqueeze(-1).float()  # (B, T, 1)
        return (seq * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)

    def forward(self, input_ids, attention_mask, audio=None, visual=None,
                audio_mask=None, visual_mask=None, return_domains=False):
        """
        Args:
            input_ids: (B, T_text)
            attention_mask: (B, T_text)
            audio: (B, T_audio, 74)
            visual: (B, T_visual, 713)
            audio_mask: (B, T_audio) optional
            visual_mask: (B, T_visual) optional
            return_domains: bool - если True, возвращает invariant/private для loss вычисления

        Returns:
            logits: (B, num_classes)
            или (logits, invariant_dict, private_dict) если return_domains=True
        """

        # =====================================================
        # STEP 1: Encode исходные данные через projection layers
        # =====================================================

        # TEXT
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_encoded = bert_out.last_hidden_state[:, 0, :]  # CLS token, (B, 768)
        text_encoded = self.text_proj(text_encoded)  # (B, 128)

        # AUDIO
        if audio is not None:
            audio_encoded = self.audio_proj(audio)  # (B, T_audio, 128)
            if audio_mask is not None:
                audio_encoded = self._masked_mean(audio_encoded, audio_mask)  # (B, 128)
            else:
                audio_encoded = audio_encoded.mean(dim=1)  # (B, 128)
        else:
            audio_encoded = None

        # VISUAL
        if visual is not None:
            visual_encoded = self.visual_proj(visual)  # (B, T_visual, 128)
            if visual_mask is not None:
                visual_encoded = self._masked_mean(visual_encoded, visual_mask)  # (B, 128)
            else:
                visual_encoded = visual_encoded.mean(dim=1)  # (B, 128)
        else:
            visual_encoded = None

        # =====================================================
        # STEP 2: Domain Separation (Invariant + Private)
        # =====================================================

        # TEXT
        text_invariant = self.text_invariant(text_encoded)  # (B, 128)
        text_private = self.text_private(text_encoded)      # (B, 128)

        # AUDIO
        audio_invariant = self.audio_invariant(audio_encoded) if audio_encoded is not None else None
        audio_private = self.audio_private(audio_encoded) if audio_encoded is not None else None

        # VISUAL
        visual_invariant = self.visual_invariant(visual_encoded) if visual_encoded is not None else None
        visual_private = self.visual_private(visual_encoded) if visual_encoded is not None else None

        # =====================================================
        # STEP 3: Bottleneck Cross-Attention
        # =====================================================
        # Информационный обмен через bottleneck tokens

        batch_size = text_encoded.size(0)
        bottleneck_tokens = self.bottleneck_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 16, 128)

        # TEXT: private domain attends to bottleneck tokens
        text_private_refined, _ = self.text_attention(
            query=text_private.unsqueeze(1),  # (B, 1, 128)
            key=bottleneck_tokens,             # (B, 16, 128)
            value=bottleneck_tokens            # (B, 16, 128)
        )
        text_private_refined = text_private_refined.squeeze(1)  # (B, 128)

        # AUDIO: private domain attends to bottleneck tokens
        if audio_private is not None:
            audio_private_refined, _ = self.audio_attention(
                query=audio_private.unsqueeze(1),  # (B, 1, 128)
                key=bottleneck_tokens,             # (B, 16, 128)
                value=bottleneck_tokens            # (B, 16, 128)
            )
            audio_private_refined = audio_private_refined.squeeze(1)  # (B, 128)
        else:
            audio_private_refined = None

        # VISUAL: private domain attends to bottleneck tokens
        if visual_private is not None:
            visual_private_refined, _ = self.visual_attention(
                query=visual_private.unsqueeze(1),  # (B, 1, 128)
                key=bottleneck_tokens,              # (B, 16, 128)
                value=bottleneck_tokens             # (B, 16, 128)
            )
            visual_private_refined = visual_private_refined.squeeze(1)  # (B, 128)
        else:
            visual_private_refined = None

        # =====================================================
        # STEP 4: Reconstruction (для reconstruction loss)
        # =====================================================

        # Объединяем invariant + refined private для реконструкции
        text_combined = torch.cat([text_invariant, text_private_refined], dim=1)  # (B, 256)
        text_recon = self.text_reconstruct(text_combined)  # (B, 768)

        if audio_encoded is not None:
            audio_combined = torch.cat([audio_invariant, audio_private_refined], dim=1)  # (B, 256)
            audio_recon = self.audio_reconstruct(audio_combined)  # (B, 74)
        else:
            audio_recon = None

        if visual_encoded is not None:
            visual_combined = torch.cat([visual_invariant, visual_private_refined], dim=1)  # (B, 256)
            visual_recon = self.visual_reconstruct(visual_combined)  # (B, 713)
        else:
            visual_recon = None

        # =====================================================
        # STEP 5: Final Fusion & Classification
        # =====================================================

        # Concatenate все invariant + refined private
        fusion_list = [text_invariant, text_private_refined]
        if audio_private_refined is not None:
            fusion_list.extend([audio_invariant, audio_private_refined])
        if visual_private_refined is not None:
            fusion_list.extend([visual_invariant, visual_private_refined])

        fused = torch.cat(fusion_list, dim=1)  # (B, hidden_dim*6) или меньше если нет audio/visual

        # Fusion head
        fused = self.fusion_head(fused)  # (B, hidden_dim)

        # Classification
        logits = self.classifier(fused)  # (B, num_classes)

        # =====================================================
        # RETURN
        # =====================================================

        if return_domains:
            # Для вычисления domain separation losses
            domain_data = {
                'text_invariant': text_invariant,
                'text_private': text_private_refined,
                'audio_invariant': audio_invariant,
                'audio_private': audio_private_refined,
                'visual_invariant': visual_invariant,
                'visual_private': visual_private_refined,
            }

            reconstruction_data = {
                'text_recon': text_recon,
                'text_original': text_encoded,  # используем projected version как reference
                'audio_recon': audio_recon,
                'audio_original': audio_encoded,
                'visual_recon': visual_recon,
                'visual_original': visual_encoded,
            }

            return logits, domain_data, reconstruction_data

        return logits
