import torch
import torch.nn as nn
from old_arch.core.encoders import BertTextEncoder, AudioCNNEncoder, VideoBiLSTMEncoder


class DomainEncoder(nn.Module):
    """2-layer Transformer encoder for domain separation (invariant / private)."""

    def __init__(self, hidden_dim=128, num_layers=2, num_heads=8, ff_dim=256, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x, padding_mask=None):
        # padding_mask: True=valid → invert for Transformer (True=ignore)
        src_key_padding_mask = ~padding_mask if padding_mask is not None else None
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


class CrossAttentionBlock(nn.Module):
    """Cross-attention with residual + LayerNorm + feedforward."""

    def __init__(self, hidden_dim=128, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, kv):
        attn_out, _ = self.attn(query, kv, kv)
        x = self.norm1(query + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class BottleneckFusion(nn.Module):
    """Domain-Separated Bottleneck Fusion for multimodal emotion recognition.

    Architecture:
        Text  → BertTextEncoder  → Linear(768→D) → text_seq  (B, T_text, D)
        Audio → AudioCNNEncoder  → audio_seq  (B, T_audio, D)
        Video → VideoBiLSTMEncoder → Linear(D*2→D) → visual_seq (B, T_visual, D)

        Each modality → DomainEncoder × 2 → invariant_seq + private_seq

        Bottleneck:
          1. Within-private: private attends to base bottleneck tokens
          2. Condition bottleneck with invariant pool
          3. Cross-domain: each private attends to other modalities' conditioned bottleneck
          4. Pool invariant + private → per-modality vectors
          5. Concat all → fusion_head → classifier

    BottleneckFusion does NOT use logits from baseline models.
    It uses the same encoder blocks (BERT / CNN / BiLSTM) directly on raw features.
    """

    def __init__(self, num_classes=6, hidden_dim=128, num_bottleneck_tokens=16,
                 dropout=0.1, freeze_bert="full", use_audio=True, use_visual=True,
                 audio_input_dim=74, visual_input_dim=713):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_bottleneck_tokens = num_bottleneck_tokens
        self.use_audio = use_audio
        self.use_visual = use_visual
        self.audio_input_dim = audio_input_dim
        self.visual_input_dim = visual_input_dim

        # ── Text encoder ──
        self.text_encoder = BertTextEncoder(freeze_bert=freeze_bert)
        self.text_proj = nn.Linear(768, hidden_dim)

        # ── Audio encoder (CNN output = hidden_dim, no extra projection) ──
        if use_audio:
            self.audio_encoder = AudioCNNEncoder(audio_input_dim, hidden_dim, dropout)

        # ── Visual encoder (BiLSTM output = hidden_dim*2, project to hidden_dim) ──
        if use_visual:
            self.visual_encoder = VideoBiLSTMEncoder(visual_input_dim, hidden_dim, dropout)
            self.visual_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # ── Domain separation: one invariant + one private encoder per modality ──
        self.text_inv_enc = DomainEncoder(hidden_dim, dropout=dropout)
        self.text_priv_enc = DomainEncoder(hidden_dim, dropout=dropout)
        if use_audio:
            self.audio_inv_enc = DomainEncoder(hidden_dim, dropout=dropout)
            self.audio_priv_enc = DomainEncoder(hidden_dim, dropout=dropout)
        if use_visual:
            self.visual_inv_enc = DomainEncoder(hidden_dim, dropout=dropout)
            self.visual_priv_enc = DomainEncoder(hidden_dim, dropout=dropout)

        # ── Learnable base bottleneck tokens ──
        self.bottleneck_tokens = nn.Parameter(
            torch.randn(num_bottleneck_tokens, hidden_dim) * 0.02
        )

        # ── Invariant pool → condition bottleneck tokens ──
        self.text_inv_to_bn = nn.Linear(hidden_dim, num_bottleneck_tokens * hidden_dim)
        if use_audio:
            self.audio_inv_to_bn = nn.Linear(hidden_dim, num_bottleneck_tokens * hidden_dim)
        if use_visual:
            self.visual_inv_to_bn = nn.Linear(hidden_dim, num_bottleneck_tokens * hidden_dim)

        # ── Within-private refinement (private → base bottleneck) ──
        self.text_within_attn = CrossAttentionBlock(hidden_dim, dropout=dropout)
        if use_audio:
            self.audio_within_attn = CrossAttentionBlock(hidden_dim, dropout=dropout)
        if use_visual:
            self.visual_within_attn = CrossAttentionBlock(hidden_dim, dropout=dropout)

        # ── Cross-domain exchange (private → other modalities' conditioned bottleneck) ──
        self.text_cross_attn = CrossAttentionBlock(hidden_dim, dropout=dropout)
        if use_audio:
            self.audio_cross_attn = CrossAttentionBlock(hidden_dim, dropout=dropout)
        if use_visual:
            self.visual_cross_attn = CrossAttentionBlock(hidden_dim, dropout=dropout)

        # ── Reconstruction heads: [inv_pool || priv_pool] → original feature dim ──
        self.text_recon_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512), nn.ReLU(), nn.Linear(512, 768),
        )
        if use_audio:
            self.audio_recon_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, 512), nn.ReLU(), nn.Linear(512, audio_input_dim),
            )
        if use_visual:
            self.visual_recon_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, 512), nn.ReLU(), nn.Linear(512, visual_input_dim),
            )

        # ── Fusion classifier ──
        num_mod = 1 + int(use_audio) + int(use_visual)
        fusion_dim = hidden_dim * 2 * num_mod  # [inv_pool || priv_pool] per modality
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    @staticmethod
    def _masked_mean(seq, mask):
        if mask is None:
            return seq.mean(dim=1)
        m = mask.unsqueeze(-1).float()
        return (seq * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)

    def forward(self, batch, return_domains=False):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        audio = batch.get("audio")
        audio_mask = batch.get("audio_mask")
        visual = batch.get("visual")
        visual_mask = batch.get("visual_mask")

        B = input_ids.size(0)
        N = self.num_bottleneck_tokens
        D = self.hidden_dim

        # ══ ENCODE (sequence-level, no pooling yet) ══════════════════════════
        bert_hidden = self.text_encoder(input_ids, attention_mask)  # (B, T_text, 768)
        text_seq = self.text_proj(bert_hidden)                       # (B, T_text, D)
        text_mask = attention_mask.bool()

        audio_seq, a_mask, audio_raw = None, None, None
        if self.use_audio and audio is not None:
            audio_raw = audio
            audio_seq = self.audio_encoder(audio)                    # (B, T_a, D)
            a_mask = audio_mask.bool() if audio_mask is not None else None

        visual_seq, v_mask, visual_raw = None, None, None
        if self.use_visual and visual is not None:
            visual_raw = visual
            visual_seq = self.visual_proj(self.visual_encoder(visual))  # (B, T_v, D)
            v_mask = visual_mask.bool() if visual_mask is not None else None

        # ══ DOMAIN SEPARATION (sequence-level) ═══════════════════════════════
        text_inv = self.text_inv_enc(text_seq, text_mask)
        text_priv = self.text_priv_enc(text_seq, text_mask)

        audio_inv = audio_priv = None
        if audio_seq is not None:
            audio_inv = self.audio_inv_enc(audio_seq, a_mask)
            audio_priv = self.audio_priv_enc(audio_seq, a_mask)

        visual_inv = visual_priv = None
        if visual_seq is not None:
            visual_inv = self.visual_inv_enc(visual_seq, v_mask)
            visual_priv = self.visual_priv_enc(visual_seq, v_mask)

        # ══ WITHIN-PRIVATE REFINEMENT (private → base bottleneck) ════════════
        base_bn = self.bottleneck_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, N, D)

        text_priv = self.text_within_attn(text_priv, base_bn)
        if audio_priv is not None:
            audio_priv = self.audio_within_attn(audio_priv, base_bn)
        if visual_priv is not None:
            visual_priv = self.visual_within_attn(visual_priv, base_bn)

        # ══ CONDITION BOTTLENECK WITH INVARIANT POOL ═════════════════════════
        text_inv_pool = self._masked_mean(text_inv, text_mask)           # (B, D)
        text_bn = base_bn + self.text_inv_to_bn(text_inv_pool).view(B, N, D)
        cond_bns = {"text": text_bn}

        audio_inv_pool = None
        if audio_inv is not None:
            audio_inv_pool = self._masked_mean(audio_inv, a_mask)
            cond_bns["audio"] = base_bn + self.audio_inv_to_bn(audio_inv_pool).view(B, N, D)

        visual_inv_pool = None
        if visual_inv is not None:
            visual_inv_pool = self._masked_mean(visual_inv, v_mask)
            cond_bns["visual"] = base_bn + self.visual_inv_to_bn(visual_inv_pool).view(B, N, D)

        # ══ CROSS-DOMAIN EXCHANGE (private → other modalities' conditioned BN) ══
        text_kv = [bn for k, bn in cond_bns.items() if k != "text"]
        if text_kv:
            text_priv = self.text_cross_attn(text_priv, torch.cat(text_kv, dim=1))

        if audio_priv is not None:
            audio_kv = [bn for k, bn in cond_bns.items() if k != "audio"]
            if audio_kv:
                audio_priv = self.audio_cross_attn(audio_priv, torch.cat(audio_kv, dim=1))

        if visual_priv is not None:
            visual_kv = [bn for k, bn in cond_bns.items() if k != "visual"]
            if visual_kv:
                visual_priv = self.visual_cross_attn(visual_priv, torch.cat(visual_kv, dim=1))

        # ══ POOL FOR FUSION ═══════════════════════════════════════════════════
        text_priv_pool = self._masked_mean(text_priv, text_mask)
        parts = [text_inv_pool, text_priv_pool]

        audio_priv_pool = None
        if audio_priv is not None:
            audio_priv_pool = self._masked_mean(audio_priv, a_mask)
            parts.extend([audio_inv_pool, audio_priv_pool])

        visual_priv_pool = None
        if visual_priv is not None:
            visual_priv_pool = self._masked_mean(visual_priv, v_mask)
            parts.extend([visual_inv_pool, visual_priv_pool])

        # ══ FUSION + CLASSIFY ════════════════════════════════════════════════
        fused = torch.cat(parts, dim=-1)           # (B, D*2*num_modalities)
        logits = self.classifier(self.fusion_head(fused))

        if not return_domains:
            return logits

        # ══ DOMAIN DATA FOR AUXILIARY LOSSES ═════════════════════════════════
        domain_data = {
            "text_inv_pool": text_inv_pool,
            "text_priv_pool": text_priv_pool,
        }
        recon_data = {
            "text_recon": self.text_recon_head(torch.cat([text_inv_pool, text_priv_pool], dim=-1)),
            "text_original": self._masked_mean(bert_hidden, text_mask),
        }

        if audio_inv_pool is not None:
            domain_data["audio_inv_pool"] = audio_inv_pool
            domain_data["audio_priv_pool"] = audio_priv_pool
            recon_data["audio_recon"] = self.audio_recon_head(
                torch.cat([audio_inv_pool, audio_priv_pool], dim=-1)
            )
            recon_data["audio_original"] = self._masked_mean(audio_raw, a_mask)

        if visual_inv_pool is not None:
            domain_data["visual_inv_pool"] = visual_inv_pool
            domain_data["visual_priv_pool"] = visual_priv_pool
            recon_data["visual_recon"] = self.visual_recon_head(
                torch.cat([visual_inv_pool, visual_priv_pool], dim=-1)
            )
            recon_data["visual_original"] = self._masked_mean(visual_raw, v_mask)

        return logits, domain_data, recon_data
