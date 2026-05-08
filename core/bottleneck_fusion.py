import torch
import torch.nn as nn
from transformers import BertModel


class BottleneckFusion(nn.Module):
    """Domain-Separated Bottleneck Architecture for multimodal emotion recognition."""

    def __init__(self, num_classes=6, hidden_dim=128, num_bottleneck_tokens=16,
                 dropout=0.1, freeze_bert=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_bottleneck_tokens = num_bottleneck_tokens
        self.num_classes = num_classes

        # Input Encoders
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.text_proj = nn.Linear(768, hidden_dim)
        self.audio_proj = nn.Linear(74, hidden_dim)
        self.visual_proj = nn.Linear(713, hidden_dim)

        # Domain Separation
        self.text_invariant = self._build_domain_encoder(hidden_dim)
        self.text_private = self._build_domain_encoder(hidden_dim)
        self.audio_invariant = self._build_domain_encoder(hidden_dim)
        self.audio_private = self._build_domain_encoder(hidden_dim)
        self.visual_invariant = self._build_domain_encoder(hidden_dim)
        self.visual_private = self._build_domain_encoder(hidden_dim)

        # Bottleneck Tokens
        self.bottleneck_tokens = nn.Parameter(
            torch.randn(num_bottleneck_tokens, hidden_dim) * 0.02
        )

        # Cross-Attention
        num_heads = 8
        self.text_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout, batch_first=True)
        self.audio_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout, batch_first=True)
        self.visual_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout, batch_first=True)

        # Reconstruction
        self.text_reconstruct = self._build_reconstruct_head(hidden_dim, 768)
        self.audio_reconstruct = self._build_reconstruct_head(hidden_dim, 74)
        self.visual_reconstruct = self._build_reconstruct_head(hidden_dim, 713)

        # Fusion & Classification
        fusion_dim = hidden_dim * 6
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
    def _build_domain_encoder(hidden_dim):
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    @staticmethod
    def _build_reconstruct_head(hidden_dim, output_dim):
        return nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    @staticmethod
    def _masked_mean(seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Masked mean pooling: seq(B,T,D) x mask(B,T) -> (B,D)"""
        m = mask.unsqueeze(-1).float()
        return (seq * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)

    def forward(self, input_ids, attention_mask, audio=None, visual=None,
                audio_mask=None, visual_mask=None, return_domains=False):

        # Encode
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_enc = self.text_proj(bert_out.last_hidden_state[:, 0, :])

        audio_enc = self.audio_proj(audio)
        if audio_mask is not None:
            audio_enc = self._masked_mean(audio_enc, audio_mask)
        else:
            audio_enc = audio_enc.mean(dim=1)

        visual_enc = self.visual_proj(visual)
        if visual_mask is not None:
            visual_enc = self._masked_mean(visual_enc, visual_mask)
        else:
            visual_enc = visual_enc.mean(dim=1)

        # Domain Separation
        text_inv = self.text_invariant(text_enc)
        text_priv = self.text_private(text_enc)
        audio_inv = self.audio_invariant(audio_enc)
        audio_priv = self.audio_private(audio_enc)
        visual_inv = self.visual_invariant(visual_enc)
        visual_priv = self.visual_private(visual_enc)

        # Reconstruction
        text_recon = self.text_reconstruct(torch.cat([text_inv, text_priv], dim=1))
        audio_recon = self.audio_reconstruct(torch.cat([audio_inv, audio_priv], dim=1))
        visual_recon = self.visual_reconstruct(torch.cat([visual_inv, visual_priv], dim=1))

        # Bottleneck Cross-Attention
        batch_size = text_enc.size(0)
        btokens = self.bottleneck_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        text_priv_refined, _ = self.text_attention(
            text_priv.unsqueeze(1), btokens, btokens
        )
        text_priv_refined = text_priv_refined.squeeze(1)

        audio_priv_refined, _ = self.audio_attention(
            audio_priv.unsqueeze(1), btokens, btokens
        )
        audio_priv_refined = audio_priv_refined.squeeze(1)

        visual_priv_refined, _ = self.visual_attention(
            visual_priv.unsqueeze(1), btokens, btokens
        )
        visual_priv_refined = visual_priv_refined.squeeze(1)

        # Fusion
        fused = torch.cat([
            text_inv, text_priv_refined,
            audio_inv, audio_priv_refined,
            visual_inv, visual_priv_refined
        ], dim=1)

        fused = self.fusion_head(fused)
        logits = self.classifier(fused)

        if return_domains:
            domain_data = {
                'text_invariant': text_inv,
                'text_private': text_priv_refined,
                'audio_invariant': audio_inv,
                'audio_private': audio_priv_refined,
                'visual_invariant': visual_inv,
                'visual_private': visual_priv_refined,
            }
            recon_data = {
                'text_recon': text_recon,
                'text_original': text_enc,
                'audio_recon': audio_recon,
                'audio_original': audio_enc,
                'visual_recon': visual_recon,
                'visual_original': visual_enc,
            }
            return logits, domain_data, recon_data

        return logits
