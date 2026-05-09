import torch.nn as nn
from core.encoders import BertTextEncoder


class TextOnlyBERT(nn.Module):
    """BERT-based text-only baseline for emotion classification.

    Architecture: BERT → CLS vector → dropout → classifier → logits
    """

    def __init__(self, num_classes=6, hidden_dim=128, dropout=0.3, freeze_bert="full"):
        super().__init__()
        self.text_encoder = BertTextEncoder(freeze_bert=freeze_bert)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch, return_domains=False):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        text_seq = self.text_encoder(input_ids, attention_mask)  # (B, T, 768)
        cls = text_seq[:, 0, :]                                   # CLS token (B, 768)
        logits = self.classifier(self.dropout(cls))               # (B, num_classes)
        if return_domains:
            return logits, {}
        return logits
