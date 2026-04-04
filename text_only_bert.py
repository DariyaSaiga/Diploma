import torch
import torch.nn as nn
from transformers import BertModel


class TextOnlyBert(nn.Module):
    def __init__(self, num_classes=6, hidden_dim=128, dropout=0.3):
        super().__init__()

        # BERT — предобученная модель, держим замороженной
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False

        # После BERT берём CLS токен — вектор [768]
        # Проецируем его в hidden_dim [128], потом в num_classes [6]
        self.classifier = nn.Sequential(
            nn.Linear(768, hidden_dim),  # 768 → 128
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),  # 128 → 6
        )

    def forward(self, input_ids, attention_mask):
        # input_ids:      (B, L) — токены
        # attention_mask: (B, L) — 1 где реальные слова, 0 где padding

        # Прогоняем через BERT
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Берём CLS токен — первая позиция в последнем hidden state
        cls = bert_out.last_hidden_state[:, 0, :]  # (B, 768)

        # Классифицируем
        logits = self.classifier(cls)  # (B, 6)

        return logits