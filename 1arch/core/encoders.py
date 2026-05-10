import torch.nn as nn
from transformers import BertModel


class BertTextEncoder(nn.Module):
    """BERT encoder returning sequence-level output (B, T, 768)."""

    def __init__(self, freeze_bert="full"):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self._apply_freeze(freeze_bert)

    def _apply_freeze(self, mode):
        if mode is True or mode == "full":
            for p in self.bert.parameters():
                p.requires_grad = False
        elif mode == "partial":
            for p in self.bert.parameters():
                p.requires_grad = False
            for layer in self.bert.encoder.layer[-4:]:
                for p in layer.parameters():
                    p.requires_grad = True
        # mode == "none" or False: all params trainable

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state  # (B, T_text, 768)


class AudioCNNEncoder(nn.Module):
    """3-layer 1D-CNN encoder for audio. Input: (B, T, D) → Output: (B, T, hidden_dim)."""

    def __init__(self, audio_input_dim=74, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(audio_input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, audio):
        # audio: (B, T, audio_input_dim)
        x = audio.transpose(1, 2)  # (B, audio_input_dim, T)
        x = self.convs(x)          # (B, hidden_dim, T)
        return x.transpose(1, 2)   # (B, T, hidden_dim)


class VideoBiLSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder for video. Input: (B, T, D) → Output: (B, T, lstm_hidden*2)."""

    def __init__(self, visual_input_dim=713, lstm_hidden=128, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=visual_input_dim,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, visual):
        # visual: (B, T, visual_input_dim)
        out, _ = self.lstm(visual)  # (B, T, lstm_hidden*2)
        return self.dropout(out)
