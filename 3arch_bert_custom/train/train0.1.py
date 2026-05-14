import pickle
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import math

# =========================
# 1. LOAD DATA
# =========================
with open("datasets/mosei_bert_custom.pkl", "rb") as f:
    data = pickle.load(f)

train_data = data["train"]
valid_data = data["valid"]
test_data  = data["test"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

EMOTION_NAMES = ["Happy", "Sad", "Anger", "Surprise", "Disgust", "Fear"]

# =========================
# 2. DATASET
# =========================
class MOSEIDataset(Dataset):
    def __init__(self, data):
        self.text   = data["text"]
        self.audio  = data["audio"]
        self.video  = data["vision"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text  = torch.tensor(self.text[idx],  dtype=torch.float32)
        audio = torch.tensor(self.audio[idx], dtype=torch.float32)
        video = torch.tensor(self.video[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return text, audio, video, label


train_dataset = MOSEIDataset(train_data)
valid_dataset = MOSEIDataset(valid_data)
test_dataset  = MOSEIDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=0)

# =========================
# 3. MULTI-LABEL POSITIVE CLASS WEIGHTS
# =========================
labels_np = np.array(train_data["labels"], dtype=np.float32)  # (N, 6)

pos_counts = labels_np.sum(axis=0)
neg_counts = labels_np.shape[0] - pos_counts

pos_weight = neg_counts / (pos_counts + 1e-6)
pos_weight = np.sqrt(pos_weight)
pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)

print("Positive counts:", pos_counts.astype(int))
print("Negative counts:", neg_counts.astype(int))
print("POS weights:    ", pos_weight.detach().cpu().numpy().round(3))

# =========================
# 3.5 FOCAL LOSS
# =========================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        # BCE с pos_weight
        bce = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight,
            reduction='none'
        )
        # вероятности
        probs    = torch.sigmoid(logits)
        # p_t = prob если target=1, (1-prob) если target=0
        p_t      = probs * targets + (1 - probs) * (1 - targets)
        # фокусирующий множитель
        focal_w  = (1 - p_t) ** self.gamma

        return (focal_w * bce).mean()

# =========================
# 4. BOTTLENECK ATTENTION FUSION
# =========================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class BottleneckAttentionFusion(nn.Module):
    """
    Bottleneck Attention Fusion (BAF).
    Reference: "Attention Bottlenecks for Multimodal Fusion" (NeurIPS 2021)
    """
    def __init__(self, d_model, num_heads=4, num_bottleneck=16, dropout=0.1):
        super().__init__()
        self.bottleneck = nn.Parameter(torch.randn(1, num_bottleneck, d_model) * 0.02)

        self.attn_t  = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_a  = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_v  = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_bt = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_ba = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_bv = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

        self.norm_t   = nn.LayerNorm(d_model)
        self.norm_a   = nn.LayerNorm(d_model)
        self.norm_v   = nn.LayerNorm(d_model)
        self.norm_b   = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.drop     = nn.Dropout(dropout)

    def forward(self, t, a, v, bn=None):
        B  = t.size(0)
        if bn is None:
            bn = self.bottleneck.expand(B, -1, -1)

        bt, _ = self.attn_bt(bn, t, t)
        ba, _ = self.attn_ba(bn, a, a)
        bv, _ = self.attn_bv(bn, v, v)
        bn = self.norm_b(bn + self.drop(bt + ba + bv) / 3.0)

        t2, _ = self.attn_t(t, bn, bn)
        a2, _ = self.attn_a(a, bn, bn)
        v2, _ = self.attn_v(v, bn, bn)

        t = self.norm_t(t + self.drop(t2))
        a = self.norm_a(a + self.drop(a2))
        v = self.norm_v(v + self.drop(v2))

        bn = self.norm_ffn(bn + self.drop(self.ffn(bn)))
        return bn, t, a, v

class ModalityEncoder(nn.Module):
    def __init__(self, in_dim, d_model, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.pe = PositionalEncoding(d_model, dropout=dropout)

    def forward(self, x):
        return self.pe(self.proj(x))

# Аудио: 1D-CNN
# COVAREP = временной ряд акустических признаков
# CNN извлекает локальные паттерны (изменения pitch, тембра)
# Обоснование: fnins (Xia et al., 2022)
class AudioCNNEncoder(nn.Module):
    def __init__(self, in_dim, d_model, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )
        self.pe = PositionalEncoding(d_model, dropout=dropout)

    def forward(self, x):
        # x: (B, seq_len, in_dim)
        x = x.transpose(1, 2)    # → (B, in_dim, seq_len)
        x = self.conv(x)          # → (B, d_model, seq_len)
        x = x.transpose(1, 2)    # → (B, seq_len, d_model)
        return self.pe(x)


# Видео: BiLSTM
# OpenFace = движения лица во времени
# BiLSTM читает последовательность вперёд и назад
# Обоснование: fnins (Xia et al., 2022)
class VideoBiLSTMEncoder(nn.Module):
    def __init__(self, in_dim, d_model, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2, 
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, seq_len, in_dim)
        x = self.proj(x)          # → (B, seq_len, d_model)
        x, _ = self.bilstm(x)     # → (B, seq_len, d_model)
        return self.norm(x)


class MultimodalEmotionModel(nn.Module):
    def __init__(self, text_dim, audio_dim, video_dim,
                 d_model=128, num_heads=4, num_bottleneck=16,
                 num_fusion_layers=2, num_classes=6, dropout=0.2):
        super().__init__()

        # Text: Linear(768→128) → Transformer
        self.text_enc = ModalityEncoder(text_dim, d_model, dropout)
        self.text_sa  = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, d_model*4, dropout, batch_first=True),
            num_layers=2
        )

        # Audio: 1D-CNN
        self.audio_enc = AudioCNNEncoder(audio_dim, d_model, dropout)

        # Video: BiLSTM
        self.video_enc = VideoBiLSTMEncoder(video_dim, d_model, dropout)

        # Bottleneck Fusion
        self.fusion_layers = nn.ModuleList([
            BottleneckAttentionFusion(d_model, num_heads, num_bottleneck, dropout)
            for _ in range(num_fusion_layers)
        ])

        # Classifier на bottleneck токенах
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self.text_classifier  = nn.Linear(d_model, num_classes)
        self.audio_classifier = nn.Linear(d_model, num_classes)
        self.video_classifier = nn.Linear(d_model, num_classes)

    def forward(self, text, audio, video):
        t = self.text_sa(self.text_enc(text))  # (B, 50, 128)
        a = self.audio_enc(audio)               # (B, 60, 128)
        v = self.video_enc(video)               # (B, 60, 128)

        # unimodal логиты — после SEM
        t_logits = self.text_classifier(t.mean(dim=1))
        a_logits = self.audio_classifier(a.mean(dim=1))
        v_logits = self.video_classifier(v.mean(dim=1))

        # bottleneck fusion
        bn = None
        for layer in self.fusion_layers:
            bn, t, a, v = layer(t, a, v, bn)

        f_logits = self.classifier(bn.mean(dim=1))
        return f_logits, t_logits, a_logits, v_logits


# =========================
# 5. INIT
# =========================
text_dim  = train_data["text"][0].shape[1]
audio_dim = train_data["audio"][0].shape[1]
video_dim = train_data["vision"][0].shape[1]
print(f"TEXT: {text_dim}, AUDIO: {audio_dim}, VIDEO: {video_dim}")

model = MultimodalEmotionModel(
    text_dim=text_dim,
    audio_dim=audio_dim,
    video_dim=video_dim,
    d_model=128,
    num_heads=4,
    num_bottleneck=16,
    num_fusion_layers=2,
    num_classes=6,
    dropout=0.2,
).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params:,}")

# Multi-label loss with positive class weights
criterion = FocalLoss(gamma=2.0, pos_weight=pos_weight)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

NUM_EPOCHS = 50
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-4,     
    steps_per_epoch=len(train_loader),
    epochs=NUM_EPOCHS,
    pct_start=0.1,
)


# =========================
# 6. TRAIN
# =========================
def train_epoch():
    model.train()
    losses = []

    for text, audio, video, labels in train_loader:
        text   = text.to(device)
        audio  = audio.to(device)
        video  = video.to(device)
        labels = labels.to(device).float()

        optimizer.zero_grad()
        f_logits, t_logits, a_logits, v_logits = model(text, audio, video)
        loss_f = criterion(f_logits, labels)
        loss_t = criterion(t_logits, labels)
        loss_a = criterion(a_logits, labels)
        loss_v = criterion(v_logits, labels)
        loss = (loss_f + 0.3*loss_t + 0.2*loss_a + 0.2*loss_v) / 1.7

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

    return np.mean(losses)
# =========================
# 7. EVALUATE
# =========================
def evaluate(loader, split_name="Valid", threshold=0.5):
    model.eval()
    preds, true = [], []

    with torch.no_grad():
        for text, audio, video, labels in loader:
            text, audio, video = text.to(device), audio.to(device), video.to(device)
            f_logits, _, _, _ = model(text, audio, video)
            probs = torch.sigmoid(f_logits)
            preds.append((probs >= threshold).int().cpu().numpy())
            true.append(labels.numpy())

    preds = np.vstack(preds)
    true  = np.vstack(true).astype(int)

    macro_f1 = f1_score(true, preds, average="macro",    zero_division=0)
    wf1      = f1_score(true, preds, average="weighted", zero_division=0)
    per_acc  = [accuracy_score(true[:, i], preds[:, i]) for i in range(6)]

    return macro_f1, wf1, np.mean(per_acc), per_acc, preds, true


# =========================
# 8. TRAIN LOOP
# =========================
best_macro_f1 = -1.0
patience, no_improve = 15, 0
os.makedirs("/content/drive/MyDrive/Дипломка_правильная/checkpoints", exist_ok=True)
best_path = "/content/drive/MyDrive/Дипломка_правильная/checkpoints/best_model_bert_cnn_bilstm.pt"

print(f"\n{'Epoch':<8} {'Loss':>8} {'Acc':>8} {'MacroF1':>10} {'WF1':>8}")
print("-" * 48)

for epoch in range(1, NUM_EPOCHS + 1):
    loss = train_epoch()
    macro_f1, wf1, acc, _, _, _ = evaluate(valid_loader)

    print(f"{epoch:02d}/{NUM_EPOCHS}  {loss:>8.4f}  {acc*100:>7.1f}%  {macro_f1*100:>9.1f}%  {wf1*100:>7.1f}%")

    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        no_improve = 0
        torch.save(model.state_dict(), best_path)
        print(f"  ✓ Best Macro-F1 = {best_macro_f1*100:.1f}%")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}.")
            break


# =========================
# 9. ФИНАЛЬНЫЙ ТЕСТ
# =========================
model.load_state_dict(torch.load(best_path, map_location=device))
macro_f1, wf1, avg_acc, per_acc, preds, true = evaluate(test_loader, "Test")

print("\n" + "="*45)
print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ (Test)")
print("="*45)
print(f"\n{'Эмоция':<12} {'Accuracy':>10} {'F1':>8}")
print("-" * 32)
for i, name in enumerate(EMOTION_NAMES):
    f1 = f1_score(true[:, i], preds[:, i], zero_division=0)
    print(f"{name:<12} {per_acc[i]*100:>9.1f}% {f1*100:>7.1f}%")
print("-" * 32)
print(f"{'Среднее':<12} {avg_acc*100:>9.1f}% {macro_f1*100:>7.1f}%")
print(f"\n  Macro-F1    : {macro_f1*100:.1f}%")
print(f"  Weighted-F1 : {wf1*100:.1f}%")


# =========================
# 10. THRESHOLD TUNING
# =========================
all_probs, all_labels = [], []
model.eval()
with torch.no_grad():
    for text, audio, video, labels in valid_loader:
        text, audio, video = text.to(device), audio.to(device), video.to(device)
        f_logits, _, _, _ = model(text, audio, video)  # ← исправлено
        all_probs.append(torch.sigmoid(f_logits).cpu().numpy())
        all_labels.append(labels.numpy())

all_probs  = np.vstack(all_probs)
all_labels = np.vstack(all_labels).astype(int)

best_thresholds = []
for i in range(6):
    best_t = max(np.arange(0.1, 0.7, 0.05),
                 key=lambda t: f1_score(all_labels[:, i],
                                        (all_probs[:, i] >= t).astype(int),
                                        zero_division=0))
    best_thresholds.append(best_t)

all_preds, all_true = [], []
with torch.no_grad():
    for text, audio, video, labels in test_loader:
        text, audio, video = text.to(device), audio.to(device), video.to(device)
        f_logits, _, _, _ = model(text, audio, video)  # ← исправлено
        probs = torch.sigmoid(f_logits).cpu().numpy()
        p = np.stack([(probs[:, i] >= best_thresholds[i]).astype(int)
                      for i in range(6)], axis=1)
        all_preds.append(p)
        all_true.append(labels.numpy())

all_preds = np.vstack(all_preds)
all_true  = np.vstack(all_true).astype(int)

print("\n" + "="*45)
print("ПОСЛЕ THRESHOLD TUNING (Test)")
print("="*45)
print(f"\n{'Эмоция':<12} {'Threshold':>10} {'Accuracy':>10} {'F1':>8}")
print("-" * 43)
for i, name in enumerate(EMOTION_NAMES):
    acc = accuracy_score(all_true[:, i], all_preds[:, i])
    f1  = f1_score(all_true[:, i], all_preds[:, i], zero_division=0)
    print(f"{name:<12} {best_thresholds[i]:>10.2f} {acc*100:>9.1f}% {f1*100:>7.1f}%")

macro_t = f1_score(all_true, all_preds, average='macro',    zero_division=0)
wf1_t   = f1_score(all_true, all_preds, average='weighted', zero_division=0)
avg_t   = np.mean([accuracy_score(all_true[:, i], all_preds[:, i]) for i in range(6)])
print("-" * 43)
print(f"\n  Средний Accuracy : {avg_t*100:.1f}%")
print(f"  Macro-F1         : {macro_t*100:.1f}%")
print(f"  Weighted-F1      : {wf1_t*100:.1f}%")