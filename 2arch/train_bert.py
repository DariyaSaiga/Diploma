import pickle
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertModel
import math

# =========================
# 1. LOAD DATA
# =========================
with open("datasets/mosei_finetune_bert.pkl", "rb") as f:
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
        self.input_ids      = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        self.audio          = data["audio"]
        self.video          = data["vision"]
        self.labels         = data["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.input_ids[idx],      dtype=torch.long),
            torch.tensor(self.attention_mask[idx],  dtype=torch.long),
            torch.tensor(self.audio[idx],           dtype=torch.float32),
            torch.tensor(self.video[idx],           dtype=torch.float32),
            torch.tensor(self.labels[idx],          dtype=torch.float32),
        )

BATCH_SIZE = 16

train_loader = DataLoader(MOSEIDataset(train_data), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
valid_loader = DataLoader(MOSEIDataset(valid_data), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(MOSEIDataset(test_data),  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# =========================
# 3. POS WEIGHTS
# =========================
labels_np  = np.array(train_data["labels"], dtype=np.float32)
pos_counts = labels_np.sum(axis=0)
neg_counts = labels_np.shape[0] - pos_counts
pos_weight = torch.tensor(np.sqrt(neg_counts / (pos_counts + 1e-6)), dtype=torch.float32).to(device)

print("POS weights:", pos_weight.detach().cpu().numpy().round(3))

# =========================
# 3.5 FOCAL LOSS
# =========================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce    = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction='none')
        probs  = torch.sigmoid(logits)
        p_t    = probs * targets + (1 - probs) * (1 - targets)
        return ((1 - p_t) ** self.gamma * bce).mean()

# =========================
# 4. АРХИТЕКТУРА
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
        return self.dropout(x + self.pe[:, :x.size(1)])


class BottleneckAttentionFusion(nn.Module):
    """Bottleneck Attention Fusion. Reference: NeurIPS 2021 (Nagrani et al.)"""
    def __init__(self, d_model, num_heads=4, num_bottleneck=16, dropout=0.1):
        super().__init__()
        self.bottleneck = nn.Parameter(torch.randn(1, num_bottleneck, d_model) * 0.02)
        self.attn_bt = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_ba = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_bv = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_t  = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_a  = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_v  = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * 4, d_model),
        )
        self.norm_t   = nn.LayerNorm(d_model)
        self.norm_a   = nn.LayerNorm(d_model)
        self.norm_v   = nn.LayerNorm(d_model)
        self.norm_b   = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.drop     = nn.Dropout(dropout)

    def forward(self, t, a, v, bn=None):
        B = t.size(0)
        if bn is None:
            bn = self.bottleneck.expand(B, -1, -1)
        bt, _ = self.attn_bt(bn, t, t)
        ba, _ = self.attn_ba(bn, a, a)
        bv, _ = self.attn_bv(bn, v, v)
        bn = self.norm_b(bn + self.drop((bt + ba + bv) / 3.0))
        t2, _ = self.attn_t(t, bn, bn)
        a2, _ = self.attn_a(a, bn, bn)
        v2, _ = self.attn_v(v, bn, bn)
        t = self.norm_t(t + self.drop(t2))
        a = self.norm_a(a + self.drop(a2))
        v = self.norm_v(v + self.drop(v2))
        bn = self.norm_ffn(bn + self.drop(self.ffn(bn)))
        return bn, t, a, v


class AudioCNNEncoder(nn.Module):
    """1D-CNN для COVAREP аудио фич. Reference: fnins (Xia et al., 2022)"""
    def __init__(self, in_dim, d_model, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model), nn.ReLU(),
        )
        self.pe = PositionalEncoding(d_model, dropout=dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return self.pe(x)


class VideoBiLSTMEncoder(nn.Module):
    """BiLSTM для OpenFace видео фич. Reference: fnins (Xia et al., 2022)"""
    def __init__(self, in_dim, d_model, dropout=0.1):
        super().__init__()
        self.proj   = nn.Linear(in_dim, d_model)
        self.bilstm = nn.LSTM(d_model, d_model // 2, num_layers=2,
                              batch_first=True, bidirectional=True, dropout=dropout)
        self.norm   = nn.LayerNorm(d_model)

    def forward(self, x):
        x, _ = self.bilstm(self.proj(x))
        return self.norm(x)


class MultimodalEmotionModelBERT(nn.Module):
    def __init__(self, audio_dim, video_dim,
                 d_model=128, num_heads=4, num_bottleneck=16,
                 num_fusion_layers=2, num_classes=6, dropout=0.2,
                 bert_finetune_layers=3):
        super().__init__()

        # BERT — частичный fine-tuning последних N слоёв
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Замораживаем все слои кроме последних bert_finetune_layers
        for param in self.bert.parameters():
            param.requires_grad = False

        # Размораживаем последние bert_finetune_layers слоёв
        total_layers = len(self.bert.encoder.layer)
        for i in range(total_layers - bert_finetune_layers, total_layers):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = True

        # Размораживаем pooler
        for param in self.bert.pooler.parameters():
            param.requires_grad = True

        # Text проекция: 768 → d_model
        self.text_proj = nn.Sequential(
            nn.Linear(768, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.text_pe = PositionalEncoding(d_model, dropout=dropout)
        self.text_sa = nn.TransformerEncoder(
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

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        # Auxiliary unimodal classifiers
        self.text_classifier  = nn.Linear(d_model, num_classes)
        self.audio_classifier = nn.Linear(d_model, num_classes)
        self.video_classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask, audio, video):
        # BERT: получаем все токены последнего слоя
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        t = bert_out.last_hidden_state          # (B, 50, 768)
        t = self.text_pe(self.text_proj(t))     # (B, 50, 128)
        t = self.text_sa(t)                     # (B, 50, 128)

        a = self.audio_enc(audio)               # (B, 60, 128)
        v = self.video_enc(video)               # (B, 60, 128)

        # Auxiliary logits
        t_logits = self.text_classifier(t.mean(dim=1))
        a_logits = self.audio_classifier(a.mean(dim=1))
        v_logits = self.video_classifier(v.mean(dim=1))

        # Bottleneck fusion
        bn = None
        for layer in self.fusion_layers:
            bn, t, a, v = layer(t, a, v, bn)

        f_logits = self.classifier(bn.mean(dim=1))
        return f_logits, t_logits, a_logits, v_logits


# =========================
# 5. INIT
# =========================
audio_dim = train_data["audio"][0].shape[1]
video_dim = train_data["vision"][0].shape[1]
print(f"AUDIO: {audio_dim}, VIDEO: {video_dim}")

model = MultimodalEmotionModelBERT(
    audio_dim=audio_dim,
    video_dim=video_dim,
    d_model=128,
    num_heads=4,
    num_bottleneck=16,
    num_fusion_layers=2,
    num_classes=6,
    dropout=0.2,
    bert_finetune_layers=3,
).to(device)

# Считаем параметры
total  = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters:     {total:,}")
print(f"Trainable parameters: {trainable:,}")

criterion = FocalLoss(gamma=2.0, pos_weight=pos_weight)

# Два отдельных optimizer с разными lr
# BERT lr = 1e-5 (маленький чтобы не сломать претренированные веса)
# Остальное lr = 1e-4
bert_params  = [p for p in model.bert.parameters() if p.requires_grad]
other_params = [p for p in model.parameters() if p.requires_grad and
                not any(p is bp for bp in bert_params)]

optimizer = torch.optim.AdamW([
    {'params': bert_params,  'lr': 1e-5},
    {'params': other_params, 'lr': 1e-4},
], weight_decay=1e-2)

NUM_EPOCHS        = 30
ACCUM_STEPS       = 2   # gradient accumulation: эффективный batch = 16*2 = 32
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[1e-5, 1e-4],
    steps_per_epoch=math.ceil(len(train_loader) / ACCUM_STEPS),
    epochs=NUM_EPOCHS,
    pct_start=0.1,
)

# =========================
# 6. TRAIN
# =========================
def train_epoch():
    model.train()
    losses = []
    optimizer.zero_grad()

    for step, (input_ids, attention_mask, audio, video, labels) in enumerate(train_loader):
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        audio          = audio.to(device)
        video          = video.to(device)
        labels         = labels.to(device).float()

        f_logits, t_logits, a_logits, v_logits = model(input_ids, attention_mask, audio, video)

        loss = (criterion(f_logits, labels)
              + 0.3 * criterion(t_logits, labels)
              + 0.2 * criterion(a_logits, labels)
              + 0.2 * criterion(v_logits, labels)) / (1.7 * ACCUM_STEPS)

        loss.backward()

        if (step + 1) % ACCUM_STEPS == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        losses.append(loss.item() * ACCUM_STEPS)

    if (step + 1) % ACCUM_STEPS != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return np.mean(losses)

# =========================
# 7. EVALUATE
# =========================
def evaluate(loader, threshold=0.5):
    model.eval()
    preds, true = [], []

    with torch.no_grad():
        for input_ids, attention_mask, audio, video, labels in loader:
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            audio          = audio.to(device)
            video          = video.to(device)

            f_logits, _, _, _ = model(input_ids, attention_mask, audio, video)
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
patience, no_improve = 10, 0
os.makedirs("/content/drive/MyDrive/Дипломка_правильная/checkpoints", exist_ok=True)
best_path = "/content/drive/MyDrive/Дипломка_правильная/checkpoints/best_model_bert_finetune.pt"

print(f"\n{'Epoch':<8} {'Loss':>8} {'Acc':>8} {'MacroF1':>10} {'WF1':>8}")
print("-" * 48)

for epoch in range(1, NUM_EPOCHS + 1):
    loss = train_epoch()
    macro_f1, wf1, acc, _, _, _ = evaluate(valid_loader)

    print(f"{epoch:02d}/{NUM_EPOCHS}  {loss:>8.4f}  {acc*100:>7.1f}%  {macro_f1*100:>9.1f}%  {wf1*100:>7.1f}%")

    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        no_improve    = 0
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
macro_f1, wf1, avg_acc, per_acc, preds, true = evaluate(test_loader)

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
    for input_ids, attention_mask, audio, video, labels in valid_loader:
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        audio, video = audio.to(device), video.to(device)
        f_logits, _, _, _ = model(input_ids, attention_mask, audio, video)
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
    for input_ids, attention_mask, audio, video, labels in test_loader:
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        audio, video = audio.to(device), video.to(device)
        f_logits, _, _, _ = model(input_ids, attention_mask, audio, video)
        probs = torch.sigmoid(f_logits).cpu().numpy()
        p = np.stack([(probs[:, i] >= best_thresholds[i]).astype(int) for i in range(6)], axis=1)
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