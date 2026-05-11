import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import math

# =========================
# 1. LOAD DATA
# =========================
with open("mosei_emotion_aligned_60.pkl", "rb") as f:
    data = pickle.load(f)

train_data = data["train"]
valid_data = data["valid"]
test_data  = data["test"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =========================
# 2. DATASET
# =========================
class MOSEIDataset(Dataset):
    def __init__(self, data):
        self.text   = data["text"]
        self.audio  = data["audio"]
        self.video = data["vision"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text  = torch.tensor(self.text[idx], dtype=torch.float32)
        audio = torch.tensor(self.audio[idx], dtype=torch.float32)
        video = torch.tensor(self.video[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return text, audio, video, label


train_loader = DataLoader(MOSEIDataset(train_data), batch_size=32, shuffle=True)
valid_loader = DataLoader(MOSEIDataset(valid_data), batch_size=32)
test_loader  = DataLoader(MOSEIDataset(test_data), batch_size=32)

# =========================
# 3. CLASS WEIGHTS (smoothed)
# =========================
labels_all = np.array([int(torch.argmax(torch.tensor(y))) for y in train_data["labels"]])

raw_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels_all),
    y=labels_all
)

class_weights = torch.tensor(raw_weights, dtype=torch.float32).to(device)
print("Class weights:", class_weights)

# =========================
# 4. MODEL PARTS
# =========================
class ModalityEncoder(nn.Module):
    def __init__(self, in_dim, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)


class BottleneckFusion(nn.Module):
    def __init__(self, d_model, num_heads=4, bottleneck=4):
        super().__init__()

        self.bn = nn.Parameter(torch.randn(1, bottleneck, d_model) * 0.02)

        self.attn_b_t = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.attn_b_a = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.attn_b_v = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

        self.attn_t = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.attn_a = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.attn_v = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, t, a, v):
        B = t.size(0)
        bn = self.bn.expand(B, -1, -1)

        bt, _ = self.attn_b_t(bn, t, t)
        ba, _ = self.attn_b_a(bn, a, a)
        bv, _ = self.attn_b_v(bn, v, v)

        bottleneck = self.norm(bn + (bt + ba + bv) / 3)

        t2, _ = self.attn_t(t, bottleneck, bottleneck)
        a2, _ = self.attn_a(a, bottleneck, bottleneck)
        v2, _ = self.attn_v(v, bottleneck, bottleneck)

        t = self.norm(t + t2)
        a = self.norm(a + a2)
        v = self.norm(v + v2)

        return t, a, v


# =========================
# 5. FULL MODEL
# =========================
class MultimodalModel(nn.Module):
    def __init__(self, text_dim, audio_dim, video_dim, d_model=128, num_classes=6):
        super().__init__()

        self.text_enc  = ModalityEncoder(text_dim, d_model)
        self.audio_enc = ModalityEncoder(audio_dim, d_model)
        self.video_enc = ModalityEncoder(video_dim, d_model)

        self.fusion = BottleneckFusion(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, text, audio, video):

        t = self.text_enc(text)
        a = self.audio_enc(audio)
        v = self.video_enc(video)

        t, a, v = self.fusion(t, a, v)

        fused = torch.cat([
            t.mean(1),
            a.mean(1),
            v.mean(1)
        ], dim=-1)

        return self.classifier(fused)


# =========================
# 6. INIT
# =========================
text_dim  = train_data["text"][0].shape[1]
audio_dim = train_data["audio"][0].shape[1]
video_dim = train_data["vision"][0].shape[1]

model = MultimodalModel(text_dim, audio_dim, video_dim).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

# =========================
# 7. TRAIN
# =========================
def train():
    model.train()
    total_loss = []

    for t, a, v, y in train_loader:
        t, a, v = t.to(device), a.to(device), v.to(device)
        y = torch.argmax(y.to(device), dim=1)

        optimizer.zero_grad()
        out = model(t, a, v)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

    return np.mean(total_loss)


# =========================
# 8. EVAL
# =========================
def eval(loader, name="Val"):
    model.eval()
    preds, true = [], []

    with torch.no_grad():
        for t, a, v, y in loader:
            t, a, v = t.to(device), a.to(device), v.to(device)
            y = torch.argmax(y.to(device), dim=1)

            out = model(t, a, v)
            p = torch.argmax(out, dim=1)

            preds.extend(p.cpu().numpy())
            true.extend(y.cpu().numpy())

    print(f"\n[{name}]")
    print(classification_report(true, preds, zero_division=0))

    return f1_score(true, preds, average="weighted")


# =========================
# 9. TRAIN LOOP
# =========================
best_f1 = 0
patience = 12
wait = 0

for epoch in range(1, 31):

    loss = train()
    f1 = eval(valid_loader, "Valid")

    print(f"\nEpoch {epoch} | Loss {loss:.4f} | Val F1 {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        wait = 0
        torch.save(model.state_dict(), "best.pt")
        print("Saved best model")
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping")
            break


# =========================
# 10. TEST
# =========================
model.load_state_dict(torch.load("best.pt", map_location=device))
eval(test_loader, "TEST")