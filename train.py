import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# =========================
# 1. LOAD DATA
# =========================

base = "/content/MELD.Features.Models/features"

def load(name):
    with open(os.path.join(base, name), "rb") as f:
        return pickle.load(f)

text_data = load("text_glove_average_emotion.pkl")
audio_data = load("audio_embeddings_feature_selection_emotion.pkl")

print("TEXT TYPE:", type(text_data))
print("AUDIO TYPE:", type(audio_data))
print("LEN TEXT:", len(text_data))
print("LEN AUDIO:", len(audio_data))

# =========================
# 2. SAFETY CHECK (CRITICAL)
# =========================

assert len(text_data) == len(audio_data), "TEXT and AUDIO length mismatch!"

# =========================
# 3. FEATURE + LABEL EXTRACT
# =========================

def get_feat(x):
    if isinstance(x, dict):
        if "feature" in x:
            return np.array(x["feature"])
        return np.array(list(x.values())[0])
    return np.array(x)

def get_label(x):
    if isinstance(x, dict):
        return x.get("label", x.get("emotion", 0))
    return 0

# =========================
# 4. SPLIT INDEXES (IMPORTANT)
# =========================

idx = np.arange(len(text_data))

train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)
train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=42)

# =========================
# 5. DATASET
# =========================

class MELDDataset(Dataset):
    def __init__(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]

        t = torch.tensor(get_feat(text_data[idx]), dtype=torch.float32)
        a = torch.tensor(get_feat(audio_data[idx]), dtype=torch.float32)
        y = torch.tensor(get_label(text_data[idx]), dtype=torch.long)

        return t, a, y

# =========================
# 6. DATALOADERS
# =========================

train_loader = DataLoader(MELDDataset(train_idx), batch_size=32, shuffle=True)
val_loader   = DataLoader(MELDDataset(val_idx), batch_size=32)
test_loader  = DataLoader(MELDDataset(test_idx), batch_size=32)

# =========================
# 7. MODEL
# =========================

class FusionModel(nn.Module):
    def __init__(self, t_dim, a_dim, hidden=128, num_classes=6):
        super().__init__()

        self.text = nn.Sequential(
            nn.Linear(t_dim, hidden),
            nn.ReLU()
        )

        self.audio = nn.Sequential(
            nn.Linear(a_dim, hidden),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, t, a):
        t = self.text(t)
        a = self.audio(a)
        x = torch.cat([t, a], dim=1)
        return self.classifier(x)

# =========================
# 8. INIT MODEL
# =========================

t_dim = get_feat(text_data[0]).shape[-1]
a_dim = get_feat(audio_data[0]).shape[-1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FusionModel(t_dim, a_dim).to(device)

# =========================
# 9. LOSS / OPTIM
# =========================

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# =========================
# 10. TRAIN
# =========================

def train_one_epoch():
    model.train()
    losses = []

    for t, a, y in train_loader:
        t, a, y = t.to(device), a.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(t, a)

        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)

# =========================
# 11. EVAL
# =========================

def evaluate(loader):
    model.eval()
    preds, true = [], []

    with torch.no_grad():
        for t, a, y in loader:
            t, a = t.to(device), a.to(device)

            out = model(t, a)
            p = torch.argmax(out, dim=1)

            preds.extend(p.cpu().numpy())
            true.extend(y.numpy())

    return accuracy_score(true, preds), f1_score(true, preds, average="weighted")

# =========================
# 12. TRAIN LOOP
# =========================

best_f1 = 0

for epoch in range(1, 21):
    loss = train_one_epoch()
    acc, f1 = evaluate(val_loader)

    print(f"Epoch {epoch} | loss {loss:.4f} | val acc {acc:.4f} | val f1 {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), "best_meld.pt")
        print("✓ saved")

# =========================
# 13. TEST
# =========================

model.load_state_dict(torch.load("best_meld.pt"))

acc, f1 = evaluate(test_loader)

print("\n===== TEST RESULTS =====")
print("Accuracy:", acc)
print("F1:", f1)

print("\n===== DATASET DEBUG =====")

print("TEXT TYPE:", type(text_data))
print("AUDIO TYPE:", type(audio_data))

print("LEN TEXT:", len(text_data))

print("LEN AUDIO:", len(audio_data))
print("\n===== SAMPLE TEXT[0] =====")
print(text_data[0])

print("\n===== SAMPLE AUDIO[0] =====")
print(audio_data[0])

for i in range(len(text_data)):
    print(i, type(text_data[i]))
    
def find_label(x):
    if isinstance(x, dict):
        for k in ["label", "emotion", "sentiment"]:
            if k in x:
                return x[k]
    return None

labels = [find_label(x) for x in text_data]

print("\n===== LABEL DEBUG =====")
print("UNIQUE:", set(labels))
print("COUNT:", len(labels))