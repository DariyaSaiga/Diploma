import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# =========================
# 1. LOAD DATA
# =========================
with open("preprocess/mosei_emotion_aligned_60.pkl", "rb") as f:
    data = pickle.load(f)

train_data = data["train"]
valid_data = data["valid"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 2. DATASET
# =========================
class MOSEIDataset(Dataset):
    def __init__(self, data):
        self.text = data["text"]
        self.audio = data["audio"]
        self.video = data["vision"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.text[idx], dtype=torch.float32),
            torch.tensor(self.audio[idx], dtype=torch.float32),
            torch.tensor(self.video[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )


# =========================
# 3. LOADERS
# =========================
train_dataset = MOSEIDataset(train_data)
valid_dataset = MOSEIDataset(valid_data)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32)


# =========================
# 4. CLASS WEIGHTS (FIXED ORDER!)
# =========================
labels_all = []

for _, _, _, labels in train_dataset:
    if labels.ndim > 0:
        labels_all.append(np.argmax(labels.numpy()))

labels_all = np.array(labels_all)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels_all),
    y=labels_all
)

class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)


# =========================
# 5. MODEL (ATTENTION FUSION)
# =========================
class AttentionFusion(nn.Module):
    def __init__(self, text_dim, audio_dim, video_dim, hidden=128, num_classes=6):
        super().__init__()

        self.text_fc = nn.Linear(text_dim * 60, hidden)
        self.audio_fc = nn.Linear(audio_dim * 60, hidden)
        self.video_fc = nn.Linear(video_dim * 60, hidden)

        self.attention = nn.Sequential(
            nn.Linear(hidden * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, text, audio, video):

        text = text.view(text.size(0), -1)
        audio = audio.view(audio.size(0), -1)
        video = video.view(video.size(0), -1)

        t = torch.relu(self.text_fc(text))
        a = torch.relu(self.audio_fc(audio))
        v = torch.relu(self.video_fc(video))

        stacked = torch.stack([t, a, v], dim=1)
        flat = torch.cat([t, a, v], dim=1)

        weights = self.attention(flat).unsqueeze(-1)

        fused = (stacked * weights).sum(dim=1)

        return self.classifier(fused)


# =========================
# 6. INIT MODEL
# =========================
text_dim = train_data["text"][0].shape[1]
audio_dim = train_data["audio"][0].shape[1]
video_dim = train_data["vision"][0].shape[1]

model = AttentionFusion(text_dim, audio_dim, video_dim).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# =========================
# 7. TRAIN
# =========================
def train_one_epoch():
    model.train()
    losses = []

    for text, audio, video, labels in train_loader:

        text = text.to(device)
        audio = audio.to(device)
        video = video.to(device)

        labels = labels.to(device)

        if labels.dim() > 1:
            labels = torch.argmax(labels, dim=1)

        labels = labels.long()

        optimizer.zero_grad()

        outputs = model(text, audio, video)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)


# =========================
# 8. EVAL
# =========================
def evaluate():
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for text, audio, video, labels in valid_loader:

            text = text.to(device)
            audio = audio.to(device)
            video = video.to(device)

            labels = labels.to(device)

            if labels.dim() > 1:
                labels = torch.argmax(labels, dim=1)

            labels = labels.long()

            outputs = model(text, audio, video)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))

    return acc, f1


# =========================
# 9. TRAIN LOOP
# =========================
epochs = 10

for epoch in range(epochs):
    loss = train_one_epoch()
    acc, f1 = evaluate()

    print(f"\nEpoch {epoch+1}")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("-" * 50)