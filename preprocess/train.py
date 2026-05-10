import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# 1. LOAD DATA
# =========================
with open("mosei_emotion_aligned_60.pkl", "rb") as f:
    data = pickle.load(f)

train_data = data["train"]
valid_data = data["valid"]

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
# 4. CHECK SHAPES
# =========================
print("TEXT:", train_dataset.text[0].shape)
print("AUDIO:", train_dataset.audio[0].shape)
print("VIDEO:", train_dataset.video[0].shape)

text_dim = train_dataset.text[0].shape[1]
audio_dim = train_dataset.audio[0].shape[1]
video_dim = train_dataset.video[0].shape[1]

# =========================
# 5. MODEL
# =========================
class FusionModel(nn.Module):
    def __init__(self, text_dim, audio_dim, video_dim, hidden=128, num_classes=6):
        super().__init__()

        # flatten 60 timesteps
        self.text_fc = nn.Linear(text_dim * 60, hidden)
        self.audio_fc = nn.Linear(audio_dim * 60, hidden)
        self.video_fc = nn.Linear(video_dim * 60, hidden)

        self.classifier = nn.Sequential(
            nn.Linear(hidden * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, text, audio, video):
        text = text.view(text.size(0), -1)
        audio = audio.view(audio.size(0), -1)
        video = video.view(video.size(0), -1)

        t = torch.relu(self.text_fc(text))
        a = torch.relu(self.audio_fc(audio))
        v = torch.relu(self.video_fc(video))

        fused = torch.cat([t, a, v], dim=1)
        return self.classifier(fused)

# =========================
# 6. DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FusionModel(
    text_dim=text_dim,
    audio_dim=audio_dim,
    video_dim=video_dim,
    num_classes=6
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# =========================
# 7. TRAIN
# =========================
def train_one_epoch():
    model.train()
    total_loss = 0

    for text, audio, video, labels in train_loader:

        text = text.to(device)
        audio = audio.to(device)
        video = video.to(device)
        labels = labels.to(device)

        # FIX LABELS (ONE-HOT → CLASS INDEX)
        if labels.dim() > 1:
            labels = torch.argmax(labels, dim=1)
        labels = labels.long()

        optimizer.zero_grad()

        outputs = model(text, audio, video)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# =========================
# 8. EVAL
# =========================
def evaluate():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for text, audio, video, labels in valid_loader:

            text = text.to(device)
            audio = audio.to(device)
            video = video.to(device)
            labels = labels.to(device)

            outputs = model(text, audio, video)
            preds = torch.argmax(outputs, dim=1)

            # FIX LABELS
            if labels.dim() > 1:
                labels = torch.argmax(labels, dim=1)

            labels = labels.long()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

# =========================
# 9. TRAIN LOOP
# =========================
epochs = 10

for epoch in range(epochs):
    loss = train_one_epoch()
    acc = evaluate()

    print(f"\nEpoch {epoch+1}/{epochs}")
    print(f"Loss: {loss:.4f}")
    print(f"Val Acc: {acc:.4f}")
    print("-" * 40)