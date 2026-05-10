import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertModel
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
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


train_dataset = MOSEIDataset(train_data)
valid_dataset = MOSEIDataset(valid_data)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16)


# =========================
# 3. CLASS WEIGHTS
# =========================
labels_all = []

for _, _, _, y in train_dataset:
    labels_all.append(int(torch.argmax(y)))

labels_all = np.array(labels_all)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels_all),
    y=labels_all
)

class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)


# =========================
# 4. FOCAL LOSS
# =========================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = self.ce(logits, targets)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma) * ce


# =========================
# 5. MODEL (REAL MULTIMODAL)
# =========================
class FinalModel(nn.Module):
    def __init__(self, audio_dim, video_dim, num_classes=6):
        super().__init__()

        # REAL TEXT ENCODER
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.text_proj = nn.Linear(768, 256)

        # AUDIO / VIDEO TEMPORAL
        self.audio_gru = nn.GRU(audio_dim, 128, batch_first=True)
        self.video_gru = nn.GRU(video_dim, 128, batch_first=True)

        # TRANSFORMER FUSION
        self.fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True),
            num_layers=2
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, text, audio, video):

        # TEXT (BERT)
        text = text.mean(dim=1)  # simplify token embeddings
        bert_out = self.bert(inputs_embeds=text).last_hidden_state[:, 0, :]
        t = self.text_proj(bert_out)

        # AUDIO
        _, a = self.audio_gru(audio)
        a = a[-1]

        # VIDEO
        _, v = self.video_gru(video)
        v = v[-1]

        # FUSION
        x = torch.stack([t, a, v], dim=1)
        x = self.fusion(x)

        x = x.reshape(x.size(0), -1)

        return self.classifier(x)


# =========================
# 6. INIT
# =========================
audio_dim = train_data["audio"][0].shape[1]
video_dim = train_data["vision"][0].shape[1]

model = FinalModel(audio_dim, video_dim).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
criterion = FocalLoss()


# =========================
# 7. TRAIN
# =========================
def train():
    model.train()
    losses = []

    for text, audio, video, labels in train_loader:

        text, audio, video = text.to(device), audio.to(device), video.to(device)

        labels = labels.to(device)
        labels = torch.argmax(labels, dim=1).long()

        optimizer.zero_grad()

        out = model(text, audio, video)
        loss = criterion(out, labels)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)


# =========================
# 8. EVAL
# =========================
def evaluate():
    model.eval()

    preds, true = [], []

    with torch.no_grad():
        for text, audio, video, labels in valid_loader:

            text, audio, video = text.to(device), audio.to(device), video.to(device)

            labels = torch.argmax(labels.to(device), dim=1)

            out = model(text, audio, video)
            p = torch.argmax(out, dim=1)

            preds.extend(p.cpu().numpy())
            true.extend(labels.cpu().numpy())

    print(classification_report(true, preds, zero_division=0))

    return accuracy_score(true, preds), f1_score(true, preds, average="weighted")


# =========================
# 9. TRAIN LOOP
# =========================
for epoch in range(10):

    loss = train()
    acc, f1 = evaluate()

    print(f"\nEpoch {epoch+1}")
    print(f"Loss: {loss:.4f}")
    print(f"Acc: {acc:.4f}")
    print(f"F1: {f1:.4f}")
    print("-" * 50)