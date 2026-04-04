import pickle
import torch
from torch.utils.data import Dataset
import numpy as np


class MoseiDataset(Dataset):
    def __init__(self, path="mosei_cleaned.pkl", split="train", max_len=50):
        self.max_len = max_len

        # загрузка
        with open(path, "rb") as f:
            data = pickle.load(f)

        # ожидаем структуру: data["train"], data["val"], data["test"]
        self.samples = data[split]

    def pad_or_truncate(self, x, max_len):
        """
        x: (T, D)
        возвращает (max_len, D)
        """
        T = x.shape[0]

        if T > max_len:
            return x[:max_len]
        elif T < max_len:
            pad = np.zeros((max_len - T, x.shape[1]))
            return np.concatenate([x, pad], axis=0)
        else:
            return x

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # ⚠️ ВАЖНО: подстрой под свои ключи, если названия другие
        text = sample["text"]
        audio = np.array(sample["audio"])     # (T, 74)
        visual = np.array(sample["visual"])   # (T, 713)
        label = sample["label"]               # int

        # pad/truncate
        audio = self.pad_or_truncate(audio, self.max_len)
        visual = self.pad_or_truncate(visual, self.max_len)

        return {
            "text": text,
            "audio": torch.tensor(audio, dtype=torch.float32),
            "visual": torch.tensor(visual, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long)
        }