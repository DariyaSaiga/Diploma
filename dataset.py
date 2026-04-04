import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import BertTokenizer


class MoseiDataset(Dataset):
    """
    CMU-MOSEI dataset wrapper.

    Поддерживает три сплита: "train", "val", "test".
    Lazy loading — грузим только нужный сплит, не весь файл целиком.
    """

    def __init__(
        self,
        path: str = "mosei_cleaned.pkl",
        split: str = "train",
        max_len: int = 128,
    ):
        assert split in ("train", "val", "test"), (
            f"split должен быть 'train', 'val' или 'test', получили: '{split}'"
        )

        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Загружаем весь pkl в память, затем выбираем нужный split
        with open(path, "rb") as f:
            data = pickle.load(f)

        if split not in data:
            available = list(data.keys())
            raise KeyError(
                f"Сплит '{split}' не найден в файле. Доступные ключи: {available}"
            )

        self.samples = data[split]
        print(f"[MoseiDataset] split='{split}' | samples={len(self.samples)}")

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _pad_or_truncate(self, x: np.ndarray, max_len: int) -> np.ndarray:
        """Приводит временную ось к фиксированной длине max_len."""
        T = x.shape[0]
        if T > max_len:
            return x[:max_len]
        if T < max_len:
            pad = np.zeros((max_len - T, x.shape[1]), dtype=x.dtype)
            return np.concatenate([x, pad], axis=0)
        return x

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        text: str = sample["text"]
        audio = self._pad_or_truncate(np.array(sample["audio"]), self.max_len)
        visual = self._pad_or_truncate(np.array(sample["visual"]), self.max_len)
        label: int = sample["label"]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "text": {
                "input_ids": encoding["input_ids"].squeeze(0),       # (L,)
                "attention_mask": encoding["attention_mask"].squeeze(0),  # (L,)
            },
            "audio": torch.tensor(audio, dtype=torch.float32),       # (T, D_a)
            "visual": torch.tensor(visual, dtype=torch.float32),      # (T, D_v)
            "label": torch.tensor(label, dtype=torch.long),
        }