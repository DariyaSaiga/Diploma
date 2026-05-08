import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import BertTokenizer

MAX_AUDIO_LEN = 100
MAX_VISUAL_LEN = 100


class MoseiDataset(Dataset):
    """
    CMU-MOSEI dataset wrapper.

    Поддерживает три сплита: "train", "val", "test".
    Audio/visual возвращаются без паддинга (truncate до 100).
    Паддинг и маски строятся в collate_fn.
    """

    def __init__(self, data, split="train", max_text_len=128):
        assert split in ("train", "val", "test")

        self.max_text_len = max_text_len
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.samples = data[split]
        print(f"[MoseiDataset] split='{split}' | samples={len(self.samples)}")

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        text: str = sample["text"]
        # Truncate to max — padding делается в collate_fn динамически
        audio = np.array(sample["audio"])[:MAX_AUDIO_LEN]   # (Ta, 74)
        visual = np.array(sample["visual"])[:MAX_VISUAL_LEN]  # (Tv, 713)
        label: int = sample["label"]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),          # (Lt,)
            "attention_mask": encoding["attention_mask"].squeeze(0), # (Lt,)
            "audio": torch.tensor(audio, dtype=torch.float32),       # (Ta, 74)
            "visual": torch.tensor(visual, dtype=torch.float32),     # (Tv, 713)
            "label": torch.tensor(label, dtype=torch.long),
        }