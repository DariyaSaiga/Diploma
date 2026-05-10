"""CMU-MOSEI multi-label emotion dataset (feature-based, aligned T=60).

Expected pickle structure (mosei_emotion_aligned_60.pkl):
    data[split] = {
        "vision":  ndarray (N, 60, 35),   # Facet
        "audio":   ndarray (N, 60, 74),   # COVAREP
        "text":    ndarray (N, 60, 300),  # GloVe-like embeddings
        "labels":  ndarray (N, 6),        # multi-label binary
        "id":      None or list,
    }

Emotion order:
    0: happy   1: sad   2: anger   3: surprise   4: disgust   5: fear
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

EMOTIONS = ["happy", "sad", "anger", "surprise", "disgust", "fear"]
NUM_CLASSES = len(EMOTIONS)

TEXT_DIM = 300
AUDIO_DIM = 74
VISION_DIM = 35
SEQ_LEN = 60


class MOSEIMultiLabelDataset(Dataset):
    """One split of MOSEI as a torch Dataset.

    Returns dict with text/audio/vision/labels (all torch.float32).
    """

    def __init__(
        self,
        text: np.ndarray,
        audio: np.ndarray,
        vision: np.ndarray,
        labels: np.ndarray,
        text_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        audio_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        vision_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        self.text = text.astype(np.float32, copy=False)
        self.audio = audio.astype(np.float32, copy=False)
        self.vision = vision.astype(np.float32, copy=False)
        self.labels = (labels > 0).astype(np.float32)  # binarize multi-label

        self.text_stats = text_stats
        self.audio_stats = audio_stats
        self.vision_stats = vision_stats

    def __len__(self) -> int:
        return self.labels.shape[0]

    @staticmethod
    def _apply_stats(x: np.ndarray, stats: Optional[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        if stats is None:
            return x
        mean, std = stats
        # Preserve zero-padding: only normalize non-zero rows.
        nonzero_mask = (np.abs(x).sum(axis=-1, keepdims=True) > 0)
        return np.where(nonzero_mask, (x - mean) / std, x)

    def __getitem__(self, idx: int):
        text = self._apply_stats(self.text[idx], self.text_stats)
        audio = self._apply_stats(self.audio[idx], self.audio_stats)
        vision = self._apply_stats(self.vision[idx], self.vision_stats)
        labels = self.labels[idx]
        return {
            "text": torch.from_numpy(np.ascontiguousarray(text)),
            "audio": torch.from_numpy(np.ascontiguousarray(audio)),
            "vision": torch.from_numpy(np.ascontiguousarray(vision)),
            "labels": torch.from_numpy(labels),
        }


def _compute_feature_stats(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mean/std over (N, T) ignoring all-zero (padded) timesteps."""
    flat = arr.reshape(-1, arr.shape[-1]).astype(np.float64)
    nonzero = flat[np.abs(flat).sum(axis=-1) > 0]
    if nonzero.shape[0] == 0:
        nonzero = flat
    mean = nonzero.mean(axis=0)
    std = nonzero.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _load_pkl(data_path: str | Path) -> dict:
    with open(data_path, "rb") as f:
        return pickle.load(f)


def positive_class_counts(labels: np.ndarray) -> np.ndarray:
    """Per-class number of positives (length 6)."""
    return (labels > 0).astype(np.int64).sum(axis=0)


def make_loaders(
    data_path: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    normalize: bool = False,
    pin_memory: bool = False,
):
    """Load pkl and build train/valid/test DataLoaders.

    Returns: (train_loader, valid_loader, test_loader, info)
        info = {
            "n_train", "n_valid", "n_test",
            "pos_counts": np.ndarray (6,)  # train positives per class
            "neg_counts": np.ndarray (6,)  # train negatives per class
            "stats": dict of (mean, std) per modality if normalize else None
        }
    """
    data = _load_pkl(data_path)

    for split in ("train", "valid", "test"):
        if split not in data:
            raise KeyError(f"Missing split '{split}' in {data_path}. Found: {list(data.keys())}")

    train_text = data["train"]["text"]
    train_audio = data["train"]["audio"]
    train_vision = data["train"]["vision"]
    train_labels = data["train"]["labels"]

    text_stats = audio_stats = vision_stats = None
    if normalize:
        text_stats = _compute_feature_stats(train_text)
        audio_stats = _compute_feature_stats(train_audio)
        vision_stats = _compute_feature_stats(train_vision)

    def build(split: str) -> MOSEIMultiLabelDataset:
        return MOSEIMultiLabelDataset(
            text=data[split]["text"],
            audio=data[split]["audio"],
            vision=data[split]["vision"],
            labels=data[split]["labels"],
            text_stats=text_stats,
            audio_stats=audio_stats,
            vision_stats=vision_stats,
        )

    train_ds = build("train")
    valid_ds = build("valid")
    test_ds = build("test")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
    )

    pos_counts = positive_class_counts(train_labels)
    neg_counts = train_labels.shape[0] - pos_counts

    info = {
        "n_train": len(train_ds),
        "n_valid": len(valid_ds),
        "n_test": len(test_ds),
        "pos_counts": pos_counts,
        "neg_counts": neg_counts,
        "stats": (
            {"text": text_stats, "audio": audio_stats, "vision": vision_stats}
            if normalize else None
        ),
    }
    return train_loader, valid_loader, test_loader, info
