"""
BottleneckDataset + collate_fn для Multimodal Bottleneck Transformer.

Поддерживает два формата pkl:

  mosei_bottleneck.pkl  (рекомендуется)
    Создаётся preprocess/preprocess_bottleneck.py.
    Каждый сэмпл уже содержит audio_len и visual_len.
    Audio/visual хранятся БЕЗ padding — реальная длина.

  mosei_cleaned.pkl  (fallback)
    Baseline-pkl. Audio/visual padded до 50 нулями.
    Реальная длина восстанавливается из нулевых строк.

В обоих случаях __getitem__ возвращает одинаковый формат:
    text, audio (Ta,74), visual (Tv,713), audio_len, visual_len, label.

collate_fn паддит audio/visual до max длины в батче и строит bool-маски.
Маски: True = реальный шаг, False = padding.
При передаче в MultiheadAttention инвертируй: key_padding_mask = ~mask.
"""

import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class BottleneckDataset(Dataset):
    """
    Загружает mosei_cleaned.pkl и возвращает сэмплы без padding.

    __getitem__ возвращает:
        text       : str          — сырая строка
        audio      : np.ndarray   — (Ta, 74),  Ta = реальная длина
        visual     : np.ndarray   — (Tv, 713), Tv = реальная длина
        audio_len  : int
        visual_len : int
        label      : int
    """

    def __init__(self, data: dict, split: str = "train"):
        assert split in ("train", "val", "test"), \
            f"split должен быть train/val/test, получили: '{split}'"

        self.samples = data[split]
        print(f"[BottleneckDataset] split='{split}' | samples={len(self.samples)}")

    # ------------------------------------------------------------------

    @staticmethod
    def _real_len(x: np.ndarray) -> int:
        """
        Возвращает реальную длину последовательности.
        Строка считается padding если ВСЕ признаки == 0.
        Ищем последний ненулевой шаг.
        """
        nonzero_rows = np.any(x != 0, axis=1)   # (T,) bool
        indices = np.where(nonzero_rows)[0]
        if len(indices) == 0:
            return 1   # крайний случай — вся последовательность нулевая
        return int(indices[-1]) + 1

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        audio  = np.array(s["audio"],  dtype=np.float32)
        visual = np.array(s["visual"], dtype=np.float32)

        # mosei_bottleneck.pkl хранит длины явно — используем их
        # mosei_cleaned.pkl — восстанавливаем из нулей (fallback)
        if "audio_len" in s:
            audio_len  = int(s["audio_len"])
            visual_len = int(s["visual_len"])
        else:
            audio_len  = self._real_len(audio)
            visual_len = self._real_len(visual)

        # Обрезаем до реальной длины (для cleaned.pkl убираем trailing нули)
        audio  = audio[:audio_len]    # (Ta, 74)
        visual = visual[:visual_len]  # (Tv, 713)

        return {
            "text":       s["text"],    # str
            "audio":      audio,        # (Ta, 74)
            "visual":     visual,       # (Tv, 713)
            "audio_len":  audio_len,    # int
            "visual_len": visual_len,   # int
            "label":      s["label"],   # int
        }


# ─────────────────────────────────────────────────────────────────────────────
# collate_fn
# ─────────────────────────────────────────────────────────────────────────────

def make_collate_fn(
    tokenizer: BertTokenizer,
    max_text_len:   int = 128,
    max_audio_len:  int = 50,
    max_visual_len: int = 50,
):
    """
    Фабрика collate_fn.

    Что делает:
      1. Токенизирует все тексты батча сразу (быстрее чем по одному)
      2. Паддит audio до max длины в батче (но не больше max_audio_len)
      3. Паддит visual аналогично
      4. Строит bool-маски: True = реальный шаг, False = padding

    Примечание по маскам:
      В nn.MultiheadAttention параметр key_padding_mask работает наоборот:
        True  = ИГНОРИРОВАТЬ (padding)
        False = ИСПОЛЬЗОВАТЬ (реальный шаг)
      Поэтому в модели при передаче маски нужно делать ~audio_mask.
      Здесь маски хранятся в "человеческом" формате (True = реальный).
    """

    def collate_fn(batch: list[dict]) -> dict:
        B = len(batch)

        texts       = [item["text"]       for item in batch]
        audios      = [item["audio"]      for item in batch]   # list of (Ta_i, 74)
        visuals     = [item["visual"]     for item in batch]   # list of (Tv_i, 713)
        audio_lens  = [item["audio_len"]  for item in batch]
        visual_lens = [item["visual_len"] for item in batch]
        labels      = [item["label"]      for item in batch]

        # ── Текст: токенизируем весь батч за один вызов ──────────────────────
        encoding = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_text_len,
            return_tensors="pt",
        )
        # encoding["input_ids"]      : (B, Lt)
        # encoding["attention_mask"] : (B, Lt) — 1=реальный, 0=padding

        # ── Аудио: pad до max длины в батче ─────────────────────────────────
        Ta = min(max(audio_lens), max_audio_len)

        audio_padded = np.zeros((B, Ta, 74), dtype=np.float32)
        audio_mask   = torch.zeros(B, Ta, dtype=torch.bool)   # False = padding

        for i, (a, L) in enumerate(zip(audios, audio_lens)):
            L = min(L, Ta)
            audio_padded[i, :L] = a[:L]
            audio_mask[i, :L]   = True   # реальные шаги

        # ── Визуал: pad до max длины в батче ────────────────────────────────
        Tv = min(max(visual_lens), max_visual_len)

        visual_padded = np.zeros((B, Tv, 713), dtype=np.float32)
        visual_mask   = torch.zeros(B, Tv, dtype=torch.bool)

        for i, (v, L) in enumerate(zip(visuals, visual_lens)):
            L = min(L, Tv)
            visual_padded[i, :L] = v[:L]
            visual_mask[i, :L]   = True

        return {
            # Текст
            "input_ids":      encoding["input_ids"],           # (B, Lt)
            "attention_mask": encoding["attention_mask"],      # (B, Lt) — 1/0
            # Аудио
            "audio":          torch.tensor(audio_padded),      # (B, Ta, 74)
            "audio_mask":     audio_mask,                      # (B, Ta) bool
            "audio_len":      torch.tensor(audio_lens),        # (B,)
            # Визуал
            "visual":         torch.tensor(visual_padded),     # (B, Tv, 713)
            "visual_mask":    visual_mask,                     # (B, Tv) bool
            "visual_len":     torch.tensor(visual_lens),       # (B,)
            # Метка
            "label":          torch.tensor(labels, dtype=torch.long),  # (B,)
        }

    return collate_fn


# ─────────────────────────────────────────────────────────────────────────────
# Удобная фабрика DataLoader-ов
# ─────────────────────────────────────────────────────────────────────────────

def make_bottleneck_loaders(
    data_path:      str = "mosei_bottleneck.pkl",
    batch_size:     int = 16,
    max_text_len:   int = 128,
    max_audio_len:  int = 50,
    max_visual_len: int = 50,
    num_workers:    int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Создаёт train/val/test DataLoader-ы для BottleneckModel.

    Пример использования:
        train_loader, val_loader, test_loader = make_bottleneck_loaders()
    """
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    collate   = make_collate_fn(tokenizer, max_text_len, max_audio_len, max_visual_len)

    def _loader(split, shuffle):
        ds = BottleneckDataset(data=data, split=split)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate,
            num_workers=num_workers,
        )

    train_loader = _loader("train", shuffle=True)
    val_loader   = _loader("val",   shuffle=False)
    test_loader  = _loader("test",  shuffle=False)

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "mosei_cleaned.pkl"
    print(f"Проверяем BottleneckDataset на {path}...")

    train_loader, _, _ = make_bottleneck_loaders(
        data_path=path, batch_size=4, num_workers=0
    )

    batch = next(iter(train_loader))

    print("\n── Shapes батча ──")
    print(f"  input_ids:      {batch['input_ids'].shape}")       # (4, 128)
    print(f"  attention_mask: {batch['attention_mask'].shape}")  # (4, 128)
    print(f"  audio:          {batch['audio'].shape}")           # (4, Ta, 74)
    print(f"  audio_mask:     {batch['audio_mask'].shape}")      # (4, Ta)
    print(f"  audio_len:      {batch['audio_len'].tolist()}")
    print(f"  visual:         {batch['visual'].shape}")          # (4, Tv, 713)
    print(f"  visual_mask:    {batch['visual_mask'].shape}")     # (4, Tv)
    print(f"  visual_len:     {batch['visual_len'].tolist()}")
    print(f"  label:          {batch['label'].tolist()}")

    # Проверяем что маски совпадают с длинами
    for i in range(len(batch["label"])):
        al = batch["audio_len"][i].item()
        assert batch["audio_mask"][i, :al].all(),  f"audio_mask не совпадает с audio_len [{i}]"
        assert not batch["audio_mask"][i, al:].any(), f"audio_mask заходит в padding [{i}]"

    print("\n✅ Все проверки пройдены!")
