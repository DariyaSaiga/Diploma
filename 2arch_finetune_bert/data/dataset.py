import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
 
PKL_PATH   = "/Users/dariyaablanova/Desktop/unic_work/Diploma/Diploma_clone/Diploma/datasets/mosei_finetune_bert.pkl"
BATCH_SIZE = 8    # XMBT, Table 2: batch_size=8 для CMU-MOSEI
NUM_WORKERS = 0   # 0 на Mac, 2-4 на Colab
 
# ─────────────────────────────────────────────────────────────────────────────
 
class MOSEIDataset(Dataset):
 
    def __init__(self, split_data):
        labels = np.array(split_data["labels"])  # [N, 6]
 
        # ── Проблема 7: фильтрация samples без ни одной эмоции ────────────────
        # Статья: XMBT (Nguyen et al., 2025) — используют 16326/1871/4659 samples,
        # что меньше полного датасета (23453). Разница — отфильтрованные all-zero samples.
        valid = labels.sum(axis=1) > 0
 
        # ── Проблема 5: конвертация float64 → float32 ─────────────────────────
        # Статья: все статьи работают с float32. float64 даёт лишний расход памяти
        # и потенциальную нестабильность при mixed precision training.
        self.text_ids   = torch.tensor(split_data["input_ids"],     dtype=torch.long  )[valid]
        self.text_mask  = torch.tensor(split_data["attention_mask"], dtype=torch.long  )[valid]
        self.audio      = torch.tensor(split_data["audio"],          dtype=torch.float32)[valid]
        self.vision     = torch.tensor(split_data["vision"],         dtype=torch.float32)[valid]
        self.labels     = torch.tensor(labels,                       dtype=torch.float32)[valid]
 
        print(f"  Samples после фильтрации: {self.labels.shape[0]}")
        print(f"  Labels shape : {self.labels.shape}")
        print(f"  Audio shape  : {self.audio.shape}")
        print(f"  Vision shape : {self.vision.shape}")
        print(f"  Text ids     : {self.text_ids.shape}")
 
    def __len__(self):
        return len(self.labels)
 
    def __getitem__(self, idx):
        audio  = self.audio[idx]   # [60, 74]
        vision = self.vision[idx]  # [60, 35]
 
        # ── Проблема 3: padding маски для audio и vision ───────────────────────
        # Статья: MER-SEM-MBT (Xia et al., 2022) — positional encoding предполагает
        # что модель знает где реальные данные. Нулевые фреймы = padding, их маскируем.
        # True = реальный фрейм, False = padding (нули).
        audio_mask  = (audio.sum(dim=-1)  != 0)  # [60]
        vision_mask = (vision.sum(dim=-1) != 0)  # [60]
 
        return {
            "input_ids"    : self.text_ids[idx],   # [50]      — токены для BERT
            "attention_mask": self.text_mask[idx],  # [50]      — маска текста из датасета
            "audio"        : audio,                 # [60, 74]  — COVAREP features
            "vision"       : vision,                # [60, 35]  — OpenFace features
            "audio_mask"   : audio_mask,            # [60]      — маска padding для audio
            "vision_mask"  : vision_mask,           # [60]      — маска padding для vision
            "labels"       : self.labels[idx],      # [6]       — multi-label binary
        }
 
 
def get_pos_weight(train_labels):
    """
    Проблема 4: class imbalance — happy 53.5%, fear 8.2%, surprise 10.1%
    Статья: MER-SEM-MBT (Xia et al., 2022) — "binary cross-entropy loss weighted
    by the ratio of positive and negative samples"
    """
    N  = len(train_labels)
    pw = [(N - train_labels[:, i].sum()) / max(train_labels[:, i].sum(), 1)
          for i in range(6)]
    return torch.tensor(pw, dtype=torch.float32)
 
 
def get_dataloaders():
 
    # ── Проблема 8: воспроизводимость ─────────────────────────────────────────
    # Статья: DBA (He et al., 2024) — фиксированные гиперпараметры и воспроизводимость.
    torch.manual_seed(42)
    np.random.seed(42)
 
    print("Загружаем датасет...")
    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f, encoding="latin1")
 
    loaders = {}
    for split in ["train", "valid", "test"]:
        print(f"\n[{split}]")
        ds = MOSEIDataset(data[split])
        loaders[split] = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=(split == "train"),
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
 
    # pos_weight считаем только по train
    train_labels = loaders["train"].dataset.labels.numpy()
    pos_weight   = get_pos_weight(train_labels)
 
    print("\n── pos_weight (BCEWithLogitsLoss) ──")
    emotions = ["happy", "sad", "anger", "surprise", "disgust", "fear"]
    for emo, pw in zip(emotions, pos_weight):
        print(f"  {emo:<10}: {pw:.2f}")
 
    return loaders, pos_weight
 
 
# ── Быстрая проверка ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    loaders, pos_weight = get_dataloaders()
 
    print("\n── Проверка одного батча ──")
    batch = next(iter(loaders["train"]))
    for k, v in batch.items():
        print(f"  {k:<16}: {v.shape}  dtype={v.dtype}")
 
    print(f"\n  pos_weight shape: {pos_weight.shape}")
    print("\n✅ Датасет готов к обучению Bottleneck модели")