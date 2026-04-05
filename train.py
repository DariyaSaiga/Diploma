import argparse
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from dataset import MoseiDataset
from utils import device, set_seed


# ──────────────────────────────────────────────────────────────────────────────
# Collate — динамический паддинг audio/visual внутри батча
# ──────────────────────────────────────────────────────────────────────────────

def collate_fn(batch: list[dict], max_audio_len: int = 100, max_visual_len: int = 100) -> dict:
    """
    Собирает батч с динамическим паддингом audio/visual.
    Text уже пофиксирован токенизатором.

    Возвращает плоский dict:
        input_ids, attention_mask  — (B, Lt)
        audio, audio_mask          — (B, Ta, 74) / (B, Ta)
        visual, visual_mask        — (B, Tv, 713) / (B, Tv)
        label                      — (B,)
    """
    input_ids      = torch.stack([s["input_ids"] for s in batch])       # (B, Lt)
    attention_mask = torch.stack([s["attention_mask"] for s in batch])  # (B, Lt)
    labels         = torch.stack([s["label"] for s in batch])           # (B,)

    # ── Audio ────────────────────────────────────────────────────────────
    audio_list = [s["audio"] for s in batch]  # list of (Ti, 74)
    audio_lens = [min(a.shape[0], max_audio_len) for a in audio_list]
    batch_audio_len = max(audio_lens)
    D_a = audio_list[0].shape[1]

    audio_padded = torch.zeros(len(batch), batch_audio_len, D_a)
    audio_mask   = torch.zeros(len(batch), batch_audio_len, dtype=torch.long)
    for i, (a, l) in enumerate(zip(audio_list, audio_lens)):
        audio_padded[i, :l] = a[:l]
        audio_mask[i, :l]   = 1

    # ── Visual ───────────────────────────────────────────────────────────
    visual_list = [s["visual"] for s in batch]  # list of (Ti, 713)
    visual_lens = [min(v.shape[0], max_visual_len) for v in visual_list]
    batch_visual_len = max(visual_lens)
    D_v = visual_list[0].shape[1]

    visual_padded = torch.zeros(len(batch), batch_visual_len, D_v)
    visual_mask   = torch.zeros(len(batch), batch_visual_len, dtype=torch.long)
    for i, (v, l) in enumerate(zip(visual_list, visual_lens)):
        visual_padded[i, :l] = v[:l]
        visual_mask[i, :l]   = 1

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "audio":          audio_padded,
        "audio_mask":     audio_mask,
        "visual":         visual_padded,
        "visual_mask":    visual_mask,
        "label":          labels,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Forward pass — единое место для всех моделей
# ──────────────────────────────────────────────────────────────────────────────

def run_batch(model, batch: dict, model_type: str) -> torch.Tensor:
    """
    Прогоняет один батч через нужную модель и возвращает logits.

    Добавляешь новую модель — просто добавь elif ниже.
    Менять train_one_epoch / evaluate не нужно.
    """
    if model_type == "text":
        return model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )

    if model_type == "av":
        return model(
            audio=batch["audio"].to(device),
            visual=batch["visual"].to(device),
        )

    if model_type == "bottleneck":
        return model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            audio=batch["audio"].to(device),
            audio_mask=batch["audio_mask"].to(device),
            visual=batch["visual"].to(device),
            visual_mask=batch["visual_mask"].to(device),
        )

    raise ValueError(f"Неизвестный model_type: '{model_type}'")


# ──────────────────────────────────────────────────────────────────────────────
# Train / Evaluate
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    model_type: str,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        labels = batch["label"].to(device)
        outputs = run_batch(model, batch, model_type)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), acc, f1


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    model_type: str,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        labels = batch["label"].to(device)
        outputs = run_batch(model, batch, model_type)

        total_loss += criterion(outputs, labels).item()
        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), acc, f1


# ──────────────────────────────────────────────────────────────────────────────
# Model factory — добавляешь модель здесь и больше нигде
# ──────────────────────────────────────────────────────────────────────────────

def build_model(model_type: str) -> nn.Module:
    if model_type == "text":
        from text_only_bert import TextOnlyBert
        return TextOnlyBert()

    if model_type == "av":
        from audio_visual_baseline import AudioVisualBaseline
        return AudioVisualBaseline()

    if model_type == "bottleneck":
        from bottleneck_fusion import BottleneckFusion
        return BottleneckFusion()

    raise ValueError(f"Неизвестный тип модели: '{model_type}'")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train CMU-MOSEI models")
    parser.add_argument(
        "--model", type=str, default="text",
        choices=["text", "av", "bottleneck"],
        help="Какую модель обучать",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data_path", type=str, default="mosei_cleaned.pkl")
    parser.add_argument("--max_len", type=int, default=128)
    args = parser.parse_args()

    set_seed(42)

    # Датасеты
    import pickle

    with open(args.data_path, "rb") as f:
        data = pickle.load(f)

    train_dataset = MoseiDataset(data=data, split="train", max_text_len=args.max_len)
    val_dataset   = MoseiDataset(data=data, split="val",   max_text_len=args.max_len)
    test_dataset  = MoseiDataset(data=data, split="test",  max_text_len=args.max_len)

    cfn = partial(collate_fn, max_audio_len=100, max_visual_len=100)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=2, collate_fn=cfn)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=cfn)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=cfn)

    # Веса классов — компенсируем дисбаланс (happy ×25 > fear)
    train_labels = np.array([s["label"] for s in train_dataset.samples])
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(6),
        y=train_labels,
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class weights: {class_weights.cpu().numpy().round(3)}")

    # Модель
    model = build_model(args.model).to(device)

    # Loss с весами + optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_f1 = 0.0
    print(f"\nМодель: {args.model} | Device: {device}")
    print("=" * 55)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, args.model
        )
        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, criterion, args.model
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train  loss={train_loss:.4f}  acc={train_acc:.4f}  f1={train_f1:.4f} | "
            f"Val    loss={val_loss:.4f}  acc={val_acc:.4f}  f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_path = f"best_{args.model}.pt"
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ Сохранена лучшая модель → {save_path}")

    print(f"\n🔥 Обучение завершено! Лучший val F1: {best_val_f1:.4f}")

    # ── Финальная оценка на test set ──────────────────────────────────────────
    print("\nЗагружаем лучшую модель для оценки на test set...")
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, args.model)
    print("=" * 55)
    print(
        f"TEST | loss={test_loss:.4f}  acc={test_acc:.4f}  f1={test_f1:.4f}"
    )
    print("=" * 55)


if __name__ == "__main__":
    main()