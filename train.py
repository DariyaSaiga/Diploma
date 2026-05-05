import argparse
import os
import pickle
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, WeightedRandomSampler
from Diploma.backend.bottleneck_fusion import BottleneckFusion

from dataset import MoseiDataset
from utils import device, set_seed

EMOTION_NAMES = ["happy", "sad", "anger", "surprise", "disgust", "fear"]


# ──────────────────────────────────────────────────────────────────────────────
# Collate — динамический паддинг audio/visual внутри батча
# ──────────────────────────────────────────────────────────────────────────────

def collate_fn(batch: list[dict], max_audio_len: int = 100, max_visual_len: int = 100) -> dict:
    input_ids      = torch.stack([s["input_ids"] for s in batch])
    attention_mask = torch.stack([s["attention_mask"] for s in batch])
    labels         = torch.stack([s["label"] for s in batch])

    # ── Audio ────────────────────────────────────────────────────────────
    audio_list = [s["audio"] for s in batch]
    audio_lens = [min(a.shape[0], max_audio_len) for a in audio_list]
    batch_audio_len = max(audio_lens)
    D_a = audio_list[0].shape[1]

    audio_padded = torch.zeros(len(batch), batch_audio_len, D_a)
    audio_mask   = torch.zeros(len(batch), batch_audio_len, dtype=torch.long)
    for i, (a, l) in enumerate(zip(audio_list, audio_lens)):
        audio_padded[i, :l] = a[:l]
        audio_mask[i, :l]   = 1

    # ── Visual ───────────────────────────────────────────────────────────
    visual_list = [s["visual"] for s in batch]
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
# Forward pass
# ──────────────────────────────────────────────────────────────────────────────

def run_batch(model, batch: dict, model_type: str) -> torch.Tensor:
    if model_type == "text":
        return model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )

    if model_type == "av":
        return model(
            audio=batch["audio"].to(device),
            visual=batch["visual"].to(device),
            audio_mask=batch["audio_mask"].to(device),
            visual_mask=batch["visual_mask"].to(device),
        )

    if model_type in ("bottleneck", "simple_fusion"):
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
    clip_grad: float = 1.0,
    dra_criterion=None,      # DRALoss instance или None
) -> tuple:
    """
    Возвращает (loss, acc, f1).
    Если dra_criterion задан — использует model.forward_dra() и DRA loss.
    Дополнительно возвращает epoch_losses_per_branch для dra.update_history().
    """
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    # Для DRA: накапливаем средние лоссы по веткам за эпоху
    dra_branch_totals = [0.0, 0.0, 0.0, 0.0]   # [text, audio, visual, fused]

    for batch in loader:
        labels = batch["label"].to(device)

        if dra_criterion is not None and model_type == "bottleneck":
            # ── DRA forward: 4 набора логитов ─────────────────────────────
            logits_f, logits_t, logits_a, logits_v = model.forward_dra(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                audio=batch["audio"].to(device),
                audio_mask=batch["audio_mask"].to(device),
                visual=batch["visual"].to(device),
                visual_mask=batch["visual_mask"].to(device),
            )
            L_t = criterion(logits_t, labels)
            L_a = criterion(logits_a, labels)
            L_v = criterion(logits_v, labels)
            L_f = criterion(logits_f, labels)

            loss = dra_criterion([L_t, L_a, L_v, L_f])
            outputs = logits_f   # для предсказаний используем fused

            dra_branch_totals[0] += L_t.item()
            dra_branch_totals[1] += L_a.item()
            dra_branch_totals[2] += L_v.item()
            dra_branch_totals[3] += L_f.item()
        else:
            # ── Стандартный forward ────────────────────────────────────────
            outputs = run_batch(model, batch, model_type)
            loss    = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n = len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    epoch_branch_losses = [x / n for x in dra_branch_totals]
    return total_loss / n, acc, f1, epoch_branch_losses


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
        labels  = batch["label"].to(device)
        outputs = run_batch(model, batch, model_type)

        total_loss += criterion(outputs, labels).item()
        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), acc, f1


@torch.no_grad()
def evaluate_full(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    model_type: str,
) -> tuple[float, list, list]:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        labels  = batch["label"].to(device)
        outputs = run_batch(model, batch, model_type)

        total_loss += criterion(outputs, labels).item()
        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), all_preds, all_labels


# ──────────────────────────────────────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────────────────────────────────────

def build_model(args) -> nn.Module:
    if args.model == "text":
        from text_only_bert import TextOnlyBert
        return TextOnlyBert()

    if args.model == "av":
        from audio_visual_baseline import AudioVisualBaseline
        return AudioVisualBaseline()

    if args.model == "simple_fusion":
        from simple_fusion import SimpleFusion
        return SimpleFusion()

    if args.model == "bottleneck":
        from Diploma.backend.bottleneck_fusion import BottleneckFusion
        return BottleneckFusion(
            num_bottleneck_tokens=args.num_bottleneck_tokens,
            num_bottleneck_layers=args.num_bottleneck_layers,  # ← новый параметр
            freeze_bert=(args.freeze_bert != "none"),
            use_audio=not args.no_audio,
            use_visual=not args.no_visual,
        )

    raise ValueError(f"Неизвестный тип модели: '{args.model}'")


# ──────────────────────────────────────────────────────────────────────────────
# Сохранение эксперимента
# ──────────────────────────────────────────────────────────────────────────────

def save_config(args, exp_dir: str) -> None:
    path = os.path.join(exp_dir, "config.txt")
    lines = [
        f"model                = {args.model}",
        f"num_classes          = 6",
        f"hidden_dim           = 256",
        f"num_bottleneck_tokens= {args.num_bottleneck_tokens}",
        f"num_bottleneck_layers= {args.num_bottleneck_layers}",
        f"batch_size           = {args.batch_size}",
        f"lr                   = {args.lr}",
        f"lr_bert              = {args.lr_bert}",
        f"epochs               = {args.epochs}",
        f"freeze_bert          = {args.freeze_bert}",
        f"no_audio             = {args.no_audio}",
        f"no_visual            = {args.no_visual}",
        f"use_sampler          = {args.use_sampler}",
        f"max_text_len         = {args.max_len}",
        f"max_audio_len        = 100",
        f"max_visual_len       = 100",
        f"class_weights        = yes",
        f"label_smoothing      = {args.label_smoothing}",
        f"early_stopping       = {args.patience} epochs",
        f"data_path            = {args.data_path}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  📄 config.txt → {path}")


def save_metrics(
    exp_dir: str,
    best_val_f1: float,
    best_epoch: int,
    test_loss: float,
    test_preds: list,
    test_labels: list,
) -> None:
    acc = accuracy_score(test_labels, test_preds)
    f1  = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    cm  = confusion_matrix(test_labels, test_preds)
    rep = classification_report(
        test_labels, test_preds,
        target_names=EMOTION_NAMES, digits=3, zero_division=0,
    )

    path = os.path.join(exp_dir, "metrics.txt")
    with open(path, "w") as f:
        f.write(f"Best epoch        : {best_epoch}\n")
        f.write(f"Best val macro F1 : {best_val_f1:.4f}\n")
        f.write(f"Test loss         : {test_loss:.4f}\n")
        f.write(f"Test accuracy     : {acc:.4f}\n")
        f.write(f"Test macro F1     : {f1:.4f}\n")
        f.write("\n── Confusion Matrix ──────────────────────────────────\n")
        header = "         " + "  ".join(f"{n[:4]:>4}" for n in EMOTION_NAMES)
        f.write(header + "\n")
        for i, row in enumerate(cm):
            f.write(f"  {EMOTION_NAMES[i]:8s}" + "  ".join(f"{v:4d}" for v in row) + "\n")
        f.write("\n── Classification Report ─────────────────────────────\n")
        f.write(rep + "\n")

    print(f"  📊 metrics.txt → {path}")
    print(f"\n{'='*55}")
    print(f"TEST | loss={test_loss:.4f}  acc={acc:.4f}  macro_f1={f1:.4f}")
    print(f"{'='*55}")
    print(rep)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train CMU-MOSEI models")

    # ── Основные параметры ────────────────────────────────────────────────
    parser.add_argument("--model",      type=str, default="text",
                        choices=["text", "av", "simple_fusion", "bottleneck"])
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--data_path",  type=str,   default="mosei_cleaned.pkl")
    parser.add_argument("--max_len",    type=int,   default=128)

    # ── Bottleneck-специфичные параметры ──────────────────────────────────
    parser.add_argument("--num_bottleneck_tokens", type=int, default=16)
    parser.add_argument("--num_bottleneck_layers", type=int, default=2,   # ← новый
                        help="Число слоёв MBT fusion (рекомендуется 2-3)")
    parser.add_argument("--freeze_bert", type=str, default="full",
                        choices=["full", "partial", "none"])
    parser.add_argument("--lr_bert", type=float, default=None)
    parser.add_argument("--no_audio",    action="store_true")
    parser.add_argument("--no_visual",   action="store_true")
    parser.add_argument("--use_sampler", action="store_true",
                        help="WeightedRandomSampler для балансировки батчей по классам")
    parser.add_argument("--use_dra",    action="store_true",
                        help="DRA loss: раздельные aux лоссы для каждой модальности")

    # ── Новые параметры обучения ──────────────────────────────────────────
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing для CrossEntropyLoss (0.0 = выкл)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping: стоп если val F1 не растёт N эпох")

    # ── Сохранение эксперимента ───────────────────────────────────────────
    parser.add_argument("--exp_dir", type=str, default=None)

    args = parser.parse_args()
    set_seed(42)

    if args.exp_dir:
        os.makedirs(args.exp_dir, exist_ok=True)
        save_config(args, args.exp_dir)

    # ── Датасеты ──────────────────────────────────────────────────────────
    with open(args.data_path, "rb") as f:
        data = pickle.load(f)

    train_dataset = MoseiDataset(data=data, split="train", max_text_len=args.max_len)
    val_dataset   = MoseiDataset(data=data, split="val",   max_text_len=args.max_len)
    test_dataset  = MoseiDataset(data=data, split="test",  max_text_len=args.max_len)

    # ── Веса классов (только train) ───────────────────────────────────────
    train_labels = np.array([s["label"] for s in train_dataset.samples])
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(6),
        y=train_labels,
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class weights: {class_weights.cpu().numpy().round(3)}")

    cfn = partial(collate_fn, max_audio_len=100, max_visual_len=100)

    # ── WeightedRandomSampler (exp_10) ────────────────────────────────────
    if args.use_sampler:
        sample_weights = class_weights[train_labels].cpu()
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=2, collate_fn=cfn)
        print("  ⚖️  WeightedRandomSampler включён")
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=2, collate_fn=cfn)

    val_loader  = DataLoader(val_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=cfn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=cfn)

    # ── Модель ────────────────────────────────────────────────────────────
    model = build_model(args).to(device)

    # ── Partial unfreeze BERT ─────────────────────────────────────────────
    if args.model == "bottleneck" and args.freeze_bert == "partial":
        unfrozen = 0
        for name, param in model.bert.named_parameters():
            if any(f"encoder.layer.{i}." in name for i in [9, 10, 11]):
                param.requires_grad = True
                unfrozen += 1
        print(f"  🔓 Partial unfreeze BERT: разморожено {unfrozen} параметров (layers 9-11)")

    # ── Optimizer ─────────────────────────────────────────────────────────
    if args.model == "bottleneck" and args.freeze_bert == "partial" and args.lr_bert is not None:
        bert_params  = [p for n, p in model.named_parameters() if "bert" in n and p.requires_grad]
        other_params = [p for n, p in model.named_parameters() if "bert" not in n and p.requires_grad]
        optimizer = torch.optim.AdamW([
            {"params": bert_params,  "lr": args.lr_bert},
            {"params": other_params, "lr": args.lr},
        ], weight_decay=1e-4)
        print(f"  ⚙️  Optimizer: BERT lr={args.lr_bert}, остальное lr={args.lr}")
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=1e-4,
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    # ── Loss с label smoothing ─────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )

    # ── DRA loss (опционально) ─────────────────────────────────────────────
    dra_criterion = None
    if args.use_dra and args.model == "bottleneck":
        from dra_loss import DRALoss
        dra_criterion = DRALoss(num_tasks=4, temperature=2.0).to(device)
        # Добавляем параметры DRA в optimizer
        optimizer.add_param_group({"params": dra_criterion.parameters(), "lr": args.lr})
        print("  🔀 DRA loss включён (text + audio + visual + fused)")

    # ── Путь для сохранения ───────────────────────────────────────────────
    save_path = os.path.join(args.exp_dir, "best_model.pt") if args.exp_dir else f"best_{args.model}.pt"

    # ── Цикл обучения ─────────────────────────────────────────────────────
    best_val_f1       = 0.0
    best_epoch        = 1
    epochs_no_improve = 0

    print(f"\nМодель: {args.model} | Device: {device}")
    print(f"Bottleneck tokens: {args.num_bottleneck_tokens} | layers: {args.num_bottleneck_layers} | freeze_bert: {args.freeze_bert}")
    print(f"label_smoothing: {args.label_smoothing} | patience: {args.patience} | use_dra: {args.use_dra}")
    print(f"use_audio: {not args.no_audio} | use_visual: {not args.no_visual}")
    print("=" * 65)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1, branch_losses = train_one_epoch(
            model, train_loader, optimizer, criterion, args.model,
            dra_criterion=dra_criterion,
        )

        # Обновляем DWA-историю после каждой эпохи
        if dra_criterion is not None:
            dra_criterion.update_history(branch_losses)
            print(f"  DRA α: {torch.exp(dra_criterion.log_alpha).detach().cpu().numpy().round(3)}"
                  f"  branch_losses(t/a/v/f): "
                  f"{branch_losses[0]:.3f}/{branch_losses[1]:.3f}/{branch_losses[2]:.3f}/{branch_losses[3]:.3f}")

        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, criterion, args.model
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train  loss={train_loss:.4f}  acc={train_acc:.4f}  f1={train_f1:.4f} | "
            f"Val    loss={val_loss:.4f}  acc={val_acc:.4f}  f1={val_f1:.4f}"
        )

        scheduler.step()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ Лучший val F1={best_val_f1:.4f} → {save_path}")
        else:
            epochs_no_improve += 1
            print(f"  ⏳ Нет улучшения {epochs_no_improve}/{args.patience}")
            if epochs_no_improve >= args.patience:
                print(f"\n🛑 Early stopping на epoch {epoch} (patience={args.patience})")
                break

    print(f"\n🔥 Обучение завершено! Лучший val F1: {best_val_f1:.4f} (epoch {best_epoch})")

    # ── Финальная оценка на test set ──────────────────────────────────────
    print("\nЗагружаем лучшую модель для оценки на test set...")
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_loss, test_preds, test_labels = evaluate_full(
        model, test_loader, criterion, args.model
    )

    if args.exp_dir:
        save_metrics(args.exp_dir, best_val_f1, best_epoch,
                     test_loss, test_preds, test_labels)
    else:
        acc = accuracy_score(test_labels, test_preds)
        f1  = f1_score(test_labels, test_preds, average="macro", zero_division=0)
        print("=" * 55)
        print(f"TEST | loss={test_loss:.4f}  acc={acc:.4f}  f1={f1:.4f}")
        print("=" * 55)


if __name__ == "__main__":
    main()