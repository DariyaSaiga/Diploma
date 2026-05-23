"""
Train — Baseline 1: BERT (frozen) + 1D-CNN (audio) + BiLSTM (vision) + Late Fusion

Тот же protocol что у Proposed (train.py):
  - AdamW, lr=1e-4, CosineAnnealingLR
  - Modality dropout p=0.05
  - Auxiliary losses (text + audio + vision)
  - Early stopping по val Avg.F1, patience=5
  - Threshold tuning на финальном тесте
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score

from dataset         import get_dataloaders
from baseline1_model import LateFusionModel

# ── Гиперпараметры (те же что в train.py) ─────────────────────────────────────
LR        = 1e-4
EPOCHS    = 50
PATIENCE  = 5
GRAD_CLIP = 1.0

SAVE_PATH = "/content/drive/MyDrive/Дипломка_правильная/results/baseline1_best.pt"

EMOTIONS = ["happy", "sad", "anger", "surprise", "disgust", "fear"]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        audio          = batch["audio"].to(device)
        vision         = batch["vision"].to(device)
        audio_mask     = batch["audio_mask"].to(device)
        vision_mask    = batch["vision_mask"].to(device)
        labels         = batch["labels"].to(device)

        # Modality dropout — то же что в Proposed (p=0.05)
        if torch.rand(1).item() < 0.05:
            audio = torch.zeros_like(audio)
        if torch.rand(1).item() < 0.05:
            vision = torch.zeros_like(vision)

        optimizer.zero_grad()

        logits_fuse, logits_text, logits_audio, logits_vision = model(
            input_ids, attention_mask, audio, vision, audio_mask, vision_mask
        )

        # Auxiliary losses — то же что в Proposed
        loss = (criterion(logits_fuse,   labels) +
                criterion(logits_text,   labels) +
                criterion(logits_audio,  labels) +
                criterion(logits_vision, labels))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        audio          = batch["audio"].to(device)
        vision         = batch["vision"].to(device)
        audio_mask     = batch["audio_mask"].to(device)
        vision_mask    = batch["vision_mask"].to(device)
        labels         = batch["labels"].to(device)

        logits_fuse, _, _, _ = model(
            input_ids, attention_mask, audio, vision, audio_mask, vision_mask
        )
        loss = criterion(logits_fuse, labels)
        total_loss += loss.item()

        preds = (torch.sigmoid(logits_fuse) > 0.5).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    avg_f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None,    zero_division=0)

    wa_list = []
    for i in range(6):
        y_true = all_labels[:, i]
        y_pred = all_preds[:, i]
        n_pos  = y_true.sum()
        n_neg  = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            wa_list.append(0.0)
            continue
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        wa_list.append(0.5 * (tp / n_pos + tn / n_neg))
    avg_wa = float(np.mean(wa_list))

    return total_loss / len(loader), avg_f1, avg_wa, per_class_f1


@torch.no_grad()
def threshold_tuning(model, valid_loader, test_loader, device):
    def collect_probs(loader):
        model.eval()
        probs_all, labels_all = [], []
        for batch in loader:
            logits_fuse, _, _, _ = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["audio"].to(device),
                batch["vision"].to(device),
                batch["audio_mask"].to(device),
                batch["vision_mask"].to(device),
            )
            probs_all.append(torch.sigmoid(logits_fuse).cpu().numpy())
            labels_all.append(batch["labels"].numpy())
        return np.vstack(probs_all), np.vstack(labels_all).astype(int)

    valid_probs, valid_labels = collect_probs(valid_loader)
    best_thresholds = []
    for i in range(6):
        best_t, best_f1 = 0.5, -1.0
        for t in np.arange(0.10, 0.71, 0.05):
            f1_i = f1_score(valid_labels[:, i],
                            (valid_probs[:, i] >= t).astype(int),
                            zero_division=0)
            if f1_i > best_f1:
                best_f1, best_t = f1_i, t
        best_thresholds.append(best_t)
    best_thresholds = np.array(best_thresholds)

    test_probs, test_labels = collect_probs(test_loader)
    test_preds   = (test_probs >= best_thresholds.reshape(1, -1)).astype(int)
    tuned_f1     = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    tuned_per_f1 = f1_score(test_labels, test_preds, average=None,    zero_division=0)

    wa_list = []
    for i in range(6):
        y_true = test_labels[:, i]
        y_pred = test_preds[:, i]
        n_pos, n_neg = y_true.sum(), len(y_true) - y_true.sum()
        if n_pos == 0 or n_neg == 0:
            wa_list.append(0.0)
            continue
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        wa_list.append(0.5 * (tp / n_pos + tn / n_neg))

    return best_thresholds, tuned_f1, float(np.mean(wa_list)), tuned_per_f1


def print_metrics(epoch, train_loss, val_loss, avg_f1, avg_wa, per_f1, best_f1):
    print(f"\nEpoch {epoch:3d} | train={train_loss:.4f} | val={val_loss:.4f}")
    print(f"         | Avg.F1={avg_f1:.4f}  Avg.WA={avg_wa:.4f}  "
          f"{'★ NEW BEST' if avg_f1 > best_f1 else ''}")
    for emo, f1 in zip(EMOTIONS, per_f1):
        bar = "█" * int(f1 * 20)
        print(f"    {emo:<10}: {f1:.4f}  {bar}")


def main():
    device = get_device()
    print(f"Device: {device}")
    print("Model: Baseline 1 — BERT(frozen) + 1D-CNN + BiLSTM + Late Fusion\n")

    loaders, pos_weight = get_dataloaders()
    model     = LateFusionModel().to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    best_f1, patience_cnt = 0.0, 0

    print(f"Обучение: {EPOCHS} эпох, patience={PATIENCE}")
    print("=" * 65)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, loaders["train"], optimizer, criterion, device)
        val_loss, avg_f1, avg_wa, per_f1 = evaluate(
            model, loaders["valid"], criterion, device
        )
        print_metrics(epoch, train_loss, val_loss, avg_f1, avg_wa, per_f1, best_f1)

        if avg_f1 > best_f1:
            best_f1, patience_cnt = avg_f1, 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"    ✅ Сохранено (best Avg.F1={best_f1:.4f})")
        else:
            patience_cnt += 1
            print(f"    patience {patience_cnt}/{PATIENCE}")

        scheduler.step()

        if patience_cnt >= PATIENCE:
            print(f"\n⏹  Early stopping на эпохе {epoch}")
            break

    # ── Финальный тест ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    _, test_f1, test_wa, test_per_f1 = evaluate(
        model, loaders["test"], criterion, device
    )

    print(f"\nФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ — BASELINE 1 (Late Fusion)")
    print(f"{'='*65}")
    print(f"  Avg. F1 (macro) : {test_f1:.4f}")
    print(f"  Avg. WA         : {test_wa:.4f}")
    print(f"\n  Per-class F1:")
    for emo, f1 in zip(EMOTIONS, test_per_f1):
        marker = " ← rare" if emo in ("fear", "surprise") else ""
        print(f"    {emo:<10}: {f1:.4f}{marker}")

    # Threshold tuning
    thresholds, tuned_f1, tuned_wa, tuned_per_f1 = threshold_tuning(
        model, loaders["valid"], loaders["test"], device
    )

    print(f"\n  После threshold tuning:")
    print(f"    Avg. F1: {test_f1:.4f} → {tuned_f1:.4f} ({tuned_f1-test_f1:+.4f})")
    print(f"    Avg. WA: {test_wa:.4f} → {tuned_wa:.4f} ({tuned_wa-test_wa:+.4f})")
    print(f"\n  Best thresholds:")
    for emo, th in zip(EMOTIONS, thresholds):
        print(f"    {emo:<10}: {th:.2f}")
    print(f"\n  Per-class F1 after tuning:")
    for emo, f1 in zip(EMOTIONS, tuned_per_f1):
        marker = " ← rare" if emo in ("fear", "surprise") else ""
        print(f"    {emo:<10}: {f1:.4f}{marker}")


if __name__ == "__main__":
    main()
