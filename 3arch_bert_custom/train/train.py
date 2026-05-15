import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score

from dataset import get_dataloaders
from models  import BottleneckFusionModel


LR_MAIN     = 1e-4   # lr основной сети — DBA (He et al., 2024): lr=1e-4
LR_BERT     = 5e-6   # lr BERT — XMBT (Nguyen et al., 2025): text lr = lr_main / 10
EPOCHS      = 80     # DBA: max 80, XMBT: 30 — берём 50 как компромисс
PATIENCE    = 15      # XMBT: early stopping patience = 6 эпох
GRAD_CLIP   = 1.0    # MulT (Tsai et al., 2019): gradient clip = 1.0


SAVE_PATH   = "best_model.pt"   # путь для сохранения лучшей модели

# =============================================================================

EMOTIONS = ["happy", "sad", "anger", "surprise", "disgust", "fear"]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_epoch(model, loader, optimizer, criterion, device):
    """Один проход по train — возвращает средний loss."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        text        = batch["text"].to(device)
        audio       = batch["audio"].to(device)
        vision      = batch["vision"].to(device)
        audio_mask  = batch["audio_mask"].to(device)
        vision_mask = batch["vision_mask"].to(device)
        labels      = batch["labels"].to(device).float()

        optimizer.zero_grad()

        text = batch["text"].to(device)
        logits_fuse, logits_text, logits_audio, logits_vision = model(
            text, audio, vision, audio_mask, vision_mask
        )

        # ── Auxiliary losses — каждая модальность учится независимо ──────────────
        # Статья: XMBT — L = L_fuse + L_text + L_audio + L_vision
        # Это регуляризует bottleneck и не даёт text доминировать
        loss = (criterion(logits_fuse,   labels) +
                criterion(logits_text,   labels) +
                criterion(logits_audio,  labels) +
                criterion(logits_vision, labels))
        loss.backward()

        # ── Gradient clipping для трансформеров ───────────────────────────────
        # Статья: MulT (Tsai et al., 2019) — gradient clip = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Валидация — считаем loss, Avg. F1, Avg. WA, per-class F1.
    Статья: MER-SEM-MBT, XMBT — главные метрики для CMU-MOSEI:
    Average F1 (macro) и Average Weighted Accuracy (Avg. WA)
    """
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    for batch in loader:
        text        = batch["text"].to(device)
        audio       = batch["audio"].to(device)
        vision      = batch["vision"].to(device)
        audio_mask  = batch["audio_mask"].to(device)
        vision_mask = batch["vision_mask"].to(device)
        labels      = batch["labels"].to(device).float()

        logits_fuse, _, _, _ = model(text, audio, vision, audio_mask, vision_mask)
        loss = criterion(logits_fuse, labels)
        total_loss += loss.item()

        # sigmoid → бинаризация порогом 0.5
        preds = (torch.sigmoid(logits_fuse) > 0.5).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    all_preds  = np.vstack(all_preds)   # [N, 6]
    all_labels = np.vstack(all_labels)  # [N, 6]

    # ── Avg. F1 (macro) — главная метрика ─────────────────────────────────────
    # Статья: MER-SEM-MBT — "Avg. F1 is used as the main evaluation indicator"
    avg_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # ── Per-class F1 — смотрим fear и surprise отдельно ──────────────────────
    # Статья: XMBT — "F1 scores exceeding 30% for fear and surprise"
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

    # ── Avg. WA (Weighted Accuracy) — дополнительная метрика ─────────────────
    # Статья: XMBT, MER-SEM-MBT — average binary weighted accuracy для CMU-MOSEI
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
        wa = 0.5 * (tp / n_pos + tn / n_neg)
        wa_list.append(wa)
    avg_wa = float(np.mean(wa_list))

    return total_loss / len(loader), avg_f1, avg_wa, per_class_f1


def print_metrics(epoch, train_loss, val_loss, avg_f1, avg_wa, per_f1, best_f1):
    print(f"\nEpoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
    print(f"         | Avg.F1={avg_f1:.4f}  Avg.WA={avg_wa:.4f}  "
          f"{'★ NEW BEST' if avg_f1 > best_f1 else ''}")
    print(f"         | Per-class F1:")
    for emo, f1 in zip(EMOTIONS, per_f1):
        bar = "█" * int(f1 * 20)
        print(f"           {emo:<10}: {f1:.4f}  {bar}")


def main():
    device = get_device()
    print(f"Device: {device}")

    # ── Загрузка данных ───────────────────────────────────────────────────────
    loaders, pos_weight = get_dataloaders()

    # ── Модель ───────────────────────────────────────────────────────────────
    model = BottleneckFusionModel().to(device)

    # ── Проблема 2: два lr — BERT и остальная сеть ────────────────────────────
    # Статья: XMBT (Nguyen et al., 2025) — "scaled learning rate strategy,
    # text learning rate factor to prevent gradient updates destabilizing training"
    # Статья: MER-SEM-MBT (Xia et al., 2022) — "smaller learning rate 1e-5 to fine-tune"
    # ── Один optimizer — BERT не обучается, все параметры с одним lr ──────────
    # Статья: DBA — AdamW lr=1e-4 для всей сети когда BERT заморожен
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAIN)

    # ── CosineAnnealingLR ─────────────────────────────────────────────────────
    # Статья: DBA, XMBT — cosine annealing постепенно снижает lr до ~0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    # ── Loss с pos_weight ─────────────────────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    # ── Early stopping ────────────────────────────────────────────────────────
    # Статья: XMBT — "training halts if F1 + WA does not improve for 6 epochs"
    best_f1      = 0.0
    patience_cnt  = 0

    print(f"\nНачинаем обучение: {EPOCHS} эпох, patience={PATIENCE}\n")
    print("=" * 65)

    for epoch in range(1, EPOCHS + 1):

        train_loss = train_epoch(model, loaders["train"], optimizer, criterion, device)
        val_loss, avg_f1, avg_wa, per_f1 = evaluate(model, loaders["valid"], criterion, device)

        print_metrics(epoch, train_loss, val_loss, avg_f1, avg_wa, per_f1, best_f1)

        # ── Сохранение лучшей модели по Avg. F1 ──────────────────────────────
        if avg_f1 > best_f1:
            best_f1      = avg_f1
            patience_cnt = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"         | ✅ Модель сохранена (best Avg.F1={best_f1:.4f})")
        else:
            patience_cnt += 1
            print(f"         | patience {patience_cnt}/{PATIENCE}")

        scheduler.step()

        # ── Ранняя остановка ──────────────────────────────────────────────────
        if patience_cnt >= PATIENCE:
            print(f"\n⏹  Early stopping на эпохе {epoch} (patience={PATIENCE})")
            break

    # ── Финальная оценка на test ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Загружаем лучшую модель для оценки на test...")
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))

    _, test_f1, test_wa, test_per_f1 = evaluate(model, loaders["test"], criterion, device)

    print(f"\n{'='*65}")
    print(f"ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ НА TEST")
    print(f"{'='*65}")
    print(f"  Avg. F1 (macro) : {test_f1:.4f}   ← главная метрика")
    print(f"  Avg. WA         : {test_wa:.4f}   ← дополнительная")
    print(f"\n  Per-class F1:")
    for emo, f1 in zip(EMOTIONS, test_per_f1):
        bar    = "█" * int(f1 * 20)
        marker = " ← следим!" if emo in ("fear", "surprise") else ""
        print(f"    {emo:<10}: {f1:.4f}  {bar}{marker}")

    print(f"\n  Сравнение с базовыми моделями (CMU-MOSEI):")
    print(f"    Late Fusion LSTM   : Avg.F1=0.433")
    print(f"    MulT (2019)        : Avg.F1=0.475")
    print(f"    MER-SEM-MBT (2022) : Avg.F1=0.509")
    print(f"    XMBT+DRA (2025)    : Avg.F1=0.496")
    print(f"    Твоя модель        : Avg.F1={test_f1:.4f}")
    delta = test_f1 - 0.475
    print(f"\n  vs MulT: {delta:+.4f}")


if __name__ == "__main__":
    main()