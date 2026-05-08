import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
import logging

logging.getLogger('transformers').setLevel(logging.ERROR)

from core.bottleneck_fusion import BottleneckFusion
from data.dataset import make_bottleneck_loaders
from training.losses import compute_separation_loss, compute_invariant_loss, compute_reconstruction_loss
from diploma_utils.utils import device, set_seed


EMOTIONS = ["happy", "sad", "anger", "surprise", "disgust", "fear"]


def train_epoch(model, loader, optimizer, criterion, model_type, device,
                use_domain_sep=False, alpha_sep=0.1, alpha_inv=0.05, alpha_rec=0.01):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        labels = batch["label"].to(device)

        if use_domain_sep and model_type == "bottleneck":
            logits, domain_data, recon_data = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                audio=batch["audio"].to(device),
                audio_mask=batch["audio_mask"].to(device),
                visual=batch["visual"].to(device),
                visual_mask=batch["visual_mask"].to(device),
                return_domains=True,
            )
            L_task = criterion(logits, labels)
            L_sep = compute_separation_loss(domain_data, device)
            L_inv = compute_invariant_loss(domain_data, labels)
            L_rec = compute_reconstruction_loss(recon_data)
            loss = L_task + alpha_sep * L_sep + alpha_inv * L_inv + alpha_rec * L_rec
        else:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                audio=batch["audio"].to(device),
                audio_mask=batch["audio_mask"].to(device),
                visual=batch["visual"].to(device),
                visual_mask=batch["visual_mask"].to(device),
            )
            loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), acc, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        labels = batch["label"].to(device)
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            audio=batch["audio"].to(device),
            audio_mask=batch["audio_mask"].to(device),
            visual=batch["visual"].to(device),
            visual_mask=batch["visual_mask"].to(device),
        )
        total_loss += criterion(logits, labels).item()
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), acc, f1, all_preds, all_labels


def build_model(args):
    """Create model."""
    return BottleneckFusion(
        num_bottleneck_tokens=args.num_bottleneck_tokens,
        freeze_bert=(args.freeze_bert != "none"),
    )


def save_results(exp_dir, best_val_f1, best_epoch, test_loss, test_preds, test_labels):
    """Save metrics and results."""
    os.makedirs(exp_dir, exist_ok=True)

    acc = accuracy_score(test_labels, test_preds)
    f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    cm = confusion_matrix(test_labels, test_preds)
    report = classification_report(test_labels, test_preds, target_names=EMOTIONS, digits=3)

    with open(os.path.join(exp_dir, "metrics.txt"), "w") as f:
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best val F1: {best_val_f1:.4f}\n")
        f.write(f"Test loss: {test_loss:.4f}\n")
        f.write(f"Test accuracy: {acc:.4f}\n")
        f.write(f"Test macro F1: {f1:.4f}\n")
        f.write(f"\n{report}\n")

    print(f"\n{'='*55}")
    print(f"TEST | loss={test_loss:.4f}  acc={acc:.4f}  f1={f1:.4f}")
    print(f"{'='*55}")


def main():
    parser = argparse.ArgumentParser(description="Train bottleneck model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data_path", type=str, default="mosei_bottleneck.pkl")
    parser.add_argument("--num_bottleneck_tokens", type=int, default=16)
    parser.add_argument("--freeze_bert", type=str, default="full", choices=["full", "partial", "none"])
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--use_domain_sep", action="store_true")
    parser.add_argument("--alpha_sep", type=float, default=0.1)
    parser.add_argument("--alpha_inv", type=float, default=0.05)
    parser.add_argument("--alpha_rec", type=float, default=0.01)
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--exp_dir", type=str, default=None)

    args = parser.parse_args()
    set_seed(42)

    # Data
    print(f"Loading data from {args.data_path}...")
    train_loader, val_loader, test_loader = make_bottleneck_loaders(
        data_path=args.data_path, batch_size=args.batch_size
    )

    with open(args.data_path, "rb") as f:
        data = pickle.load(f)
    train_labels = np.array([s["label"] for s in data["train"]])
    class_weights = compute_class_weight("balanced", classes=np.arange(6), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Model
    model = build_model(args).to(device)
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"Loading pretrained weights from {args.pretrained_path}")
        model.load_state_dict(torch.load(args.pretrained_path, map_location=device))

    # Training
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    best_val_f1 = 0.0
    best_epoch = 1
    epochs_no_improve = 0
    save_path = os.path.join(args.exp_dir, "best_model.pt") if args.exp_dir else "best_model.pt"

    print(f"\nTraining: epochs={args.epochs}, lr={args.lr}, domain_sep={args.use_domain_sep}")
    print("=" * 65)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, "bottleneck", device,
            use_domain_sep=args.use_domain_sep,
            alpha_sep=args.alpha_sep, alpha_inv=args.alpha_inv, alpha_rec=args.alpha_rec
        )
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"Train: loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f} | "
              f"Val: loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f}")

        scheduler.step()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ Best F1={best_val_f1:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Test
    print(f"\nLoading best model from epoch {best_epoch}...")
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

    if args.exp_dir:
        save_results(args.exp_dir, best_val_f1, best_epoch, test_loss, test_preds, test_labels)


if __name__ == "__main__":
    main()
