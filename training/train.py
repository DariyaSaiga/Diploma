import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
import logging

logging.getLogger('transformers').setLevel(logging.ERROR)

from core.bottleneck_fusion import BottleneckFusion
from data.dataset import make_bottleneck_loaders
from training.losses import compute_separation_loss, compute_invariant_loss, compute_reconstruction_loss
from diploma_utils.utils import device, set_seed


EMOTIONS = ["happy", "sad", "anger", "surprise", "disgust", "fear"]


def train_epoch(model, loader, optimizer, criterion, device,
                use_domain_sep=False, alpha_sep=0.1, alpha_inv=0.05, alpha_rec=0.01):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        labels = batch["label"].to(device)

        if use_domain_sep:
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
    return BottleneckFusion(
        num_bottleneck_tokens=args.num_bottleneck_tokens,
        freeze_bert=args.freeze_bert,
        use_audio=not args.no_audio,
        use_visual=not args.no_visual,
        audio_input_dim=args.audio_input_dim,
        visual_input_dim=args.visual_input_dim,
        dropout=args.dropout,
    )


def make_optimizer(model, args):
    if args.lr_bert is not None and args.freeze_bert != "full":
        bert_params = [p for p in model.bert.parameters() if p.requires_grad]
        non_bert_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and not n.startswith("bert.")
        ]
        return torch.optim.AdamW([
            {"params": non_bert_params, "lr": args.lr},
            {"params": bert_params, "lr": args.lr_bert},
        ], weight_decay=1e-4)
    else:
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=1e-4,
        )


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_f1, args):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_f1": best_val_f1,
        "args": vars(args),
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        return ckpt.get("epoch", 0), ckpt.get("best_val_f1", 0.0)
    else:
        model.load_state_dict(ckpt)
        return 0, 0.0


def save_results(exp_dir, best_val_f1, best_epoch, test_loss, test_preds, test_labels):
    os.makedirs(exp_dir, exist_ok=True)
    acc = accuracy_score(test_labels, test_preds)
    f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
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
    parser.add_argument("--lr_bert", type=float, default=None,
                        help="Separate LR for BERT (used when freeze_bert != full)")
    parser.add_argument("--data_path", type=str, default="mosei_bottleneck.pkl")
    parser.add_argument("--num_bottleneck_tokens", type=int, default=16)
    parser.add_argument("--freeze_bert", type=str, default="full",
                        choices=["full", "partial", "none"])
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--use_domain_sep", action="store_true")
    parser.add_argument("--alpha_sep", type=float, default=0.1)
    parser.add_argument("--alpha_inv", type=float, default=0.05)
    parser.add_argument("--alpha_rec", type=float, default=0.01)
    parser.add_argument("--no_audio", action="store_true")
    parser.add_argument("--no_visual", action="store_true")
    parser.add_argument("--audio_input_dim", type=int, default=74)
    parser.add_argument("--visual_input_dim", type=int, default=713)
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="Load only model weights (new optimizer) — for next stage")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from full checkpoint (optimizer + scheduler + epoch)")
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

    # Model + optimizer + scheduler
    model = build_model(args).to(device)
    optimizer = make_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    best_val_f1 = 0.0
    best_epoch = 1
    start_epoch = 1
    epochs_no_improve = 0

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        ep, f1 = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = ep + 1
        best_val_f1 = f1
        print(f"  Epoch {start_epoch}, best val F1 = {best_val_f1:.4f}")
    elif args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"Loading weights from {args.pretrained_path} (fresh optimizer)")
        load_checkpoint(args.pretrained_path, model)

    if args.exp_dir:
        os.makedirs(args.exp_dir, exist_ok=True)

    best_model_path = os.path.join(args.exp_dir, "best_model.pt") if args.exp_dir else "best_model.pt"
    last_checkpoint_path = os.path.join(args.exp_dir, "last_checkpoint.pt") if args.exp_dir else "last_checkpoint.pt"

    # Save config
    if args.exp_dir:
        with open(os.path.join(args.exp_dir, "config.txt"), "w") as f:
            for k, v in sorted(vars(args).items()):
                f.write(f"{k}: {v}\n")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParams: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Training: epochs={start_epoch}-{args.epochs}, lr={args.lr}, "
          f"lr_bert={args.lr_bert}, domain_sep={args.use_domain_sep}")
    print("=" * 65)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, device,
            use_domain_sep=args.use_domain_sep,
            alpha_sep=args.alpha_sep, alpha_inv=args.alpha_inv, alpha_rec=args.alpha_rec,
        )
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"Train: loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f} | "
              f"Val: loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} | "
              f"LR={lr_now:.2e}")

        save_checkpoint(last_checkpoint_path, model, optimizer, scheduler,
                        epoch, best_val_f1, args)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            save_checkpoint(best_model_path, model, optimizer, scheduler,
                            epoch, best_val_f1, args)
            print(f"  -> Best val F1={best_val_f1:.4f}, saved to {best_model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Test
    print(f"\nLoading best model (epoch {best_epoch})...")
    load_checkpoint(best_model_path, model)
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    if args.exp_dir:
        save_results(args.exp_dir, best_val_f1, best_epoch, test_loss, test_preds, test_labels)
        with open(os.path.join(args.exp_dir, "model_info.txt"), "w") as f:
            f.write(f"Best F1: {best_val_f1:.4f} (epoch {best_epoch})\n")
            f.write(f"Test Accuracy: {test_acc:.4f}\n")
            f.write(f"Test F1: {test_f1:.4f}\n")
            f.write(f"\n--- Resume / next stage ---\n")
            f.write(f"Resume:     --resume {last_checkpoint_path}\n")
            f.write(f"Next stage: --pretrained_path {best_model_path}\n")
    else:
        print(f"\nTEST | loss={test_loss:.4f}  acc={test_acc:.4f}  f1={test_f1:.4f}")


if __name__ == "__main__":
    main()
