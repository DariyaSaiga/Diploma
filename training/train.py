"""Train one of three multi-label MOSEI models.

Usage:
    PYTHONPATH=. python training/train.py \
        --model bottleneck \
        --data_path datasets/mosei_emotion_aligned_60.pkl \
        --exp_dir experiments/exp03_bottleneck \
        --epochs 60 --batch_size 16 --lr 3e-4 \
        --use_domain_sep
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Make `python training/train.py` work too (not only `python -m training.train`).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.mosei_multilabel_dataset import EMOTIONS, NUM_CLASSES, make_loaders
from models import build_model
from training.losses import (
    compute_total_loss,
    make_bce_criterion,
    make_pos_weight,
)
from training.metrics import (
    EMOTIONS as M_EMOTIONS,
    compute_metrics,
    find_best_thresholds_on_validation,
    format_per_class_report,
)
from training.utils import (
    AverageMeter,
    append_log,
    count_parameters,
    ensure_dir,
    get_device,
    save_config,
    set_seed,
)

assert EMOTIONS == M_EMOTIONS, "Emotion order mismatch"


# ─────────────────────────── argparse ────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train multi-label MOSEI model")
    p.add_argument("--model", choices=["late_fusion", "mult", "bottleneck"], required=True)
    p.add_argument("--data_path", default="datasets/mosei_emotion_aligned_60.pkl")
    p.add_argument("--exp_dir", required=True)

    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--normalize", action="store_true",
                   help="Train-mean/std normalization for features")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                   help="Force device (auto skips MPS — see README)")

    # Model
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--num_bottleneck_tokens", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.3)

    # Loss / multi-label
    p.add_argument("--pos_weight_mode", choices=["none", "balanced", "sqrt"], default="sqrt")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--threshold_search", action="store_true",
                   help="Pick best global threshold on validation after training")
    p.add_argument("--monitor_metric", choices=["weighted_f1", "macro_f1", "micro_f1"],
                   default="weighted_f1")

    # Domain separation (only used by bottleneck)
    p.add_argument("--use_domain_sep", action="store_true")
    p.add_argument("--alpha_sep", type=float, default=0.03)
    p.add_argument("--alpha_inv", type=float, default=0.005)
    p.add_argument("--alpha_rec", type=float, default=0.001)

    return p.parse_args()


# ─────────────────────────── helpers ─────────────────────────────────────────

def build_model_from_args(args) -> nn.Module:
    return build_model(
        args.model,
        num_classes=NUM_CLASSES,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_bottleneck_tokens=args.num_bottleneck_tokens,
        dropout=args.dropout,
        use_domain_sep=args.use_domain_sep,
    )


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def model_forward(model, batch, *, return_domains: bool):
    """Unified forward: bottleneck supports return_domains; others ignore it."""
    if isinstance(model, torch.nn.Module) and hasattr(model, "forward"):
        try:
            return model(batch, return_domains=return_domains)
        except TypeError:
            return model(batch)
    return model(batch)


def run_epoch(model, loader, optimizer, criterion, device, args, train: bool):
    model.train(train)

    loss_meter = AverageMeter()
    aux_meters = {k: AverageMeter() for k in ("task", "sep", "inv", "rec")}
    all_logits, all_labels = [], []

    use_dom = bool(args.use_domain_sep) and args.model == "bottleneck"

    for batch in loader:
        batch = to_device(batch, device)
        labels = batch["labels"]

        with torch.set_grad_enabled(train):
            if use_dom:
                logits, domain_data, recon_data = model_forward(model, batch, return_domains=True)
            else:
                logits = model_forward(model, batch, return_domains=False)
                domain_data, recon_data = {}, {}

            loss, log = compute_total_loss(
                logits, labels, domain_data, recon_data, criterion,
                use_domain_sep=use_dom,
                alpha_sep=args.alpha_sep,
                alpha_inv=args.alpha_inv,
                alpha_rec=args.alpha_rec,
            )

        if train:
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        bs = labels.size(0)
        loss_meter.update(loss.item(), bs)
        for k, v in log.items():
            aux_meters[k].update(v, bs)

        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    return loss_meter.avg, aux_meters, logits_cat, labels_cat


def collect_logits(model, loader, device):
    """Forward pass without gradient/loss for threshold search and final eval."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            logits = model_forward(model, batch, return_domains=False)
            all_logits.append(logits.cpu())
            all_labels.append(batch["labels"].cpu())
    return torch.cat(all_logits, 0), torch.cat(all_labels, 0)


def save_metrics_txt(exp_dir, *, best_epoch, best_val, threshold, val_metrics, test_metrics):
    lines = [
        f"best_epoch: {best_epoch}",
        f"monitor_metric: {best_val['monitor_metric']}",
        f"best_val_{best_val['monitor_metric']}: {best_val['score']:.4f}",
        f"threshold: {threshold:.4f}",
        "",
        "── VALIDATION (best epoch) ──",
        f"micro_f1:        {val_metrics.micro_f1:.4f}",
        f"macro_f1:        {val_metrics.macro_f1:.4f}",
        f"weighted_f1:     {val_metrics.weighted_f1:.4f}",
        f"samples_f1:      {val_metrics.samples_f1:.4f}",
        f"hamming_loss:    {val_metrics.hamming_loss:.4f}",
        f"subset_accuracy: {val_metrics.subset_accuracy:.4f}",
        "",
        "── TEST ──",
        f"micro_f1:        {test_metrics.micro_f1:.4f}",
        f"macro_f1:        {test_metrics.macro_f1:.4f}",
        f"weighted_f1:     {test_metrics.weighted_f1:.4f}",
        f"samples_f1:      {test_metrics.samples_f1:.4f}",
        f"hamming_loss:    {test_metrics.hamming_loss:.4f}",
        f"subset_accuracy: {test_metrics.subset_accuracy:.4f}",
        "",
        format_per_class_report(test_metrics, EMOTIONS),
    ]
    with open(Path(exp_dir) / "metrics.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


def save_metrics_json(exp_dir, *, best_epoch, threshold, val_metrics, test_metrics):
    payload = {
        "best_epoch": best_epoch,
        "threshold": threshold,
        "valid": val_metrics.to_dict(),
        "test": test_metrics.to_dict(),
        "emotions": EMOTIONS,
    }
    with open(Path(exp_dir) / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)


# ─────────────────────────── main ────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    exp_dir = ensure_dir(args.exp_dir)

    save_config(exp_dir, vars(args))
    log_path = exp_dir / "train_log.txt"
    if log_path.exists():
        log_path.unlink()
    append_log(log_path, f"# device={device}  model={args.model}")

    print(f"Loading data from {args.data_path} ...")
    train_loader, valid_loader, test_loader, info = make_loaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=args.normalize,
        pin_memory=(device.type == "cuda"),
    )
    print(f"  train={info['n_train']}  valid={info['n_valid']}  test={info['n_test']}")
    print(f"  per-class positives (train): {info['pos_counts'].tolist()}")

    pos_weight = make_pos_weight(info["pos_counts"], info["neg_counts"], args.pos_weight_mode)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
        print(f"  pos_weight ({args.pos_weight_mode}): "
              f"{[round(float(w), 3) for w in pos_weight]}")
    criterion = make_bce_criterion(pos_weight)

    model = build_model_from_args(args).to(device)
    total, trainable = count_parameters(model)
    print(f"Model: {args.model}  | params: {total:,} total, {trainable:,} trainable")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )

    best_score = -float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    best_path = exp_dir / "best_model.pt"
    last_path = exp_dir / "last_checkpoint.pt"

    print("=" * 70)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, _, train_logits, train_labels = run_epoch(
            model, train_loader, optimizer, criterion, device, args, train=True
        )
        val_loss, _, val_logits, val_labels = run_epoch(
            model, valid_loader, optimizer, criterion, device, args, train=False
        )

        train_m = compute_metrics(train_logits, train_labels, threshold=args.threshold)
        val_m = compute_metrics(val_logits, val_labels, threshold=args.threshold)
        score = getattr(val_m, args.monitor_metric)

        dt = time.time() - t0
        line = (
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_w_f1={train_m.weighted_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_w_f1={val_m.weighted_f1:.4f} "
            f"val_macro_f1={val_m.macro_f1:.4f} val_micro_f1={val_m.micro_f1:.4f} | "
            f"{dt:.1f}s"
        )
        print(line)
        append_log(log_path, line)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_score": best_score,
            "args": vars(args),
        }, last_path)

        if score > best_score:
            best_score = score
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_score": best_score,
                "args": vars(args),
            }, best_path)
            msg = f"  ↳ new best {args.monitor_metric}={best_score:.4f} (epoch {epoch})"
            print(msg)
            append_log(log_path, msg)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                msg = f"Early stopping at epoch {epoch} (no improve for {args.patience})"
                print(msg)
                append_log(log_path, msg)
                break

    # ── load best, run final eval ────────────────────────────────────────────
    print("=" * 70)
    print(f"Loading best checkpoint (epoch {best_epoch}) for final evaluation ...")
    state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])

    val_logits, val_labels = collect_logits(model, valid_loader, device)
    threshold = args.threshold
    if args.threshold_search:
        threshold, ts_score = find_best_thresholds_on_validation(
            val_logits, val_labels, monitor=args.monitor_metric
        )
        print(f"Threshold search → best {args.monitor_metric}={ts_score:.4f} at t={threshold:.2f}")

    val_metrics = compute_metrics(val_logits, val_labels, threshold=threshold)
    test_logits, test_labels = collect_logits(model, test_loader, device)
    test_metrics = compute_metrics(test_logits, test_labels, threshold=threshold)

    save_metrics_txt(
        exp_dir,
        best_epoch=best_epoch,
        best_val={"monitor_metric": args.monitor_metric, "score": best_score},
        threshold=threshold,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )
    save_metrics_json(
        exp_dir,
        best_epoch=best_epoch,
        threshold=threshold,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )

    with open(exp_dir / "per_class_report.txt", "w") as f:
        f.write(format_per_class_report(test_metrics, EMOTIONS) + "\n")

    print("\n=== TEST ===")
    print(f"weighted_f1={test_metrics.weighted_f1:.4f}  "
          f"macro_f1={test_metrics.macro_f1:.4f}  "
          f"micro_f1={test_metrics.micro_f1:.4f}  "
          f"hamming={test_metrics.hamming_loss:.4f}  "
          f"subset_acc={test_metrics.subset_accuracy:.4f}")
    print(format_per_class_report(test_metrics, EMOTIONS))


if __name__ == "__main__":
    main()
