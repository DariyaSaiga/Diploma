"""Forward / backward sanity check for all three models on a fake batch.

Usage:
    PYTHONPATH=. python scripts/check_forward.py --model late_fusion
    PYTHONPATH=. python scripts/check_forward.py --model mult
    PYTHONPATH=. python scripts/check_forward.py --model bottleneck
    PYTHONPATH=. python scripts/check_forward.py --model all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from old_arch.data.mosei_multilabel_dataset import (
    AUDIO_DIM,
    NUM_CLASSES,
    SEQ_LEN,
    TEXT_DIM,
    VISION_DIM,
)
from old_arch.models import build_model
from old_arch.training.losses import (
    compute_invariant_loss,
    compute_reconstruction_loss,
    compute_separation_loss,
)
from old_arch.training.utils import count_parameters, get_device, set_seed

B = 2


def make_fake_batch(device: torch.device) -> dict:
    return {
        "text": torch.randn(B, SEQ_LEN, TEXT_DIM, device=device),
        "audio": torch.randn(B, SEQ_LEN, AUDIO_DIM, device=device),
        "vision": torch.randn(B, SEQ_LEN, VISION_DIM, device=device),
        "labels": torch.randint(0, 2, (B, NUM_CLASSES), device=device).float(),
    }


def assert_no_nan(t: torch.Tensor, name: str) -> None:
    assert torch.isfinite(t).all(), f"non-finite values in {name}"


def check_one(name: str, device: torch.device, *, use_domain_sep: bool = False) -> None:
    print(f"\n=== {name}{' [domain_sep]' if use_domain_sep else ''} ===")
    kwargs = dict(num_classes=NUM_CLASSES, dropout=0.3)
    if name == "mult":
        kwargs.update(hidden_dim=64, num_heads=4, num_layers=2)
    elif name == "bottleneck":
        kwargs.update(hidden_dim=128, num_heads=8, num_layers=2,
                      num_bottleneck_tokens=16, use_domain_sep=use_domain_sep)
    else:
        kwargs.update(hidden_dim=128)

    model = build_model(name, **kwargs).to(device)
    total, trainable = count_parameters(model)
    print(f"  params: {total:,} total, {trainable:,} trainable")

    batch = make_fake_batch(device)

    # forward
    if name == "bottleneck":
        out = model(batch, return_domains=use_domain_sep)
        if use_domain_sep:
            logits, dom, recon = out
            assert isinstance(dom, dict) and isinstance(recon, dict)
            for k, v in dom.items():
                assert v.shape == (B, kwargs["hidden_dim"]), f"{k}: {v.shape}"
            print(f"  domain_data keys: {sorted(dom.keys())}")
            print(f"  recon_data keys: {sorted(recon.keys())}")
            L_sep = compute_separation_loss(dom)
            L_inv = compute_invariant_loss(dom)
            L_rec = compute_reconstruction_loss(recon)
            print(f"  aux losses: sep={L_sep:.4f}  inv={L_inv:.4f}  rec={L_rec:.4f}")
        else:
            logits = out
    else:
        logits = model(batch)

    assert logits.shape == (B, NUM_CLASSES), f"got {tuple(logits.shape)}"
    assert_no_nan(logits, "logits")
    print(f"  forward OK: logits shape = {tuple(logits.shape)}")

    # backward via BCE
    loss = F.binary_cross_entropy_with_logits(logits, batch["labels"])
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "no gradients flowed"
    for p in grads:
        assert torch.isfinite(p).all(), "non-finite gradient"
    print(f"  backward OK: BCE={loss.item():.4f}, grads={len(grads)} tensors")
    print(f"  {name}: PASSED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all",
                        choices=["late_fusion", "mult", "bottleneck", "all"])
    args = parser.parse_args()

    set_seed(42)
    device = get_device("auto")
    print(f"Device: {device}")

    if args.model == "all":
        check_one("late_fusion", device)
        check_one("mult", device)
        check_one("bottleneck", device, use_domain_sep=False)
        check_one("bottleneck", device, use_domain_sep=True)
    elif args.model == "bottleneck":
        check_one("bottleneck", device, use_domain_sep=False)
        check_one("bottleneck", device, use_domain_sep=True)
    else:
        check_one(args.model, device)

    print("\n" + "=" * 50)
    print("All checks passed.")
    print("=" * 50)


if __name__ == "__main__":
    main()
