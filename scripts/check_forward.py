"""Forward pass sanity check for all three models.

Usage:
    python scripts/check_forward.py --model text_only
    python scripts/check_forward.py --model audio_visual
    python scripts/check_forward.py --model bottleneck
"""
import sys
import argparse
import torch

sys.path.insert(0, ".")

from diploma_utils.utils import device, set_seed

B = 2
T_TEXT = 16
T_AUDIO = 50
T_VISUAL = 50
AUDIO_DIM = 74
VISUAL_DIM = 713
NUM_CLASSES = 6


def make_fake_batch():
    return {
        "input_ids": torch.randint(0, 1000, (B, T_TEXT)).to(device),
        "attention_mask": torch.ones(B, T_TEXT, dtype=torch.long).to(device),
        "audio": torch.randn(B, T_AUDIO, AUDIO_DIM).to(device),
        "audio_mask": torch.ones(B, T_AUDIO, dtype=torch.bool).to(device),
        "visual": torch.randn(B, T_VISUAL, VISUAL_DIM).to(device),
        "visual_mask": torch.ones(B, T_VISUAL, dtype=torch.bool).to(device),
        "labels": torch.randint(0, NUM_CLASSES, (B,)).to(device),
    }


def check_no_nan(tensor, name):
    assert not torch.isnan(tensor).any(), f"NaN detected in {name}!"


def check_text_only():
    from core.text_only_bert import TextOnlyBERT

    print("\n=== TextOnlyBERT ===")
    model = TextOnlyBERT(num_classes=NUM_CLASSES, freeze_bert="full").to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {total:,} total, {trainable:,} trainable")

    batch = make_fake_batch()

    with torch.no_grad():
        logits = model(batch)
    assert logits.shape == (B, NUM_CLASSES), f"Expected ({B}, {NUM_CLASSES}), got {logits.shape}"
    check_no_nan(logits, "logits")
    print(f"  Forward (no domains): logits {logits.shape} OK")

    with torch.no_grad():
        logits2, domains = model(batch, return_domains=True)
    assert isinstance(domains, dict)
    print(f"  Forward (return_domains=True): logits {logits2.shape}, domains={domains} OK")

    logits3, _ = model(batch, return_domains=True)
    loss = logits3.mean()
    loss.backward()
    print("  Backward: OK")
    print("  TextOnlyBERT: PASSED")


def check_audio_visual():
    from core.audio_visual_baseline import AudioVisualBaseline

    print("\n=== AudioVisualBaseline ===")
    model = AudioVisualBaseline(
        num_classes=NUM_CLASSES,
        audio_input_dim=AUDIO_DIM,
        visual_input_dim=VISUAL_DIM,
    ).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {total:,} total, {trainable:,} trainable")

    batch = make_fake_batch()

    with torch.no_grad():
        logits = model(batch)
    assert logits.shape == (B, NUM_CLASSES), f"Expected ({B}, {NUM_CLASSES}), got {logits.shape}"
    check_no_nan(logits, "logits")
    print(f"  Forward (no domains): logits {logits.shape} OK")

    logits2, domains = model(batch, return_domains=True)
    loss = logits2.mean()
    loss.backward()
    print("  Backward: OK")
    print("  AudioVisualBaseline: PASSED")


def check_bottleneck():
    from core.bottleneck_fusion import BottleneckFusion
    from training.losses import compute_separation_loss, compute_invariant_loss, compute_reconstruction_loss

    print("\n=== BottleneckFusion ===")
    model = BottleneckFusion(
        num_classes=NUM_CLASSES,
        audio_input_dim=AUDIO_DIM,
        visual_input_dim=VISUAL_DIM,
        freeze_bert="full",
    ).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {total:,} total, {trainable:,} trainable")

    batch = make_fake_batch()

    with torch.no_grad():
        logits = model(batch)
    assert logits.shape == (B, NUM_CLASSES), f"Expected ({B}, {NUM_CLASSES}), got {logits.shape}"
    check_no_nan(logits, "logits")
    print(f"  Forward (no domains): logits {logits.shape} OK")

    with torch.no_grad():
        logits2, domain_data, recon_data = model(batch, return_domains=True)
    assert logits2.shape == (B, NUM_CLASSES)
    check_no_nan(logits2, "logits")
    print(f"  Forward (with domains): logits {logits2.shape}")
    print(f"    domain_data keys: {list(domain_data.keys())}")
    print(f"    recon_data  keys: {list(recon_data.keys())}")

    for k, v in domain_data.items():
        assert v.shape == (B, 128), f"  domain_data[{k}]: expected (B, 128), got {v.shape}"
    assert recon_data["text_recon"].shape == (B, 768), \
        f"text_recon shape mismatch: {recon_data['text_recon'].shape}"
    assert recon_data["audio_recon"].shape == (B, AUDIO_DIM), \
        f"audio_recon shape mismatch: {recon_data['audio_recon'].shape}"
    assert recon_data["visual_recon"].shape == (B, VISUAL_DIM), \
        f"visual_recon shape mismatch: {recon_data['visual_recon'].shape}"

    # Shapes of recon vs originals must match
    for mod in ("text", "audio", "visual"):
        r = recon_data[f"{mod}_recon"]
        o = recon_data[f"{mod}_original"]
        assert r.shape == o.shape, f"recon/original shape mismatch for {mod}: {r.shape} vs {o.shape}"
    print("  Recon/original shapes: OK")

    # Loss functions
    L_sep = compute_separation_loss(domain_data)
    L_inv = compute_invariant_loss(domain_data)
    L_rec = compute_reconstruction_loss(recon_data)
    print(f"  Losses: sep={L_sep.item():.4f}  inv={L_inv.item():.4f}  rec={L_rec.item():.4f}")

    # Backward
    logits3, dom3, rec3 = model(batch, return_domains=True)
    loss = logits3.mean() + dom3["text_inv_pool"].mean()
    loss.backward()
    print("  Backward: OK")

    print("  BottleneckFusion: PASSED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bottleneck",
                        choices=["text_only", "audio_visual", "bottleneck", "all"])
    args = parser.parse_args()

    set_seed(42)
    print(f"Device: {device}")

    if args.model == "all":
        check_text_only()
        check_audio_visual()
        check_bottleneck()
    elif args.model == "text_only":
        check_text_only()
    elif args.model == "audio_visual":
        check_audio_visual()
    elif args.model == "bottleneck":
        check_bottleneck()

    print("\n" + "=" * 50)
    print("All checks passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
