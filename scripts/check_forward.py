#!/usr/bin/env python
"""Check model forward pass and dataset loading."""

import argparse
import torch
from core.bottleneck_fusion import BottleneckFusion
from data.dataset import make_bottleneck_loaders
from utils.utils import device


def main():
    parser = argparse.ArgumentParser(description="Check model architecture")
    parser.add_argument("--data_path", type=str, default="mosei_bottleneck.pkl")
    args = parser.parse_args()

    print("🔧 Checking model and dataset...")

    # Load data
    print("\n1️⃣  Loading dataset...")
    train_loader, val_loader, test_loader = make_bottleneck_loaders(args.data_path)
    print(f"   ✅ Train: {len(train_loader)} batches")
    print(f"   ✅ Val: {len(val_loader)} batches")
    print(f"   ✅ Test: {len(test_loader)} batches")

    # Create model
    print("\n2️⃣  Creating model...")
    model = BottleneckFusion(num_bottleneck_tokens=16).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Model created: {total_params:,} parameters")

    # Forward pass
    print("\n3️⃣  Testing forward pass...")
    batch = next(iter(train_loader))
    with torch.no_grad():
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            audio=batch["audio"].to(device),
            audio_mask=batch["audio_mask"].to(device),
            visual=batch["visual"].to(device),
            visual_mask=batch["visual_mask"].to(device),
        )
    print(f"   ✅ Logits shape: {logits.shape}")

    # Test with domain separation
    print("\n4️⃣  Testing with domain separation...")
    with torch.no_grad():
        logits, domain_data, recon_data = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            audio=batch["audio"].to(device),
            audio_mask=batch["audio_mask"].to(device),
            visual=batch["visual"].to(device),
            visual_mask=batch["visual_mask"].to(device),
            return_domains=True,
        )
    print(f"   ✅ Logits: {logits.shape}")
    print(f"   ✅ Invariant domains: {list(domain_data.keys())[:3]}")
    print(f"   ✅ Reconstruction: {list(recon_data.keys())[:3]}")

    # Backward pass
    print("\n5️⃣  Testing backward pass...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = logits.mean()
    loss.backward()
    optimizer.step()
    print("   ✅ Backward pass successful")

    print("\n" + "="*50)
    print("✅ All checks passed! Ready to train.")
    print("="*50)


if __name__ == "__main__":
    main()
