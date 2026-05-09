"""Quick sanity check: create model, run forward + backward, verify shapes."""
import sys
import torch

sys.path.insert(0, ".")
from core.bottleneck_fusion import BottleneckFusion
from diploma_utils.utils import device


def main():
    B, T_text, T_av = 4, 32, 20

    print("1. Creating model...")
    model = BottleneckFusion(num_bottleneck_tokens=16, freeze_bert=True).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   {total:,} params total, {trainable:,} trainable")

    # Fake batch
    batch = {
        "input_ids": torch.randint(0, 1000, (B, T_text)).to(device),
        "attention_mask": torch.ones(B, T_text, dtype=torch.long).to(device),
        "audio": torch.randn(B, T_av, 74).to(device),
        "audio_mask": torch.ones(B, T_av, dtype=torch.bool).to(device),
        "visual": torch.randn(B, T_av, 713).to(device),
        "visual_mask": torch.ones(B, T_av, dtype=torch.bool).to(device),
    }

    print("\n2. Forward pass (no domains)...")
    with torch.no_grad():
        logits = model(**batch)
    print(f"   logits: {logits.shape}")
    assert logits.shape == (B, 6), f"Expected (B, 6), got {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN in logits!"

    print("\n3. Forward pass (with domains)...")
    with torch.no_grad():
        logits, domain_data, recon_data = model(**batch, return_domains=True)
    print(f"   logits: {logits.shape}")
    print(f"   domain keys: {list(domain_data.keys())}")
    print(f"   recon keys: {list(recon_data.keys())}")

    for k, v in domain_data.items():
        assert v.shape == (B, 128), f"{k}: expected (B, 128), got {v.shape}"
    assert recon_data["text_recon"].shape == (B, 768)
    assert recon_data["audio_recon"].shape == (B, 74)
    assert recon_data["visual_recon"].shape == (B, 713)

    print("\n4. Backward pass...")
    logits, domain_data, recon_data = model(**batch, return_domains=True)
    loss = logits.mean() + domain_data["text_inv_pool"].mean()
    loss.backward()
    print("   Backward OK")

    print("\n5. Testing with disabled modalities...")
    model_text_only = BottleneckFusion(
        use_audio=False, use_visual=False, freeze_bert=True
    ).to(device)
    with torch.no_grad():
        logits = model_text_only(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
    print(f"   text-only logits: {logits.shape}")
    assert logits.shape == (B, 6)

    print("\n" + "=" * 50)
    print("All checks passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
