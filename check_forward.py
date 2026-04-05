"""
Forward-проверка BottleneckFusion перед обучением.

Запуск:
    python check_forward.py --data_path mosei_cleaned.pkl

Должно вывести shape-и на каждом шаге и завершиться без ошибок.
"""

import argparse
import pickle
from functools import partial

import torch
from torch.utils.data import DataLoader

from dataset import MoseiDataset
from bottleneck_fusion import BottleneckFusion
from train import collate_fn
from utils import device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="mosei_cleaned.pkl")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    print(f"Device: {device}")

    with open(args.data_path, "rb") as f:
        data = pickle.load(f)

    dataset = MoseiDataset(data=data, split="train", max_text_len=128)
    cfn = partial(collate_fn, max_audio_len=100, max_visual_len=100)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=cfn)

    batch = next(iter(loader))

    print("\n── Batch shapes ──────────────────────────────────────")
    print(f"  input_ids:      {batch['input_ids'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    print(f"  audio:          {batch['audio'].shape}")
    print(f"  audio_mask:     {batch['audio_mask'].shape}")
    print(f"  visual:         {batch['visual'].shape}")
    print(f"  visual_mask:    {batch['visual_mask'].shape}")
    print(f"  label:          {batch['label'].shape}")

    # ── Forward ─────────────────────────────────────────────────────────
    model = BottleneckFusion(
        num_classes=6,
        hidden_dim=128,
        num_bottleneck_tokens=8,
        num_heads=4,
        dropout=0.3,
        freeze_bert=True,
    ).to(device)
    model.eval()

    with torch.no_grad():
        # Запускаем вручную, чтобы видеть промежуточные shape-и
        B = batch["input_ids"].size(0)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        audio          = batch["audio"].to(device)
        audio_mask     = batch["audio_mask"].to(device)
        visual         = batch["visual"].to(device)
        visual_mask    = batch["visual_mask"].to(device)
        labels         = batch["label"].to(device)

        bert_out  = model.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_seq  = model.text_proj(bert_out.last_hidden_state)
        audio_seq = model.audio_proj(audio)
        visual_seq = model.visual_proj(visual)

        fused_seq  = torch.cat([text_seq, audio_seq, visual_seq], dim=1)
        text_pad   = (attention_mask == 0)
        audio_pad  = (audio_mask == 0)
        visual_pad = (visual_mask == 0)
        fused_pad  = torch.cat([text_pad, audio_pad, visual_pad], dim=1)

        bn_tokens = model.bottleneck_tokens.expand(B, -1, -1)
        attn_out, _ = model.cross_attn(
            query=bn_tokens, key=fused_seq, value=fused_seq,
            key_padding_mask=fused_pad,
        )
        attn_out = model.attn_norm(attn_out + bn_tokens)
        pooled   = attn_out.mean(dim=1)
        logits   = model.classifier(pooled)

    print("\n── Intermediate shapes ───────────────────────────────")
    print(f"  text_seq:         {text_seq.shape}   (B, Lt, 128)")
    print(f"  audio_seq:        {audio_seq.shape}   (B, Ta, 128)")
    print(f"  visual_seq:       {visual_seq.shape}  (B, Tv, 128)")
    print(f"  fused_seq:        {fused_seq.shape}   (B, Lt+Ta+Tv, 128)")
    print(f"  fused_pad_mask:   {fused_pad.shape}   (B, Lt+Ta+Tv)")
    print(f"  bottleneck_tokens:{bn_tokens.shape}  (B, 8, 128)")
    print(f"  attn_out:         {attn_out.shape}   (B, 8, 128)")
    print(f"  pooled:           {pooled.shape}      (B, 128)")
    print(f"  logits:           {logits.shape}      (B, 6)")

    # ── Loss + backward ──────────────────────────────────────────────────
    model.train()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    criterion = torch.nn.CrossEntropyLoss()

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        audio=audio,
        audio_mask=audio_mask,
        visual=visual,
        visual_mask=visual_mask,
    )
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    print(f"\n── Loss & backward ───────────────────────────────────")
    print(f"  loss = {loss.item():.4f}  ✓")
    print("\n✅ Forward check passed!")


if __name__ == "__main__":
    main()
