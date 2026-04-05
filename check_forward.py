"""
Forward-проверка BottleneckFusion перед обучением.

Запуск:
    python check_forward.py --data_path mosei_bottleneck.pkl

Проверяет:
  - shapes на входе и выходе каждого шага
  - loss + backward не падают
  - совместимость с текущей архитектурой BottleneckFusion
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
    parser.add_argument("--data_path", default="mosei_bottleneck.pkl")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_bottleneck_tokens", type=int, default=16)
    parser.add_argument("--num_bottleneck_layers", type=int, default=2)
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

    # ── Создаём модель ────────────────────────────────────────────────────
    model = BottleneckFusion(
        num_classes=6,
        hidden_dim=128,
        num_bottleneck_tokens=args.num_bottleneck_tokens,
        num_bottleneck_layers=args.num_bottleneck_layers,
        num_heads=4,
        dropout=0.3,
        freeze_bert=True,
    ).to(device)
    model.eval()

    B = batch["input_ids"].size(0)
    input_ids      = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    audio          = batch["audio"].to(device)
    audio_mask     = batch["audio_mask"].to(device)
    visual         = batch["visual"].to(device)
    visual_mask    = batch["visual_mask"].to(device)
    labels         = batch["label"].to(device)

    # ── Промежуточные shapes ──────────────────────────────────────────────
    print("\n── Intermediate shapes ───────────────────────────────")
    with torch.no_grad():
        bert_out   = model.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_seq   = model.text_proj(bert_out.last_hidden_state)
        audio_seq  = model.audio_proj(audio)
        visual_seq = model.visual_proj(visual)

        print(f"  text_seq:           {text_seq.shape}   (B, Lt, 128)")
        print(f"  audio_seq:          {audio_seq.shape}   (B, Ta, 128)")
        print(f"  visual_seq:         {visual_seq.shape}  (B, Tv, 128)")

        text_pad   = (attention_mask == 0)
        audio_pad  = (audio_mask == 0)
        visual_pad = (visual_mask == 0)

        bn = model.bottleneck_tokens.expand(B, -1, -1).clone()
        print(f"  bottleneck_tokens:  {bn.shape}   (B, n_bn, 128)")
        print(f"  num_layers:         {len(model.layers)}")

        # Прогоняем через все слои
        for i, layer in enumerate(model.layers):
            text_seq, audio_seq, visual_seq, bn = layer(
                text_seq, audio_seq, visual_seq, bn,
                text_pad, audio_pad, visual_pad
            )
            print(f"  after layer[{i}] bn:  {bn.shape}")

        pooled = bn.mean(dim=1)
        print(f"  pooled:             {pooled.shape}      (B, 128)")

        logits_check = model.classifier(pooled)
        print(f"  logits:             {logits_check.shape}      (B, 6)")

    # ── Full forward pass ─────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio=audio,
            audio_mask=audio_mask,
            visual=visual,
            visual_mask=visual_mask,
        )
    print(f"\n  model() output:     {logits.shape}  ✓")

    # ── Loss + backward ───────────────────────────────────────────────────
    print("\n── Loss & backward ───────────────────────────────────")
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4
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

    print(f"  loss = {loss.item():.4f}  ✓")
    print(f"\n✅ Forward check passed!")
    print(f"   tokens={args.num_bottleneck_tokens}  layers={args.num_bottleneck_layers}  device={device}")


if __name__ == "__main__":
    main()
