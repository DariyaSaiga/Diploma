import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MoseiDataset
from utils import set_seed, device
from sklearn.metrics import accuracy_score, f1_score
import argparse


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    all_preds = []
    all_labels = []

    for batch in loader:
        audio = batch["audio"].to(device)
        visual = batch["visual"].to(device)
        labels = batch["label"].to(device)

        # текст пока не используем (для AV baseline)
        text = batch["text"]

        outputs = model(audio=audio, visual=visual, text=text)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return total_loss / len(loader), acc, f1


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(device)
            visual = batch["visual"].to(device)
            labels = batch["label"].to(device)
            text = batch["text"]

            outputs = model(audio=audio, visual=visual, text=text)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return total_loss / len(loader), acc, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="av")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()

    set_seed(42)

    # 📦 Датасеты
    train_dataset = MoseiDataset(split="train")
    val_dataset = MoseiDataset(split="val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # 🧠 Выбор модели
    if args.model == "av":
        from audio_visual_baseline import AVModel
        model = AVModel()
    elif args.model == "text":
        from text_only_bert import TextModel
        model = TextModel()
    elif args.model == "bottleneck":
        from bottleneck_fusion import BottleneckModel
        model = BottleneckModel()
    else:
        raise ValueError("Unknown model type")

    model.to(device)

    # ⚖️ Loss и optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_f1 = 0

    print(f"\nTraining model: {args.model}")
    print("=" * 50)

    for epoch in range(args.epochs):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion
        )

        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, criterion
        )

        print(f"""
Epoch {epoch+1}/{args.epochs}
Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}
Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}
        """)

        # 💾 сохраняем лучшую модель
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), f"best_{args.model}.pt")
            print("✅ Saved best model!")

    print("\n🔥 Training finished!")


if __name__ == "__main__":
    main()