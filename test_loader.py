from torch.utils.data import DataLoader
from dataset import MoseiDataset

# создаём датасет
dataset = MoseiDataset(split="train")

# создаём loader
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# берём один batch
batch = next(iter(loader))

print("TEXT:", batch["text"])
print("AUDIO SHAPE:", batch["audio"].shape)
print("VISUAL SHAPE:", batch["visual"].shape)
print("LABEL:", batch["label"])