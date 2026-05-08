"""
Тестирование DataLoader с BottleneckDataset.
Используем mosei_bottleneck.pkl и динамический паддинг.
"""

import pickle
from torch.utils.data import DataLoader
from bottleneck_dataset import BottleneckDataset, make_collate_fn
from transformers import BertTokenizer

# ──────────────────────────────────────────────────────────────────
# Загружаем данные из pkl
# ──────────────────────────────────────────────────────────────────

print("Загружаю данные...")
with open("mosei_bottleneck.pkl", "rb") as f:
    data = pickle.load(f)

# ──────────────────────────────────────────────────────────────────
# Создаём датасет и collate_fn
# ──────────────────────────────────────────────────────────────────

dataset = BottleneckDataset(data=data, split="train")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
collate_fn = make_collate_fn(tokenizer, max_text_len=128, max_audio_len=100, max_visual_len=100)

# ──────────────────────────────────────────────────────────────────
# Создаём DataLoader с collate_fn
# ──────────────────────────────────────────────────────────────────

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn,  # ✅ ВАЖНО: collate_fn для динамического паддинга
    num_workers=0
)

# ──────────────────────────────────────────────────────────────────
# Берём один батч и выводим информацию
# ──────────────────────────────────────────────────────────────────

batch = next(iter(loader))

print("\n✅ Батч загружен успешно!")
print("\n── Shapes ──")
print(f"  input_ids:      {batch['input_ids'].shape}")        # (4, 128)
print(f"  attention_mask: {batch['attention_mask'].shape}")    # (4, 128)
print(f"  audio:          {batch['audio'].shape}")             # (4, Ta, 74)
print(f"  audio_mask:     {batch['audio_mask'].shape}")        # (4, Ta)
print(f"  audio_len:      {batch['audio_len'].tolist()}")      # [15, 20, 18, 25]
print(f"  visual:         {batch['visual'].shape}")            # (4, Tv, 713)
print(f"  visual_mask:    {batch['visual_mask'].shape}")       # (4, Tv)
print(f"  visual_len:     {batch['visual_len'].tolist()}")     # [12, 22, 19, 28]
print(f"  label:          {batch['label'].tolist()}")          # [0, 2, 1, 3]

print("\n── Проверка масок ──")
for i in range(len(batch["label"])):
    al = batch["audio_len"][i].item()
    vl = batch["visual_len"][i].item()

    # Проверяем что маски совпадают с длинами
    assert batch["audio_mask"][i, :al].all(), f"audio_mask не совпадает с audio_len [{i}]"
    assert not batch["audio_mask"][i, al:].any(), f"audio_mask заходит в padding [{i}]"

    assert batch["visual_mask"][i, :vl].all(), f"visual_mask не совпадает с visual_len [{i}]"
    assert not batch["visual_mask"][i, vl:].any(), f"visual_mask заходит в padding [{i}]"

print("✅ Все проверки пройдены!")