import pickle
import numpy as np

path = "/Users/dariyaablanova/Desktop/unic_work/Diploma/Diploma_clone/Diploma/datasets/mosei_combined.pkl"

with open(path, 'rb') as f:
    data = pickle.load(f)

train = data['train']

# 1. Базовая структура
print("=== СТРУКТУРА ===")
for key, val in train.items():
    arr = np.array(val)
    print(f"  {key}: {arr.shape}, dtype={arr.dtype}")

# 2. Проверка text (BERT) фич
print("\n=== BERT TEXT ФИЧИ ===")
text = np.array(train['text'])
print(f"  Shape: {text.shape}")
print(f"  Min: {text.min():.4f}")
print(f"  Max: {text.max():.4f}")
print(f"  Mean: {text.mean():.4f}")
print(f"  Std: {text.std():.4f}")
print(f"  NaN count: {np.isnan(text).sum()}")
print(f"  Zero rows (padding): {(text.sum(axis=2) == 0).sum()}")

# 3. Проверка audio фич
print("\n=== AUDIO ФИЧИ ===")
audio = np.array(train['audio'])
print(f"  Shape: {audio.shape}")
print(f"  Min: {audio.min():.4f}")
print(f"  Max: {audio.max():.4f}")
print(f"  NaN count: {np.isnan(audio).sum()}")

# 4. Проверка vision фич
print("\n=== VISION ФИЧИ ===")
vision = np.array(train['vision'])
print(f"  Shape: {vision.shape}")
print(f"  Min: {vision.min():.4f}")
print(f"  Max: {vision.max():.4f}")
print(f"  NaN count: {np.isnan(vision).sum()}")

# 5. Дисбаланс классов
print("\n=== ДИСБАЛАНС КЛАССОВ ===")
labels = np.array(train['labels'])
EMOTION_NAMES = ["Happy", "Sad", "Anger", "Surprise", "Disgust", "Fear"]
total = len(labels)
for i, name in enumerate(EMOTION_NAMES):
    count = int(labels[:, i].sum())
    pct = count / total * 100
    bar = '█' * int(pct / 2)
    print(f"  [{i}] {name:<10}: {count:5d} ({pct:5.1f}%)  {bar}")

# 6. BERT vs GloVe — информативность
print("\n=== BERT vs GloVe СРАВНЕНИЕ ===")
print(f"  GloVe размерность: 300")
print(f"  BERT размерность:  768  (+156% больше информации)")
print(f"  GloVe: статичные векторы без контекста")
print(f"  BERT:  контекстные векторы, понимает смысл предложения")

# 7. Проверка нулевых строк в BERT (padding)
print("\n=== PADDING В BERT ===")
text = np.array(train['text'])
zero_mask = (text.sum(axis=2) == 0)
zeros_per_sample = zero_mask.sum(axis=1)
print(f"  Среднее нулевых токенов на сэмпл: {zeros_per_sample.mean():.1f} из 50")
print(f"  Макс нулевых токенов: {zeros_per_sample.max()}")
print(f"  Мин нулевых токенов: {zeros_per_sample.min()}")
print(f"  Сэмплов без padding: {(zeros_per_sample == 0).sum()}")