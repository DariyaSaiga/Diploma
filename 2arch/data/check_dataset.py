import pickle
import numpy as np

path = "/Users/dariyaablanova/Desktop/unic_work/Diploma/Diploma_clone/Diploma/datasets/mosei_emotion_aligned_60.pkl"

with open(path, 'rb') as f:
    data = pickle.load(f)

labels = data['train']['labels']

print("=== СОПОСТАВЛЯЕМ С ОРИГИНАЛЬНЫМ CMU-MOSEI ===")
print()
print("Оригинальный порядок в CMU-MOSEI (из документации):")
print("  [happy, sad, anger, surprise, disgust, fear]")
print()

candidate_order = ['Happy', 'Sad', 'Anger', 'Surprise', 'Disgust', 'Fear']

for i, name in enumerate(candidate_order):
    count = int(labels[:, i].sum())
    pct = count / len(labels) * 100
    bar = '█' * int(pct / 2)
    print(f"  [{i}] {name:<10}: {count:5d} ({pct:5.1f}%)  {bar}")

print()
print("=== СРАВНЕНИЕ С ЛИТЕРАТУРОЙ (fnins статья, таблица 1) ===")
print("Из статьи MER-SEM-MBT (train split):")
print("  Happy:    7587")
print("  Anger:    3267")  
print("  Disgust:  2738")
print("  Surprise: 1465")
print("  Sadness:  4026")
print("  Fear:     1263")
print()
print("Наши данные по индексам:")
for i in range(6):
    count = int(labels[:, i].sum())
    print(f"  [{i}]: {count}")

print()
print("=== ВЫВОД ===")
counts = [int(labels[:, i].sum()) for i in range(6)]
sorted_idx = np.argsort(counts)[::-1]
print(f"По убыванию: {sorted_idx.tolist()}")
print(f"Значения:    {[counts[i] for i in sorted_idx]}")
print()
print("Из статьи по убыванию: Happy(7587) > Sad(4026) > Anger(3267) > Disgust(2738) > Surprise(1465) > Fear(1263)")
print("Наши по убыванию:      ", sorted([counts[i] for i in range(6)], reverse=True))