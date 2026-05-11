import pickle
import numpy as np

EMOTION_NAMES = ["Happy", "Sad", "Anger", "Surprise", "Disgust", "Fear"]

paths = {
    "GloVe (aligned_60)": "/Users/dariyaablanova/Desktop/unic_work/Diploma/Diploma_clone/Diploma/mosei_emotion_aligned_60.pkl",
    "aligned_50":         "/Users/dariyaablanova/Desktop/unic_work/Diploma/Diploma_clone/Diploma/datasets/aligned_50.pkl",
}

for name, path in paths.items():
    print(f"\n{'='*60}")
    print(f"ДАТАСЕТ: {name}")
    print('='*60)

    with open(path, 'rb') as f:
        data = pickle.load(f)

    print("\n--- Структура ---")
    for split in ['train', 'valid', 'test']:
        print(f"  {split}:")
        for key, val in data[split].items():
            try:
                arr = np.array(val)
                print(f"    {key}: {arr.shape}, dtype={arr.dtype}")
            except:
                print(f"    {key}: {type(val)}, len={len(val)}")

    # Лейблы
    if 'labels' in data['train']:
        labels = np.array(data['train']['labels'])
        total = len(labels)
        print(f"\n--- Дисбаланс классов (train) ---")
        for i, emo in enumerate(EMOTION_NAMES):
            pos = int(labels[:, i].sum())
            neg = total - pos
            pct = pos / total * 100
            ratio = neg / (pos + 1e-6)
            bar = '█' * int(pct / 2)
            print(f"  {emo:<10}: {pos:5d} ({pct:5.1f}%)  ratio={ratio:.1f}:1  {bar}")

        print(f"\n--- Мультилейбловость ---")
        emotions_per_sample = labels.sum(axis=1)
        for n in range(5):
            count = (emotions_per_sample == n).sum()
            print(f"  {n} эмоций: {count} ({count/total*100:.1f}%)")

        print(f"\n--- Fear сэмплы ---")
        fear_mask = labels[:, 5] == 1
        fear_labels = labels[fear_mask]
        print(f"  Всего Fear: {fear_mask.sum()}")
        print(f"  Fear alone:      {(fear_labels.sum(axis=1)==1).sum()}")
        print(f"  Fear + Happy:    {(fear_labels[:,0]==1).sum()}")
        print(f"  Fear + Sad:      {(fear_labels[:,1]==1).sum()}")
        print(f"  Fear + Anger:    {(fear_labels[:,2]==1).sum()}")
        print(f"  Fear + Surprise: {(fear_labels[:,3]==1).sum()}")
        print(f"  Fear + Disgust:  {(fear_labels[:,4]==1).sum()}")

    elif 'regression_labels' in data['train']:
        print("\n--- Sentiment датасет (нет emotion labels) ---")
        reg = np.array(data['train']['regression_labels'])
        print(f"  regression_labels: min={reg.min():.2f}, max={reg.max():.2f}, mean={reg.mean():.2f}")

    del data
    print("\n✓ Готово")