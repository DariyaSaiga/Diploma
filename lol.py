
import pickle
import numpy as np

with open("preprocess/mosei_emotion_aligned_60.pkl", "rb") as f:
    data = pickle.load(f)

labels = np.array(data["train"]["labels"])
print("Shape:", labels.shape)
print("Примеры с нулями (пустые метки):", np.sum(np.all(labels == 0, axis=1)))
print("Распределение по каждому классу:", labels.sum(axis=0))