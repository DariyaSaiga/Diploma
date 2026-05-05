"""
preprocess_bottleneck.py
========================
Создаёт mosei_bottleneck.pkl для Multimodal Bottleneck Transformer.

Запуск:
    python3 preprocess/preprocess_bottleneck.py

Если HDF5 лежит в другом месте — поменяй HDF5_PATH ниже.
"""

import pickle
with open("scaler_audio.pkl", "wb") as f:
    pickle.dump(audio_scaler, f)
with open("scaler_visual.pkl", "wb") as f:
    pickle.dump(visual_scaler, f)
from collections import Counter

import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# 👇 Поменяй путь если нужно
# ─────────────────────────────────────────────────────────────────────────────

HDF5_PATH = "/Users/dariyaablanova/Desktop/unic_work/Diploma/mosei.hdf5"
OUT_PATH  = "mosei_bottleneck.pkl"

# ─────────────────────────────────────────────────────────────────────────────

EMOTION_NAMES = ["happy", "sad", "anger", "surprise", "disgust", "fear"]
EMOTION_IDX   = [1, 2, 3, 4, 5, 6]

CLIP_AUDIO              = 1000.0
CLIP_VISUAL             = 1000.0
CLIP_VISUAL_AFTER_SCALE = 5.0


# ─────────────────────────────────────────────────────────────────────────────
# ШАГ 1 — Метки и split
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("ШАГ 1 — Читаем метки, фильтруем, делаем split")
print("=" * 60)

samples = []
with h5py.File(HDF5_PATH, "r") as f:
    all_ids = list(f["All Labels"].keys())
    print(f"Всего сэмплов: {len(all_ids)}")

    skipped = 0
    for sid in all_ids:
        label_raw    = np.array(f["All Labels"][sid]["features"]).flatten()
        emotion_vals = label_raw[EMOTION_IDX]
        if emotion_vals.max() == 0:
            skipped += 1
            continue
        samples.append({"id": sid, "label": int(np.argmax(emotion_vals))})

print(f"После фильтрации: {len(samples)}  (пропущено: {skipped})")

counts = Counter(s["label"] for s in samples)
for i, name in enumerate(EMOTION_NAMES):
    print(f"  {i} {name}: {counts[i]}")

# Тот же split что у baseline
all_ids_list    = [s["id"]    for s in samples]
all_labels_list = [s["label"] for s in samples]

train_val_ids, test_ids, train_val_labels, _ = train_test_split(
    all_ids_list, all_labels_list,
    test_size=0.15, random_state=42, stratify=all_labels_list,
)
train_ids, val_ids, train_labels, _ = train_test_split(
    train_val_ids, train_val_labels,
    test_size=0.176, random_state=42, stratify=train_val_labels,
)
print(f"\nSplit: train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")


# ─────────────────────────────────────────────────────────────────────────────
# ШАГ 2 — Аудио (COVAREP) без padding
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("ШАГ 2 — Читаем COVAREP (аудио) без padding")
print("=" * 60)

raw_audio = {}
with h5py.File(HDF5_PATH, "r") as f:
    for s in samples:
        sid = s["id"]
        arr = np.array(f["COVAREP"][sid]["features"], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, -CLIP_AUDIO, CLIP_AUDIO)
        raw_audio[sid] = arr   # (Ta, 74) — реальная длина

audio_lens = [raw_audio[s["id"]].shape[0] for s in samples]
print(f"Длины: min={min(audio_lens)}  max={max(audio_lens)}  mean={np.mean(audio_lens):.1f}")

print("StandardScaler на train...")
audio_scaler = StandardScaler()
audio_scaler.fit(np.vstack([raw_audio[sid] for sid in train_ids]))

audio_data = {}
for s in samples:
    sid = s["id"]
    arr = audio_scaler.transform(raw_audio[sid]).astype(np.float32)
    audio_data[sid] = (arr, arr.shape[0])

print("Готово.")


# ─────────────────────────────────────────────────────────────────────────────
# ШАГ 3 — Визуал (OpenFace_2) без padding
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("ШАГ 3 — Читаем OpenFace_2 (визуал) без padding")
print("=" * 60)

raw_visual = {}
with h5py.File(HDF5_PATH, "r") as f:
    for s in samples:
        sid = s["id"]
        arr = np.array(f["OpenFace_2"][sid]["features"], dtype=np.float32)
        arr = np.clip(arr, -CLIP_VISUAL, CLIP_VISUAL)
        raw_visual[sid] = arr   # (Tv, 713)

visual_lens = [raw_visual[s["id"]].shape[0] for s in samples]
print(f"Длины: min={min(visual_lens)}  max={max(visual_lens)}  mean={np.mean(visual_lens):.1f}")

print("StandardScaler на train...")
visual_scaler = StandardScaler()
visual_scaler.fit(np.vstack([raw_visual[sid] for sid in train_ids]))

visual_data = {}
for s in samples:
    sid = s["id"]
    arr = visual_scaler.transform(raw_visual[sid])
    arr = np.clip(arr, -CLIP_VISUAL_AFTER_SCALE, CLIP_VISUAL_AFTER_SCALE).astype(np.float32)
    visual_data[sid] = (arr, arr.shape[0])

print("Готово.")


# ─────────────────────────────────────────────────────────────────────────────
# ШАГ 4 — Текст (words → строка)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("ШАГ 4 — Читаем words (bytes → строка)")
print("=" * 60)

text_data = {}
with h5py.File(HDF5_PATH, "r") as f:
    for s in samples:
        sid = s["id"]
        raw = np.array(f["words"][sid]["features"]).flatten()
        words = []
        for token in raw:
            word = (token.decode("utf-8", errors="ignore").strip()
                    if isinstance(token, bytes) else str(token).strip())
            if word and word not in ("sp", "sil", "<unk>", "SP", "SIL"):
                words.append(word)
        text_data[sid] = " ".join(words)

for s in samples[:3]:
    print(f"  {s['id']}: '{text_data[s['id']][:70]}'")


# ─────────────────────────────────────────────────────────────────────────────
# ШАГ 5 — Собираем и сохраняем
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("ШАГ 5 — Собираем и сохраняем")
print("=" * 60)

label_by_id = {s["id"]: s["label"] for s in samples}

def make_split(ids):
    result = []
    for sid in ids:
        audio_arr,  audio_len  = audio_data[sid]
        visual_arr, visual_len = visual_data[sid]
        result.append({
            "id":         sid,
            "text":       text_data[sid],
            "audio":      audio_arr,      # (Ta, 74)
            "audio_len":  audio_len,
            "visual":     visual_arr,     # (Tv, 713)
            "visual_len": visual_len,
            "label":      label_by_id[sid],
        })
    return result

dataset = {
    "train": make_split(train_ids),
    "val":   make_split(val_ids),
    "test":  make_split(test_ids),
}

with open(OUT_PATH, "wb") as f:
    pickle.dump(dataset, f)

print(f"\nСохранено → {OUT_PATH}")
print(f"  train: {len(dataset['train'])}  val: {len(dataset['val'])}  test: {len(dataset['test'])}")

ex = dataset["train"][0]
print(f"\nПример:")
print(f"  text:       '{ex['text'][:60]}'")
print(f"  audio:      {ex['audio'].shape}  len={ex['audio_len']}")
print(f"  visual:     {ex['visual'].shape}  len={ex['visual_len']}")
print(f"  label:      {ex['label']} ({EMOTION_NAMES[ex['label']]})")
print("\nГОТОВО!")
