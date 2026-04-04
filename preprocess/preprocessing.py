import h5py
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

FILE_PATH = "/Users/dariyaablanova/Desktop/unic_work/Diploma/mosei.hdf5"


SAVE_PATH = "mosei_cleaned.pkl"

# Длина последовательности которую ожидает модель
TARGET_LEN = 50

print("=" * 60)
print("ШАГ 1 — Читаем метки и выбираем 7 классов эмоций")
print("=" * 60)

EMOTION_NAMES = ["happy", "sad", "anger", "surprise", "disgust", "fear"]
EMOTION_IDX = [1, 2, 3, 4, 5, 6]

samples = []  # сюда будем складывать готовые данные

with h5py.File(FILE_PATH, "r") as f:
    all_ids = list(f["All Labels"].keys())
    print(f"Всего сэмплов в датасете: {len(all_ids)}")

    skipped = 0
    for sid in all_ids:
        # Читаем метки для этого сэмпла
        label_raw = np.array(f["All Labels"][sid]["features"]).flatten()
        # Берём только 6 эмоций (без sentiment)
        emotion_vals = label_raw[EMOTION_IDX]  # форма (6,)

        # Если все эмоции = 0 — непонятный сэмпл, пропускаем
        if emotion_vals.max() == 0:
            skipped += 1
            continue

        # Класс = индекс эмоции с максимальным значением
        class_id = int(np.argmax(emotion_vals))

        samples.append({
            "id": sid,
            "label": class_id,
        })

    print(f"Сэмплов после фильтрации: {len(samples)}")
    print(f"Пропущено (все нули):     {skipped}")

# Считаем сколько сэмплов каждого класса
from collections import Counter
label_counts = Counter(s["label"] for s in samples)
print("\nРаспределение классов:")
for class_id, name in enumerate(EMOTION_NAMES):
    print(f"  {class_id} {name}: {label_counts[class_id]}")


print("\n" + "=" * 60)
print("ШАГ 2 — Делим ID на train / val / test")
print("Пропорция: 70% train, 15% val, 15% test")
print("=" * 60)

all_sample_ids = [s["id"] for s in samples]
all_labels     = [s["label"] for s in samples]

# Сначала отделяем test (15%)
train_val_ids, test_ids, train_val_labels, test_labels = train_test_split(
    all_sample_ids, all_labels,
    test_size=0.15,
    random_state=42,
    stratify=all_labels  # чтобы классы были равномерно в каждом сплите
)

# Потом отделяем val от train (15% от всего = ~17.6% от оставшегося)
train_ids, val_ids, train_labels, val_labels = train_test_split(
    train_val_ids, train_val_labels,
    test_size=0.176,
    random_state=42,
    stratify=train_val_labels
)

print(f"Train: {len(train_ids)} сэмплов")
print(f"Val:   {len(val_ids)} сэмплов")
print(f"Test:  {len(test_ids)} сэмплов")

# Превращаем в set для быстрого поиска
train_set = set(train_ids)
val_set   = set(val_ids)
test_set  = set(test_ids)


print("\n" + "=" * 60)
print("ШАГ 3 — Читаем COVAREP (аудио) и делаем padding + scaling")
print(f"Исходная длина: 22, целевая: {TARGET_LEN}")
print("=" * 60)

# --- Функция: дополнить или обрезать до TARGET_LEN ---
def pad_or_truncate(arr, target_len):
    # arr имеет форму (T, features)
    T = arr.shape[0]
    if T >= target_len:
        # Обрезаем если длиннее
        return arr[:target_len]
    else:
        # Дополняем нулями снизу если короче
        pad_rows = target_len - T
        padding = np.zeros((pad_rows, arr.shape[1]), dtype=np.float32)
        return np.vstack([arr, padding])

# Читаем COVAREP для всех сэмплов
print("Читаем аудио данные...")
audio_data = {}
with h5py.File(FILE_PATH, "r") as f:
    for s in samples:
        sid = s["id"]
        arr = np.array(f["COVAREP"][sid]["features"], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)  # заменяем inf на 0
        arr = np.clip(arr, -1000.0, 1000.0)   
        audio_data[sid] = pad_or_truncate(arr, TARGET_LEN)  # -> (50, 74)

print(f"Форма одного аудио сэмпла: {audio_data[samples[0]['id']].shape}")

# Fit StandardScaler только на train
print("Обучаем нормализатор на train...")
train_audio = np.vstack([audio_data[sid] for sid in train_ids])
# train_audio форма: (N_train * 50, 74)

audio_scaler = StandardScaler()
audio_scaler.fit(train_audio)

# Применяем scaler ко всем сэмплам
for sid in audio_data:
    arr = audio_data[sid]                        # (50, 74)
    arr_scaled = audio_scaler.transform(arr)     # (50, 74)
    audio_data[sid] = arr_scaled.astype(np.float32)

print("COVAREP готов. Пример после нормализации:")
example = audio_data[samples[0]["id"]]
print(f"  Форма: {example.shape}")
print(f"  Мин: {example.min():.4f}  Макс: {example.max():.4f}")


print("\n" + "=" * 60)
print("ШАГ 4 — Читаем OpenFace_2 (визуал): clip + padding + scaling")
print("Сначала обрезаем выбросы (±1000), потом нормализуем")
print("=" * 60)

CLIP_VAL = 1000.0  # всё что больше 1000 или меньше -1000 — обрезаем

print("Читаем визуальные данные...")
visual_data = {}
with h5py.File(FILE_PATH, "r") as f:
    for s in samples:
        sid = s["id"]
        arr = np.array(f["OpenFace_2"][sid]["features"], dtype=np.float32)
        # Сначала clip — убираем безумные значения ±33 миллиона
        arr = np.clip(arr, -CLIP_VAL, CLIP_VAL)
        # Потом padding до 50
        visual_data[sid] = pad_or_truncate(arr, TARGET_LEN)  # -> (50, 713)

print(f"Форма одного визуального сэмпла: {visual_data[samples[0]['id']].shape}")
print(f"Значения после clip: мин={np.min(list(visual_data.values())[0]):.2f}, макс={np.max(list(visual_data.values())[0]):.2f}")

# Fit StandardScaler только на train
print("Обучаем нормализатор на train...")
train_visual = np.vstack([visual_data[sid] for sid in train_ids])
# train_visual форма: (N_train * 50, 713)

visual_scaler = StandardScaler()
visual_scaler.fit(train_visual)

# Применяем ко всем
for sid in visual_data:
    arr = visual_data[sid]
    arr_scaled = visual_scaler.transform(arr)
    # Дополнительный clip после scaling — убираем если всё ещё есть выбросы
    arr_scaled = np.clip(arr_scaled, -5.0, 5.0)
    visual_data[sid] = arr_scaled.astype(np.float32)

print("OpenFace_2 готов. Пример после обработки:")
example = visual_data[samples[0]["id"]]
print(f"  Форма: {example.shape}")
print(f"  Мин: {example.min():.4f}  Макс: {example.max():.4f}")


print("\n" + "=" * 60)
print("ШАГ 5 — Читаем words (текст): bytes → строка")
print("BERT запустим отдельно, пока просто декодируем слова")
print("=" * 60)

print("Читаем текстовые данные...")
text_data = {}
with h5py.File(FILE_PATH, "r") as f:
    for s in samples:
        sid = s["id"]
        raw = np.array(f["words"][sid]["features"]).flatten()
        # raw — это массив байтов вида [b'i', b'am', b'happy', b'sp', b'sp']

        words = []
        for token in raw:
            # Декодируем байты в строку
            if isinstance(token, bytes):
                word = token.decode("utf-8", errors="ignore").strip()
            else:
                word = str(token).strip()

            # Убираем служебные токены и пустые строки
            if word and word not in ("sp", "sil", "<unk>", "SP", "SIL"):
                words.append(word)

        # Склеиваем слова в одно предложение
        sentence = " ".join(words)
        text_data[sid] = sentence

print("Примеры декодированных предложений:")
for s in samples[:5]:
    print(f"  {s['id']}: '{text_data[s['id']]}'")


print("\n" + "=" * 60)
print("ШАГ 6 — Собираем всё в один словарь и сохраняем")
print("=" * 60)


def build_split(ids, label_list):
    result = []
    label_dict = {s["id"]: s["label"] for s in samples}
    for sid in ids:
        result.append({
            "id":     sid,
            "audio":  audio_data[sid],    # numpy (50, 74)
            "visual": visual_data[sid],   # numpy (50, 713)
            "text":   text_data[sid],     # строка
            "label":  label_dict[sid],    # число 0-5
        })
    return result

dataset = {
    "train": build_split(train_ids, train_labels),
    "val":   build_split(val_ids,   val_labels),
    "test":  build_split(test_ids,  test_labels),
}

# Сохраняем в файл
with open(SAVE_PATH, "wb") as f:
    pickle.dump(dataset, f)

print(f"Датасет сохранён в: {SAVE_PATH}")
print(f"  train: {len(dataset['train'])} сэмплов")
print(f"  val:   {len(dataset['val'])} сэмплов")
print(f"  test:  {len(dataset['test'])} сэмплов")
print(f"\nПример одного сэмпла из train:")
ex = dataset["train"][0]
print(f"  id:     {ex['id']}")
print(f"  audio:  {ex['audio'].shape}")
print(f"  visual: {ex['visual'].shape}")
print(f"  text:   '{ex['text']}'")
print(f"  label:  {ex['label']} ({EMOTION_NAMES[ex['label']]})")

print("\nГОТОВО!")