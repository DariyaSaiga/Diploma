"""
prepare_bert_dataset.py
=======================
Скрипт для загрузки CMU-MOSEI через mmsdk и создания .pkl с BERT эмбеддингами.

Установка зависимостей:
    pip install mmsdk transformers torch tqdm

Запуск:
    python prepare_bert_dataset.py
"""

import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

# =============================================================================
# 1. ЗАГРУЗКА CMU-MOSEI ЧЕРЕЗ MMSDK
# =============================================================================
# mmsdk автоматически скачает датасет при первом запуске (~несколько ГБ)
# Данные сохраняются в папку ./data/

try:
    import mmsdk
    from mmsdk import mmdatasdk as md
except ImportError:
    raise ImportError(
        "Установи mmsdk:\n  pip install mmsdk\n"
        "Или вручную: pip install git+https://github.com/A2Zadeh/CMU-MultimodalSDK.git"
    )

DATA_DIR = "./data/cmu_mosei"
OUTPUT_PKL = "./mosei_bert_aligned.pkl"
BERT_MODEL_NAME = "bert-base-uncased"
MAX_TEXT_LEN = 64       # максимальная длина текста для BERT (в токенах)
MAX_SEQ_LEN = 60        # длина временной оси для audio/video (padding/truncation)
BATCH_SIZE = 32         # батч для BERT inference

os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 60)
print("Шаг 1: Загрузка CMU-MOSEI через mmsdk")
print("=" * 60)

# Рецепты содержат URL для скачивания каждой модальности
MOSEI_HIGHLEVEL = md.cmu_mosei.highlevel  # audio + video features
MOSEI_RAW = md.cmu_mosei.raw             # raw text (слова)
MOSEI_LABELS = md.cmu_mosei.labels       # метки эмоций
print(MOSEI_HIGHLEVEL.keys())
# Скачиваем модальности (если уже скачаны — пропускает)
dataset = md.mmdataset(
    recipe={
        "CMU_MOSEI_TimestampedWords": MOSEI_RAW["words"],
        "CMU_MOSEI_COVAREP": MOSEI_HIGHLEVEL["COVAREP"],           # audio (74-dim)
        "CMU_MOSEI_OpenFace2.0": MOSEI_HIGHLEVEL["OpenFace2.0"],     # video (35-dim)
        "CMU_MOSEI_Labels": MOSEI_LABELS["Sentiment Labels"],       # labels
    },
    destination=DATA_DIR,
)

# Выравниваем все модальности по тексту (слова — опорная ось)
print("\nВыравниваем модальности по тексту...")
dataset.align("CMU_MOSEI_TimestampedWords", collapse_functions=[md.avg])

# Разбиваем на train/valid/test по официальным id
print("Разбиваем на splits...")
train_split, valid_split, test_split = dataset.get_splits(
    md.cmu_mosei.standard_folds.standard_train_fold,
    md.cmu_mosei.standard_folds.standard_valid_fold,
    md.cmu_mosei.standard_folds.standard_test_fold,
)

print(f"Train: {len(train_split)} | Valid: {len(valid_split)} | Test: {len(test_split)}")

# =============================================================================
# 2. ИЗВЛЕЧЕНИЕ ДАННЫХ ИЗ mmsdk
# =============================================================================

def extract_modality(split, key, max_seq=MAX_SEQ_LEN):
    """
    Извлекает числовую модальность (audio/video) из split.
    Возвращает список массивов shape (max_seq, feat_dim).
    """
    result = []
    for vid in split.keys():
        for seg in split[vid][key]["intervals"]:
            raw = split[vid][key]["features"]   # (T, feat_dim)
            T, D = raw.shape

            # Padding / Truncation до max_seq
            if T >= max_seq:
                seq = raw[:max_seq]
            else:
                pad = np.zeros((max_seq - T, D), dtype=np.float32)
                seq = np.concatenate([raw, pad], axis=0)

            result.append(seq.astype(np.float32))
            break   # одна запись на сегмент
    return result


def extract_words(split):
    """
    Извлекает сырые слова из split и собирает предложения.
    Возвращает список строк.
    """
    sentences = []
    for vid in split.keys():
        for seg_id in split[vid]["CMU_MOSEI_TimestampedWords"]["intervals"]:
            words = split[vid]["CMU_MOSEI_TimestampedWords"]["features"]
            # features — массив слов shape (T, 1), каждое слово — строка
            sentence = " ".join(
                [w[0].decode("utf-8") if isinstance(w[0], bytes) else str(w[0])
                 for w in words if w[0] not in ["", b"", "sp", b"sp"]]
            )
            sentences.append(sentence.strip())
            break
    return sentences


def extract_labels(split, num_classes=6):
    """
    Извлекает метки эмоций (one-hot) из split.
    CMU-MOSEI: [happiness, sadness, anger, fear, disgust, surprise]
    """
    labels = []
    for vid in split.keys():
        raw = split[vid]["CMU_MOSEI_Labels"]["features"]  # (1, 7) — последний — sentiment
        emotions = raw[0, :6]  # берём первые 6 эмоций, отбрасываем sentiment

        # Преобразуем в one-hot по argmax (или можно оставить мультилейбл)
        one_hot = np.zeros(num_classes, dtype=np.float32)
        idx = int(np.argmax(emotions))
        one_hot[idx] = 1.0
        labels.append(one_hot)
    return labels


print("\nИзвлекаем модальности...")
splits_raw = {
    "train": train_split,
    "valid": valid_split,
    "test":  test_split,
}

extracted = {}
for split_name, split_data in splits_raw.items():
    print(f"\n  Обработка {split_name}...")
    extracted[split_name] = {
        "audio":  extract_modality(split_data, "CMU_MOSEI_COVAREP"),
        "vision": extract_modality(split_data, "CMU_MOSEI_OpenFace_2"),
        "labels": extract_labels(split_data),
        "raw_text": extract_words(split_data),
    }
    print(f"    Размер: {len(extracted[split_name]['labels'])} примеров")
    print(f"    Пример текста: '{extracted[split_name]['raw_text'][0]}'")

# =============================================================================
# 3. BERT ЭМБЕДДИНГИ
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*60}")
print(f"Шаг 2: Извлечение BERT эмбеддингов (device: {device})")
print("=" * 60)

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
bert = BertModel.from_pretrained(BERT_MODEL_NAME).to(device)
bert.eval()

print(f"Модель: {BERT_MODEL_NAME} | Max токенов: {MAX_TEXT_LEN}")


def get_bert_embeddings(sentences, tokenizer, bert_model, device,
                        max_len=MAX_TEXT_LEN, batch_size=BATCH_SIZE):
    """
    Прогоняет список предложений через BERT батчами.
    Возвращает список тензоров shape (max_len, 768).
    """
    all_embeddings = []

    for i in tqdm(range(0, len(sentences), batch_size), desc="BERT inference"):
        batch = sentences[i : i + batch_size]

        encoded = tokenizer(
            batch,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output = bert_model(**encoded)

        # last_hidden_state: (batch, max_len, 768)
        hidden = output.last_hidden_state.cpu().numpy()  # (B, max_len, 768)

        for j in range(len(batch)):
            all_embeddings.append(hidden[j].astype(np.float32))  # (max_len, 768)

    return all_embeddings


for split_name in ["train", "valid", "test"]:
    print(f"\nОбработка {split_name}...")
    sentences = extracted[split_name]["raw_text"]
    bert_embeds = get_bert_embeddings(sentences, tokenizer, bert, device)
    extracted[split_name]["text"] = bert_embeds
    del extracted[split_name]["raw_text"]   # сырой текст больше не нужен

    print(f"  text shape пример:  {bert_embeds[0].shape}")     # (64, 768)
    print(f"  audio shape пример: {extracted[split_name]['audio'][0].shape}")
    print(f"  video shape пример: {extracted[split_name]['vision'][0].shape}")

# =============================================================================
# 4. СОХРАНЕНИЕ В PKL
# =============================================================================

print(f"\n{'='*60}")
print(f"Шаг 3: Сохранение в {OUTPUT_PKL}")
print("=" * 60)

with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(extracted, f, protocol=4)

# Проверка
size_mb = os.path.getsize(OUTPUT_PKL) / 1024 / 1024
print(f"Готово! Файл: {OUTPUT_PKL} ({size_mb:.1f} MB)")

print("\nСтруктура датасета:")
for split_name in ["train", "valid", "test"]:
    n = len(extracted[split_name]["labels"])
    print(f"  {split_name}: {n} примеров")
    for key, val in extracted[split_name].items():
        shape = np.array(val[0]).shape
        print(f"    {key}: {n} x {shape}")

print("\nДатасет готов к использованию!")
print("В train.py измени:")
print('  with open("mosei_bert_aligned.pkl", "rb") as f: ...')
print("  text_dim = 768")