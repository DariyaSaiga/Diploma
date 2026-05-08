# 🗑️ Анализ бесполезных файлов

Статус: Проект содержит несколько файлов, которые можно удалить для очистки.

---

## 📊 Файлы для удаления

### 🔴 КАТЕГОРИЯ 1: Файлы только для экспериментов (НЕ используются в основном коде)

#### 1. `text_only_bert.py` (4 KB)
- **Использование:** Импортируется в train.py как опция `--model text`
- **Назначение:** Текстовая модель (только BERT, без аудио/видео)
- **Нужен ли?** ⚠️ Опционально — если вы не сравниваете с baseline
- **Статус:** МОЖНО УДАЛИТЬ если не нужны эксперименты с текстом

---

#### 2. `dataset.py` (4 KB)
- **Использование:** ❌ НЕ используется! Заменён на `bottleneck_dataset.py`
- **Назначение:** Старая реализация DataLoader (с фиксированным padding)
- **Почему заменён?** `bottleneck_dataset.py` имеет:
  - ✅ Динамический паддинг (лучше для памяти)
  - ✅ Bool маски для attention
  - ✅ Сохранение реальной длины (audio_len, visual_len)
- **Статус:** ✅ БЕЗОПАСНО УДАЛИТЬ

---

#### 3. `compare_preprocess.py` (4 KB)
- **Использование:** ❌ НЕ используется
- **Назначение:** Создаёт красивую картинку с информацией о препроцессинге
- **Содержит:** matplotlib код для визуализации (не функционал)
- **Статус:** ✅ БЕЗОПАСНО УДАЛИТЬ (есть README.md)

---

### 🟡 КАТЕГОРИЯ 2: Папки для экспериментов

#### 4. `preprocess/` (16 MB)
```
preprocess/
├── preprocessing.py       — создаёт mosei_cleaned.pkl (старый подход)
├── preprocess_bottleneck.py  — создаёт mosei_bottleneck.pkl (нужно!)
├── inspect_dataset.py     — инспекция данных (опционально)
├── check.py              — проверка структуры pkl (опционально)
├── result.txt, result2.txt, result_prep.txt  — логи препроцессинга
```

**Что можно удалить:**
- ✅ `preprocess/result*.txt` — это логи, их уже не нужно
- ⚠️ `preprocess/preprocessing.py` — создаёт mosei_cleaned.pkl (старый способ)
- ⚠️ `preprocess/check.py` и `preprocess/inspect_dataset.py` — опционально

**Что НУЖНО оставить:**
- ✅ `preprocess/preprocess_bottleneck.py` — создаёт нужный датасет!

---

#### 5. `experiments/` (64 KB)
```
experiments/
├── compare_results.py     — сравнивает результаты эксперимента
├── compare_baselines.py   — сравнивает базовые модели
```

**Статус:**
- ⚠️ Опционально — используются только если сравниваете результаты разных экспериментов
- ✅ МОЖНО УДАЛИТЬ если не нужны статистики экспериментов

---

### 🟠 КАТЕГОРИЯ 3: Большие файлы данных

#### 6. `mosei_cleaned.pkl` (3.0 GB) ⚠️ БОЛЬШОЙ!
- **Использование:** ❌ НЕ рекомендуется (замена: mosei_bottleneck.pkl)
- **Проблема:** 
  - Фиксированный padding (теряется информация)
  - Нет масок для attention
  - Занимает 3 GB!
- **Статус:** ✅ МОЖНО УДАЛИТЬ (используйте mosei_bottleneck.pkl)

---

#### 7. `after_preprocessing.png` (176 KB)
- **Использование:** ❌ НЕ используется (только для документации)
- **Назначение:** Скриншот/диаграмма из compare_preprocess.py
- **Статус:** ✅ МОЖНО УДАЛИТЬ

---

#### 8. `__pycache__/` (80 KB)
- **Использование:** Автоматические compiled файлы Python
- **Статус:** ✅ ВСЕГДА БЕЗОПАСНО УДАЛИТЬ (пересоздаётся автоматически)
- **Команда:** `rm -rf __pycache__/`

---

### 🟢 КАТЕГОРИЯ 4: Файлы, которые НУЖНО ОСТАВИТЬ

| Файл | Размер | Статус | Причина |
|---|---|---|---|
| `bottleneck_dataset.py` | 16 KB | ✅ НУЖЕН | Загрузка данных с масками |
| `train.py` | 24 KB | ✅ НУЖЕН | Основной скрипт обучения |
| `check_forward.py` | 8 KB | ✅ НУЖЕН | Проверка модели перед обучением |
| `test_loader.py` | 8 KB | ✅ НУЖЕН | Проверка загрузки данных |
| `utils.py` | 4 KB | ✅ НУЖЕН | Helper функции (device, seed) |
| `simple_fusion.py` | 4 KB | ✅ НУЖЕН | Model для `--model simple_fusion` |
| `audio_visual_baseline.py` | 4 KB | ✅ НУЖЕН | Model для `--model av` |
| `text_only_bert.py` | 4 KB | ⚠️ ОПЦИОНАЛЬНО | Model для `--model text` |
| `dra_loss.py` | 8 KB | ✅ НУЖЕН | DRA loss для обучения |
| `requirements.txt` | 4 KB | ✅ НУЖЕН | Зависимости |
| `preprocess/preprocess_bottleneck.py` | - | ✅ НУЖЕН | Создание датасета |
| `mosei_bottleneck.pkl` | 1.6 GB | ✅ НУЖЕН | Основной датасет |

---

## 🧹 Рекомендуемая очистка

### Минимальная очистка (освобождает ~3.2 GB):
```bash
# Удалить большой ненужный датасет
rm mosei_cleaned.pkl

# Удалить кэш Python
rm -rf __pycache__/

# Удалить логи препроцессинга
rm preprocess/result*.txt
```

**Освобождено: ~3.0 GB** ✅

---

### Полная очистка (освобождает ~3.3 GB, удаляет "лишнее"):
```bash
# Большие файлы
rm mosei_cleaned.pkl              # 3.0 GB
rm after_preprocessing.png        # 176 KB
rm -rf __pycache__/              # 80 KB

# Опциональные файлы (если не нужны эксперименты)
rm -rf experiments/               # 64 KB
rm compare_preprocess.py          # 4 KB
rm text_only_bert.py              # 4 KB (если не используете --model text)
rm dataset.py                     # 4 KB (уже замещён на bottleneck_dataset.py)

# Препроцессинг (если данные уже готовы)
rm preprocess/preprocessing.py    # старый способ создания mosei_cleaned.pkl
rm preprocess/check.py            # опционально
rm preprocess/inspect_dataset.py  # опционально
rm preprocess/result*.txt         # логи
```

**Освобождено: ~3.3 GB** ✅

---

## 📋 Итоговый чек-лист

### ✅ ОСТАВИТЬ (обязательно):
- [ ] `train.py`
- [ ] `bottleneck_dataset.py`
- [ ] `check_forward.py`
- [ ] `test_loader.py`
- [ ] `utils.py`
- [ ] `dra_loss.py`
- [ ] `simple_fusion.py`
- [ ] `audio_visual_baseline.py`
- [ ] `requirements.txt`
- [ ] `mosei_bottleneck.pkl`
- [ ] `preprocess/preprocess_bottleneck.py` (для пересоздания данных)

### ⚠️ ОПЦИОНАЛЬНО (если не используете):
- [ ] `text_only_bert.py` — если не нужно `--model text`
- [ ] `experiments/` — если не сравниваете результаты
- [ ] `preprocess/preprocessing.py` — если не используете mosei_cleaned.pkl

### ✅ УДАЛИТЬ (безопасно):
- [ ] `mosei_cleaned.pkl` (3.0 GB!) 🎯 ГЛАВНЫЙ КАНДИДАТ
- [ ] `after_preprocessing.png`
- [ ] `compare_preprocess.py`
- [ ] `dataset.py` (замещён на bottleneck_dataset.py)
- [ ] `preprocess/result*.txt` (логи)
- [ ] `__pycache__/`

---

## 🎯 Финальная рекомендация

### Минимум для работы:
```
./
├── train.py                    ✅ Обучение
├── check_forward.py            ✅ Проверка
├── test_loader.py              ✅ Проверка данных
├── bottleneck_dataset.py       ✅ Датасет
├── simple_fusion.py            ✅ Модель
├── audio_visual_baseline.py    ✅ Модель
├── dra_loss.py                 ✅ Loss
├── utils.py                    ✅ Helpers
├── requirements.txt            ✅ Зависимости
├── preprocess/
│   └── preprocess_bottleneck.py ✅ Создание датасета
└── mosei_bottleneck.pkl        ✅ Данные (1.6 GB)

Всего: ~1.75 GB ✅
Читаемость: ⭐⭐⭐⭐⭐
```

### НЕ нужны (удалить):
```
mosei_cleaned.pkl               ❌ 3.0 GB! (замена: mosei_bottleneck.pkl)
dataset.py                      ❌ Замещён на bottleneck_dataset.py
compare_preprocess.py           ❌ Визуализация
after_preprocessing.png         ❌ Скриншот
experiments/                    ❌ Если не сравниваете
preprocess/preprocessing.py     ❌ Старый способ
preprocess/result*.txt          ❌ Логи
__pycache__/                    ❌ Кэш
```

---

## 🚀 Команда для очистки

```bash
# Быстрая очистка (минимум)
rm mosei_cleaned.pkl
rm -rf __pycache__/
rm preprocess/result*.txt

# Полная очистка
rm mosei_cleaned.pkl after_preprocessing.png
rm -rf __pycache__/ experiments/
rm preprocess/{preprocessing.py,check.py,inspect_dataset.py,result*.txt}
rm dataset.py compare_preprocess.py
# Опционально:
# rm text_only_bert.py (если не нужен --model text)
```

**После очистки:** из 4.8 GB → 1.75 GB ✅ (экономия 3.05 GB!)
