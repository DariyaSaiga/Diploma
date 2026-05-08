# 🎯 Стратегия обучения для достижения 80%+ Accuracy

## ❓ ГЛАВНЫЙ ВОПРОС

**Нужно ли обучать каждую модель отдельно, потом соединять? Или только Bottleneck?**

→ **Ответ:** Правильная стратегия - **Progressive Fine-tuning** (постепенная оптимизация)

---

## 🎬 ПРАВИЛЬНАЯ АРХИТЕКТУРА ОБУЧЕНИЯ

```
ВАРИАНТ 1: ❌ НЕПРАВИЛЬНО (обучать модали отдельно)
═══════════════════════════════════════════════════════════

Text model обучается отдельно → SavedModel_Text
  ↓
Audio model обучается отдельно → SavedModel_Audio
  ↓
Visual model обучается отдельно → SavedModel_Visual
  ↓
Bottleneck Fusion обучается с замороженными весами
  ❌ ПРОБЛЕМА: Модали не знают о fusion, обучаются друг от друга независимо
  ❌ РЕЗУЛЬТАТ: Плохая интеграция информации, accuracy ~70%


ВАРИАНТ 2: ✅ ПРАВИЛЬНО (End-to-End + Progressive tuning)
═══════════════════════════════════════════════════════════

Шаг 1: Обучить ВСЕ ВМЕСТЕ (Bottleneck + все модали) с замороженным BERT
        → 20 эпох, lr=1e-3

Шаг 2: Сохранить веса, потом обучить еще 20 эпох с ДРУГИМИ параметрами
        → 20 эпох, lr=3e-4 (пониженный)
        → Начинается с сохраненных весов из Шага 1

Шаг 3: Разморозить BERT (partial unfreeze) и fine-tune еще 20 эпох
        → 20 эпох, lr_bert=5e-5, lr_other=5e-4
        → Начинается с весов из Шага 2

Шаг 4: Fine-tune на полных параметрах с регуляризацией
        → 10-15 эпох, lr_bert=1e-5, lr_other=1e-4
        → Label smoothing, dropout увеличить

✅ РЕЗУЛЬТАТ: Модали обучаются вместе, лучше интеграция, accuracy ~80%+
```

---

## 📊 ПОЧЕМУ ВСЕ ВМЕСТЕ, А НЕ ОТДЕЛЬНО?

### ❌ Если обучать отдельно:

```
Text обучается
  ↓ (не видит Audio/Visual)
Audio обучается
  ↓ (не видит Text/Visual)
Visual обучается
  ↓ (не видит Text/Audio)
Bottleneck fusion
  ↓ (пытается что-то сделать с информацией, которая не была обучена для fusion)
  ❌ КАЧЕСТВО ПЛОХОЕ
```

**Проблема:** Каждая модальность оптимизируется для СВОЕЙ задачи, а не для СОВМЕСТНОЙ решения задачи.

### ✅ Если обучать вместе (End-to-End):

```
Text + Audio + Visual + Bottleneck
       ↓
   Все обучаются ВМЕСТЕ
       ↓
   Каждая модальность узнаёт:
   • Как лучше представлять информацию
   • Как лучше интегрироваться с другими
   • Как вместе решить задачу распознавания эмоций
       ↓
   ✅ КАЧЕСТВО ОТЛИЧНОЕ (80%+)
```

**Преимущество:** Gradient flow идёт через все модали одновременно, каждая модальность "видит" какая помощь ей нужна от других.

---

## 🚀 ПОШАГОВАЯ СТРАТЕГИЯ ДЛЯ 80%+

### Этап 1️⃣: Базовое обучение (Exp02-style)
```bash
python train.py \
    --model bottleneck \
    --data_path mosei_bottleneck.pkl \
    --epochs 20 \
    --batch_size 32 \
    --lr 1e-3 \
    --num_bottleneck_tokens 16 \
    --num_bottleneck_layers 2 \
    --freeze_bert full \
    --label_smoothing 0.1 \
    --exp_dir exp_stage1_baseline
```

**Результат:** Best model saved → `exp_stage1_baseline/best_model.pt`
**Ожидание:** Val F1 ~0.62-0.65, Test accuracy ~72-75%
**Время:** ~30 минут

---

### Этап 2️⃣: Fine-tune с оптимизированной LR (Exp08-style)
```bash
python train.py \
    --model bottleneck \
    --data_path mosei_bottleneck.pkl \
    --epochs 20 \
    --batch_size 32 \
    --lr 3e-4 \              # ← ПОНИЖЕНА!
    --num_bottleneck_tokens 16 \
    --num_bottleneck_layers 2 \
    --freeze_bert full \
    --label_smoothing 0.15 \  # ← УВЕЛИЧЕНА!
    --patience 5 \
    --exp_dir exp_stage2_refined
    # ⚠️ НЕ ЗАБЫТЬ: Загрузить best_model.pt из stage1!
```

**Загрузка весов от Stage1:**
```python
# В начале train.py добавить:
model.load_state_dict(torch.load("exp_stage1_baseline/best_model.pt"))
print("Loaded pretrained weights from Stage1")
```

**Результат:** Better model saved → `exp_stage2_refined/best_model.pt`
**Ожидание:** Val F1 ~0.64-0.67, Test accuracy ~74-77%
**Время:** ~30 минут

---

### Этап 3️⃣: BERT Fine-tuning (partial unfreeze)
```bash
python train.py \
    --model bottleneck \
    --data_path mosei_bottleneck.pkl \
    --epochs 15 \           # ← МЕНЬШЕ ЭПОХ (можно переобучить)
    --batch_size 32 \
    --lr 5e-4 \             # ← Для остального
    --lr_bert 5e-5 \        # ← Очень низкий для BERT
    --num_bottleneck_tokens 16 \
    --num_bottleneck_layers 2 \
    --freeze_bert partial \ # ← РАЗМОРОЗИТЬ!
    --label_smoothing 0.15 \
    --patience 4 \
    --exp_dir exp_stage3_bert_tuning
    # Загрузить best_model.pt из stage2!
```

**Результат:** Even better model → `exp_stage3_bert_tuning/best_model.pt`
**Ожидание:** Val F1 ~0.66-0.69, Test accuracy ~76-79%
**Время:** ~20 минут

---

### Этап 4️⃣: Финальная оптимизация (последний push к 80%)
```bash
python train.py \
    --model bottleneck \
    --data_path mosei_bottleneck.pkl \
    --epochs 10 \           # ← СОВСЕМ НЕМНОГО
    --batch_size 32 \
    --lr 2e-4 \             # ← Еще ниже
    --lr_bert 2e-5 \        # ← Еще ниже для BERT
    --num_bottleneck_tokens 16 \
    --num_bottleneck_layers 2 \
    --freeze_bert partial \
    --label_smoothing 0.2 \  # ← МАКСИМУМ
    --dropout 0.4 \         # ← УВЕЛИЧИТЬ DROPOUT
    --patience 3 \
    --use_dra \             # ← ВКЛЮЧИТЬ DRA LOSS!
    --exp_dir exp_stage4_final_push
    # Загрузить best_model.pt из stage3!
```

**Результат:** FINAL model → `exp_stage4_final_push/best_model.pt`
**Ожидание:** Val F1 ~0.68-0.71, Test accuracy ~78-81% 🎯
**Время:** ~15 минут

---

## 📈 ОЖИДАЕМАЯ ПРОГРЕССИЯ

```
Этап          Val F1    Test Acc    Rare Classes    Time
───────────────────────────────────────────────────────────
Начало        0.50      0.65        0.38            (baseline)
Этап 1        0.63      0.74        0.45            30 мин
Этап 2        0.65      0.76        0.50            +30 мин
Этап 3        0.68      0.78        0.54            +20 мин
Этап 4        0.70      0.80        0.58            +15 мин
───────────────────────────────────────────────────────────
ИТОГО                                                95 мин (1.5 часа)

🎯 Целевая метрика достигнута: Test Accuracy ≥ 80%!
```

---

## ⚙️ КАК ТЕХНИЧЕСКИ ЗАГРУЖАТЬ ВЕСА

### Способ 1: Модифицировать train.py

```python
# В главной функции (перед optimizer):

# Если есть сохраненные веса, загружаем
if args.pretrained_path:
    print(f"Loading pretrained model from {args.pretrained_path}")
    state_dict = torch.load(args.pretrained_path, map_location=device)
    model.load_state_dict(state_dict)
    print("✅ Loaded pretrained weights")
else:
    print("Training from scratch")

# Потом обучаем как обычно
# model.train()
# for epoch in range(args.epochs):
#     ...
```

### Способ 2: Запуск с флагом

```bash
# Добавить в argparse:
parser.add_argument("--pretrained_path", type=str, default=None,
                    help="Path to pretrained model weights")

# Запуск:
python train.py \
    --model bottleneck \
    --data_path mosei_bottleneck.pkl \
    --epochs 20 \
    --lr 3e-4 \
    --pretrained_path exp_stage1_baseline/best_model.pt \  # ← ЗАГРУЗИТЬ!
    --exp_dir exp_stage2_refined
```

### Способ 3: Bash скрипт (проще)

```bash
#!/bin/bash

# Stage 1
python train.py --model bottleneck --epochs 20 --lr 1e-3 \
    --exp_dir exp_stage1 --data_path mosei_bottleneck.pkl

# Stage 2 (начиная с весов Stage 1)
python train.py --model bottleneck --epochs 20 --lr 3e-4 \
    --exp_dir exp_stage2 --data_path mosei_bottleneck.pkl \
    --pretrained_path exp_stage1/best_model.pt

# Stage 3 (начиная с весов Stage 2)
python train.py --model bottleneck --epochs 15 --lr 5e-4 \
    --freeze_bert partial --lr_bert 5e-5 \
    --exp_dir exp_stage3 --data_path mosei_bottleneck.pkl \
    --pretrained_path exp_stage2/best_model.pt

# Stage 4 (финальный push)
python train.py --model bottleneck --epochs 10 --lr 2e-4 \
    --freeze_bert partial --lr_bert 2e-5 \
    --label_smoothing 0.2 --use_dra \
    --exp_dir exp_stage4 --data_path mosei_bottleneck.pkl \
    --pretrained_path exp_stage3/best_model.pt
```

---

## 🔑 КЛЮЧЕВЫЕ ПАРАМЕТРЫ ДЛЯ 80%+

### Learning Rate Strategy:
```
Stage 1: lr = 1e-3    (базовое обучение, быстро)
Stage 2: lr = 3e-4    (рефайн, медленнее)
Stage 3: lr = 5e-4    (fine-tuning, осторожно)
Stage 4: lr = 2e-4    (финальная настройка, очень осторожно)

Правило: LR → ↓ когда переходим на следующий stage
         Модель уже "знает" хорошее направление
         Нужно тонко настраивать, а не переучивать
```

### Batch Size:
```
Держите CONSTANT: batch_size = 32
Не менять между этапами!
(Разные batch size = разные gradients)
```

### BERT Freezing:
```
Stage 1: freeze_bert = full
  └─ BERT замерзен, обучаем только fusion + projections

Stage 2: freeze_bert = full
  └─ BERT все еще замерзен

Stage 3: freeze_bert = partial
  └─ Размораживаем последние 3 слоя BERT
  └─ lr_bert должен быть НАМНОГО ниже (5e-5)

Stage 4: freeze_bert = partial
  └─ lr_bert еще ниже (2e-5)
```

### Label Smoothing:
```
Stage 1: 0.10  (небольшая регуляризация)
Stage 2: 0.15  (побольше)
Stage 3: 0.15  (жесткая регуляризация против переобучения)
Stage 4: 0.20  (максимум, финальная защита)
```

### Early Stopping:
```
Stage 1: patience = 5
Stage 2: patience = 5
Stage 3: patience = 4  (раньше останавливаемся)
Stage 4: patience = 3  (еще раньше)
```

---

## 🎯 ЦЕЛЕВЫЕ МЕТРИКИ КАЖДОГО ЭТАПА

### Stage 1 (должны получить):
```
Val Macro F1:    0.62-0.64
Test Accuracy:   0.73-0.75
Test Macro F1:   0.62-0.64
Happy F1:        0.45-0.48  (редкий класс улучшается)
Surprise F1:     0.42-0.46
Fear F1:         0.40-0.44
```

### Stage 2 (должны получить):
```
Val Macro F1:    0.64-0.66  ← улучшение +2%
Test Accuracy:   0.75-0.77
Test Macro F1:   0.64-0.66
Happy F1:        0.50-0.53
Surprise F1:     0.48-0.51
Fear F1:         0.46-0.49
```

### Stage 3 (должны получить):
```
Val Macro F1:    0.66-0.69  ← улучшение +2-3%
Test Accuracy:   0.77-0.79
Test Macro F1:   0.66-0.69
Happy F1:        0.53-0.56
Surprise F1:     0.51-0.54
Fear F1:         0.50-0.53
```

### Stage 4 (ЦЕЛЕВОЕ):
```
Val Macro F1:    0.68-0.71  ← ФИНАЛЬНОЕ улучшение +2%
Test Accuracy:   0.79-0.82  🎯 ≥ 80%!
Test Macro F1:   0.68-0.71
Happy F1:        0.55-0.59
Surprise F1:     0.53-0.57
Fear F1:         0.52-0.56
```

---

## ⚠️ ЧАСТЫЕ ОШИБКИ

### ❌ Ошибка 1: Использовать слишком высокий LR на позже стадиях
```
Stage 2: lr = 1e-3 ❌ (слишком высокий!)
         Модель "разучится" того, что выучила на Stage 1
         
Правильно: lr = 3e-4
```

### ❌ Ошибка 2: Оставлять BERT замороженным на последних этапах
```
Stage 4: freeze_bert = full ❌
         BERT не обновляется, не может адаптироваться

Правильно: freeze_bert = partial
```

### ❌ Ошибка 3: Слишком много эпох на позже этапах
```
Stage 4: epochs = 50 ❌
         Слишком много → переобучение

Правильно: epochs = 10-15
```

### ❌ Ошибка 4: Не использовать label_smoothing
```
Без label_smoothing: train/val разница большая
С label_smoothing: модель обобщается лучше

Использовать: label_smoothing = 0.1-0.2
```

### ❌ Ошибка 5: Обучать модали отдельно
```
Text отдельно → Audio отдельно → Visual отдельно → Fusion
❌ Не работает, каждая модаль оптимизируется для себя

✅ Правильно: Все вместе через Bottleneck Fusion
```

---

## 📊 МОНИТОРИНГ ПРОГРЕССА

### Что смотреть после каждого этапа:

```python
# В metrics.txt ищите:

1. Test Accuracy
   Stage 1: 73-75%
   Stage 2: 75-77%
   Stage 3: 77-79%
   Stage 4: 79-82% ← ЦЕЛЕВОЕ

2. Test Macro F1
   Stage 1: 0.62-0.64
   Stage 2: 0.64-0.66
   Stage 3: 0.66-0.69
   Stage 4: 0.68-0.71

3. Per-class F1
   Happy:    улучшается с каждым этапом?
   Surprise: улучшается с каждым этапом?
   Fear:     улучшается с каждым этапом?

4. Val-Test Gap
   Val F1 - Test F1 < 0.03? ← хорошее обобщение
   > 0.05?                  ← переобучение, нужна регуляризация
```

---

## 🚀 ГОТОВЫЙ BASH СКРИПТ

```bash
#!/bin/bash
# train_to_80percent.sh

set -e

DATA_PATH="mosei_bottleneck.pkl"

echo "════════════════════════════════════════════════════════════════"
echo "СТАДИЯ 1: БАЗОВОЕ ОБУЧЕНИЕ"
echo "════════════════════════════════════════════════════════════════"
python train.py \
    --model bottleneck \
    --data_path "$DATA_PATH" \
    --epochs 20 \
    --batch_size 32 \
    --lr 1e-3 \
    --num_bottleneck_tokens 16 \
    --num_bottleneck_layers 2 \
    --freeze_bert full \
    --label_smoothing 0.1 \
    --exp_dir experiments/stage1_baseline

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "СТАДИЯ 2: РЕФАЙН С ОПТИМИЗИРОВАННЫМ LR"
echo "════════════════════════════════════════════════════════════════"
python train.py \
    --model bottleneck \
    --data_path "$DATA_PATH" \
    --epochs 20 \
    --batch_size 32 \
    --lr 3e-4 \
    --num_bottleneck_tokens 16 \
    --num_bottleneck_layers 2 \
    --freeze_bert full \
    --label_smoothing 0.15 \
    --patience 5 \
    --exp_dir experiments/stage2_refined \
    --pretrained_path experiments/stage1_baseline/best_model.pt

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "СТАДИЯ 3: BERT FINE-TUNING (PARTIAL UNFREEZE)"
echo "════════════════════════════════════════════════════════════════"
python train.py \
    --model bottleneck \
    --data_path "$DATA_PATH" \
    --epochs 15 \
    --batch_size 32 \
    --lr 5e-4 \
    --lr_bert 5e-5 \
    --num_bottleneck_tokens 16 \
    --num_bottleneck_layers 2 \
    --freeze_bert partial \
    --label_smoothing 0.15 \
    --patience 4 \
    --exp_dir experiments/stage3_bert_tuning \
    --pretrained_path experiments/stage2_refined/best_model.pt

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "СТАДИЯ 4: ФИНАЛЬНАЯ ОПТИМИЗАЦИЯ (PUSH К 80%)"
echo "════════════════════════════════════════════════════════════════"
python train.py \
    --model bottleneck \
    --data_path "$DATA_PATH" \
    --epochs 10 \
    --batch_size 32 \
    --lr 2e-4 \
    --lr_bert 2e-5 \
    --num_bottleneck_tokens 16 \
    --num_bottleneck_layers 2 \
    --freeze_bert partial \
    --label_smoothing 0.2 \
    --patience 3 \
    --use_dra \
    --exp_dir experiments/stage4_final_push \
    --pretrained_path experiments/stage3_bert_tuning/best_model.pt

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Результаты:"
echo "  Stage 1: $(grep 'Test accuracy' experiments/stage1_baseline/metrics.txt 2>/dev/null || echo 'не найдены')"
echo "  Stage 2: $(grep 'Test accuracy' experiments/stage2_refined/metrics.txt 2>/dev/null || echo 'не найдены')"
echo "  Stage 3: $(grep 'Test accuracy' experiments/stage3_bert_tuning/metrics.txt 2>/dev/null || echo 'не найдены')"
echo "  Stage 4: $(grep 'Test accuracy' experiments/stage4_final_push/metrics.txt 2>/dev/null || echo 'не найдены')"
echo ""
echo "Финальная модель: experiments/stage4_final_push/best_model.pt"
```

---

## 📋 ЧЕК-ЛИСТ

- [ ] Модифицировать train.py для загрузки pretrained weights
- [ ] Создать bash скрипт с 4 этапами
- [ ] Stage 1: 20 эпох, lr=1e-3 (ожидание ~74%)
- [ ] Stage 2: 20 эпох, lr=3e-4 (ожидание ~76%)
- [ ] Stage 3: 15 эпох, partial unfreeze (ожидание ~78%)
- [ ] Stage 4: 10 эпох, use_dra (ожидание ~80%+)
- [ ] Проверить per-class F1 на редких классах
- [ ] Сравнить результаты каждого этапа
- [ ] Финальная модель: ≥ 80% accuracy 🎯

---

## 💾 ФИНАЛЬНАЯ МОДЕЛЬ

После успешного Stage 4:
```
Final model: experiments/stage4_final_push/best_model.pt
Expected: Test Accuracy ≥ 80%
Expected: Per-class на редких классах улучшилась на +10-15% vs baseline
```

---

**Главное:** Обучайте ВСЕ ВМЕСТЕ, не отдельно!
Это обеспечит правильную интеграцию информации и 80%+ accuracy.

Good luck! 🚀
