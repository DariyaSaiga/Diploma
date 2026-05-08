# 🎯 БЫСТРЫЙ ВИЗУАЛЬНЫЙ ГАЙД: От 75% к 80%+

## ❓ ОТВЕТ НА ГЛАВНЫЙ ВОПРОС

```
ВОПРОС: Нужно ли обучать каждую модель отдельно?
        Потом соединять через bottleneck?

ОТВЕТ:  ❌ НЕТ!
        ✅ Обучайте ВСЕ ВМЕСТЕ через Bottleneck
```

---

## 📊 ВИЗУАЛЬНО: Как работает обучение

### ❌ НЕПРАВИЛЬНО (обучать отдельно):

```
Text    Audio    Visual
  |       |        |
  v       v        v
[Обучение отдельно]
  |       |        |
  v       v        v
Text    Audio    Visual
Model   Model    Model
  
         ↓ результаты не совместимы!
         
    [Bottleneck Fusion]
         ↓
      Плохо! (~70%)
```

**Проблема:** Каждая модель учится сама по себе, не "видит" других


### ✅ ПРАВИЛЬНО (обучать вместе):

```
         [Bottleneck Fusion]
              ↓
    [Progressive Training]
              ↓
         [Stage 1]        [Stage 2]       [Stage 3]      [Stage 4]
         20 epochs        20 epochs        15 epochs      10 epochs
         lr=1e-3          lr=3e-4         lr=5e-4        lr=2e-4
         frozen BERT      frozen BERT     unfrozen       unfrozen
              ↓                ↓               ↓              ↓
           ~74%            ~76%            ~78%         ~80%+ ✅
           
Все вместе обучаются на КАЖДОМ этапе!
```

---

## 🚀 4-ЭТАПНАЯ СТРАТЕГИЯ

### Простой способ понять:

```
Этап 1: Быстрое обучение
└─ Цель: Научить модель базовой интеграции
   Параметры: lr=1e-3 (большой), 20 эпох
   Результат: ~74% accuracy
   Время: 30 мин

Этап 2: Уточнение (Fine-tuning)
└─ Цель: Улучшить качество, не разучиться
   Параметры: lr=3e-4 (меньше!), 20 эпох
   Результат: ~76% accuracy
   Время: 30 мин

Этап 3: Разморозить BERT
└─ Цель: BERT начинает помогать
   Параметры: lr_bert=5e-5 (очень мало!), 15 эпох
   Результат: ~78% accuracy
   Время: 20 мин

Этап 4: ФИНАЛЬНЫЙ PUSH
└─ Цель: Максимальное качество
   Параметры: lr=2e-4 (еще меньше), label_smoothing=0.2, 10 эпох
   Результат: ~80%+ accuracy ✅
   Время: 15 мин

ИТОГО: ~1.5 часа обучения
```

---

## 🎓 АНАЛОГИЯ ДЛЯ ПОНИМАНИЯ

```
Представьте трёхчленный оркестр:

❌ НЕПРАВИЛЬНО (обучить отдельно):
  • Скрипач 30 дней репетирует ОДИН
  • Пианист 30 дней репетирует ОДИН
  • Виолончелист 30 дней репетирует ОДИН
  • Потом они впервые встречаются → КАКОФОНИЯ!
    (Они не знают, как играть ВМЕСТЕ)

✅ ПРАВИЛЬНО (обучить вместе, 4 этапа):
  • День 1-5: БЫСТРО учатся играть вместе (громко, быстро)
  • День 6-10: УточняЮТ вместе (медленнее, аккуратнее)
  • День 11-15: СКРИПАЧ УСИЛИВАЕТ ПАРТИЮ
  • День 16-20: ФИНАЛЬНАЯ ОТРАБОТКА
  • Результат: КРАСИВАЯ МУЗЫКА! ✨

Ключ: Все учатся ВМЕСТЕ, как одна команда!
```

---

## 📈 ПРОГРЕССИЯ МЕТРИК (ожидаемые результаты)

```
                Базовая   Stage 1   Stage 2   Stage 3   Stage 4
                         (20 эпох) (20 эпох) (15 эпох) (10 эпох)
────────────────────────────────────────────────────────────────
Test Accuracy    0%  →    74%   →    76%   →    78%   →   80%+ 🎯
Test Macro F1    0%  →   0.62  →   0.64  →   0.67  →   0.70
Happy (F1)      0%  →   0.45  →   0.50  →   0.54  →   0.57 ✨
Surprise (F1)   0%  →   0.42  →   0.48  →   0.52  →   0.55 ✨
Fear (F1)       0%  →   0.40  →   0.46  →   0.51  →   0.54 ✨

Видно улучшение на КАЖДОМ этапе? → Правильная стратегия!
```

---

## 🔑 ГЛАВНЫЕ ПАРАМЕТРЫ

### Learning Rate (самый важный!)

```
Stage 1: lr = 1e-3    ✅ Большой (быстрое обучение)
Stage 2: lr = 3e-4    ✅ В 3 раза меньше (аккуратнее)
Stage 3: lr = 5e-4    ✅ Такой же (осторожно)
         lr_bert = 5e-5 ✅ Очень мало (BERT чувствительный)
Stage 4: lr = 2e-4    ✅ Еще меньше (финальная настройка)
         lr_bert = 2e-5 ✅ Еще меньше для BERT

ПРАВИЛО: Когда LR падает, точность растет медленнее, но стабильнее
```

### BERT Freezing

```
Stage 1-2: freeze_bert = full
           └─ BERT замерз, не обновляется
           └─ Обучаем только Bottleneck + projections

Stage 3-4: freeze_bert = partial
           └─ Размораживаем последние 3 слоя BERT
           └─ BERT начинает помогать!
           └─ lr_bert должен быть НАМНОГО ниже обычного
```

### Label Smoothing (против переобучения)

```
Stage 1: label_smoothing = 0.10  (легкая регуляризация)
Stage 2: label_smoothing = 0.15  (посильнее)
Stage 3: label_smoothing = 0.15  (защита)
Stage 4: label_smoothing = 0.20  (максимум!)
```

---

## ⚡ ТЕХНИЧЕСКИ: КАК ЗАГРУЖАТЬ ВЕСА

### Самый простой способ (скопируй-вставь):

```bash
# Stage 1 - с нуля
python train.py --model bottleneck --epochs 20 --lr 1e-3 \
    --exp_dir stage1 --data_path mosei_bottleneck.pkl

# Stage 2 - начиная с весов Stage 1
python train.py --model bottleneck --epochs 20 --lr 3e-4 \
    --exp_dir stage2 --data_path mosei_bottleneck.pkl \
    --pretrained_path stage1/best_model.pt    # ← ЗАГРУЗИТЬ!

# Stage 3 - начиная с весов Stage 2
python train.py --model bottleneck --epochs 15 --lr 5e-4 \
    --freeze_bert partial --lr_bert 5e-5 \
    --exp_dir stage3 --data_path mosei_bottleneck.pkl \
    --pretrained_path stage2/best_model.pt    # ← ЗАГРУЗИТЬ!

# Stage 4 - финальный push
python train.py --model bottleneck --epochs 10 --lr 2e-4 \
    --freeze_bert partial --lr_bert 2e-5 \
    --label_smoothing 0.2 --use_dra \
    --exp_dir stage4 --data_path mosei_bottleneck.pkl \
    --pretrained_path stage3/best_model.pt    # ← ЗАГРУЗИТЬ!
```

---

## 🎯 ЧТО СМОТРЕТЬ ПОСЛЕ КАЖДОГО ЭТАПА

```
Откройте: stage1/metrics.txt (потом stage2/metrics.txt, etc)

Ищите строку:
  Test accuracy: 0.XXXX

Stage 1: Должно быть ~0.73-0.75? ✅
Stage 2: Должно быть ~0.75-0.77? ✅ (улучшилось)
Stage 3: Должно быть ~0.77-0.79? ✅ (еще улучшилось)
Stage 4: Должно быть ~0.79-0.82? ✅✅✅ (ЦЕЛЬ ДОСТИГНУТА!)

Если на каком-то этапе нет улучшения:
  → Может быть, нужна другая LR
  → Может быть, нужны другие параметры
  → Не отчаивайтесь, попробуйте еще раз!
```

---

## ❌ ЧТО НЕ ДЕЛАТЬ

```
❌ НЕ обучайте модали отдельно!
   └─ Это не поможет достичь 80%

❌ НЕ используйте LR=1e-3 на Stage 3-4!
   └─ Модель разучится того, что выучила

❌ НЕ пропускайте Stage 3 (BERT fine-tuning)!
   └─ Именно разморозка BERT дает +2-3%

❌ НЕ тренируйте 50+ эпох на поздних этапах!
   └─ Будет переобучение

❌ НЕ забывайте про label_smoothing!
   └─ Он критичен для обобщения (val/test разница)
```

---

## ✅ ЧЕКЛИСТ УСПЕХА

```
Перед началом:
☐ Понимаю, что все модали обучаются ВМЕСТЕ? (не отдельно)
☐ Знаю, что LR уменьшается на каждом этапе?
☐ Знаю, как загружать сохраненные веса?

Во время обучения:
☐ Stage 1 завершилась, получил результаты (~74%)
☐ Stage 2 завершилась, результаты улучшились (~76%)
☐ Stage 3 завершилась, BERT размораживался (~78%)
☐ Stage 4 завершилась, достиг ЦЕЛИ (~80%)

Проверка результатов:
☐ Test Accuracy ≥ 80%?
☐ Happy F1 улучшилась на +10-15%? (редкие классы!)
☐ Val-Test gap < 3%? (хорошее обобщение)
☐ Каждый этап показал улучшение?
```

---

## 🎬 ГОТОВО К ЗАПУСКУ!

**Начните с этого скрипта (скопируйте в terminal):**

```bash
# Stage 1
python train.py --model bottleneck --data_path mosei_bottleneck.pkl \
    --epochs 20 --batch_size 32 --lr 1e-3 \
    --num_bottleneck_tokens 16 --num_bottleneck_layers 2 \
    --freeze_bert full --label_smoothing 0.1 \
    --exp_dir stage1_baseline

# Когда Stage 1 готова, запустите Stage 2 (измените --pretrained_path):
python train.py --model bottleneck --data_path mosei_bottleneck.pkl \
    --epochs 20 --batch_size 32 --lr 3e-4 \
    --num_bottleneck_tokens 16 --num_bottleneck_layers 2 \
    --freeze_bert full --label_smoothing 0.15 --patience 5 \
    --exp_dir stage2_refined \
    --pretrained_path stage1_baseline/best_model.pt

# И так далее для Stage 3 и 4...
```

---

**TL;DR:**
1. Обучай ВСЕ ВМЕСТЕ (не отдельно!) ✅
2. 4 этапа с разными LR (1e-3 → 3e-4 → 5e-4 → 2e-4) ✅
3. Stage 3: разморозь BERT ✅
4. Stage 4: максимум label_smoothing + DRA loss ✅
5. Итог: ~80%+ accuracy! 🎯

Good luck! 🚀
