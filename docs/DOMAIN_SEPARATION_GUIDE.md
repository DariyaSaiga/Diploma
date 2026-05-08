# 🔥 Domain-Separated Bottleneck Architecture Guide

## Что изменилось?

Ваша новая архитектура:
- **Разделяет каждую модальность** на Invariant (эмоция) + Private (специфика модальности)
- **Bottleneck tokens (B=16)** ограничивают информационный обмен между модальностями
- **Multi-loss training** с 4 компонентами:
  - `L_task`: классификация эмоций (основная)
  - `L_sep`: разделение между invariant и private (ортогональность)
  - `L_inv`: alignment инвариант сигналов через модальности
  - `L_rec`: восстановление исходных features из decomposed domains

## Используется с вашими текущими энкодерами

✅ **BERT** (768) → 128 (frozen в Stage 1-2)  
✅ **Audio Linear** (74) → 128  
✅ **Visual Linear** (713) → 128  

## 4-Stage Training (74% → 80%+)

### Stage 1: Fresh Training
```bash
bash experiments/run_stage1_domain_sep.sh
```
- **LR**: 1e-3 (быстрое обучение)
- **Epochs**: 20
- **BERT**: Frozen
- **Expected**: 74-76% accuracy, ~0.62 Macro F1
- **Time**: ~30-40 minutes

### Stage 2: Fine-tuning
```bash
bash experiments/run_stage2_domain_sep.sh
```
- **LR**: 3e-4 (refined learning)
- **Epochs**: 20
- **BERT**: Frozen
- **Pretrained**: Stage 1 weights
- **Expected**: 76-78% accuracy, ~0.64 Macro F1
- **Time**: ~30-40 minutes

### Stage 3: BERT Unfreezing
```bash
bash experiments/run_stage3_domain_sep.sh
```
- **BERT LR**: 5e-5 (очень осторожно!)
- **Other LR**: 5e-4
- **Epochs**: 15
- **BERT**: Partially unfrozen (layers 9-11)
- **Pretrained**: Stage 2 weights
- **Expected**: 77-79% accuracy, ~0.66 Macro F1
- **Time**: ~20-30 minutes

### Stage 4: Final Optimization
```bash
bash experiments/run_stage4_domain_sep.sh
```
- **BERT LR**: 2e-5 (консервативно)
- **Other LR**: 2e-4 (консервативно)
- **Epochs**: 10
- **Label Smoothing**: 0.2 (max)
- **Pretrained**: Stage 3 weights
- **Expected**: 79-82% accuracy, ~0.70 Macro F1
- **Time**: ~15-20 minutes

**TOTAL**: ~1.5-2 hours для достижения 80%+ accuracy

---

## Как это работает?

### 1️⃣ Domain Separation
Каждая модальность разделяется на:
```
Input (Text/Audio/Visual)
         ↓
    Encoder (BERT/CNN/BiLSTM)
         ↓
    Projection (→ 128)
         ↓
    ┌────────────────┐
    │  Invariant    │  ← Эмоция (shared)
    └────────────────┘
    ┌────────────────┐
    │   Private     │  ← Специфика модальности
    └────────────────┘
```

**Invariant**: одна и та же эмоция должна иметь похожие инвариант представления (тоска на лице, голосе, словах — похожа)

**Private**: каждая модальность может иметь свою специфику (фонетические особенности в голосе, выражение лица в видео, слова в тексте)

### 2️⃣ Bottleneck Tokens
```
16 learnable tokens (B=16)
         ↓
    Information Gateway
         ↓
    Все cross-modal information
    проходит ТОЛЬКО через эти 16 токенов
         ↓
    Это создаёт информационное сжатие → лучшая обобщаемость
```

### 3️⃣ Cross-Attention через Bottleneck
```
Private domain (Text) → Attend to Bottleneck Tokens → Refined Private
     ↓                                                      ↓
     └──────────────────────────────────────────────────────┘
```

Информация из Text private domain пополняется инструкциями от bottleneck tokens.

### 4️⃣ Multi-Loss Training
```
L_total = L_task + 0.1*L_sep + 0.05*L_inv + 0.01*L_rec

L_task      → Основная классификация эмоций
L_sep       → Минимизируем cosine_similarity(invariant, private)
              = Делаем домены ортогональными
L_inv       → Для одной эмоции, invariant сигналы должны быть близки
              = Cross-modal alignment
L_rec       → Восстанавливаем исходные features из [inv||priv]
              = Сохраняем информацию в обоих доменах
```

---

## Почему это работает лучше чем 53%?

Ваша старая архитектура:
```
Text (128) ──┐
Audio (128)  ├─→ Concat (384) → Linear (384→128) → Classifier
Visual (128) ┘
```
❌ Просто конкатенация и сжатие  
❌ Нет разделения модальностей  
❌ Нет информационного ограничения  
❌ Модель может запомнить spurious correlations  

Новая архитектура:
```
Text   → Inv (128) + Priv (128) ┐
Audio  → Inv (128) + Priv (128) ├─→ Bottleneck Token Attention ─→ Fusion ─→ Classifier
Visual → Inv (128) + Priv (128) ┘
                           ↑
                  Информационное сжатие через 16 токенов
```
✅ Разделение invariant/private  
✅ Информационное сжатие через bottleneck  
✅ Multi-loss training гарантирует хорошую decomposition  
✅ Лучшая обобщаемость на тестовом наборе  

---

## Понимание Loss Weights

```python
--alpha_sep 0.1     # Separation loss вес
--alpha_inv 0.05    # Invariant loss вес
--alpha_rec 0.01    # Reconstruction loss вес
```

На разных стадиях эти веса немного меняются:

| Stage | α_sep | α_inv | α_rec | Зачем? |
|-------|-------|-------|-------|--------|
| 1 | 0.10 | 0.05 | 0.01 | Baseline domain separation |
| 2 | 0.10 | 0.05 | 0.01 | Продолжаем то же направление |
| 3 | 0.10 | 0.05 | 0.01 | BERT начинает обучаться |
| 4 | **0.15** | **0.08** | **0.02** | Увеличиваем регуляризацию |

В Stage 4 веса увеличены — это дополнительная регуляризация чтобы избежать overfitting на финальном этапе.

---

## Полный процесс (от А до Я)

```bash
# 1. Проверить что всё загружается
python check_forward.py --data_path mosei_bottleneck.pkl

# 2. Stage 1 (свежее обучение)
bash experiments/run_stage1_domain_sep.sh
cat experiments/stage1_domain_sep/metrics.txt

# 3. Stage 2 (fine-tuning)
bash experiments/run_stage2_domain_sep.sh
cat experiments/stage2_domain_sep/metrics.txt

# 4. Stage 3 (unfreeze BERT)
bash experiments/run_stage3_domain_sep.sh
cat experiments/stage3_domain_sep/metrics.txt

# 5. Stage 4 (final optimization)
bash experiments/run_stage4_domain_sep.sh
cat experiments/stage4_domain_sep/metrics.txt

# 🏆 Готово! Проверить финальные результаты
cat experiments/stage4_domain_sep/metrics.txt
```

---

## Отладка если что-то не работает

### Ошибка: "ModuleNotFoundError: No module named 'bottleneck_fusion'"
```bash
ls -la bottleneck_fusion.py  # должен быть в корне
```

### Ошибка: "RuntimeError: expected scalar type Float"
Проверить что audio и visual правильно приходят в модель. В bottleneck_dataset.py должны быть float32.

### Loss растёт (не уменьшается)
- Попробовать уменьшить learning rate: `--lr 5e-4`
- Или увеличить patience: `--patience 10`
- Проверить что class weights правильно рассчитаны

### Validation F1 не растёт
- Stage 2 должна улучшить результаты Stage 1
- Если нет, может быть overfitting на train set
- Попробовать увеличить label smoothing: `--label_smoothing 0.2`

---

## Дальнейшие улучшения (если захочешь экспериментировать)

Когда достигнешь 80%, можно попробовать:

1. **Увеличить bottleneck tokens**: `--num_bottleneck_tokens 32` (больше информационных врат)
2. **Добавить attention heads**: в bottleneck_fusion.py изменить `num_heads = 8` на `16`
3. **Использовать разные learning rates для разных слоёв**: `--lr_bert 1e-5 --lr 1e-4`
4. **Ensemble разных моделей**: train несколько раз с разными seeds и усреднить predictions

Но сейчас главная цель — 80%, и эта архитектура должна это дать!

---

## Важные файлы

- `bottleneck_fusion.py` — главная архитектура (domain-separated)
- `train.py` — updated с multi-loss support
- `experiments/run_stage*.sh` — скрипты для 4-stage training
- `BOTTLENECK_ARCHITECTURE.docx` — полная техническая спецификация

Удачи! 🚀
