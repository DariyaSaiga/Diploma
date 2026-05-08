# 🚀 Быстрый старт (после исправлений)

## ✅ Все ошибки исправлены!

Все критичные ошибки в коде устранены. Вот как начать работать с проектом.

---

## 1️⃣ Установка зависимостей

```bash
pip install -r requirements.txt
```

Это установит все необходимые пакеты с правильными версиями.

---

## 2️⃣ Проверка загрузки данных

```bash
python test_loader.py
```

✅ Должен вывести:
```
✅ Батч загружен успешно!

── Shapes ──
  input_ids:      torch.Size([4, 128])
  attention_mask: torch.Size([4, 128])
  audio:          torch.Size([4, 25, 74])      # Ta варьируется!
  audio_mask:     torch.Size([4, 25])
  visual:         torch.Size([4, 30, 713])     # Tv варьируется!
  visual_mask:    torch.Size([4, 30])

── Проверка масок ──
✅ Все проверки пройдены!
```

---

## 3️⃣ Проверка модели (Forward pass)

```bash
python check_forward.py --data_path mosei_bottleneck.pkl --batch_size 4
```

✅ Должен вывести:
```
Device: cuda (или cpu)

── Batch shapes ──────────────────────────────────────
  input_ids:      torch.Size([4, 128])
  attention_mask: torch.Size([4, 128])
  audio:          torch.Size([4, 25, 74])
  audio_mask:     torch.Size([4, 25])
  visual:         torch.Size([4, 30, 713])
  visual_mask:    torch.Size([4, 30])
  label:          torch.Size([4])

── Intermediate shapes ───────────────────────────────
  text_seq:           torch.Size([4, 128, 128])
  audio_seq:          torch.Size([4, 25, 128])
  visual_seq:         torch.Size([4, 30, 128])
  bottleneck_tokens:  torch.Size([4, 16, 128])

── Loss & backward ───────────────────────────────────
  loss = 1.7895  ✓

✅ Forward check passed!
```

---

## 4️⃣ Обучение базовой модели

### Простой запуск (по умолчанию):
```bash
python train.py --model bottleneck --data_path mosei_bottleneck.pkl
```

### С конфигурацией:
```bash
python train.py \
    --model bottleneck \
    --data_path mosei_bottleneck.pkl \
    --batch_size 32 \
    --lr 1e-4 \
    --epochs 20 \
    --num_bottleneck_tokens 16 \
    --num_bottleneck_layers 2 \
    --freeze_bert partial \
    --label_smoothing 0.1 \
    --patience 5 \
    --exp_dir ./results/exp_001
```

### Описание параметров:

| Параметр | Значение | Описание |
|---|---|---|
| `--model` | bottleneck | Используемая модель |
| `--data_path` | mosei_bottleneck.pkl | Путь к датасету |
| `--batch_size` | 32 | Размер батча |
| `--lr` | 1e-4 | Learning rate |
| `--epochs` | 20 | Количество эпох |
| `--num_bottleneck_tokens` | 16 | Количество bottleneck токенов |
| `--num_bottleneck_layers` | 2 | Количество слоёв bottleneck |
| `--freeze_bert` | partial | none / full / partial |
| `--label_smoothing` | 0.1 | Label smoothing (0.0 = отключено) |
| `--patience` | 5 | Early stopping patience |
| `--exp_dir` | ./results/exp_001 | Директория для сохранения |

---

## 5️⃣ Сравнение моделей

### Текстовая модель (BERT только):
```bash
python train.py --model text --data_path mosei_bottleneck.pkl --epochs 10
```

### Аудио-видео модель:
```bash
python train.py --model av --data_path mosei_bottleneck.pkl --epochs 10
```

### Простая fusion (без bottleneck):
```bash
python train.py --model simple_fusion --data_path mosei_bottleneck.pkl --epochs 10
```

### Bottleneck fusion (рекомендуется):
```bash
python train.py --model bottleneck --data_path mosei_bottleneck.pkl --epochs 10
```

---

## 🎯 Рекомендуемая конфигурация для лучшего результата

```bash
python train.py \
    --model bottleneck \
    --data_path mosei_bottleneck.pkl \
    --batch_size 32 \
    --lr 1e-4 \
    --lr_bert 2e-5 \
    --epochs 30 \
    --num_bottleneck_tokens 16 \
    --num_bottleneck_layers 3 \
    --freeze_bert partial \
    --label_smoothing 0.15 \
    --patience 8 \
    --use_dra \
    --exp_dir ./results/best_model
```

**Параметры:**
- `--use_dra` — включить Dynamic Rate Adjustment Loss (для улучшения качества)
- `--freeze_bert partial` — разморозить последние 3 слоя BERT
- `--lr_bert 2e-5` — низкий learning rate для BERT

---

## 📊 Вывод результатов

После обучения в директории `--exp_dir` будут сохранены:

```
./results/best_model/
├── config.txt          # Конфигурация эксперимента
├── best_model.pt       # Сохранённые веса модели
└── metrics.txt         # Итоговые метрики (Test F1, accuracy и т.д.)
```

Смотреть результаты:
```bash
cat ./results/best_model/metrics.txt
```

---

## 🔧 Что исправлено

✅ **Критичные ошибки:**
- Исправлены импорты `from Diploma.backend.bottleneck_fusion` → `from bottleneck_fusion`
- Добавлена загрузка данных перед использованием датасета
- Добавлена collate_fn для динамического паддинга и масок

✅ **Логические ошибки:**
- Исправлена нормализация слоёв в simple_fusion.py
- Стабилизировано деление в dra_loss.py
- Согласована работа с BottleneckDataset

✅ **Улучшения:**
- Добавлены версии зависимостей в requirements.txt
- Добавлены проверки масок в test_loader.py
- Улучшена обработка ошибок импорта

---

## 📝 Полная документация

- [DATASET_COMPARISON.md](DATASET_COMPARISON.md) — сравнение датасетов
- [ISSUES_REPORT.md](ISSUES_REPORT.md) — подробный анализ проблем
- [FIXES_APPLIED.md](FIXES_APPLIED.md) — список всех исправлений

---

## ❓ FAQ

**Q: Какой датасет использовать?**
A: Используйте `mosei_bottleneck.pkl` — он лучше сохраняет информацию и работает с масками для attention.

**Q: Можно ли использовать `mosei_cleaned.pkl`?**
A: Технически да, но нужно будет переписать load_loaders. Рекомендуется `mosei_bottleneck.pkl`.

**Q: Модель не загружается — ошибка bottleneck_fusion.py?**
A: Нужно найти исходный файл `bottleneck_fusion.py`. Сейчас есть только скомпилированный .pyc.

**Q: Как ускорить обучение?**
A: Используйте `--batch_size 64`, `--freeze_bert full` или уменьшите `--num_bottleneck_layers`.

**Q: Как получить лучшие результаты?**
A: Используйте конфигурацию из раздела "Рекомендуемая конфигурация" с `--use_dra`.

---

✅ **Готово к работе!** Запустите `python test_loader.py` для проверки.
