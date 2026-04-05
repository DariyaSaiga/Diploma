#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Эксперимент 10 — WeightedRandomSampler
#
# Что меняем относительно exp_08:
#   --use_sampler   : включает WeightedRandomSampler вместо shuffle=True
#
# Гипотеза:
#   class_weights в loss штрафует за неверные предсказания редких классов,
#   но каждый батч всё равно формируется с натуральным дисбалансом (56% anger).
#   WeightedRandomSampler делает батчи более сбалансированными по классам
#   — это отдельный сигнал от class_weights, не дублирование.
#
# Осторожно:
#   Sampler + class_weights вместе могут перевесить редкие классы слишком сильно.
#   Если val F1 на anger/disgust упал — это признак перекоса.
#
# Что смотреть:
#   - per-class recall для happy, surprise, fear
#   - не упал ли F1 для anger (класс 0, самый частый)
#
# ─────────────────────────────────────────────────────────────────────────────

EXP_DIR="/content/drive/MyDrive/Дипломка_правильная/exp_10_sampler"
mkdir -p "$EXP_DIR"

python train.py \
  --model bottleneck \
  --epochs 20 \
  --batch_size 16 \
  --lr 3e-4 \
  --num_bottleneck_tokens 16 \
  --num_bottleneck_layers 2 \
  --freeze_bert full \
  --patience 5 \
  --label_smoothing 0.1 \
  --use_sampler \
  --data_path mosei_bottleneck.pkl \
  --exp_dir "$EXP_DIR" \
  | tee "$EXP_DIR/train_log.txt"
