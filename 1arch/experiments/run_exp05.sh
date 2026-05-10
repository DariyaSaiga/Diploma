#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Эксперимент 05 — Частичная разморозка BERT (layers 9, 10, 11)
#
# Меняем:
#   freeze_bert: full → partial  (размораживаем последние 3 encoder-слоя)
#   lr_bert    : 2e-5            (отдельный маленький lr для BERT-слоёв)
#   lr         : 5e-4            (lr для остальной части модели)
#   tokens     : 16              (берём лучший из предыдущих экспериментов)
#
# Гипотеза: адаптация последних слоёв BERT под эмоциональную задачу
#           улучшит текстовый branch
#
# Осторожно: если val F1 растёт, но train F1 >> val F1 → переобучение
# ─────────────────────────────────────────────────────────────────────────────

EXP_DIR="experiments/exp_05_partial_unfreeze"
mkdir -p "$EXP_DIR"

python train.py \
  --model bottleneck \
  --epochs 10 \
  --batch_size 16 \
  --lr 5e-4 \
  --lr_bert 2e-5 \
  --num_bottleneck_tokens 16 \
  --freeze_bert partial \
  --data_path mosei_bottleneck.pkl \
  --exp_dir "$EXP_DIR" \
  | tee "$EXP_DIR/train_log.txt"
