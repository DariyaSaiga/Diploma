#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Эксперимент 03 — Уменьшить learning rate: 1e-3 → 5e-4
#
# Меняем только одно:
#   lr: 1e-3 → 5e-4
#
# Гипотеза: lr=1e-3 слишком шумный, меньший lr даст стабильнее сходимость
# Смотреть: более гладкая val-кривая, лучший финальный val/test F1
# ─────────────────────────────────────────────────────────────────────────────

EXP_DIR="experiments/exp_03_lr5e4"
mkdir -p "$EXP_DIR"

python train.py \
  --model bottleneck \
  --epochs 10 \
  --batch_size 16 \
  --lr 5e-4 \
  --num_bottleneck_tokens 8 \
  --freeze_bert full \
  --data_path mosei_bottleneck.pkl \
  --exp_dir "$EXP_DIR" \
  | tee "$EXP_DIR/train_log.txt"
