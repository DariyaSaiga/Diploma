#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Эксперимент 04 — Комбинация: tokens=16 + lr=5e-4
#
# Меняем два параметра сразу (оба уже проверены по отдельности):
#   num_bottleneck_tokens: 8  → 16
#   lr:                  1e-3 → 5e-4
#
# Гипотеза: лучшая комбинация из exp_02 и exp_03
# Это кандидат на итоговую лучшую модель
# ─────────────────────────────────────────────────────────────────────────────

EXP_DIR="experiments/exp_04_tokens16_lr5e4"
mkdir -p "$EXP_DIR"

python train.py \
  --model bottleneck \
  --epochs 10 \
  --batch_size 16 \
  --lr 5e-4 \
  --num_bottleneck_tokens 16 \
  --freeze_bert full \
  --data_path mosei_bottleneck.pkl \
  --exp_dir "$EXP_DIR" \
  | tee "$EXP_DIR/train_log.txt"
