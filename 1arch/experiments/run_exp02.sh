#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Эксперимент 02 — Увеличить bottleneck tokens: 8 → 16
#
# Меняем только одно:
#   num_bottleneck_tokens: 8 → 16
#
# Гипотеза: больше токенов = больше capacity для fusion
# Сравнить с exp_01 по: val F1, test F1, редким классам (happy, surprise, fear)
# ─────────────────────────────────────────────────────────────────────────────

EXP_DIR="experiments/exp_02_tokens16"
mkdir -p "$EXP_DIR"

python train.py \
  --model bottleneck \
  --epochs 10 \
  --batch_size 16 \
  --lr 1e-3 \
  --num_bottleneck_tokens 16 \
  --freeze_bert full \
  --data_path mosei_bottleneck.pkl \
  --exp_dir "$EXP_DIR" \
  | tee "$EXP_DIR/train_log.txt"
