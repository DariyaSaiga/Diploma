#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Эксперимент 01 — Базовый bottleneck (reference point)
#
# Что фиксируем:
#   num_bottleneck_tokens = 8
#   lr                   = 1e-3
#   freeze_bert          = full
#   use_audio            = yes
#   use_visual           = yes
# ─────────────────────────────────────────────────────────────────────────────

EXP_DIR="experiments/exp_01_base_bottleneck"

python train.py \
  --model bottleneck \
  --epochs 10 \
  --batch_size 16 \
  --lr 1e-3 \
  --num_bottleneck_tokens 8 \
  --freeze_bert full \
  --data_path mosei_bottleneck.pkl \
  --exp_dir "$EXP_DIR" \
  | tee "$EXP_DIR/train_log.txt"
