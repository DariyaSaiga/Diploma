#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Эксперимент 06A — Ablation: без visual
#
# Меняем только одно:
#   --no_visual  (visual branch исключается из fusion)
#
# Остальное берём из baseline (exp_01):
#   tokens = 8, lr = 1e-3, freeze_bert = full
#
# Гипотеза / вопрос: visual помогает или шумит?
#
# Интерпретация:
#   Если test F1 ЛУЧШЕ без visual → visual branch мешает
#   Если test F1 ХУЖЕ  без visual → visual полезен
# ─────────────────────────────────────────────────────────────────────────────

EXP_DIR="experiments/exp_06A_no_visual"
mkdir -p "$EXP_DIR"

python train.py \
  --model bottleneck \
  --epochs 10 \
  --batch_size 16 \
  --lr 1e-3 \
  --num_bottleneck_tokens 8 \
  --freeze_bert full \
  --no_visual \
  --data_path mosei_bottleneck.pkl \
  --exp_dir "$EXP_DIR" \
  | tee "$EXP_DIR/train_log.txt"
