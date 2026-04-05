#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Эксперимент 06B — Ablation: без audio
#
# Меняем только одно:
#   --no_audio  (audio branch исключается из fusion)
#
# Остальное берём из baseline (exp_01):
#   tokens = 8, lr = 1e-3, freeze_bert = full
#
# Гипотеза / вопрос: audio помогает или шумит?
#
# Интерпретация:
#   Если test F1 ЛУЧШЕ без audio → audio branch мешает
#   Если test F1 ХУЖЕ  без audio → audio полезен
# ─────────────────────────────────────────────────────────────────────────────

EXP_DIR="experiments/exp_06B_no_audio"
mkdir -p "$EXP_DIR"

python train.py \
  --model bottleneck \
  --epochs 10 \
  --batch_size 16 \
  --lr 1e-3 \
  --num_bottleneck_tokens 8 \
  --freeze_bert full \
  --no_audio \
  --data_path mosei_bottleneck.pkl \
  --exp_dir "$EXP_DIR" \
  | tee "$EXP_DIR/train_log.txt"
