#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Эксперимент 08 — Multi-layer MBT: 2 слоя + lr=3e-4 + 20 эпох
#
# Что меняем относительно exp_01:
#   num_bottleneck_layers : 1 → 2   (итеративный обмен токенами между модальностями)
#   lr                    : 1e-3 → 3e-4
#   epochs                : 10 → 20
#
# Гипотеза:
#   2-слойный MBT чувствителен к lr — 3e-4 стабильнее чем 1e-3
#   20 эпох даёт полноценно сойтись multi-layer attention
#   Early stopping (patience=5) остановит раньше если нет прогресса
#
# Ожидаем улучшение по редким классам (happy, surprise, fear)
# т.к. bottleneck токены теперь итеративно обновляются через оба слоя
# ─────────────────────────────────────────────────────────────────────────────

EXP_DIR="experiments/exp_08_multilayer_lr3e4"
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
  --data_path mosei_bottleneck.pkl \
  --exp_dir "$EXP_DIR" \
  | tee "$EXP_DIR/train_log.txt"
