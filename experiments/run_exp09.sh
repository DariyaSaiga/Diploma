#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Эксперимент 09 — Multi-layer MBT + жёсткий early stopping (patience=3)
#
# Что меняем относительно exp_08:
#   patience : 5 → 3   (остановка раньше если модель перестала расти)
#   epochs   : 20 → 30  (лимит выше, но early stopping не даст уйти далеко)
#
# Гипотеза:
#   Multi-layer attention может быстро переобучиться на дисбалансных данных
#   Patience=3 страхует от плато и лишних эпох после пика
#   Сравниваем с exp_08: если лучший epoch тот же — patience не влияет
#   Если лучше — значит exp_08 шёл лишние эпохи без пользы
#
# Что смотреть:
#   - best_epoch в metrics.txt (если <10 — модель быстро сходится)
#   - сравнить val F1 curve с exp_08
# ─────────────────────────────────────────────────────────────────────────────

EXP_DIR="experiments/exp_09_early_stop_p3"
mkdir -p "$EXP_DIR"

python train.py \
  --model bottleneck \
  --epochs 30 \
  --batch_size 16 \
  --lr 3e-4 \
  --num_bottleneck_tokens 16 \
  --num_bottleneck_layers 2 \
  --freeze_bert full \
  --patience 3 \
  --label_smoothing 0.1 \
  --data_path mosei_bottleneck.pkl \
  --exp_dir "$EXP_DIR" \
  | tee "$EXP_DIR/train_log.txt"
