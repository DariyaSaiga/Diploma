#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  run_baselines.sh — полное сравнение за ~2 часа
#
#  Уровень 1 — отдельные модальности:
#    B01: Text-only BERT
#    B02: Audio + Visual
#
#  Уровень 2 — простой мультимодальный fusion:
#    B03: SimpleFusion (concat без bottleneck)
#
#  Уровень 3 — Bottleneck:
#    B04: Bottleneck 1 layer
#    B05: Bottleneck 2 layers (основная модель)
#
#  Запуск: cd Diploma/ && bash experiments/run_baselines.sh
#  Результат: python experiments/compare_baselines.py
# ═══════════════════════════════════════════════════════════════

set -e
DATA="mosei_bottleneck.pkl"

echo "══════════════════════════════════════════════════════════"
echo "  Baseline comparison — 3 models"
echo "══════════════════════════════════════════════════════════"

# ──────────────────────────────────────────────────────────────
# B01: Text-only BERT
# Ожидаемое время: ~20 мин (BERT frozen, 6 эпох, batch=32)
# ──────────────────────────────────────────────────────────────
echo ""
echo ">>> [B01] Text-only BERT  (уровень 1)"
EXP="experiments/B01_text_only"
mkdir -p "$EXP"
python train.py \
  --model text \
  --epochs 6 \
  --batch_size 32 \
  --lr 1e-3 \
  --patience 3 \
  --label_smoothing 0.1 \
  --data_path "$DATA" \
  --exp_dir "$EXP" \
  | tee "$EXP/train_log.txt"

# ──────────────────────────────────────────────────────────────
# B02: Audio + Visual baseline
# Ожидаемое время: ~5 мин (нет BERT, быстро)
# ──────────────────────────────────────────────────────────────
echo ""
echo ">>> [B02] Audio+Visual baseline  (уровень 1)"
EXP="experiments/B02_audio_visual"
mkdir -p "$EXP"
python train.py \
  --model av \
  --epochs 10 \
  --batch_size 32 \
  --lr 1e-3 \
  --patience 3 \
  --label_smoothing 0.1 \
  --data_path "$DATA" \
  --exp_dir "$EXP" \
  | tee "$EXP/train_log.txt"

# ──────────────────────────────────────────────────────────────
# B03: Simple Fusion (concat text+audio+visual без bottleneck)
# Ожидаемое время: ~20 мин (BERT frozen, 6 эпох, batch=32)
# ──────────────────────────────────────────────────────────────
echo ""
echo ">>> [B03] Simple Fusion — concat без bottleneck  (уровень 2)"
EXP="experiments/B03_simple_fusion"
mkdir -p "$EXP"
python train.py \
  --model simple_fusion \
  --epochs 6 \
  --batch_size 32 \
  --lr 1e-3 \
  --patience 3 \
  --label_smoothing 0.1 \
  --data_path "$DATA" \
  --exp_dir "$EXP" \
  | tee "$EXP/train_log.txt"

