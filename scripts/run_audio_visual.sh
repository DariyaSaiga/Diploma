#!/bin/bash
# Baseline 2: Audio + Visual (no text, no bottleneck)
# Target: accuracy ~65-72%, macro F1 >= 0.50

echo "AUDIO-VISUAL BASELINE"

python training/train.py \
  --model audio_visual \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-3 \
  --patience 5 \
  --label_smoothing 0.1 \
  --data_path "/content/drive/MyDrive/Дипломка_правильная/mosei_bottleneck.pkl" \
  --exp_dir experiments/audio_visual_baseline

echo "Done. Results: cat experiments/audio_visual_baseline/metrics.txt"
