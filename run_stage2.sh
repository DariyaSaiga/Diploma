#!/bin/bash
# Stage 2: Fine-tuning (Domain-Separated)

echo "🚀 STAGE 2: FINE-TUNING"
echo "Time: ~40 minutes | Expected: 76-78% accuracy"
echo ""

python training/train.py \
  --epochs 20 \
  --batch_size 32 \
  --lr 3e-4 \
  --num_bottleneck_tokens 16 \
  --freeze_bert full \
  --patience 5 \
  --label_smoothing 0.15 \
  --use_domain_sep \
  --alpha_sep 0.1 \
  --alpha_inv 0.05 \
  --alpha_rec 0.01 \
  --pretrained_path experiments/stage1_domain_sep/best_model.pt \
  --data_path mosei_bottleneck.pkl \
  --exp_dir experiments/stage2_domain_sep

echo ""
echo "✅ Stage 2 completed!"
echo "📊 Results: cat experiments/stage2_domain_sep/metrics.txt"
