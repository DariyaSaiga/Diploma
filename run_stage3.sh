#!/bin/bash
# Stage 3: BERT Unfreezing

echo "🚀 STAGE 3: BERT UNFREEZING"
echo "Time: ~30 minutes | Expected: 77-79% accuracy"
echo ""

python training/train.py \
  --epochs 15 \
  --batch_size 32 \
  --lr 5e-4 \
  --num_bottleneck_tokens 16 \
  --freeze_bert partial \
  --patience 5 \
  --label_smoothing 0.15 \
  --use_domain_sep \
  --alpha_sep 0.1 \
  --alpha_inv 0.05 \
  --alpha_rec 0.01 \
  --pretrained_path experiments/stage2_domain_sep/best_model.pt \
  --data_path mosei_bottleneck.pkl \
  --exp_dir experiments/stage3_domain_sep

echo ""
echo "✅ Stage 3 completed!"
echo "📊 Results: cat experiments/stage3_domain_sep/metrics.txt"
