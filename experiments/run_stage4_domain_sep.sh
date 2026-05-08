#!/bin/bash

# Stage 4: Final Optimization (Polish)
# BERT LR: 2e-5 (very conservative)
# Other LR: 2e-4 (conservative)
# Epochs: 10
# Domain Separation: Enabled (increased weights for regularization)
# Dropout: increased to 0.15
# Label smoothing: increased to 0.2
# Pretrained: stage3_domain_sep/best_model.pt
# Expected: 79-82% accuracy (PEAK PERFORMANCE)

echo "🚀 STAGE 4: FINAL OPTIMIZATION + DOMAIN SEPARATION"
echo "=================================================="

python train.py \
  --model bottleneck \
  --epochs 10 \
  --batch_size 32 \
  --lr 2e-4 \
  --lr_bert 2e-5 \
  --num_bottleneck_tokens 16 \
  --num_bottleneck_layers 2 \
  --freeze_bert partial \
  --patience 5 \
  --label_smoothing 0.2 \
  --use_domain_sep \
  --alpha_sep 0.15 \
  --alpha_inv 0.08 \
  --alpha_rec 0.02 \
  --data_path mosei_bottleneck.pkl \
  --pretrained_path experiments/stage3_domain_sep/best_model.pt \
  --exp_dir experiments/stage4_domain_sep

echo ""
echo "🎉 TRAINING COMPLETE!"
echo "📊 Results saved to: experiments/stage4_domain_sep/"
echo ""
echo "📈 Check final results:"
echo "   cat experiments/stage4_domain_sep/metrics.txt"
echo ""
echo "🏆 Expected accuracy: 79-82% (vs baseline 74-76%)"
echo "📊 Expected Macro F1: ~0.70"
