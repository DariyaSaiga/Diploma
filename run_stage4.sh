#!/bin/bash
# Stage 4: Final Optimization

echo "🚀 STAGE 4: FINAL OPTIMIZATION"
echo "Time: ~20 minutes | Expected: 79-82% accuracy"
echo ""

python training/train.py \
  --epochs 10 \
  --batch_size 32 \
  --lr 2e-4 \
  --num_bottleneck_tokens 16 \
  --freeze_bert partial \
  --patience 5 \
  --label_smoothing 0.2 \
  --use_domain_sep \
  --alpha_sep 0.15 \
  --alpha_inv 0.08 \
  --alpha_rec 0.02 \
  --pretrained_path experiments/stage3_domain_sep/best_model.pt \
  --data_path mosei_bottleneck.pkl \
  --exp_dir experiments/stage4_domain_sep

echo ""
echo "🎉 ALL STAGES COMPLETED!"
echo "📊 Final Results: cat experiments/stage4_domain_sep/metrics.txt"
echo "🏆 Expected: 80%+ accuracy"
