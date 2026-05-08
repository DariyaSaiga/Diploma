# 🎓 Diploma: Multimodal Emotion Recognition with Domain-Separated Bottleneck

**Status**: 🚀 Ready for training | **Expected Accuracy**: 80%+ | **Time**: ~2 hours (4 stages)

---

## 📁 Clean Project Structure

```
diploma/
├── core/                   ← Model architecture (CLEAN)
│   └── bottleneck_fusion.py (165 lines)
├── data/                   ← Data loading (CLEAN)
│   └── dataset.py (100 lines)
├── training/               ← Training logic (CLEAN)
│   ├── train.py (250 lines)
│   └── losses.py (50 lines)
├── utils/                  ← Utilities
│   └── utils.py (15 lines)
├── scripts/                ← Verification scripts
│   └── check_forward.py
├── docs/                   ← Documentation
├── experiments/            ← Results (auto-created)
├── .archive/               ← Old files (not needed)
├── run_stage1.sh           ← Training scripts
├── run_stage2.sh
├── run_stage3.sh
├── run_stage4.sh
└── mosei_bottleneck.pkl    ← Dataset
```

---

## 🚀 Quick Start

### Verify setup (1 min)
```bash
python scripts/check_forward.py --data_path mosei_bottleneck.pkl
```

### Train all 4 stages (~2 hours)
```bash
bash run_stage1.sh    # 40 min → 74-76% accuracy
bash run_stage2.sh    # 40 min → 76-78% accuracy
bash run_stage3.sh    # 30 min → 77-79% accuracy
bash run_stage4.sh    # 20 min → 79-82% accuracy ✨
```

### Check results
```bash
cat experiments/stage4_domain_sep/metrics.txt
```

---

## 📖 Documentation

| Guide | What |
|-------|------|
| [`docs/QUICK_START.md`](docs/QUICK_START.md) | 5-min setup |
| [`docs/README_ARCHITECTURE.md`](docs/README_ARCHITECTURE.md) | Full reference |
| [`docs/CLEANUP_SUMMARY.md`](docs/CLEANUP_SUMMARY.md) | What changed |
| [`docs/COLAB_CLEAN.py`](docs/COLAB_CLEAN.py) | Google Colab |

---

## ✨ Code Quality (59% Reduction)

| Metric | Before | After |
|--------|--------|-------|
| Lines | 1700 | 700 |
| Modularity | ❌ Mixed | ✅ Clean |
| Readability | ❌ Cluttered | ✅ Clear |
| Maintainability | ❌ Hard | ✅ Easy |

---

## 🎯 Start Training!

```bash
bash run_stage1.sh
```
