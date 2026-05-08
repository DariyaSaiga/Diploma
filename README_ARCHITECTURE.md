# 🏗️ Project Architecture

Clean and modular structure for multimodal emotion recognition.

```
diploma/
├── core/                    # Model architecture
│   ├── __init__.py
│   └── bottleneck_fusion.py # Domain-separated bottleneck model
│
├── data/                    # Data loading & processing
│   ├── __init__.py
│   └── dataset.py          # Dataset class + loaders
│
├── training/               # Training logic
│   ├── __init__.py
│   ├── train.py            # Main training loop
│   └── losses.py           # Loss functions (separation, invariant, reconstruction)
│
├── utils/                  # Utilities
│   ├── __init__.py
│   └── utils.py            # Device setup, seeding
│
├── scripts/                # Executable scripts
│   ├── check_forward.py    # Verify model & data
│   └── test_loader.py      # Test data loading
│
├── experiments/            # Results & checkpoints
│   ├── stage1_domain_sep/
│   │   ├── best_model.pt
│   │   ├── metrics.txt
│   │   └── config.txt
│   ├── stage2_domain_sep/
│   └── ...
│
├── run_stage1.sh           # Training scripts
├── run_stage2.sh
├── run_stage3.sh
├── run_stage4.sh
│
└── COLAB_INSTRUCTIONS.md   # Colab setup guide
```

---

## Core Modules

### `core/bottleneck_fusion.py` (165 lines)
**Domain-Separated Bottleneck model**

Main components:
- **Input encoders**: BERT(text), Linear(audio), Linear(visual)
- **Domain separation**: Invariant (shared) + Private (per-modality) encoders
- **Bottleneck tokens**: 16 learnable tokens as information gateways
- **Cross-attention**: Private → Bottleneck tokens → Refined private
- **Reconstruction heads**: Recover original features from [inv||priv]
- **Fusion head**: Concatenate all domains + classify

Methods:
- `forward()`: Standard inference
- `forward(..., return_domains=True)`: Returns domain info for loss calculation

### `data/dataset.py` (100 lines)
**Dataset and DataLoaders**

Classes:
- `BottleneckDataset`: Handles text/audio/visual with dynamic padding + masks
- `collate_fn()`: Custom collate for batching
- `make_bottleneck_loaders()`: Creates train/val/test loaders

Features:
- Bool masks for attention (1=real, 0=padding)
- Automatic padding to fixed lengths
- BertTokenizer integration

### `training/losses.py` (50 lines)
**Loss functions for domain separation**

Functions:
- `compute_separation_loss()`: Minimize cosine similarity between invariant/private
- `compute_invariant_loss()`: Pull same-emotion representations
- `compute_reconstruction_loss()`: Reconstruct original features

Usage in train loop:
```python
loss = L_task + 0.1*L_sep + 0.05*L_inv + 0.01*L_rec
```

### `training/train.py` (250 lines)
**Main training loop**

Functions:
- `train_epoch()`: Train one epoch with optional multi-loss
- `evaluate()`: Validation/test evaluation
- `build_model()`: Create model from args
- `save_results()`: Save metrics & report
- `main()`: Entry point with argparse

Features:
- Early stopping with patience
- Model checkpointing (best F1)
- Class-weighted loss
- Support for domain separation training
- Pretrained weight loading for stage-wise training

### `utils/utils.py` (15 lines)
**Utilities**

- `device`: Auto-detect CUDA vs CPU
- `set_seed()`: Reproducible random initialization

---

## Usage Examples

### 1. Verify setup
```bash
python scripts/check_forward.py --data_path mosei_bottleneck.pkl
```

### 2. Stage 1 training
```bash
python training/train.py \
  --epochs 20 \
  --lr 1e-3 \
  --use_domain_sep \
  --alpha_sep 0.1 \
  --alpha_inv 0.05 \
  --alpha_rec 0.01 \
  --data_path mosei_bottleneck.pkl \
  --exp_dir experiments/stage1_domain_sep
```

### 3. Stage 2 training (continue from Stage 1)
```bash
python training/train.py \
  --epochs 20 \
  --lr 3e-4 \
  --use_domain_sep \
  --pretrained_path experiments/stage1_domain_sep/best_model.pt \
  --data_path mosei_bottleneck.pkl \
  --exp_dir experiments/stage2_domain_sep
```

---

## Key Design Decisions

### ✨ Modularity
- **One responsibility per file**
  - `bottleneck_fusion.py`: Only model architecture
  - `dataset.py`: Only data loading
  - `losses.py`: Only loss computation
  - `train.py`: Only training loop

### 📚 Imports
Clean import paths:
```python
from core.bottleneck_fusion import BottleneckFusion
from data.dataset import make_bottleneck_loaders
from training.losses import compute_separation_loss
from utils.utils import device
```

### 🔄 Flexibility
Easy to change:
- Model: Modify only `core/bottleneck_fusion.py`
- Data: Modify only `data/dataset.py`
- Losses: Modify only `training/losses.py`
- Training: Modify only `training/train.py`

### 📊 Reproducibility
- `set_seed(42)` in train.py
- Class weights from training data
- All hyperparams as command-line args
- Metrics saved per experiment

---

## File Sizes (After Cleanup)

| File | Before | After | Reduction |
|------|--------|-------|-----------|
| bottleneck_fusion.py | 347 | 165 | 53% ↓ |
| train.py | 642 | 250 | 61% ↓ |
| dataset.py | 245 | 100 | 59% ↓ |
| **Total** | **1700** | **700** | **59% ↓** |

---

## Adding Features

### Add new loss function
1. Open `training/losses.py`
2. Add function: `def compute_new_loss():`
3. Import in `training/train.py`
4. Use in `train_epoch()`

### Add hyperparameter
1. Open `training/train.py`
2. Add to argparse: `parser.add_argument("--new_param", ...)`
3. Pass to `train_epoch(..., new_param=args.new_param)`

### Change model architecture
1. Open `core/bottleneck_fusion.py`
2. Modify `__init__()` or `forward()`
3. No other files need to change!

### Change dataset format
1. Open `data/dataset.py`
2. Modify `__getitem__()` or `__init__()`
3. Update `collate_fn()` if needed

---

## Next Steps

1. **Verify setup**: `python scripts/check_forward.py`
2. **Stage 1**: `bash run_stage1.sh` (40 min)
3. **Stage 2-4**: Follow stage scripts (90 min total)
4. **Expected result**: 80%+ accuracy on CMU-MOSEI

🚀 Ready to train!
