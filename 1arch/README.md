# Multimodal Emotion Recognition on CMU-MOSEI (multi-label)

Diploma project comparing three multimodal architectures on **feature-aligned
CMU-MOSEI** (`mosei_emotion_aligned_60.pkl`).

> **Task:** multi-label emotion recognition over 6 classes
> (`happy, sad, anger, surprise, disgust, fear`) — a sample can carry several
> labels at once, e.g. `[1, 1, 0, 0, 0, 1]`.

---

## Why no BERT

The dataset stores **GloVe-like word embeddings** (`text shape = [N, 60, 300]`),
not raw text. BERT requires raw token IDs, and you cannot recover token IDs
from continuous embeddings. We therefore use a **feature-based** setup —
exactly the format used in MulT and the broader CMU-SDK literature:

| Modality | Encoder of features | Dim |
|----------|---------------------|----:|
| Text     | GloVe word embeddings | 300 |
| Audio    | COVAREP               |  74 |
| Vision   | Facet (FACS-like AUs) |  35 |

`scripts/check_bert_possible.py` already verified this for the file shipped
with this repo.

---

## Project layout

```
Diploma/
├── datasets/
│   └── mosei_emotion_aligned_60.pkl       # symlink to the actual file (4.49 GB)
├── data/
│   ├── __init__.py
│   └── mosei_multilabel_dataset.py        # loader + DataLoader factory
├── models/
│   ├── __init__.py
│   ├── common_layers.py                   # shared blocks
│   ├── late_fusion_cnn_bilstm.py          # Model 1
│   ├── mult_cross_attention.py            # Model 2
│   └── bottleneck_fusion.py               # Model 3 (proposed)
├── training/
│   ├── __init__.py
│   ├── losses.py                          # BCE + pos_weight + aux losses
│   ├── metrics.py                         # multi-label F1, hamming, ...
│   ├── train.py                           # main training script
│   └── utils.py                           # seed, device, logging
├── experiments/
│   ├── exp01_late_fusion/                 # auto-populated outputs
│   ├── exp02_mult_attention/
│   └── exp03_bottleneck/
├── scripts/
│   ├── check_forward.py                   # forward+backward sanity check
│   ├── compare_results.py                 # build comparison table
│   ├── inspect_dataset.py                 # explore the .pkl
│   ├── run_exp01_late_fusion.sh
│   ├── run_exp02_mult_attention.sh
│   └── run_exp03_bottleneck.sh
├── requirements.txt
└── README.md
```

The legacy BERT-based files in `core/`, `data/dataset.py`, `diploma_utils/`
and old `experiments/run_*.sh` are kept around for reference but are **not
used** by the new pipeline. They expect raw text and single-label data.

---

## Three models

### 1. `late_fusion` — Late Fusion CNN-BiLSTM (baseline)

```
text  [B,60,300] → BiLSTM(128, bidir) → masked mean →  [B,256]
audio [B,60, 74] → Conv1d(74→128)→Conv1d(128→128) → mean →  [B,128]
vision[B,60, 35] → BiLSTM(128, bidir) → masked mean →  [B,256]
concat([text, audio, vision])  →  FC(640→256) → ReLU → Dropout → FC(256→6)
```

### 2. `mult` — MulT-like Cross-Modal Attention

Each modality is projected to `hidden_dim` (default 64), then refined via:

* per-modality self-attention encoder (`num_layers=2`, `num_heads=4`),
* six pairwise cross-attention streams (`T←A, T←V, A←T, A←V, V←T, V←A`),
* mean pooling per modality, concat, classifier `[B, hidden*3] → 128 → 6`.

### 3. `bottleneck` — Bottleneck Fusion (proposed)

* Each modality is projected to `hidden_dim=128` and processed by a
  self-attention encoder.
* A bank of **learnable bottleneck tokens** (default `16`) mediates *all*
  cross-modal information exchange — modalities never directly attend to
  each other's tokens.
* Two layers of `BottleneckBlock`: bottleneck collects info from each
  modality, then each modality reads back from the updated bottleneck.
* Classifier consumes `[ pool(text) | pool(audio) | pool(vision) | pool(BN) ]`.

#### Optional domain separation (`--use_domain_sep`)

Each modality is split into:

* **invariant** branch — aligned across modalities,
* **private** branch — modality-specific, the bottleneck operates on this branch.

Three auxiliary losses:

| Loss            | Meaning                                             |
|-----------------|-----------------------------------------------------|
| `L_sep`         | invariant ⊥ private per modality (cosine² → 0)      |
| `L_inv`         | MSE alignment between modalities' invariant pools   |
| `L_rec`         | reconstruct original pooled features from `[inv,priv]` |

Total loss: `L_BCE + α_sep · L_sep + α_inv · L_inv + α_rec · L_rec`.

Defaults are intentionally **soft** (`0.03 / 0.005 / 0.001`) — aggressive
weights cause overfitting on this dataset.

---

## Loss & class imbalance

Target distribution (train split): happy 53.5%, sad 26.1%, anger 21.6%,
disgust 18.1%, surprise 10.1%, fear 8.2%.

`--pos_weight_mode {none, balanced, sqrt}` (default `sqrt`):

* `none` — plain `BCEWithLogitsLoss`.
* `balanced` — `pos_weight = neg / pos` per class.
* `sqrt` — `pos_weight = sqrt(neg / pos)` then re-scaled so `mean = 1.0`. This
  is the recommended default: it pulls up rare classes (`fear`, `surprise`)
  without destabilising training.

---

## Metrics (multi-label)

Implemented in `training/metrics.py`. After every epoch we log both
`weighted_f1` (default monitor) and `macro_f1` for monitoring rare classes.
Final test report includes:

* `micro_f1`, `macro_f1`, `weighted_f1`, `samples_f1`,
* `hamming_loss`, `subset_accuracy`,
* per-class precision / recall / F1.

Use `--threshold_search` to pick the best global threshold from
`[0.30, 0.35, 0.40, 0.45, 0.50]` on the validation set.

> **Never** use `argmax` or `CrossEntropyLoss` here — labels are multi-label
> binary vectors. Predictions = `sigmoid(logits) >= threshold`.

---

## Quick start (local / VS Code)

```bash
# 1. Install
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Make sure the dataset is reachable
ls -lh datasets/mosei_emotion_aligned_60.pkl   # symlink to the .pkl in the repo root

# 3. Sanity check (no real data, ~1 sec)
PYTHONPATH=. python3 scripts/check_forward.py --model all

# 4. Inspect the actual file
PYTHONPATH=. python3 scripts/inspect_dataset.py --path datasets/mosei_emotion_aligned_60.pkl

# 5. Train
bash scripts/run_exp01_late_fusion.sh
bash scripts/run_exp02_mult_attention.sh
bash scripts/run_exp03_bottleneck.sh

# 6. Compare
PYTHONPATH=. python3 scripts/compare_results.py
```

---

## Quick start (Google Colab)

```python
# 1. Mount Drive and put mosei_emotion_aligned_60.pkl somewhere persistent.
from google.colab import drive; drive.mount("/content/drive")

# 2. Clone the repo
!git clone <your-repo-url> /content/Diploma
%cd /content/Diploma
!pip install -q -r requirements.txt

# 3. Symlink the dataset (do NOT copy the 4.49 GB file)
!mkdir -p datasets
!ln -sf /content/drive/MyDrive/path/to/mosei_emotion_aligned_60.pkl datasets/

# 4. Forward sanity check
!PYTHONPATH=. python scripts/check_forward.py --model all

# 5. Train (use a GPU runtime). Bottleneck is the most demanding.
!bash scripts/run_exp01_late_fusion.sh
!bash scripts/run_exp02_mult_attention.sh
!bash scripts/run_exp03_bottleneck.sh

# 6. Compare
!PYTHONPATH=. python scripts/compare_results.py
```

---

## What gets saved per experiment

```
experiments/exp03_bottleneck/
├── best_model.pt          # checkpoint with best monitor metric on validation
├── last_checkpoint.pt     # last-epoch full state
├── config.json            # all CLI args
├── config.txt             # human-readable form
├── train_log.txt          # per-epoch line of losses + metrics
├── metrics.txt            # validation + test metrics + per-class report
├── metrics.json           # same data, machine-readable
└── per_class_report.txt   # convenience copy of the per-class table
```

---

## How to verify everything works

1. **Module-level smoke test** — passes on every model:
   ```bash
   PYTHONPATH=. python3 scripts/check_forward.py --model all
   ```
2. **Tiny real-data run** to make sure the loader + train loop hold:
   ```bash
   PYTHONPATH=. python3 training/train.py \
       --model late_fusion \
       --data_path datasets/mosei_emotion_aligned_60.pkl \
       --exp_dir experiments/_smoke \
       --epochs 1 --batch_size 32 --patience 1
   ```
3. **Compare** after at least one experiment finishes:
   ```bash
   PYTHONPATH=. python3 scripts/compare_results.py
   cat experiments/comparison_summary.csv
   ```

---

## Troubleshooting

**The first model gives a bad result.** Try in this order:
1. Check that `--pos_weight_mode sqrt` is enabled. Without it `fear` and
   `surprise` collapse.
2. Run with `--threshold_search` and look at `metrics.txt` — sometimes the
   network is fine and only the global threshold needs nudging.
3. Try `--normalize` (per-feature train mean/std). Default off because the
   features in the shipped pickle are already roughly standardised.
4. Reduce capacity: `--hidden_dim 64 --num_layers 1` for `mult` /
   `bottleneck` if you see overfitting on a small validation set.
5. For `bottleneck`, run **without** `--use_domain_sep` first to check the
   base architecture; turn it on once the baseline numbers are stable.
6. Lower `--lr` by 2× if loss is unstable; raise `--patience` by 2 if you
   see late convergence.

**Out of memory.** Drop `--batch_size` to 8 (or 16 for `mult`/`late_fusion`),
or set `--num_workers 0`.

**Apple Silicon (MPS).** `nn.LSTM` is numerically unstable on PyTorch's MPS
backend — the loss can blow up to `±1e20`. The training script defaults to
CPU on macOS for that reason. To opt in:
`--device mps` (only fast for `mult` / `bottleneck`, which are
attention-only). For best speed, train on CUDA in Colab.

---

## Risks / non-goals

* This pipeline is **not chasing SOTA**. The goal is a clean, runnable,
  reproducible comparison of three architectures.
* **Multi-label**: never use `CrossEntropyLoss` / `argmax` — the loss must be
  `BCEWithLogitsLoss` and predictions are thresholded sigmoid scores.
* **Subset accuracy** is intentionally a strict, hard-to-improve metric and
  should be read as an upper-bound indicator only — `weighted_f1` is the
  primary number to report.
