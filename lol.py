"""
═══════════════════════════════════════════════════════════════════════════════
  MULTIMODAL EMOTION RECOGNITION — PRODUCTION TRAINING PIPELINE
  Architecture: Cross-Modal Gated Transformer Fusion (CMGTF)
  Dataset:      MELD (pre-extracted GloVe + audio features)
  Target:       Macro F1 ≥ 0.45–0.55
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import pickle
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import math

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE = "/content/MELD.Features.Models/features"

TEXT_FILE  = os.path.join(BASE, "text_glove_average_emotion.pkl")
AUDIO_FILE = os.path.join(BASE, "audio_embeddings_feature_selection_emotion.pkl")
LABEL_FILE = os.path.join(BASE, "data_emotion.p")

SEED          = 42
BATCH_SIZE    = 64
NUM_EPOCHS    = 80
LR            = 3e-4
WEIGHT_DECAY  = 1e-2
GRAD_CLIP     = 1.0
PATIENCE      = 15           # early stopping patience
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model dims
TEXT_DIM      = 300
HIDDEN_DIM    = 256
NUM_HEADS     = 8
NUM_LAYERS    = 3
DROPOUT       = 0.3
NUM_BOTTLENECK = 4           # bottleneck/fusion tokens

TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
# TEST         = 1 - TRAIN - VAL = 0.15

CHECKPOINT    = "best_model.pt"

CLASS_NAMES   = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & ALIGNMENT
# ─────────────────────────────────────────────────────────────────────────────
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def safe_flatten_audio(feat):
    """
    Handles variable-shape audio:
      - 1-D  → return as-is
      - 2-D  → mean-pool over time axis (T×D → D)
      - 3-D  → mean-pool first two dims
    Returns a 1-D numpy array.
    """
    feat = np.array(feat, dtype=np.float32)
    if feat.ndim == 1:
        return feat
    elif feat.ndim == 2:
        return feat.mean(axis=0)
    elif feat.ndim == 3:
        return feat.mean(axis=(0, 1))
    else:
        return feat.flatten()


def build_lookup(data_list, feature_array):
    """
    Returns dict: (dialog, utterance) → feature vector (np.ndarray)
    Assumes data_list order matches feature_array rows.
    """
    lookup = {}
    for idx, meta in enumerate(data_list):
        key = (int(meta["dialog"]), int(meta["utterance"]))
        lookup[key] = feature_array[idx]
    return lookup


def load_and_align():

    print("▶ Loading feature files …")

    text_raw   = load_pickle(TEXT_FILE)
    audio_raw  = load_pickle(AUDIO_FILE)
    label_meta = load_pickle(LABEL_FILE)

    # =========================
    # RAW SPLITS
    # =========================
    train_text, val_text, test_text = text_raw
    train_audio, val_audio, test_audio = audio_raw

    train_meta = label_meta[0]

    print("text train:", len(train_text))
    print("audio train:", len(train_audio))
    print("meta train:", len(train_meta))

    # =========================
    # LABEL MAP
    # =========================
    label_map = {}

    for item in train_meta:

        if not isinstance(item, dict):
            continue

        d = str(item.get("dialog"))
        u = str(item.get("utterance"))
        y = item.get("y")

        if y is None:
            continue

        label_map[(d, u)] = y

    # =========================
    # MERGE ALL SPLITS
    # =========================
    all_text_splits = [
        train_text,
        val_text,
        test_text
    ]

    all_audio_splits = [
        train_audio,
        val_audio,
        test_audio
    ]

    texts = []
    audios = []
    labels_raw = []

    matched = 0

    for split_text, split_audio in zip(all_text_splits, all_audio_splits):

        for key in split_text.keys():

            if key not in split_audio:
                continue

            try:
                d, u = key.split("_")
            except:
                continue

            if (d, u) not in label_map:
                continue

            label = label_map[(d, u)]

            text_feat = np.array(split_text[key], dtype=np.float32)
            audio_feat = np.array(split_audio[key], dtype=np.float32)

            # safety pooling
            if text_feat.ndim == 2:
                text_feat = text_feat.mean(axis=0)

            if audio_feat.ndim == 2:
                audio_feat = audio_feat.mean(axis=0)

            texts.append(text_feat)
            audios.append(audio_feat)
            labels_raw.append(label)

            matched += 1

    print("\nMATCHED:", matched)

    texts = np.stack(texts)
    audios = np.stack(audios)

    # =========================
    # LABEL ENCODER
    # =========================
    le = LabelEncoder()
    labels = le.fit_transform(labels_raw)

    print("TEXT:", texts.shape)
    print("AUDIO:", audios.shape)
    print("LABELS:", labels.shape)

    return texts, audios, labels, le

# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
class MELDDataset(Dataset):
    def __init__(self, texts, audios, labels,
                 text_mean=None, text_std=None,
                 audio_mean=None, audio_std=None,
                 fit=False):
        # Normalise (z-score)
        if fit:
            self.text_mean  = texts.mean(0)
            self.text_std   = texts.std(0)  + 1e-8
            self.audio_mean = audios.mean(0)
            self.audio_std  = audios.std(0) + 1e-8
        else:
            self.text_mean  = text_mean
            self.text_std   = text_std
            self.audio_mean = audio_mean
            self.audio_std  = audio_std

        self.texts  = (texts  - self.text_mean)  / self.text_std
        self.audios = (audios - self.audio_mean) / self.audio_std
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.texts[idx],  dtype=torch.float32),
            torch.tensor(self.audios[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ─────────────────────────────────────────────────────────────────────────────
# FOCAL LOSS
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Multi-class focal loss with optional class weights.
    gamma: focusing parameter (2.0 is standard)
    weight: per-class weight tensor for class imbalance
    """
    def __init__(self, gamma: float = 2.0, weight=None, reduction="mean"):
        super().__init__()
        self.gamma     = gamma
        self.weight    = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets,
                                  weight=self.weight,
                                  reduction="none")
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


# ─────────────────────────────────────────────────────────────────────────────
# MODEL — Cross-Modal Gated Transformer Fusion (CMGTF)
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


class ModalityEncoder(nn.Module):
    """
    Projects a raw modality vector into HIDDEN_DIM space.
    Uses a 2-layer FF with residual + layer-norm to add capacity.
    """
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = DROPOUT):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm  = nn.LayerNorm(hidden_dim)
        self.skip  = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()

    def forward(self, x):
        return self.norm(self.proj(x) + self.skip(x))


class CrossAttentionBlock(nn.Module):
    """
    Standard multi-head cross-attention: query from A, key/value from B.
    Used for bidirectional cross-modal attention.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = DROPOUT):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, num_heads,
                                           dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, query, kv):
        """query: (B, 1, D), kv: (B, 1, D)"""
        attn_out, _ = self.attn(query, kv, kv)
        query = self.norm1(query + attn_out)
        query = self.norm2(query + self.ff(query))
        return query


class GatedFusion(nn.Module):
    """
    Soft gating: learns a sigmoid gate to interpolate two modality vectors.
    g = σ(W·[t; a])
    fused = g·t + (1-g)·a  (plus a learned residual path)
    """
    def __init__(self, d: int, dropout: float = DROPOUT):
        super().__init__()
        self.gate  = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.Sigmoid(),
        )
        self.proj  = nn.Linear(d * 2, d)
        self.norm  = nn.LayerNorm(d)
        self.drop  = nn.Dropout(dropout)

    def forward(self, t, a):
        cat = torch.cat([t, a], dim=-1)
        g   = self.gate(cat)
        out = g * t + (1 - g) * a + self.drop(self.proj(cat))
        return self.norm(out)


class BottleneckFusionLayer(nn.Module):
    """
    MBT-style bottleneck: shared latent tokens attend to each modality,
    then modalities attend to the (updated) bottleneck tokens.
    """
    def __init__(self, d: int, num_heads: int, num_bn: int, dropout: float = DROPOUT):
        super().__init__()
        # Bottleneck ← modality A
        self.bn_attend_t  = CrossAttentionBlock(d, num_heads, dropout)
        # Bottleneck ← modality B
        self.bn_attend_a  = CrossAttentionBlock(d, num_heads, dropout)
        # Bottleneck self-attention
        enc_layer = nn.TransformerEncoderLayer(d, num_heads, d * 4,
                                               dropout=dropout, batch_first=True,
                                               activation="gelu")
        self.bn_self_attn = nn.TransformerEncoder(enc_layer, num_layers=1)

    def forward(self, t_tok, a_tok, bn_tok):
        """
        t_tok : (B, 1, D)
        a_tok : (B, 1, D)
        bn_tok: (B, num_bn, D)
        """
        # Each bottleneck token queries each modality
        bn_from_t = self.bn_attend_t(bn_tok, t_tok)   # (B, num_bn, D)
        bn_from_a = self.bn_attend_a(bn_tok, a_tok)   # (B, num_bn, D)
        bn_tok    = (bn_from_t + bn_from_a) / 2.0
        bn_tok    = self.bn_self_attn(bn_tok)
        return bn_tok


class CMGTF(nn.Module):
    """
    Cross-Modal Gated Transformer Fusion (CMGTF)

    Architecture:
        1. Per-modality encoders   → project to HIDDEN_DIM
        2. N × bottleneck layers   → shared latent tokens fuse information
        3. Bidirectional cross-attention between text & audio tokens
        4. Gated fusion of token outputs
        5. Classifier head with label smoothing support
    """
    def __init__(self,
                 text_dim:   int = TEXT_DIM,
                 audio_dim:  int = 128,        # set dynamically in main()
                 hidden_dim: int = HIDDEN_DIM,
                 num_heads:  int = NUM_HEADS,
                 num_layers: int = NUM_LAYERS,
                 num_bn:     int = NUM_BOTTLENECK,
                 num_classes: int = 7,
                 dropout:    float = DROPOUT):
        super().__init__()

        # ── Modality encoders ────────────────────────────────────────────
        self.text_enc  = ModalityEncoder(text_dim,  hidden_dim, dropout)
        self.audio_enc = ModalityEncoder(audio_dim, hidden_dim, dropout)

        # ── Learnable bottleneck tokens ──────────────────────────────────
        self.bottleneck = nn.Parameter(
            torch.randn(1, num_bn, hidden_dim) * 0.02
        )

        # ── Bottleneck fusion layers ─────────────────────────────────────
        self.bn_layers = nn.ModuleList([
            BottleneckFusionLayer(hidden_dim, num_heads, num_bn, dropout)
            for _ in range(num_layers)
        ])

        # ── Bidirectional cross-attention ────────────────────────────────
        self.t2a = CrossAttentionBlock(hidden_dim, num_heads, dropout)
        self.a2t = CrossAttentionBlock(hidden_dim, num_heads, dropout)

        # ── Gated fusion ─────────────────────────────────────────────────
        self.gate = GatedFusion(hidden_dim, dropout)

        # ── Aggregate bottleneck output ──────────────────────────────────
        self.bn_pool = nn.Linear(num_bn * hidden_dim, hidden_dim)
        self.bn_norm = nn.LayerNorm(hidden_dim)

        # ── Classifier head ──────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),   # gated + bn + text_skip
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, text_feat, audio_feat):
        """
        text_feat  : (B, text_dim)
        audio_feat : (B, audio_dim)
        returns    : logits (B, num_classes)
        """
        B = text_feat.size(0)

        # ── Encode ───────────────────────────────────────────────────────
        t = self.text_enc(text_feat).unsqueeze(1)    # (B, 1, D)
        a = self.audio_enc(audio_feat).unsqueeze(1)  # (B, 1, D)

        # ── Bottleneck fusion ────────────────────────────────────────────
        bn = self.bottleneck.expand(B, -1, -1)       # (B, num_bn, D)
        for layer in self.bn_layers:
            bn = layer(t, a, bn)

        # ── Bidirectional cross-attention ────────────────────────────────
        t_ca = self.t2a(t, a)   # text attends to audio
        a_ca = self.a2t(a, t)   # audio attends to text

        # ── Gated fusion of cross-attended tokens ────────────────────────
        fused = self.gate(t_ca.squeeze(1), a_ca.squeeze(1))  # (B, D)

        # ── Pool bottleneck tokens ───────────────────────────────────────
        bn_flat  = bn.reshape(B, -1)                          # (B, num_bn*D)
        bn_out   = self.bn_norm(self.bn_pool(bn_flat))        # (B, D)

        # ── Text residual (bypass) ───────────────────────────────────────
        t_res    = t.squeeze(1)

        # ── Classify ─────────────────────────────────────────────────────
        combined = torch.cat([fused, bn_out, t_res], dim=-1) # (B, 3D)
        return self.classifier(combined)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(all_labels, all_preds, class_names, prefix=""):
    macro_f1    = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    per_class   = f1_score(all_labels, all_preds, average=None,       zero_division=0)

    tag = f"[{prefix}] " if prefix else ""
    print(f"\n{tag}Macro F1    : {macro_f1:.4f}  ← primary metric")
    print(f"{tag}Weighted F1 : {weighted_f1:.4f}")
    print(f"{tag}Per-class F1:")
    for name, score in zip(class_names, per_class):
        bar = "█" * int(score * 30)
        print(f"    {name:<10}: {score:.4f}  {bar}")

    return macro_f1, weighted_f1


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / EVAL LOOPS
# ─────────────────────────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, scheduler, criterion, device, train=True):
    model.train() if train else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for text_x, audio_x, labels in loader:
            text_x  = text_x.to(device)
            audio_x = audio_x.to(device)
            labels  = labels.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(text_x, audio_x)
            loss   = criterion(logits, labels)

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if train and scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / len(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1, all_labels, all_preds


# ─────────────────────────────────────────────────────────────────────────────
# CLASS WEIGHT COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
def compute_class_weights(labels, num_classes, device):
    """Inverse-frequency class weights, clipped to [0.5, 10]."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    weights = np.clip(weights, 0.5, 10.0)
    print(f"\n   Class weights: {dict(zip(CLASS_NAMES, weights.round(3)))}")
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHTED SAMPLER (over-sample minority classes in every epoch)
# ─────────────────────────────────────────────────────────────────────────────
def make_sampler(labels, num_classes):
    counts  = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts  = np.maximum(counts, 1.0)
    w_per_class = 1.0 / counts
    sample_weights = torch.tensor(w_per_class[labels], dtype=torch.float64)
    return WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  MULTIMODAL EMOTION RECOGNITION  —  CMGTF Pipeline")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # ── Load & align data ─────────────────────────────────────────────────
    texts, audios, labels, le = load_and_align()
    num_classes  = len(le.classes_)
    class_names  = list(le.classes_)
    audio_dim    = audios.shape[1]
    N            = len(labels)

    print(f"\n   N={N}, text_dim={texts.shape[1]}, audio_dim={audio_dim}, classes={num_classes}")

    # ── Stratified split ──────────────────────────────────────────────────
    from sklearn.model_selection import train_test_split

    idx = np.arange(N)
    train_idx, temp_idx = train_test_split(idx, test_size=1 - TRAIN_RATIO,
                                           stratify=labels, random_state=SEED)
    val_size_adjusted   = VAL_RATIO / (1 - TRAIN_RATIO)
    val_idx, test_idx   = train_test_split(temp_idx, test_size=1 - val_size_adjusted,
                                           stratify=labels[temp_idx], random_state=SEED)

    print(f"\n   Split → train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

    # ── Datasets ──────────────────────────────────────────────────────────
    train_ds = MELDDataset(texts[train_idx], audios[train_idx], labels[train_idx], fit=True)
    val_ds   = MELDDataset(texts[val_idx],   audios[val_idx],   labels[val_idx],
                           text_mean=train_ds.text_mean,   text_std=train_ds.text_std,
                           audio_mean=train_ds.audio_mean, audio_std=train_ds.audio_std)
    test_ds  = MELDDataset(texts[test_idx],  audios[test_idx],  labels[test_idx],
                           text_mean=train_ds.text_mean,   text_std=train_ds.text_std,
                           audio_mean=train_ds.audio_mean, audio_std=train_ds.audio_std)

    # ── DataLoaders (WeightedRandomSampler for training) ──────────────────
    sampler    = make_sampler(labels[train_idx], num_classes)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model = CMGTF(
        text_dim=texts.shape[1],
        audio_dim=audio_dim,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        num_bn=NUM_BOTTLENECK,
        num_classes=num_classes,
        dropout=DROPOUT,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n   Model parameters: {total_params:,}")

    # ── Loss: Focal + class weights ───────────────────────────────────────
    class_weights = compute_class_weights(labels[train_idx], num_classes, DEVICE)
    criterion     = FocalLoss(gamma=2.0, weight=class_weights)

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LR * 0.01)

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_f1    = 0.0
    patience_count = 0

    print("\n" + "=" * 70)
    print(f"{'Ep':>4}  {'Train Loss':>11}  {'Train F1':>9}  {'Val Loss':>9}  {'Val F1':>8}  {'★':>2}")
    print("=" * 70)

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_f1, _, _          = run_epoch(model, train_loader, optimizer,
                                                   scheduler, criterion, DEVICE, train=True)
        va_loss, va_f1, va_lbl, va_pr = run_epoch(model, val_loader, None,
                                                   None, criterion, DEVICE, train=False)

        star = ""
        if va_f1 > best_val_f1:
            best_val_f1    = va_f1
            patience_count = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_macro_f1": va_f1,
                "text_mean":   train_ds.text_mean,
                "text_std":    train_ds.text_std,
                "audio_mean":  train_ds.audio_mean,
                "audio_std":   train_ds.audio_std,
                "le_classes":  le.classes_,
            }, CHECKPOINT)
            star = "★"
        else:
            patience_count += 1

        print(f"{epoch:>4}  {tr_loss:>11.4f}  {tr_f1:>9.4f}  {va_loss:>9.4f}  "
              f"{va_f1:>8.4f}  {star:>2}")

        # Verbose per-class every 10 epochs
        if epoch % 10 == 0:
            compute_metrics(va_lbl, va_pr, class_names, prefix=f"Val Ep{epoch}")

        if patience_count >= PATIENCE:
            print(f"\n⏹  Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

    # ── Test evaluation ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL TEST EVALUATION")
    print("=" * 70)

    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"   Loaded best checkpoint from epoch {ckpt['epoch']} "
          f"(val macro F1 = {ckpt['val_macro_f1']:.4f})")

    _, _, te_lbl, te_pr = run_epoch(model, test_loader, None, None,
                                     criterion, DEVICE, train=False)
    macro_f1, weighted_f1 = compute_metrics(te_lbl, te_pr, class_names, prefix="TEST")

    print("\n" + "─" * 70)
    print("  Classification Report (TEST):")
    print("─" * 70)
    print(classification_report(te_lbl, te_pr, target_names=class_names, zero_division=0))
    print("=" * 70)
    print(f"  ✔  Final Test Macro F1    : {macro_f1:.4f}")
    print(f"  ✔  Final Test Weighted F1 : {weighted_f1:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()