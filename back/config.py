"""
config.py — Centralized configuration for Emotion Recognition Service.
"""

from pathlib import Path
from typing import List
import torch


# ── Model ────────────────────────────────────────────────────────────────────
MODEL_PATH: str = "best_model.pt"

EMOTIONS: List[str] = ["happy", "sad", "anger", "surprise", "disgust", "fear"]
NUM_CLASSES: int = len(EMOTIONS)

# BottleneckFusionModel.__init__ takes NO arguments — all hyperparams are
# hard-coded as module-level constants in models.py (HIDDEN_DIM, N_HEADS, etc.)
# Text sequence length used by BERT (must match dataset tokenizer max_length)
TEXT_SEQ_LEN: int = 50

# ── Device ───────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE: torch.device = get_device()

# ── Image preprocessing ───────────────────────────────────────────────────────
IMAGE_SIZE: int = 224
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float]  = [0.229, 0.224, 0.225]

# ── Video sampling ───────────────────────────────────────────────────────────
MAX_FRAMES: int = 30

# ── Dummy audio/vision tensors (inference without real streams) ───────────────
AUDIO_SEQ_LEN: int = 60
AUDIO_FEAT_DIM: int = 74
VISION_SEQ_LEN: int = 60
VISION_FEAT_DIM: int = 35

# ── API ──────────────────────────────────────────────────────────────────────
API_VERSION: str = "1.0.0"
CORS_ORIGINS: List[str] = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]