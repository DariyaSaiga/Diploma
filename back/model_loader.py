"""
model_loader.py — Safe loader for BottleneckFusionModel.

Key facts from models.py (DO NOT CHANGE):
──────────────────────────────────────────
• BottleneckFusionModel.__init__() takes NO arguments.
• forward() signature:
    (input_ids, attention_mask, audio, vision, audio_mask, vision_mask)
• forward() returns a TUPLE, not a dict:
    (logits_fuse, logits_text, logits_audio, logits_vision)
• audio  shape: [B, 60, 74]
• vision shape: [B, 60, 35]
• text   shape: [B, 50]  — matches dataset tokenizer max_length=50
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import os
from huggingface_hub import hf_hub_download
import numpy as np
import torch
import torch.nn as nn

from config import (
    AUDIO_FEAT_DIM,
    AUDIO_SEQ_LEN,
    DEVICE,
    EMOTIONS,
    MODEL_PATH,
    NUM_CLASSES,
    TEXT_SEQ_LEN,
    VISION_FEAT_DIM,
    VISION_SEQ_LEN,
)

logger = logging.getLogger(__name__)

# ── Attempt to import the real model ─────────────────────────────────────────
_model_import_ok: bool = False
_model_import_error: Optional[str] = None

try:
    from models import BottleneckFusionModel  # type: ignore
    _model_import_ok = True
except Exception as _exc:
    _model_import_error = str(_exc)
    logger.warning(
        "Could not import BottleneckFusionModel: %s — mock mode will be active.",
        _model_import_error,
    )

# ── Global state ──────────────────────────────────────────────────────────────
_model: Optional[nn.Module] = None
_mock_mode: bool = False
_load_info: Dict = {}

MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", "DariyaSaiga/multimood-best-model")
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "best_model.pt")


def get_model_path() -> Path:
    """
    Return local model path if available.
    Otherwise download best_model.pt from Hugging Face Hub.
    """
    local_path = Path(MODEL_FILENAME)

    if local_path.exists():
        logger.info("Using local model weights: %s", local_path)
        return local_path

    logger.info(
        "Local model weights not found. Downloading '%s' from Hugging Face repo '%s'...",
        MODEL_FILENAME,
        MODEL_REPO_ID,
    )

    downloaded_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILENAME,
    )

    logger.info("Model downloaded to: %s", downloaded_path)
    return Path(downloaded_path)

# ── Mock fallback ──────────────────────────────────────────────────────────────
class _MockModel(nn.Module):
    """Returns random logits — used when real weights are unavailable."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, **kwargs) -> Tuple[torch.Tensor, ...]:
        logits = torch.randn(1, self.num_classes)
        return logits, logits, logits, logits  # (fuse, text, audio, vision)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _looks_like_state_dict(d: dict) -> bool:
    return any(isinstance(v, torch.Tensor) for v in d.values())


def _set_mock(reason: str) -> None:
    global _model, _mock_mode, _load_info
    _model = _MockModel(NUM_CLASSES).to(DEVICE).eval()
    _mock_mode = True
    _load_info = {"status": "mock", "reason": reason}
    logger.warning("Mock mode active: %s", reason)


# ── Public loader ─────────────────────────────────────────────────────────────
def load_model() -> None:
    """
    Load BottleneckFusionModel weights from MODEL_PATH.

    Falls back to _MockModel (random predictions) on ANY failure.
    Never raises — logs everything instead.
    """
    global _model, _mock_mode, _load_info

    # Step 1: import available?
    if not _model_import_ok:
        _set_mock(f"models.py import failed: {_model_import_error}")
        return

    # Step 2: resolve weights path
    try:
        pt_path = get_model_path()
    except Exception as exc:
        _set_mock(f"weights download failed: {exc}")
        logger.error("Failed to resolve/download model weights", exc_info=True)
        return

    # Step 3: instantiate — NO constructor arguments
    try:
        real_model: nn.Module = BottleneckFusionModel()
    except Exception as exc:
        _set_mock(f"BottleneckFusionModel() init failed: {exc}")
        logger.error("Model init error", exc_info=True)
        return

    # Step 4: load checkpoint
    try:
        checkpoint = torch.load(pt_path, map_location=DEVICE, weights_only=False)

        # Normalise checkpoint format
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            meta = {k: str(v) for k, v in checkpoint.items() if k != "model_state_dict"}
        elif isinstance(checkpoint, dict) and not _looks_like_state_dict(checkpoint):
            state_dict = checkpoint.get("state_dict", checkpoint)
            meta = {}
        else:
            state_dict = checkpoint
            meta = {}

        load_result = real_model.load_state_dict(state_dict, strict=False)

        missing    = load_result.missing_keys
        unexpected = load_result.unexpected_keys

        if missing:
            logger.warning(
                "load_state_dict: %d missing key(s) "
                "(auxiliary heads may not have been saved — safe):\n  %s",
                len(missing), "\n  ".join(missing[:20]),
            )
        if unexpected:
            logger.warning(
                "load_state_dict: %d unexpected key(s) — ignored:\n  %s",
                len(unexpected), "\n  ".join(unexpected[:20]),
            )

        real_model.to(DEVICE).eval()
        _model     = real_model
        _mock_mode = False
        _load_info = {
            "status":          "loaded",
            "path":            str(pt_path.resolve()),
            "device":          str(DEVICE),
            "missing_keys":    len(missing),
            "unexpected_keys": len(unexpected),
            "checkpoint_meta": meta,
        }
        logger.info("Model loaded from '%s' on %s", pt_path, DEVICE)

    except Exception as exc:
        _set_mock(f"load error: {exc}")
        logger.error("Failed to load weights from '%s'", pt_path, exc_info=True)


def is_mock() -> bool:
    return _mock_mode


def get_load_info() -> Dict:
    return _load_info


# ── Dummy tensors for missing modalities ──────────────────────────────────────
def _dummy_audio(batch: int = 1) -> torch.Tensor:
    return torch.zeros(batch, AUDIO_SEQ_LEN, AUDIO_FEAT_DIM, device=DEVICE)


def _dummy_vision(batch: int = 1) -> torch.Tensor:
    return torch.zeros(batch, VISION_SEQ_LEN, VISION_FEAT_DIM, device=DEVICE)


def _dummy_text(batch: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """seq_len=50 matches dataset tokenizer max_length from training."""
    ids  = torch.zeros(batch, TEXT_SEQ_LEN, dtype=torch.long, device=DEVICE)
    mask = torch.zeros(batch, TEXT_SEQ_LEN, dtype=torch.long, device=DEVICE)
    return ids, mask


def _dummy_av_mask(batch: int = 1, seq_len: int = 60) -> torch.Tensor:
    """Ones = all positions valid."""
    return torch.ones(batch, seq_len, dtype=torch.long, device=DEVICE)


# ── Inference entry-point ─────────────────────────────────────────────────────
@torch.inference_mode()
def run_inference(
    *,
    image_tensor: Optional[torch.Tensor] = None,   # API compat — not used by model
    input_ids: Optional[torch.Tensor] = None,       # [1, 50]
    attention_mask: Optional[torch.Tensor] = None,  # [1, 50]
    audio: Optional[torch.Tensor] = None,            # [1, 60, 74]
    vision_feats: Optional[torch.Tensor] = None,     # [1, 60, 35]
    audio_mask: Optional[torch.Tensor] = None,       # [1, 60]
    vision_mask: Optional[torch.Tensor] = None,      # [1, 60]
) -> Dict:
    """
    Run one forward pass and return emotion prediction.

    NOTE: image_tensor is accepted for REST API compatibility but is NOT used by
    BottleneckFusionModel — it operates on pre-extracted COVAREP/OpenFace
    features (audio/vision_feats), not raw pixels.  Pass real features when
    available; zero-filled dummies are used otherwise.

    Returns
    ───────
    {"emotion": str, "confidence": float (0-100), "probabilities": {str: float}}
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    batch = 1

    if input_ids is None or attention_mask is None:
        input_ids, attention_mask = _dummy_text(batch)
    else:
        input_ids      = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)

    if audio is None:
        audio = _dummy_audio(batch)
    else:
        audio = audio.to(DEVICE)

    if audio_mask is None:
        audio_mask = _dummy_av_mask(batch, AUDIO_SEQ_LEN)
    else:
        audio_mask = audio_mask.to(DEVICE)

    if vision_feats is None:
        vision_feats = _dummy_vision(batch)
    else:
        vision_feats = vision_feats.to(DEVICE)

    if vision_mask is None:
        vision_mask = _dummy_av_mask(batch, VISION_SEQ_LEN)
    else:
        vision_mask = vision_mask.to(DEVICE)

    # ── Forward ───────────────────────────────────────────────────────────────
    if _mock_mode:
        outputs = _model()  # _MockModel ignores all args
    else:
        try:
            outputs = _model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio=audio,
                vision=vision_feats,
                audio_mask=audio_mask,
                vision_mask=vision_mask,
            )
        except Exception as exc:
            logger.error("Forward pass failed: %s", exc, exc_info=True)
            raise

    # ── Decode ── forward() returns TUPLE (fuse, text, audio, vision) ─────────
    # ── Decode ─────────────────────────────────────────────────────────────
    logits_fuse = outputs[0]

    if torch.isnan(logits_fuse).any() or torch.isinf(logits_fuse).any():
        logger.error("NaN/Inf detected in logits")

        logits_fuse = torch.nan_to_num(
            logits_fuse,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

    # Для multiclass classification
    probs = torch.softmax(logits_fuse, dim=-1)[0]

    # tensor → numpy
    probs_np = probs.detach().cpu().float().numpy()

    # защита от NaN/Inf
    probs_np = np.nan_to_num(
        probs_np,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    # top prediction
    top_idx = int(np.argmax(probs_np))
    top_label = EMOTIONS[top_idx]

    confidence = float(probs_np[top_idx]) * 100.0

    # финальная защита
    if not np.isfinite(confidence):
        confidence = 0.0

    return {
        "emotion": top_label,
        "confidence": round(confidence, 2),
        "probabilities": {
            label: round(float(p) * 100, 2)
            for label, p in zip(EMOTIONS, probs_np)
        },
    }