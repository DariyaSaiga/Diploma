"""
utils.py — Preprocessing utilities for Emotion Recognition Service.

Covers
──────
• base64 image → PIL → normalized torch.Tensor [1, 3, 224, 224]
• mp4/webm bytes → uniform frame sample → list of tensors
• safe temp-file management
"""

from __future__ import annotations

import base64
import io
import logging
import os
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError

from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, MAX_FRAMES

logger = logging.getLogger(__name__)

# ── Image transform (ImageNet normalisation) ──────────────────────────────────
_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── base64 helpers ────────────────────────────────────────────────────────────
def _strip_data_uri(b64: str) -> str:
    """Remove 'data:image/...;base64,' prefix if present."""
    if "," in b64:
        return b64.split(",", 1)[1]
    return b64


def decode_base64_image(b64_str: str) -> Image.Image:
    """
    Decode a base64-encoded image string into a PIL Image (RGB).

    Raises
    ──────
    ValueError  — if the string is not valid base64 or not a recognisable image.
    """
    clean = _strip_data_uri(b64_str.strip())
    try:
        raw = base64.b64decode(clean)
    except Exception as exc:
        raise ValueError(f"Invalid base64 data: {exc}") from exc

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"Cannot decode image bytes: {exc}") from exc

    return img


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """
    Apply ImageNet normalisation and return a float tensor [1, 3, H, W].
    """
    return _transform(img).unsqueeze(0)  # [1, 3, 224, 224]


def base64_to_tensor(b64_str: str) -> torch.Tensor:
    """Convenience wrapper: base64 str → [1, 3, 224, 224] tensor."""
    img = decode_base64_image(b64_str)
    return pil_to_tensor(img)


# ── Temp-file context manager ─────────────────────────────────────────────────
@contextmanager
def temp_video_file(data: bytes, suffix: str = ".mp4") -> Generator[Path, None, None]:
    """
    Write *data* to a uniquely named temp file, yield its Path, then delete it.

    Safe even if the caller raises an exception.
    """
    tmp_dir = Path(tempfile.gettempdir())
    tmp_path = tmp_dir / f"emotion_{uuid.uuid4().hex}{suffix}"
    try:
        tmp_path.write_bytes(data)
        logger.debug("Temp video written to %s (%d bytes)", tmp_path, len(data))
        yield tmp_path
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
            logger.debug("Temp file %s deleted.", tmp_path)
        except Exception as exc:
            logger.warning("Could not delete temp file %s: %s", tmp_path, exc)


# ── Video frame extraction ─────────────────────────────────────────────────────
def extract_frames(video_path: Path, max_frames: int = MAX_FRAMES) -> List[np.ndarray]:
    """
    Open a video file and return a uniformly sampled list of BGR frames (numpy).

    Parameters
    ──────────
    video_path  Path to the video file (must exist).
    max_frames  Maximum number of frames to return.

    Returns
    ───────
    List of np.ndarray in BGR uint8 format, length ≤ max_frames.

    Raises
    ──────
    ValueError if the file cannot be opened or contains no frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video has no readable frames: {video_path}")

    # Uniform sampling indices
    sample_count = min(max_frames, total_frames)
    indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)

    frames: List[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError("No frames could be read from video.")

    logger.debug("Extracted %d / %d frames from %s", len(frames), total_frames, video_path)
    return frames


def bgr_frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
    """Convert a single BGR numpy frame → [1, 3, 224, 224] normalised tensor."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    return pil_to_tensor(img)


def frames_to_tensors(frames: List[np.ndarray]) -> List[torch.Tensor]:
    """Batch-convert BGR frames to tensors."""
    return [bgr_frame_to_tensor(f) for f in frames]


# ── Aggregate multi-frame predictions ────────────────────────────────────────
def aggregate_predictions(results: List[dict]) -> dict:
    """
    Mean-pool per-class probabilities from multiple frame predictions.

    Parameters
    ──────────
    results  list of dicts, each with keys:
             'emotion', 'confidence', 'probabilities' ({label: float})

    Returns
    ───────
    Single aggregated dict in the same format.
    """
    if not results:
        raise ValueError("No predictions to aggregate.")

    all_probs: dict[str, List[float]] = {}
    for r in results:
        for label, prob in r["probabilities"].items():
            all_probs.setdefault(label, []).append(prob)

    mean_probs = {label: round(float(np.mean(vals)), 2) for label, vals in all_probs.items()}
    top_label  = max(mean_probs, key=mean_probs.__getitem__)
    confidence = mean_probs[top_label]

    return {
        "emotion":       top_label,
        "confidence":    confidence,
        "probabilities": mean_probs,
    }
