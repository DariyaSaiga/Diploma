"""
EmotionAI Backend — FastAPI
────────────────────────────────────────────────────────────────────────────────
Подключён к твоей обученной модели MultimodalEmotionModel из train-файла.

Что изменилось по сравнению с mock-версией:
  - Полная архитектура модели скопирована из train.py (BAF, CNN, BiLSTM и т.д.)
  - 6 эмоций вместо 7 (MOSEI: Happy, Sad, Anger, Surprise, Disgust, Fear)
  - Multi-label inference (sigmoid, не softmax)
  - Per-emotion thresholds (оптимальные из threshold tuning)
  - infer_frame работает с видео-кадрами из браузера
  - infer_video обрабатывает загруженный файл через OpenCV

Эндпоинты:
  GET  /                   — health check
  POST /analyze/video      — загрузить видео → emotion результат
  WS   /ws/camera          — WebSocket для real-time камеры
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
log = logging.getLogger("emotionai")

# ── Config ────────────────────────────────────────────────────────────────────
# Положи свой .pt файл в папку checkpoints/ рядом с main.py
MODEL_PATH = os.getenv("MODEL_PATH", "checkpoints/best_model_bert_cnn_bilstm_sem.pt")
DEVICE     = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MAX_UPLOAD = int(os.getenv("MAX_UPLOAD_MB", "200")) * 1024 * 1024
ALLOWED_ORIGINS = os.getenv(
    "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
).split(",")

# ── Размерности из твоего датасета MOSEI ─────────────────────────────────────
# Если у тебя другие — поменяй здесь или через env vars
TEXT_DIM  = int(os.getenv("TEXT_DIM",  "768"))   # BERT embeddings
AUDIO_DIM = int(os.getenv("AUDIO_DIM", "74"))    # COVAREP features
VIDEO_DIM = int(os.getenv("VIDEO_DIM", "35"))    # OpenFace features

TEXT_SEQ_LEN  = int(os.getenv("TEXT_SEQ_LEN",  "50"))
AUDIO_SEQ_LEN = int(os.getenv("AUDIO_SEQ_LEN", "60"))
VIDEO_SEQ_LEN = int(os.getenv("VIDEO_SEQ_LEN", "60"))

# 6 эмоций из MOSEI (порядок должен совпадать с обучением!)
EMOTION_LABELS = ["Happy", "Sad", "Anger", "Surprise", "Disgust", "Fear"]

# Per-emotion thresholds — замени на свои из threshold tuning если они другие
DEFAULT_THRESHOLDS = {
    "Happy":    0.45,
    "Sad":      0.35,
    "Anger":    0.35,
    "Surprise": 0.30,
    "Disgust":  0.30,
    "Fear":     0.30,
}


# ── Pydantic schema ───────────────────────────────────────────────────────────
class EmotionResult(BaseModel):
    dominant:  str                  # эмоция с наибольшей вероятностью
    scores:    dict[str, float]     # вероятности 0..1 для каждой эмоции
    active:    list[str]            # эмоции, превысившие порог threshold
    timestamp: Optional[float] = None


# ═══════════════════════════════════════════════════════════════════════════════
#  АРХИТЕКТУРА МОДЕЛИ (точная копия из train.py)
# ═══════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class BottleneckAttentionFusion(nn.Module):
    def __init__(self, d_model, num_heads=4, num_bottleneck=16, dropout=0.1):
        super().__init__()
        self.bottleneck = nn.Parameter(
            torch.randn(1, num_bottleneck, d_model) * 0.02
        )
        self.attn_t  = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_a  = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_v  = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_bt = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_ba = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_bv = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm_t   = nn.LayerNorm(d_model)
        self.norm_a   = nn.LayerNorm(d_model)
        self.norm_v   = nn.LayerNorm(d_model)
        self.norm_b   = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.drop     = nn.Dropout(dropout)

    def forward(self, t, a, v, bn=None):
        B = t.size(0)
        if bn is None:
            bn = self.bottleneck.expand(B, -1, -1)
        bt, _ = self.attn_bt(bn, t, t)
        ba, _ = self.attn_ba(bn, a, a)
        bv, _ = self.attn_bv(bn, v, v)
        bn = self.norm_b(bn + self.drop(bt + ba + bv) / 3.0)
        t2, _ = self.attn_t(t, bn, bn)
        a2, _ = self.attn_a(a, bn, bn)
        v2, _ = self.attn_v(v, bn, bn)
        t = self.norm_t(t + self.drop(t2))
        a = self.norm_a(a + self.drop(a2))
        v = self.norm_v(v + self.drop(v2))
        bn = self.norm_ffn(bn + self.drop(self.ffn(bn)))
        return bn, t, a, v


class ModalityEncoder(nn.Module):
    def __init__(self, in_dim, d_model, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.pe = PositionalEncoding(d_model, dropout=dropout)

    def forward(self, x):
        return self.pe(self.proj(x))


class AudioCNNEncoder(nn.Module):
    def __init__(self, in_dim, d_model, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )
        self.pe = PositionalEncoding(d_model, dropout=dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return self.pe(x)


class VideoBiLSTMEncoder(nn.Module):
    def __init__(self, in_dim, d_model, dropout=0.1):
        super().__init__()
        self.proj   = nn.Linear(in_dim, d_model)
        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.proj(x)
        x, _ = self.bilstm(x)
        return self.norm(x)


class MultimodalEmotionModel(nn.Module):
    def __init__(
        self,
        text_dim, audio_dim, video_dim,
        d_model=128, num_heads=4, num_bottleneck=16,
        num_fusion_layers=2, num_classes=6, dropout=0.2,
    ):
        super().__init__()
        self.text_enc = ModalityEncoder(text_dim, d_model, dropout)
        self.text_sa  = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, num_heads, d_model * 4, dropout, batch_first=True
            ),
            num_layers=2,
        )
        self.audio_enc    = AudioCNNEncoder(audio_dim, d_model, dropout)
        self.video_enc    = VideoBiLSTMEncoder(video_dim, d_model, dropout)
        self.fusion_layers = nn.ModuleList([
            BottleneckAttentionFusion(d_model, num_heads, num_bottleneck, dropout)
            for _ in range(num_fusion_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self.text_classifier  = nn.Linear(d_model, num_classes)
        self.audio_classifier = nn.Linear(d_model, num_classes)
        self.video_classifier = nn.Linear(d_model, num_classes)

    def forward(self, text, audio, video):
        t = self.text_sa(self.text_enc(text))
        a = self.audio_enc(audio)
        v = self.video_enc(video)
        bn = None
        for layer in self.fusion_layers:
            bn, t, a, v = layer(t, a, v, bn)
        f_logits = self.classifier(bn.mean(dim=1))
        t_logits = self.text_classifier(t.mean(dim=1))
        a_logits = self.audio_classifier(a.mean(dim=1))
        v_logits = self.video_classifier(v.mean(dim=1))
        return f_logits, t_logits, a_logits, v_logits


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def _load_model() -> Optional[MultimodalEmotionModel]:
    if not Path(MODEL_PATH).exists():
        log.warning("Checkpoint not found at '%s' — running in mock mode", MODEL_PATH)
        return None
    try:
        net = MultimodalEmotionModel(
            text_dim=TEXT_DIM, audio_dim=AUDIO_DIM, video_dim=VIDEO_DIM,
            d_model=128, num_heads=4, num_bottleneck=16,
            num_fusion_layers=2, num_classes=6, dropout=0.2,
        ).to(DEVICE)
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        net.load_state_dict(state)
        net.eval()
        log.info("✓ Model loaded from %s on %s", MODEL_PATH, DEVICE)
        return net
    except Exception as exc:
        log.error("Model load failed: %s — running in mock mode", exc)
        return None


_model: Optional[MultimodalEmotionModel] = _load_model()


# ═══════════════════════════════════════════════════════════════════════════════
#  INFERENCE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _probs_to_result(probs: np.ndarray, thresholds: dict = DEFAULT_THRESHOLDS) -> EmotionResult:
    scores   = {e: round(float(p), 4) for e, p in zip(EMOTION_LABELS, probs)}
    dominant = max(scores, key=lambda k: scores[k])
    active   = [e for e in EMOTION_LABELS if scores[e] >= thresholds.get(e, 0.5)]
    return EmotionResult(dominant=dominant, scores=scores, active=active, timestamp=time.time())


def _mock_result() -> EmotionResult:
    """Используется только если модель не загрузилась."""
    import random
    raw   = [random.random() for _ in EMOTION_LABELS]
    total = sum(raw)
    probs = np.array([v / total for v in raw], dtype=np.float32)
    return _probs_to_result(probs)


def _frames_to_video_tensor(frames: list[np.ndarray]) -> torch.Tensor:
    """
    Список BGR-кадров → тензор (1, VIDEO_SEQ_LEN, VIDEO_DIM).

    Здесь используются простые пиксельные статистики вместо OpenFace,
    потому что в браузере OpenFace недоступен.
    Если хочешь точный инференс — прогони видео через OpenFace отдельно
    и передавай готовые фичи напрямую.
    """
    features = []
    for frame in frames:
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (VIDEO_DIM, 1))
        feat    = resized.astype(np.float32).flatten() / 255.0
        features.append(feat)

    # Приводим к VIDEO_SEQ_LEN
    if len(features) < VIDEO_SEQ_LEN:
        pad      = [np.zeros(VIDEO_DIM, dtype=np.float32)] * (VIDEO_SEQ_LEN - len(features))
        features = features + pad
    else:
        idx      = np.linspace(0, len(features) - 1, VIDEO_SEQ_LEN, dtype=int)
        features = [features[i] for i in idx]

    return torch.tensor(np.stack(features), dtype=torch.float32).unsqueeze(0)


@torch.no_grad()
def _run_inference(frames: list[np.ndarray]) -> EmotionResult:
    """Главная функция инференса. Принимает список BGR-кадров."""
    if _model is None:
        return _mock_result()

    video  = _frames_to_video_tensor(frames).to(DEVICE)
    # Текст и аудио заполняем нулями (только видео-модальность из браузера)
    text   = torch.zeros(1, TEXT_SEQ_LEN,  TEXT_DIM,  dtype=torch.float32).to(DEVICE)
    audio  = torch.zeros(1, AUDIO_SEQ_LEN, AUDIO_DIM, dtype=torch.float32).to(DEVICE)

    f_logits, _, _, _ = _model(text, audio, video)
    probs = torch.sigmoid(f_logits).cpu().numpy()[0]  # shape: (6,)
    return _probs_to_result(probs)


# ═══════════════════════════════════════════════════════════════════════════════
#  FastAPI ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="EmotionAI API", version="2.0.0", docs_url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["health"])
async def health():
    return {
        "status":   "ok",
        "model":    "loaded" if _model is not None else "mock",
        "device":   DEVICE,
        "emotions": EMOTION_LABELS,
    }


@app.post("/analyze/video", response_model=EmotionResult, tags=["inference"])
async def analyze_video(file: UploadFile = File(...)):
    """
    Загрузи видеофайл → получи emotion анализ.
    Семплирует до 120 кадров равномерно по всей длине видео.
    """
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        return JSONResponse(status_code=413, content={"error": "File too large."})

    log.info("Video '%s' received (%.1f MB)", file.filename, len(data) / 1024 / 1024)

    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        result = await asyncio.to_thread(_process_video_file, tmp_path)
        log.info("Done: dominant=%s active=%s", result.dominant, result.active)
        return result
    except Exception as exc:
        log.exception("Video inference error")
        return JSONResponse(status_code=500, content={"error": str(exc)})
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _process_video_file(path: str) -> EmotionResult:
    """Читает видео → собирает кадры → запускает inference. Синхронная функция для thread."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open: {path}")

    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    sample_at = set(np.linspace(0, total - 1, min(120, total), dtype=int).tolist())
    frames, idx = [], 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in sample_at:
            frames.append(frame)
        idx += 1
    cap.release()

    if not frames:
        raise ValueError("No frames could be extracted.")

    return _run_inference(frames)


@app.websocket("/ws/camera")
async def camera_ws(ws: WebSocket):
    """
    Real-time inference через WebSocket.

    Клиент → { "frame": "<base64 JPEG>" }
    Сервер → EmotionResult JSON

    Использует скользящий буфер из 8 кадров для сглаживания результатов.
    """
    await ws.accept()
    log.info("WS connected")

    frame_buffer: list[np.ndarray] = []
    BUFFER = 8  # количество кадров для усреднения

    try:
        while True:
            raw     = await ws.receive_text()
            payload = json.loads(raw)
            b64     = payload.get("frame", "")

            if not b64:
                await ws.send_json({"error": "empty frame"})
                continue

            # base64 JPEG → numpy BGR
            img_bytes = base64.b64decode(b64)
            arr       = np.frombuffer(img_bytes, dtype=np.uint8)
            frame     = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if frame is None:
                await ws.send_json({"error": "bad frame"})
                continue

            # Скользящий буфер
            frame_buffer.append(frame)
            if len(frame_buffer) > BUFFER:
                frame_buffer.pop(0)

            result = await asyncio.to_thread(_run_inference, list(frame_buffer))
            await ws.send_text(result.model_dump_json())

    except WebSocketDisconnect:
        log.info("WS disconnected")
    except Exception as exc:
        log.exception("WS error: %s", exc)
        await ws.close(code=1011)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")