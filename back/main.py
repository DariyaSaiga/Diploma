"""
EmotionAI Backend — FastAPI (Multimodal: Video + Audio + Text)
────────────────────────────────────────────────────────────────────────────────
Все три модальности извлекаются прямо из видеофайла:

  VIDEO  → OpenCV кадры → признаки лица (landmarks + статистики) → BiLSTM
  AUDIO  → ffmpeg извлекает wav → librosa → MFCC/спектральные фичи → CNN
  TEXT   → whisper транскрибирует речь → BERT embeddings → Transformer

Эндпоинты:
  GET  /                   — health check
  POST /analyze/video      — файл → все 3 модальности → EmotionResult
  WS   /ws/camera          — real-time кадры камеры (только video модальность,
                              аудио/текст из браузера пока недоступны)
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import librosa
import numpy as np
import torch
import torch.nn as nn
import uvicorn
import whisper
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import BertModel, BertTokenizer

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
log = logging.getLogger("emotionai")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "best_model_bert_cnn_bilstm.pt")
DEVICE     = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MAX_UPLOAD = int(os.getenv("MAX_UPLOAD_MB", "200")) * 1024 * 1024
ALLOWED_ORIGINS = os.getenv(
    "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
).split(",")

# ── Размерности (должны совпадать с обучением!) ────────────────────────────────
TEXT_DIM      = int(os.getenv("TEXT_DIM",  "768"))  # BERT-base hidden size
AUDIO_DIM     = int(os.getenv("AUDIO_DIM", "74"))   # COVAREP-like фичи
VIDEO_DIM     = int(os.getenv("VIDEO_DIM", "35"))   # OpenFace-like фичи
TEXT_SEQ_LEN  = int(os.getenv("TEXT_SEQ_LEN",  "50"))
AUDIO_SEQ_LEN = int(os.getenv("AUDIO_SEQ_LEN", "60"))
VIDEO_SEQ_LEN = int(os.getenv("VIDEO_SEQ_LEN", "60"))

EMOTION_LABELS = ["Happy", "Sad", "Anger", "Surprise", "Disgust", "Fear"]

# Per-emotion thresholds из threshold tuning — замени на свои реальные значения
DEFAULT_THRESHOLDS = {
    "Happy":    0.45,
    "Sad":      0.35,
    "Anger":    0.35,
    "Surprise": 0.30,
    "Disgust":  0.30,
    "Fear":     0.30,
}


# ── Pydantic schema ────────────────────────────────────────────────────────────
class EmotionResult(BaseModel):
    dominant:   str
    scores:     dict[str, float]
    active:     list[str]
    transcript: Optional[str] = None   # текст из whisper (для отображения в UI)
    timestamp:  Optional[float] = None


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
        self.bottleneck = nn.Parameter(torch.randn(1, num_bottleneck, d_model) * 0.02)
        self.attn_t  = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_a  = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_v  = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_bt = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_ba = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_bv = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * 4, d_model),
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
            nn.Linear(in_dim, d_model), nn.LayerNorm(d_model), nn.GELU(),
        )
        self.pe = PositionalEncoding(d_model, dropout=dropout)

    def forward(self, x):
        return self.pe(self.proj(x))


class AudioCNNEncoder(nn.Module):
    def __init__(self, in_dim, d_model, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model), nn.ReLU(),
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
            input_size=d_model, hidden_size=d_model // 2,
            num_layers=2, batch_first=True, bidirectional=True, dropout=dropout,
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
        self.text_enc  = ModalityEncoder(text_dim,  d_model, dropout)
        self.text_sa   = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, d_model * 4, dropout, batch_first=True),
            num_layers=2,
        )
        self.audio_enc = AudioCNNEncoder(audio_dim, d_model, dropout)
        self.video_enc = VideoBiLSTMEncoder(video_dim, d_model, dropout)
        self.fusion_layers = nn.ModuleList([
            BottleneckAttentionFusion(d_model, num_heads, num_bottleneck, dropout)
            for _ in range(num_fusion_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, num_classes),
        )
        self.text_classifier  = nn.Linear(d_model, num_classes)
        self.audio_classifier = nn.Linear(d_model, num_classes)
        self.video_classifier = nn.Linear(d_model, num_classes)

    def forward(self, text, audio, video):
        t  = self.text_sa(self.text_enc(text))
        a  = self.audio_enc(audio)
        v  = self.video_enc(video)
        bn = None
        for layer in self.fusion_layers:
            bn, t, a, v = layer(t, a, v, bn)
        f_logits = self.classifier(bn.mean(dim=1))
        t_logits = self.text_classifier(t.mean(dim=1))
        a_logits = self.audio_classifier(a.mean(dim=1))
        v_logits = self.video_classifier(v.mean(dim=1))
        return f_logits, t_logits, a_logits, v_logits


# ═══════════════════════════════════════════════════════════════════════════════
#  ЗАГРУЗКА ВСПОМОГАТЕЛЬНЫХ МОДЕЛЕЙ (BERT + Whisper)
# ═══════════════════════════════════════════════════════════════════════════════

log.info("Loading BERT tokenizer & model...")
_bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
_bert_model     = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)
_bert_model.eval()
log.info("✓ BERT loaded")

log.info("Loading Whisper (base)...")
# Используем 'base' для баланса скорость/качество.
# Замени на 'small' или 'medium' для лучшей транскрипции.
_whisper_model = whisper.load_model("base", device=DEVICE)
log.info("✓ Whisper loaded")


# ═══════════════════════════════════════════════════════════════════════════════
#  ЗАГРУЗКА ОСНОВНОЙ МОДЕЛИ
# ═══════════════════════════════════════════════════════════════════════════════

def _load_emotion_model() -> Optional[MultimodalEmotionModel]:
    if not Path(MODEL_PATH).exists():
        log.warning("Checkpoint not found at '%s' — mock mode", MODEL_PATH)
        return None
    try:
        net = MultimodalEmotionModel(
            text_dim=TEXT_DIM, audio_dim=AUDIO_DIM, video_dim=VIDEO_DIM,
        ).to(DEVICE)
        net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        net.eval()
        log.info("✓ EmotionModel loaded from %s on %s", MODEL_PATH, DEVICE)
        return net
    except Exception as exc:
        log.error("EmotionModel load failed: %s — mock mode", exc)
        return None


_emotion_model: Optional[MultimodalEmotionModel] = _load_emotion_model()


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTORS
# ═══════════════════════════════════════════════════════════════════════════════

def extract_audio_wav(video_path: str) -> Optional[str]:
    """
    Извлекает аудио из видеофайла в временный WAV через ffmpeg.
    Возвращает путь к WAV или None если ffmpeg недоступен / аудио нет.
    """
    wav_path = video_path.replace(Path(video_path).suffix, "_audio.wav")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",              # только аудио
        "-ar", "16000",     # 16kHz — нужно Whisper
        "-ac", "1",         # моно
        "-f", "wav",
        wav_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode == 0 and Path(wav_path).exists():
            return wav_path
        log.warning("ffmpeg failed: %s", result.stderr.decode()[:200])
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        log.warning("ffmpeg not available: %s", e)
        return None


def extract_text_features(wav_path: Optional[str]) -> tuple[np.ndarray, str]:
    """
    WAV → Whisper транскрипция → BERT embeddings.

    Возвращает:
        features  : (TEXT_SEQ_LEN, TEXT_DIM) float32
        transcript: строка транскрипции (для отображения в UI)
    """
    transcript = ""

    if wav_path is None:
        # Нет аудио — возвращаем нули
        return np.zeros((TEXT_SEQ_LEN, TEXT_DIM), dtype=np.float32), transcript

    # ── Транскрипция через Whisper ─────────────────────────────────────────
    try:
        result     = _whisper_model.transcribe(wav_path, language=None, fp16=False)
        transcript = result.get("text", "").strip()
        log.info("Whisper transcript: '%s'", transcript[:80])
    except Exception as e:
        log.warning("Whisper failed: %s", e)
        return np.zeros((TEXT_SEQ_LEN, TEXT_DIM), dtype=np.float32), transcript

    if not transcript:
        return np.zeros((TEXT_SEQ_LEN, TEXT_DIM), dtype=np.float32), transcript

    # ── BERT embeddings ────────────────────────────────────────────────────
    # Токенизируем с max_length=TEXT_SEQ_LEN, получаем hidden states
    try:
        encoded = _bert_tokenizer(
            transcript,
            return_tensors="pt",
            max_length=TEXT_SEQ_LEN,
            padding="max_length",
            truncation=True,
        )
        input_ids      = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = _bert_model(input_ids=input_ids, attention_mask=attention_mask)
            # last_hidden_state: (1, TEXT_SEQ_LEN, 768)
            hidden = outputs.last_hidden_state.squeeze(0).cpu().numpy()

        return hidden.astype(np.float32), transcript

    except Exception as e:
        log.warning("BERT failed: %s", e)
        return np.zeros((TEXT_SEQ_LEN, TEXT_DIM), dtype=np.float32), transcript


def extract_audio_features(wav_path: Optional[str]) -> np.ndarray:
    """
    WAV → librosa → COVAREP-like признаки формы (AUDIO_SEQ_LEN, AUDIO_DIM).

    Признаки (итого 74):
      - MFCC 1-13 + delta + delta-delta  = 39
      - Chroma STFT                      = 12
      - Spectral contrast (7 bands)      = 7
      - ZCR + RMS + spectral centroid/bw = 4
      - Pitch (F0) + voiced probability  = 2
      - Mel-energy (10 bands)            = 10
    Итого: 39 + 12 + 7 + 4 + 2 + 10 = 74  ✓  совпадает с AUDIO_DIM
    """
    if wav_path is None:
        return np.zeros((AUDIO_SEQ_LEN, AUDIO_DIM), dtype=np.float32)

    try:
        y, sr = librosa.load(wav_path, sr=16000, mono=True)

        # MFCC 13 + delta + delta-delta → (39, T)
        mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        d_mfcc = librosa.feature.delta(mfcc)
        d2_mfcc= librosa.feature.delta(mfcc, order=2)
        mfcc_full = np.vstack([mfcc, d_mfcc, d2_mfcc])  # (39, T)

        # Chroma → (12, T)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)

        # Spectral contrast → (7, T)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        # ZCR, RMS, centroid, bandwidth → (4, T)
        zcr      = librosa.feature.zero_crossing_rate(y)
        rms      = librosa.feature.rms(y=y)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        bandwidth= librosa.feature.spectral_bandwidth(y=y, sr=sr)
        misc     = np.vstack([zcr, rms, centroid, bandwidth])  # (4, T)

        # Pitch + voiced probability → (2, T)
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y, fmin=80, fmax=400, sr=sr,
            frame_length=2048, fill_na=0.0
        )
        f0_norm = (f0 / 400.0).reshape(1, -1)
        voiced  = voiced_prob.reshape(1, -1)
        pitch   = np.vstack([f0_norm, voiced])  # (2, T)

        # Mel energy (10 bands) → (10, T)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=10)
        mel = librosa.power_to_db(mel, ref=np.max)

        # Привести все к одной длине T (min)
        T = min(
            mfcc_full.shape[1], chroma.shape[1], contrast.shape[1],
            misc.shape[1], pitch.shape[1], mel.shape[1]
        )
        features = np.vstack([
            mfcc_full[:, :T], chroma[:, :T], contrast[:, :T],
            misc[:, :T], pitch[:, :T], mel[:, :T],
        ])  # (74, T)

        # Транспонируем → (T, 74) и нормализуем
        features = features.T.astype(np.float32)  # (T, AUDIO_DIM)
        features = (features - features.mean(0)) / (features.std(0) + 1e-8)

        # Ресемплируем до AUDIO_SEQ_LEN
        if len(features) < AUDIO_SEQ_LEN:
            pad      = np.zeros((AUDIO_SEQ_LEN - len(features), AUDIO_DIM), dtype=np.float32)
            features = np.vstack([features, pad])
        else:
            idx      = np.linspace(0, len(features) - 1, AUDIO_SEQ_LEN, dtype=int)
            features = features[idx]

        return features  # (AUDIO_SEQ_LEN, AUDIO_DIM)

    except Exception as e:
        log.warning("Audio feature extraction failed: %s", e)
        return np.zeros((AUDIO_SEQ_LEN, AUDIO_DIM), dtype=np.float32)


def extract_video_features(frames: list[np.ndarray]) -> np.ndarray:
    """
    Список BGR-кадров → OpenFace-like признаки (VIDEO_SEQ_LEN, VIDEO_DIM=35).

    Признаки на кадр (35 значений):
      - HOG-like дескриптор лица (16) — захватывает форму и текстуру
      - LBP гистограмма (8)           — локальная текстура кожи
      - Статистики яркости (4)        — mean/std/min/max серого
      - Симметрия лица (1)            — левая vs правая половина
      - Edge density (1)              — насыщенность краями (мимика)
      - Размер/позиция bbox лица (5)  — через детектор Haar

    Примечание: для полной точности (как при обучении на MOSEI) нужен
    реальный OpenFace. Это proxy-приближение для демонстрации.
    """
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    def frame_features(frame: np.ndarray) -> np.ndarray:
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w    = gray.shape
        feat    = np.zeros(VIDEO_DIM, dtype=np.float32)

        # 1. Детектируем лицо
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
        if len(faces) > 0:
            x, y, fw, fh = faces[0]
            face_roi = gray[y:y+fh, x:x+fw]
            # BBox normalized
            feat[0]  = x / w
            feat[1]  = y / h
            feat[2]  = fw / w
            feat[3]  = fh / h
            feat[4]  = (fw * fh) / (w * h)  # relative area
        else:
            # Лицо не найдено — берём центральный ROI
            face_roi = gray[h//4:3*h//4, w//4:3*w//4]
            feat[0:5] = 0.0

        face_resized = cv2.resize(face_roi, (32, 32))
        face_norm    = face_resized.astype(np.float32) / 255.0

        # 2. HOG-like: разбить на 4x4 патча, взять mean градиентов → 16 значений
        patch_size = 8
        hog_feats  = []
        for pi in range(4):
            for pj in range(4):
                patch  = face_norm[pi*patch_size:(pi+1)*patch_size,
                                   pj*patch_size:(pj+1)*patch_size]
                gx     = np.diff(patch, axis=1)
                gy     = np.diff(patch, axis=0)
                hog_feats.append(np.abs(gx).mean())
                hog_feats.append(np.abs(gy).mean())
        hog_feats = np.array(hog_feats[:16], dtype=np.float32)
        feat[5:21] = hog_feats

        # 3. LBP-подобная гистограмма (8 значений)
        lbp   = np.zeros_like(face_norm)
        for di, dj in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
            shifted = np.roll(np.roll(face_norm, di, axis=0), dj, axis=1)
            lbp    += (face_norm >= shifted).astype(np.float32)
        lbp_hist, _ = np.histogram(lbp.flatten(), bins=8, range=(0,8), density=True)
        feat[21:29]  = lbp_hist.astype(np.float32)

        # 4. Статистики яркости (4)
        feat[29] = face_norm.mean()
        feat[30] = face_norm.std()
        feat[31] = face_norm.min()
        feat[32] = face_norm.max()

        # 5. Симметрия (1)
        left    = face_norm[:, :16]
        right   = face_norm[:, 16:][:, ::-1]
        feat[33] = 1.0 - np.abs(left - right).mean()

        # 6. Edge density (1) — Canny на ROI
        edges   = cv2.Canny(face_resized, 50, 150)
        feat[34] = edges.mean() / 255.0

        return feat  # (35,)

    features = [frame_features(f) for f in frames]

    # Нормализуем до VIDEO_SEQ_LEN
    if len(features) < VIDEO_SEQ_LEN:
        pad      = [np.zeros(VIDEO_DIM, dtype=np.float32)] * (VIDEO_SEQ_LEN - len(features))
        features = features + pad
    else:
        idx      = np.linspace(0, len(features) - 1, VIDEO_SEQ_LEN, dtype=int)
        features = [features[i] for i in idx]

    arr = np.stack(features).astype(np.float32)  # (VIDEO_SEQ_LEN, VIDEO_DIM)
    # Нормализация
    arr = (arr - arr.mean(0)) / (arr.std(0) + 1e-8)
    return arr


# ═══════════════════════════════════════════════════════════════════════════════
#  INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def _probs_to_result(
    probs: np.ndarray,
    transcript: str = "",
    thresholds: dict = DEFAULT_THRESHOLDS,
) -> EmotionResult:
    scores   = {e: round(float(p), 4) for e, p in zip(EMOTION_LABELS, probs)}
    dominant = max(scores, key=lambda k: scores[k])
    active   = [e for e in EMOTION_LABELS if scores[e] >= thresholds.get(e, 0.5)]
    return EmotionResult(
        dominant=dominant, scores=scores, active=active,
        transcript=transcript or None, timestamp=time.time(),
    )


def _mock_result(transcript: str = "") -> EmotionResult:
    import random
    raw   = [random.random() for _ in EMOTION_LABELS]
    total = sum(raw)
    probs = np.array([v / total for v in raw], dtype=np.float32)
    return _probs_to_result(probs, transcript)


@torch.no_grad()
def run_multimodal_inference(
    text_feat:  np.ndarray,   # (TEXT_SEQ_LEN,  TEXT_DIM)
    audio_feat: np.ndarray,   # (AUDIO_SEQ_LEN, AUDIO_DIM)
    video_feat: np.ndarray,   # (VIDEO_SEQ_LEN, VIDEO_DIM)
    transcript: str = "",
) -> EmotionResult:
    """Прогоняет все три модальности через модель и возвращает EmotionResult."""
    if _emotion_model is None:
        return _mock_result(transcript)

    text  = torch.tensor(text_feat,  dtype=torch.float32).unsqueeze(0).to(DEVICE)
    audio = torch.tensor(audio_feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    video = torch.tensor(video_feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    f_logits, _, _, _ = _emotion_model(text, audio, video)
    probs = torch.sigmoid(f_logits).cpu().numpy()[0]  # (6,)
    return _probs_to_result(probs, transcript)


# ═══════════════════════════════════════════════════════════════════════════════
#  VIDEO FILE PROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def process_video_file(video_path: str) -> EmotionResult:
    """
    Полный пайплайн для загруженного видеофайла:
      1. OpenCV → семплируем до 120 кадров → extract_video_features
      2. ffmpeg  → WAV → extract_audio_features
      3. Whisper → транскрипция → BERT → extract_text_features
      4. Все три тензора → MultimodalEmotionModel → EmotionResult
    """
    # ── 1. Видео: читаем кадры ─────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    sample_at = set(np.linspace(0, total - 1, min(120, total), dtype=int).tolist())
    frames    = []
    idx       = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in sample_at:
            frames.append(frame)
        idx += 1
    cap.release()

    if not frames:
        raise ValueError("No frames extracted.")

    log.info("Extracted %d frames for video features", len(frames))
    video_feat = extract_video_features(frames)

    # ── 2. Аудио: извлекаем WAV через ffmpeg ──────────────────────────────
    wav_path   = extract_audio_wav(video_path)
    audio_feat = extract_audio_features(wav_path)
    log.info("Audio features extracted: %s", audio_feat.shape)

    # ── 3. Текст: Whisper → BERT ───────────────────────────────────────────
    text_feat, transcript = extract_text_features(wav_path)
    log.info("Text features extracted: %s | '%s'", text_feat.shape, transcript[:60])

    # Чистим временный WAV
    if wav_path:
        Path(wav_path).unlink(missing_ok=True)

    # ── 4. Inference ───────────────────────────────────────────────────────
    return run_multimodal_inference(text_feat, audio_feat, video_feat, transcript)


# ═══════════════════════════════════════════════════════════════════════════════
#  FastAPI
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="EmotionAI API", version="3.0.0", docs_url="/docs")

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
        "model":    "loaded" if _emotion_model is not None else "mock",
        "device":   DEVICE,
        "emotions": EMOTION_LABELS,
        "modalities": ["text (BERT+Whisper)", "audio (librosa)", "video (OpenCV)"],
    }


@app.post("/analyze/video", response_model=EmotionResult, tags=["inference"])
async def analyze_video(file: UploadFile = File(...)):
    """
    Полный мультимодальный анализ загруженного видеофайла.
    Извлекает video + audio + text модальности и прогоняет через модель.
    """
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        return JSONResponse(status_code=413, content={"error": "File too large."})

    log.info("Received '%s' (%.1f MB)", file.filename, len(data) / 1024 / 1024)

    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        result = await asyncio.to_thread(process_video_file, tmp_path)
        log.info("✓ Done: dominant=%s active=%s", result.dominant, result.active)
        return result
    except Exception as exc:
        log.exception("Inference failed")
        return JSONResponse(status_code=500, content={"error": str(exc)})
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.websocket("/ws/camera")
async def camera_ws(ws: WebSocket):
    """
    Real-time камера через WebSocket.

    Клиент → { "frame": "<base64 JPEG>" }
    Сервер → EmotionResult JSON

    Использует только video-модальность (text/audio из одного кадра недоступны).
    Для production: отправлять аудио-чанки вместе с кадрами.
    """
    await ws.accept()
    log.info("WS connected")

    frame_buffer: list[np.ndarray] = []
    BUFFER = 8  # скользящее окно кадров

    try:
        while True:
            raw     = await ws.receive_text()
            payload = json.loads(raw)
            b64     = payload.get("frame", "")

            if not b64:
                await ws.send_json({"error": "empty frame"})
                continue

            img_bytes = base64.b64decode(b64)
            arr       = np.frombuffer(img_bytes, dtype=np.uint8)
            frame     = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if frame is None:
                await ws.send_json({"error": "bad frame"})
                continue

            frame_buffer.append(frame)
            if len(frame_buffer) > BUFFER:
                frame_buffer.pop(0)

            def _infer_camera():
                # Только video модальность (нет аудио/текста из одного кадра)
                video_feat = extract_video_features(list(frame_buffer))
                text_feat  = np.zeros((TEXT_SEQ_LEN,  TEXT_DIM),  dtype=np.float32)
                audio_feat = np.zeros((AUDIO_SEQ_LEN, AUDIO_DIM), dtype=np.float32)
                return run_multimodal_inference(text_feat, audio_feat, video_feat)

            result = await asyncio.to_thread(_infer_camera)
            await ws.send_text(result.model_dump_json())

    except WebSocketDisconnect:
        log.info("WS disconnected")
    except Exception as exc:
        log.exception("WS error: %s", exc)
        await ws.close(code=1011)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")