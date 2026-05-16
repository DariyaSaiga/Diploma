"""
main.py — Production FastAPI backend for Multimodal Emotion Recognition.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1

Установка зависимостей для полного pipeline:
    pip install opensmile feat faster-whisper
    + ffmpeg: winget install ffmpeg

Endpoints
─────────
GET  /health                        — liveness + model status
POST /api/analyze/frame             — single base64 image → emotions (vision=zeros, честный режим)
POST /api/analyze/audio             — аудиофайл → COVAREP через opensmile → emotions
POST /api/analyze/multimodal        — полный ввод: текст + COVAREP CSV + OpenFace CSV → emotions
POST /api/analyze/video             — видео → COVAREP + OpenFace AU + Whisper → полный инференс
WS   /ws/camera                     — real-time base64 frame stream → emotions
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from transformers import BertTokenizer

import model_loader
from feature_extractor import extract_all_features
from config import (
    API_VERSION,
    AUDIO_FEAT_DIM,
    AUDIO_SEQ_LEN,
    CORS_ORIGINS,
    DEVICE,
    MAX_FRAMES,
    TEXT_SEQ_LEN,
    VISION_FEAT_DIM,
    VISION_SEQ_LEN,
)
from utils import (
    aggregate_predictions,
    base64_to_tensor,
    extract_frames,
    frames_to_tensors,
    temp_video_file,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("emotion_api")

# ── BERT tokenizer (глобальный, грузится один раз) ────────────────────────────
_tokenizer: Optional[BertTokenizer] = None


def _get_tokenizer() -> BertTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return _tokenizer


def _tokenize(text: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Токенизировать текст → (input_ids [1,50], attention_mask [1,50])."""
    tok = _get_tokenizer()
    enc = tok(
        text,
        max_length=TEXT_SEQ_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)


# ── opensmile COVAREP extractor ───────────────────────────────────────────────
def _extract_covarep(audio_path: str) -> torch.Tensor:
    """
    Извлечь COVAREP-совместимые признаки через opensmile.
    Возвращает тензор [1, AUDIO_SEQ_LEN, AUDIO_FEAT_DIM] (zeros при ошибке).
    """
    try:
        import opensmile

        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )
        df = smile.process_file(audio_path)

        # ComParE_2016 LLD даёт 65 признаков, нам нужно AUDIO_FEAT_DIM (74)
        # Паддинг нулями до нужной размерности
        arr = df.values.astype(np.float32)  # [T, F]

        # Нормализуем длину до AUDIO_SEQ_LEN фреймов
        T = arr.shape[0]
        if T == 0:
            raise ValueError("opensmile вернул 0 фреймов")

        if T >= AUDIO_SEQ_LEN:
            # равномерная выборка
            indices = np.linspace(0, T - 1, AUDIO_SEQ_LEN, dtype=int)
            arr = arr[indices]
        else:
            # паддинг нулями снизу
            pad = np.zeros((AUDIO_SEQ_LEN - T, arr.shape[1]), dtype=np.float32)
            arr = np.vstack([arr, pad])

        # Паддинг/обрезка по признакам до AUDIO_FEAT_DIM
        F = arr.shape[1]
        if F >= AUDIO_FEAT_DIM:
            arr = arr[:, :AUDIO_FEAT_DIM]
        else:
            pad_f = np.zeros((AUDIO_SEQ_LEN, AUDIO_FEAT_DIM - F), dtype=np.float32)
            arr = np.hstack([arr, pad_f])

        tensor = torch.from_numpy(arr).unsqueeze(0).to(DEVICE)  # [1, 60, 74]
        logger.info("COVAREP извлечён: shape=%s", tensor.shape)
        return tensor

    except ImportError:
        logger.warning("opensmile не установлен — audio будет zeros")
        return torch.zeros(1, AUDIO_SEQ_LEN, AUDIO_FEAT_DIM, device=DEVICE)
    except Exception as exc:
        logger.error("Ошибка извлечения COVAREP: %s", exc, exc_info=True)
        return torch.zeros(1, AUDIO_SEQ_LEN, AUDIO_FEAT_DIM, device=DEVICE)


def _parse_feature_csv(
    csv_bytes: bytes, seq_len: int, feat_dim: int
) -> torch.Tensor:
    """
    Парсить CSV с признаками (строка = фрейм, столбец = признак).
    Возвращает [1, seq_len, feat_dim].
    """
    import io
    import pandas as pd

    df = pd.read_csv(io.BytesIO(csv_bytes))

    # убрать нечисловые столбцы (timestamp, frame и т.п.)
    df = df.select_dtypes(include=[np.number])
    arr = df.values.astype(np.float32)

    T, F = arr.shape

    # нормализовать длину
    if T >= seq_len:
        indices = np.linspace(0, T - 1, seq_len, dtype=int)
        arr = arr[indices]
    else:
        pad = np.zeros((seq_len - T, F), dtype=np.float32)
        arr = np.vstack([arr, pad])

    # нормализовать размерность признаков
    if F >= feat_dim:
        arr = arr[:, :feat_dim]
    else:
        pad_f = np.zeros((seq_len, feat_dim - F), dtype=np.float32)
        arr = np.hstack([arr, pad_f])

    return torch.from_numpy(arr).unsqueeze(0).to(DEVICE)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("═══ Emotion Recognition Service starting up ═══")
    logger.info("Device: %s", DEVICE)

    loop = asyncio.get_event_loop()

    # Загружаем модель и токенизатор параллельно
    await loop.run_in_executor(None, model_loader.load_model)
    await loop.run_in_executor(None, _get_tokenizer)

    info = model_loader.get_load_info()
    if model_loader.is_mock():
        logger.warning("⚠  Running in MOCK mode — %s", info.get("reason", "unknown"))
    else:
        logger.info("✔  Model loaded — %s", info)

    yield

    logger.info("═══ Emotion Recognition Service shutting down ═══")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Emotion Recognition API",
    description=(
        "Multimodal (text + COVAREP audio + OpenFace vision) emotion recognition "
        "via BottleneckFusionModel.\n\n"
        "**Важно:** модель обучена на COVAREP (74-dim) и OpenFace (35-dim) признаках. "
        "Endpoint `/api/analyze/frame` принимает картинку, но vision branch получает нули — "
        "используйте `/api/analyze/multimodal` для полноценного инференса."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        logger.error("Unhandled exception on %s %s: %s", request.method, request.url, exc)
        logger.debug(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )


# ── Schemas ───────────────────────────────────────────────────────────────────
class FrameRequest(BaseModel):
    image: str = Field(
        ...,
        description="Base64-encoded image (JPEG/PNG). Data-URI prefix stripped automatically.",
    )

    @field_validator("image")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("image field must not be empty")
        return v.strip()


class EmotionResponse(BaseModel):
    emotion: str
    confidence: float = Field(..., ge=0, le=100)
    probabilities: Dict[str, float]
    mock: bool = False
    latency_ms: float
    # Дополнительная диагностика — какие модальности реально использованы
    modalities_used: Dict[str, bool] = Field(
        default_factory=lambda: {"text": False, "audio": False, "vision": False}
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    mock_mode: bool
    device: str
    version: str
    load_info: Dict[str, Any]
    tokenizer_loaded: bool


class MultimodalRequest(BaseModel):
    """
    Полный мультимодальный запрос с предварительно извлечёнными признаками.

    Используй этот endpoint когда у тебя есть:
    - text: транскрипция utterance
    - audio_features: список фреймов COVAREP (каждый = список из 74 float)
    - vision_features: список фреймов OpenFace (каждый = список из 35 float)

    Все поля опциональны — недостающие модальности заполняются нулями.
    """

    text: Optional[str] = Field(None, description="Текст utterance для BERT")
    audio_features: Optional[List[List[float]]] = Field(
        None,
        description=f"COVAREP признаки: список фреймов, каждый = {AUDIO_FEAT_DIM} float. "
                    f"Будет ресемплирован до {AUDIO_SEQ_LEN} фреймов.",
    )
    vision_features: Optional[List[List[float]]] = Field(
        None,
        description=f"OpenFace признаки: список фреймов, каждый = {VISION_FEAT_DIM} float. "
                    f"Будет ресемплирован до {VISION_SEQ_LEN} фреймов.",
    )


# ── /health ───────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=model_loader._model is not None,
        mock_mode=model_loader.is_mock(),
        device=str(DEVICE),
        version=API_VERSION,
        load_info=model_loader.get_load_info(),
        tokenizer_loaded=_tokenizer is not None,
    )


# ── POST /api/analyze/frame ───────────────────────────────────────────────────
@app.post(
    "/api/analyze/frame",
    response_model=EmotionResponse,
    summary="Анализ по одному изображению (vision branch = zeros)",
    description=(
        "⚠️ **Ограничение**: модель обучена на OpenFace AU-признаках (35-dim), а не на пикселях. "
        "Vision branch получает нули. Результат определяется только text branch "
        "(тоже пустой, если текст не передан) — что даёт почти равномерное распределение.\n\n"
        "Для реального инференса используйте `/api/analyze/multimodal`."
    ),
    tags=["Inference"],
)
async def analyze_frame(body: FrameRequest) -> EmotionResponse:
    t0 = time.perf_counter()

    try:
        tensor = base64_to_tensor(body.image)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    # image_tensor передаём для совместимости, но модель его не использует
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: model_loader.run_inference(image_tensor=tensor),
        )
    except Exception as exc:
        logger.error("Inference failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Frame inference → %s (%.1f%%) | %.1f ms | mock=%s | "
        "⚠ vision=zeros (OpenFace недоступен для raw image)",
        result["emotion"], result["confidence"], latency_ms, model_loader.is_mock(),
    )

    return EmotionResponse(
        **result,
        mock=model_loader.is_mock(),
        latency_ms=round(latency_ms, 2),
        modalities_used={"text": False, "audio": False, "vision": False},
    )


# ── POST /api/analyze/audio ───────────────────────────────────────────────────
ALLOWED_AUDIO_TYPES = {
    "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp4",
    "audio/ogg", "audio/flac", "application/octet-stream",
}
MAX_AUDIO_MB = 50


@app.post(
    "/api/analyze/audio",
    response_model=EmotionResponse,
    summary="Анализ эмоций из аудиофайла через opensmile COVAREP",
    description=(
        "Загрузи WAV/MP3/FLAC файл. Backend извлекает COVAREP-совместимые признаки "
        "через opensmile и прогоняет через модель. Text и vision branch получают нули.\n\n"
        "**Требует**: `pip install opensmile`"
    ),
    tags=["Inference"],
)
async def analyze_audio(file: UploadFile = File(...)) -> EmotionResponse:
    t0 = time.perf_counter()

    if file.content_type and file.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Неподдерживаемый тип: {file.content_type}. Используй WAV, MP3 или FLAC.",
        )

    data = await file.read()
    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_AUDIO_MB:
        raise HTTPException(
            status_code=413,
            detail=f"Файл слишком большой ({size_mb:.1f} MB). Максимум {MAX_AUDIO_MB} MB.",
        )

    # определяем расширение
    suffix = ".wav"
    if file.filename:
        ext = "." + file.filename.rsplit(".", 1)[-1].lower()
        if ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
            suffix = ext

    def _process_sync() -> dict:
        import tempfile, uuid
        from pathlib import Path

        tmp = Path(tempfile.gettempdir()) / f"emotion_audio_{uuid.uuid4().hex}{suffix}"
        try:
            tmp.write_bytes(data)
            audio_tensor = _extract_covarep(str(tmp))
            return model_loader.run_inference(audio=audio_tensor)
        finally:
            tmp.unlink(missing_ok=True)

    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _process_sync)
    except Exception as exc:
        logger.error("Audio inference failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Audio processing error: {exc}")

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Audio inference → %s (%.1f%%) | %.1f ms | mock=%s",
        result["emotion"], result["confidence"], latency_ms, model_loader.is_mock(),
    )

    return EmotionResponse(
        **result,
        mock=model_loader.is_mock(),
        latency_ms=round(latency_ms, 2),
        modalities_used={"text": False, "audio": True, "vision": False},
    )


# ── POST /api/analyze/multimodal ──────────────────────────────────────────────
@app.post(
    "/api/analyze/multimodal",
    response_model=EmotionResponse,
    summary="Полный мультимодальный инференс: текст + COVAREP + OpenFace",
    description=(
        "**Это основной endpoint для production-инференса.**\n\n"
        "Принимает предварительно извлечённые признаки:\n"
        "- `text` → токенизируется через BERT (bert-base-uncased, max_length=50)\n"
        "- `audio_features` → список фреймов COVAREP (74 float каждый)\n"
        "- `vision_features` → список фреймов OpenFace AU (35 float каждый)\n\n"
        "Все поля опциональны. Недостающие модальности = zeros."
    ),
    tags=["Inference"],
)
async def analyze_multimodal(body: MultimodalRequest) -> EmotionResponse:
    t0 = time.perf_counter()
    modalities_used = {"text": False, "audio": False, "vision": False}

    def _build_and_infer() -> dict:
        # ── Текст ──────────────────────────────────────────────────────────────
        if body.text and body.text.strip():
            input_ids, attention_mask = _tokenize(body.text.strip())
            modalities_used["text"] = True
        else:
            input_ids, attention_mask = None, None

        # ── Аудио ──────────────────────────────────────────────────────────────
        if body.audio_features:
            arr = np.array(body.audio_features, dtype=np.float32)
            T, F = arr.shape

            if T >= AUDIO_SEQ_LEN:
                idx = np.linspace(0, T - 1, AUDIO_SEQ_LEN, dtype=int)
                arr = arr[idx]
            else:
                pad = np.zeros((AUDIO_SEQ_LEN - T, F), dtype=np.float32)
                arr = np.vstack([arr, pad])

            if F >= AUDIO_FEAT_DIM:
                arr = arr[:, :AUDIO_FEAT_DIM]
            else:
                pad_f = np.zeros((AUDIO_SEQ_LEN, AUDIO_FEAT_DIM - F), dtype=np.float32)
                arr = np.hstack([arr, pad_f])

            audio_tensor = torch.from_numpy(arr).unsqueeze(0).to(DEVICE)
            modalities_used["audio"] = True
        else:
            audio_tensor = None

        # ── Vision ─────────────────────────────────────────────────────────────
        if body.vision_features:
            arr = np.array(body.vision_features, dtype=np.float32)
            T, F = arr.shape

            if T >= VISION_SEQ_LEN:
                idx = np.linspace(0, T - 1, VISION_SEQ_LEN, dtype=int)
                arr = arr[idx]
            else:
                pad = np.zeros((VISION_SEQ_LEN - T, F), dtype=np.float32)
                arr = np.vstack([arr, pad])

            if F >= VISION_FEAT_DIM:
                arr = arr[:, :VISION_FEAT_DIM]
            else:
                pad_f = np.zeros((VISION_SEQ_LEN, VISION_FEAT_DIM - F), dtype=np.float32)
                arr = np.hstack([arr, pad_f])

            vision_tensor = torch.from_numpy(arr).unsqueeze(0).to(DEVICE)
            modalities_used["vision"] = True
        else:
            vision_tensor = None

        return model_loader.run_inference(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio=audio_tensor,
            vision_feats=vision_tensor,
        )

    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _build_and_infer)
    except Exception as exc:
        logger.error("Multimodal inference failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    latency_ms = (time.perf_counter() - t0) * 1000
    active = [k for k, v in modalities_used.items() if v]
    logger.info(
        "Multimodal inference → %s (%.1f%%) | %.1f ms | modalities=%s | mock=%s",
        result["emotion"], result["confidence"], latency_ms, active, model_loader.is_mock(),
    )

    return EmotionResponse(
        **result,
        mock=model_loader.is_mock(),
        latency_ms=round(latency_ms, 2),
        modalities_used=modalities_used,
    )


# ── POST /api/analyze/multimodal/csv ─────────────────────────────────────────
@app.post(
    "/api/analyze/multimodal/csv",
    response_model=EmotionResponse,
    summary="Мультимодальный инференс через загрузку CSV файлов",
    description=(
        "Альтернатива JSON endpoint — загружай CSV файлы напрямую.\n\n"
        "- `audio_csv`: CSV с COVAREP признаками (строка = фрейм, 74 столбца)\n"
        "- `vision_csv`: CSV с OpenFace AU признаками (строка = фрейм, 35 столбцов)\n"
        "- `text`: текст utterance (form field)\n\n"
        "Нечисловые столбцы (timestamp, frame_id и т.п.) удаляются автоматически."
    ),
    tags=["Inference"],
)
async def analyze_multimodal_csv(
    text: Optional[str] = None,
    audio_csv: Optional[UploadFile] = File(None),
    vision_csv: Optional[UploadFile] = File(None),
) -> EmotionResponse:
    t0 = time.perf_counter()
    modalities_used = {"text": False, "audio": False, "vision": False}

    def _build_and_infer() -> dict:
        input_ids, attention_mask = None, None
        audio_tensor, vision_tensor = None, None

        if text and text.strip():
            input_ids, attention_mask = _tokenize(text.strip())
            modalities_used["text"] = True

        return model_loader.run_inference(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio=audio_tensor,
            vision_feats=vision_tensor,
        )

    # Читаем CSV файлы (async, перед executor)
    audio_bytes = await audio_csv.read() if audio_csv else None
    vision_bytes = await vision_csv.read() if vision_csv else None

    def _full_process() -> dict:
        input_ids, attention_mask = None, None
        audio_tensor, vision_tensor = None, None

        if text and text.strip():
            input_ids, attention_mask = _tokenize(text.strip())
            modalities_used["text"] = True

        if audio_bytes:
            audio_tensor = _parse_feature_csv(audio_bytes, AUDIO_SEQ_LEN, AUDIO_FEAT_DIM)
            modalities_used["audio"] = True

        if vision_bytes:
            vision_tensor = _parse_feature_csv(vision_bytes, VISION_SEQ_LEN, VISION_FEAT_DIM)
            modalities_used["vision"] = True

        return model_loader.run_inference(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio=audio_tensor,
            vision_feats=vision_tensor,
        )

    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _full_process)
    except Exception as exc:
        logger.error("CSV multimodal inference failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    latency_ms = (time.perf_counter() - t0) * 1000
    active = [k for k, v in modalities_used.items() if v]
    logger.info(
        "CSV Multimodal → %s (%.1f%%) | %.1f ms | modalities=%s",
        result["emotion"], result["confidence"], latency_ms, active,
    )

    return EmotionResponse(
        **result,
        mock=model_loader.is_mock(),
        latency_ms=round(latency_ms, 2),
        modalities_used=modalities_used,
    )


# ── POST /api/analyze/video ───────────────────────────────────────────────────
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/webm", "video/quicktime", "application/octet-stream"}
MAX_VIDEO_MB = 100


@app.post(
    "/api/analyze/video",
    response_model=EmotionResponse,
    summary="Полный мультимодальный анализ видео",
    description=(
        "Загрузи видеофайл. Backend автоматически:\n\n"
        "1. **ffmpeg** → извлекает аудиодорожку (WAV 16kHz)\n"
        "2. **opensmile** → COVAREP признаки [60, 74] из аудио\n"
        "3. **py-feat** → OpenFace AU признаки [60, 35] из видеофреймов\n"
        "4. **faster-whisper** → транскрипция речи → BERT токены [1, 50]\n"
        "5. **BottleneckFusionModel** → предсказание эмоции по всем трём модальностям\n\n"
        "Ответ содержит `modalities_used` — видно какие ветки реально получили данные.\n\n"
        "**Требует**: `pip install opensmile feat faster-whisper` + ffmpeg в PATH\n\n"
        "Опционально: передай `text` form-field если речь на другом языке "
        "или Whisper недоступен."
    ),
    tags=["Inference"],
)
async def analyze_video(
    file: UploadFile = File(...),
    text: Optional[str] = None,   # опциональная транскрипция вручную
) -> EmotionResponse:
    t0 = time.perf_counter()

    if file.content_type and file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {file.content_type}. Use mp4 or webm.",
        )

    data = await file.read()
    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_VIDEO_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum is {MAX_VIDEO_MB} MB.",
        )

    suffix = ".mp4"
    if file.filename:
        ext = "." + file.filename.rsplit(".", 1)[-1].lower()
        if ext in (".mp4", ".webm", ".mov"):
            suffix = ext

    def _process_sync() -> dict:
        # extract_all_features: видео → COVAREP + OpenFace AU + Whisper
        features = extract_all_features(
            video_bytes=data,
            suffix=suffix,
            text_override=text,   # None = используем Whisper
            device=DEVICE,
        )

        result = model_loader.run_inference(**features["inference_kwargs"])
        result["modalities_used"] = features["modalities_used"]
        result["transcript"]      = features["transcript"]
        return result

    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _process_sync)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Video inference failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Video processing error: {exc}")

    latency_ms = (time.perf_counter() - t0) * 1000
    modalities_used = result.pop("modalities_used", {})
    transcript      = result.pop("transcript", None)

    active = [k for k, v in modalities_used.items() if v]
    logger.info(
        "Video inference → %s (%.1f%%) | %.1f ms | modalities=%s | mock=%s",
        result["emotion"], result["confidence"], latency_ms, active, model_loader.is_mock(),
    )

    return EmotionResponse(
        **result,
        mock=model_loader.is_mock(),
        latency_ms=round(latency_ms, 2),
        modalities_used=modalities_used,
    )


# ── WebSocket /ws/camera ──────────────────────────────────────────────────────
@app.websocket("/ws/camera")
async def camera_websocket(websocket: WebSocket) -> None:
    """
    Real-time emotion recognition over WebSocket.

    Client sends JSON:
        {"type": "frame", "image": "<base64>"}
        {"type": "frame", "image": "<base64>", "text": "utterance text"}
        {"type": "ping"}

    Server replies:
        {"type": "prediction", "emotion": ..., "confidence": ...,
         "probabilities": {...}, "modalities_used": {...}, "latency_ms": ...}
        {"type": "pong"}
        {"type": "error", "detail": "..."}
    """
    await websocket.accept()
    client = websocket.client
    logger.info("WebSocket connected: %s", client)

    loop = asyncio.get_event_loop()

    try:
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"type": "pong", "reason": "timeout_keepalive"}))
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "detail": "Invalid JSON"}))
                continue

            msg_type = msg.get("type", "")

            if msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                continue

            if msg_type == "frame":
                b64 = msg.get("image")
                text_input = msg.get("text")  # опциональный текст

                if not b64:
                    await websocket.send_text(
                        json.dumps({"type": "error", "detail": "Missing 'image' field"})
                    )
                    continue

                t0 = time.perf_counter()

                try:
                    tensor = base64_to_tensor(b64)
                except ValueError as exc:
                    await websocket.send_text(
                        json.dumps({"type": "error", "detail": f"Image decode error: {exc}"})
                    )
                    continue

                modalities_used = {"text": False, "audio": False, "vision": False}

                def _ws_infer():
                    ids, mask = None, None
                    if text_input and text_input.strip():
                        ids, mask = _tokenize(text_input.strip())
                        modalities_used["text"] = True
                    return model_loader.run_inference(
                        image_tensor=tensor,
                        input_ids=ids,
                        attention_mask=mask,
                    )

                try:
                    result = await loop.run_in_executor(None, _ws_infer)
                except Exception as exc:
                    logger.error("WS inference error: %s", exc)
                    await websocket.send_text(
                        json.dumps({"type": "error", "detail": f"Inference error: {exc}"})
                    )
                    continue

                latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                payload = {
                    "type":            "prediction",
                    "emotion":         result["emotion"],
                    "confidence":      result["confidence"],
                    "probabilities":   result["probabilities"],
                    "mock":            model_loader.is_mock(),
                    "modalities_used": modalities_used,
                    "latency_ms":      latency_ms,
                }
                await websocket.send_text(json.dumps(payload))

            else:
                await websocket.send_text(
                    json.dumps({"type": "error", "detail": f"Unknown message type: {msg_type!r}"})
                )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: %s", client)
    except Exception as exc:
        logger.error("WebSocket fatal error (%s): %s", client, exc, exc_info=True)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


# ── Dev entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info",
    )