"""
app.py — FastAPI сервер для распознавания эмоций.

Endpoints:
  GET  /api/health   — проверка сервера
  POST /api/predict   — предсказание эмоции

Запуск:
  cd backend/
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from predictor import EmotionPredictor
from extractor import process_upload

# ── Инициализация ──────────────────────────────────────────────────────────
app = FastAPI(
    title="MERS — Multimodal Emotion Recognition System",
    version="1.0.0",
)

# CORS для frontend (dev-режим — разрешаем всё)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Модель загружается один раз при старте сервера
predictor: EmotionPredictor | None = None


@app.on_event("startup")
async def load_model():
    global predictor
    print("=" * 60)
    print("  MERS Backend — Starting up")
    print("=" * 60)
    predictor = EmotionPredictor()
    print("=" * 60)
    print("  Server ready!")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
    }


@app.post("/api/predict")
async def predict(
    file: UploadFile | None = File(default=None),
    text: str = Form(default=""),
):
    """
    Предсказание эмоции.

    Принимает:
      - file: изображение, видео или аудио (optional)
      - text: текстовый ввод (optional)

    Хотя бы одно из двух (file или text) должно быть предоставлено.
    """
    if predictor is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model not loaded yet"},
        )

    if not file and not text.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "Provide a file or text input"},
        )

    start = time.time()

    # ── Обработка файла ────────────────────────────────────────────────
    audio_features = None
    visual_features = None

    if file and file.filename:
        file_bytes = await file.read()
        content_type = file.content_type or ""

        extracted = process_upload(
            file_bytes=file_bytes,
            filename=file.filename,
            content_type=content_type,
            text=text,
        )
        text = extracted["text"] or text
        audio_features = extracted["audio"]
        visual_features = extracted["visual"]

    # ── Предсказание ───────────────────────────────────────────────────
    result = predictor.predict(
        text=text,
        audio=audio_features,
        visual=visual_features,
    )

    elapsed = round((time.time() - start) * 1000)
    result["latency_ms"] = elapsed

    # Какие модальности были реально использованы
    result["modalities_used"] = {
        "text": bool(text and text.strip() and text != "[UNK]"),
        "audio": audio_features is not None and audio_features.sum() != 0,
        "visual": visual_features is not None and visual_features.sum() != 0,
    }

    return result
