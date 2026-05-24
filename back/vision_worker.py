#!/usr/bin/env python3
"""
vision_worker.py — Изолированный subprocess для извлечения OpenFace AU признаков через py-feat.

Запускается из feature_extractor.py через subprocess.run(..., timeout=60).
Каждый запрос — чистый процесс: py-feat не зависает при повторных вызовах.

Аргументы:
    sys.argv[1] = абсолютный путь к видео (сжатый .mp4)

Вывод (stdout):
    {"status": "ok",    "features": [[...], ...]}   — успех
    {"status": "error", "message":  "..."}           — ошибка

Весь logging py-feat идёт в stderr и не мешает JSON в stdout.
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import uuid
from pathlib import Path

import cv2
import numpy as np

# ── Настройки (должны совпадать с config.py) ──────────────────────────────────
VISION_SEQ_LEN  = 60
VISION_FEAT_DIM = 35

# Перенаправляем логи py-feat в stderr, чтобы не замусорить stdout
logging.basicConfig(level=logging.WARNING, stream=sys.stderr)


def _ok(arr: np.ndarray) -> None:
    print(json.dumps({"status": "ok", "features": arr.tolist()}), flush=True)


def _err(msg: str) -> None:
    print(json.dumps({"status": "error", "message": msg}), flush=True)


def main() -> None:
    if len(sys.argv) < 2:
        _err("Не передан путь к видео (sys.argv[1])")
        sys.exit(1)

    video_path = sys.argv[1]

    if not Path(video_path).exists():
        _err(f"Файл не найден: {video_path}")
        sys.exit(1)

    frame_paths: list[str] = []

    try:
        # ── 1. Загружаем Detector ──────────────────────────────────────────────
        try:
            from feat import Detector
        except ImportError:
            _err("py-feat не установлен: pip install feat")
            sys.exit(1)

        detector = Detector(
            face_model="retinaface",
            landmark_model="mobilefacenet",
            au_model="xgb",
            facepose_model="img2pose",
            emotion_model="resmasknet",
            device="cpu",
        )

        # ── 2. Читаем 1 кадр из видео ─────────────────────────────────────────
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            _err("Не удалось прочитать видео (total_frames=0)")
            sys.exit(1)

        frame_indices = np.linspace(0, total_frames - 1, 1, dtype=int)
        tmp_dir = Path(tempfile.gettempdir())

        for i, idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            # Минимальный размер для img2pose — 400px
            frame = cv2.resize(frame, (400, 400))
            frame_path = tmp_dir / f"feat_frame_{uuid.uuid4().hex}_{i}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(str(frame_path))

        cap.release()

        if not frame_paths:
            _err("Не удалось извлечь ни одного кадра из видео")
            sys.exit(1)

        # ── 3. Детектируем AU ──────────────────────────────────────────────────
        result = detector.detect_image(frame_paths)

        au_cols = [c for c in result.columns if "AU" in c]
        if not au_cols:
            _err("py-feat не вернул AU-колонки")
            sys.exit(1)

        arr = result[au_cols].fillna(0).values.astype(np.float32)

        # ── 4. Нормализация (только если больше 1 кадра) ──────────────────────
        if arr.shape[0] > 1:
            mean = arr.mean(axis=0, keepdims=True)
            std  = arr.std(axis=0, keepdims=True) + 1e-8
            arr  = (arr - mean) / std

        # ── 5. Ресемплинг по времени → VISION_SEQ_LEN ─────────────────────────
        T, F = arr.shape
        if T >= VISION_SEQ_LEN:
            arr = arr[np.linspace(0, T - 1, VISION_SEQ_LEN, dtype=int)]
        else:
            pad = np.zeros((VISION_SEQ_LEN - T, F), dtype=np.float32)
            arr = np.vstack([arr, pad])

        # ── 6. Ресемплинг по признакам → VISION_FEAT_DIM ──────────────────────
        F = arr.shape[1]
        if F < VISION_FEAT_DIM:
            pad_f = np.zeros((VISION_SEQ_LEN, VISION_FEAT_DIM - F), dtype=np.float32)
            arr = np.hstack([arr, pad_f])
        elif F > VISION_FEAT_DIM:
            arr = arr[:, :VISION_FEAT_DIM]

        _ok(arr)

    except Exception as exc:
        _err(str(exc))
        sys.exit(1)

    finally:
        # Удаляем временные кадры
        for p in frame_paths:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    main()
