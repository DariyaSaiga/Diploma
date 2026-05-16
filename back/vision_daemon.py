#!/usr/bin/env python3
"""
vision_daemon.py — Долгоживущий subprocess для py-feat AU extraction.

Протокол (stdin/stdout):
  Stdout → родителю:
    "READY\\n"                          — модели загружены
    JSON {"status":"ok","features":[]}\\n  — результат
    JSON {"status":"error","message":""}\\n — ошибка

  Stdin ← от родителя:
    <абсолютный путь к .mp4>\\n

Логи py-feat → stderr (не мешают JSON в stdout).
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

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
for _lg in ("feat", "feat.detector", "py_feat", "huggingface_hub", "transformers"):
    logging.getLogger(_lg).setLevel(logging.ERROR)

VISION_SEQ_LEN  = 60
VISION_FEAT_DIM = 35


def _extract(detector, video_path: str) -> np.ndarray:
    frame_paths: list[str] = []
    tmp_dir = Path(tempfile.gettempdir())
    try:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise ValueError(f"Не удалось прочитать видео: {video_path}")

        for i, idx in enumerate(np.linspace(0, total - 1, 1, dtype=int)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.resize(frame, (400, 400))
            p = tmp_dir / f"feat_{uuid.uuid4().hex}_{i}.jpg"
            cv2.imwrite(str(p), frame)
            frame_paths.append(str(p))
        cap.release()

        if not frame_paths:
            raise ValueError("Не удалось извлечь кадры")

        result = detector.detect_image(frame_paths)
        au_cols = [c for c in result.columns if "AU" in c]
        if not au_cols:
            raise ValueError("py-feat не вернул AU-колонки")

        arr = result[au_cols].fillna(0).values.astype(np.float32)

        T, F = arr.shape
        if T >= VISION_SEQ_LEN:
            arr = arr[np.linspace(0, T - 1, VISION_SEQ_LEN, dtype=int)]
        else:
            arr = np.vstack([arr, np.zeros((VISION_SEQ_LEN - T, F), dtype=np.float32)])

        F = arr.shape[1]
        if F < VISION_FEAT_DIM:
            arr = np.hstack([arr, np.zeros((VISION_SEQ_LEN, VISION_FEAT_DIM - F), dtype=np.float32)])
        elif F > VISION_FEAT_DIM:
            arr = arr[:, :VISION_FEAT_DIM]

        return arr
    finally:
        for p in frame_paths:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass


def main() -> None:
    try:
        from feat import Detector
    except ImportError:
        print(json.dumps({"status": "error", "message": "py-feat не установлен"}), flush=True)
        sys.exit(1)

    try:
        print("LOADING", flush=True)
        detector = Detector(
            face_model="retinaface",
            landmark_model="mobilefacenet",
            au_model="xgb",
            facepose_model="img2pose",
            emotion_model="resmasknet",
            device="cpu",
        )
    except Exception as exc:
        print(json.dumps({"status": "error", "message": f"Detector init failed: {exc}"}), flush=True)
        sys.exit(1)

    print("READY", flush=True)

    for line in sys.stdin:
        video_path = line.strip()
        if not video_path:
            continue
        try:
            arr = _extract(detector, video_path)
            print(json.dumps({"status": "ok", "features": arr.tolist()}), flush=True)
        except Exception as exc:
            print(json.dumps({"status": "error", "message": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
