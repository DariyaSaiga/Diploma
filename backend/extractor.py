"""
extractor.py — извлечение признаков из загруженных файлов.

Поддерживает:
  - text:  прямой ввод (передаётся как есть в BERT)
  - image: извлечение одного кадра → visual features (T=1)
  - video: извлечение кадров + аудиодорожки → visual + audio features
  - audio: извлечение аудио-признаков из аудиофайла

Особенности:
  - Модель обучена на COVAREP (74-dim) и OpenFace_2 (713-dim) признаках
  - Для демо мы используем приблизительные извлечённые признаки
  - Текстовая ветка (BERT) — самая сильная, работает точно
"""

import io
import tempfile
import os
import numpy as np

# Опциональные зависимости — fallback к нулевым признакам если не установлены
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[extractor] WARNING: opencv not found, visual features will be zeros")

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("[extractor] WARNING: librosa not found, audio features will be zeros")


AUDIO_DIM = 74
VISUAL_DIM = 713


# ══════════════════════════════════════════════════════════════════════════════
# Audio feature extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_audio_features(audio_path: str, sr: int = 16000) -> np.ndarray:
    """
    Извлекает аудио-признаки из файла.

    Приблизительная замена COVAREP-признаков:
      - MFCC (20 коэф.)
      - Delta MFCC (20)
      - Chroma (12)
      - Spectral contrast (7)
      - Spectral centroid (1)
      - Spectral bandwidth (1)
      - Spectral rolloff (1)
      - ZCR (1)
      - RMS energy (1)
      - F0 оценка (1)
      - Padding до 74 нулями (9)

    Returns:
        np.ndarray shape (T, 74) — T фреймов по 74 признака
    """
    if not HAS_LIBROSA:
        return np.zeros((1, AUDIO_DIM), dtype=np.float32)

    try:
        y, sr_loaded = librosa.load(audio_path, sr=sr, mono=True)
        if len(y) == 0:
            return np.zeros((1, AUDIO_DIM), dtype=np.float32)

        hop = 512
        n_fft = 2048

        # Основные признаки (per frame)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop, n_fft=n_fft)             # (20, T)
        delta_mfcc = librosa.feature.delta(mfcc)                                                      # (20, T)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop, n_fft=n_fft)                # (12, T)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop, n_fft=n_fft)        # (7, T)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop, n_fft=n_fft)        # (1, T)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop, n_fft=n_fft)      # (1, T)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop, n_fft=n_fft)          # (1, T)
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop)                                  # (1, T)
        rms = librosa.feature.rms(y=y, hop_length=hop)                                               # (1, T)

        # F0 (fundamental frequency)
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr, hop_length=hop)                               # (T,)
        f0 = np.nan_to_num(f0, nan=0.0).reshape(1, -1)                                              # (1, T)

        # Собираем: 20+20+12+7+1+1+1+1+1+1 = 65 признаков
        T = mfcc.shape[1]
        # Выравниваем длины (f0 может быть чуть другой длины)
        min_T = min(T, delta_mfcc.shape[1], chroma.shape[1], contrast.shape[1],
                     centroid.shape[1], bandwidth.shape[1], rolloff.shape[1],
                     zcr.shape[1], rms.shape[1], f0.shape[1])

        features = np.vstack([
            mfcc[:, :min_T],
            delta_mfcc[:, :min_T],
            chroma[:, :min_T],
            contrast[:, :min_T],
            centroid[:, :min_T],
            bandwidth[:, :min_T],
            rolloff[:, :min_T],
            zcr[:, :min_T],
            rms[:, :min_T],
            f0[:, :min_T],
        ])  # (65, T)

        # Дополняем до 74 нулями
        if features.shape[0] < AUDIO_DIM:
            pad = np.zeros((AUDIO_DIM - features.shape[0], min_T), dtype=np.float32)
            features = np.vstack([features, pad])

        features = features[:AUDIO_DIM].T.astype(np.float32)  # (T, 74)

        # Нормализация (z-score per feature)
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + 1e-8
        features = (features - mean) / std
        features = np.clip(features, -5.0, 5.0)

        return features

    except Exception as e:
        print(f"[extractor] Audio extraction failed: {e}")
        return np.zeros((1, AUDIO_DIM), dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Visual feature extraction
# ══════════════════════════════════════════════════════════════════════════════

def _extract_face_features_from_frame(frame: np.ndarray) -> np.ndarray:
    """
    Извлекает визуальные признаки из одного кадра.

    Приблизительная замена OpenFace_2:
      - Гистограмма ориентированных градиентов (HOG) лица
      - Пиксельные интенсивности лицевой области
      - Нормализованные координаты лица

    Returns:
        np.ndarray shape (713,) — один фрейм визуальных признаков
    """
    if not HAS_CV2:
        return np.zeros(VISUAL_DIM, dtype=np.float32)

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # Пытаемся найти лицо
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            # Берём самое большое лицо
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_roi = gray[y:y+h, x:x+w]
        else:
            # Нет лица → берём центральную область
            h, w = gray.shape[:2]
            cx, cy = w // 2, h // 2
            size = min(h, w) // 2
            face_roi = gray[max(0, cy-size):cy+size, max(0, cx-size):cx+size]

        if face_roi.size == 0:
            return np.zeros(VISUAL_DIM, dtype=np.float32)

        # Resize лица в фиксированный размер и извлекаем features
        face_resized = cv2.resize(face_roi, (32, 32)).astype(np.float32) / 255.0

        # Flatten + дополнительные статистики
        pixel_features = face_resized.flatten()  # 1024

        # Добавляем Sobel-градиенты (горизонтальные + вертикальные)
        sobelx = cv2.Sobel(face_resized, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(face_resized, cv2.CV_32F, 0, 1, ksize=3)
        grad_features = np.concatenate([
            sobelx.flatten()[:200],
            sobely.flatten()[:200],
        ])

        # Собираем всё вместе
        all_features = np.concatenate([pixel_features, grad_features])

        # Truncate или pad до 713
        if len(all_features) >= VISUAL_DIM:
            result = all_features[:VISUAL_DIM]
        else:
            result = np.zeros(VISUAL_DIM, dtype=np.float32)
            result[:len(all_features)] = all_features

        return result.astype(np.float32)

    except Exception as e:
        print(f"[extractor] Face feature extraction failed: {e}")
        return np.zeros(VISUAL_DIM, dtype=np.float32)


def extract_visual_features_from_image(image_bytes: bytes) -> np.ndarray:
    """
    Извлекает визуальные признаки из изображения.

    Returns:
        np.ndarray shape (1, 713) — один кадр
    """
    if not HAS_CV2:
        return np.zeros((1, VISUAL_DIM), dtype=np.float32)

    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return np.zeros((1, VISUAL_DIM), dtype=np.float32)

        features = _extract_face_features_from_frame(frame)

        # Нормализация
        mean = features.mean()
        std = features.std() + 1e-8
        features = (features - mean) / std
        features = np.clip(features, -5.0, 5.0)

        return features.reshape(1, VISUAL_DIM)

    except Exception as e:
        print(f"[extractor] Image extraction failed: {e}")
        return np.zeros((1, VISUAL_DIM), dtype=np.float32)


def extract_visual_features_from_video(video_path: str, max_frames: int = 50) -> np.ndarray:
    """
    Извлекает визуальные признаки из видео (каждый N-й кадр).

    Returns:
        np.ndarray shape (T, 713)
    """
    if not HAS_CV2:
        return np.zeros((1, VISUAL_DIM), dtype=np.float32)

    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return np.zeros((1, VISUAL_DIM), dtype=np.float32)

        # Выбираем равномерно max_frames кадров
        step = max(1, total_frames // max_frames)
        features_list = []

        frame_idx = 0
        while cap.isOpened() and len(features_list) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                feat = _extract_face_features_from_frame(frame)
                features_list.append(feat)
            frame_idx += 1

        cap.release()

        if not features_list:
            return np.zeros((1, VISUAL_DIM), dtype=np.float32)

        features = np.stack(features_list)  # (T, 713)

        # Нормализация per-feature
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + 1e-8
        features = (features - mean) / std
        features = np.clip(features, -5.0, 5.0).astype(np.float32)

        return features

    except Exception as e:
        print(f"[extractor] Video visual extraction failed: {e}")
        return np.zeros((1, VISUAL_DIM), dtype=np.float32)


def extract_audio_from_video(video_path: str) -> str | None:
    """
    Извлекает аудиодорожку из видео во временный WAV-файл.

    Returns:
        путь к временному WAV-файлу или None
    """
    if not HAS_LIBROSA:
        return None

    try:
        # librosa может читать аудио прямо из видеофайлов (через soundfile/ffmpeg)
        y, sr = librosa.load(video_path, sr=16000, mono=True)
        if len(y) == 0:
            return None

        # Сохраняем во временный файл
        import soundfile as sf
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, y, sr)
        return tmp.name

    except Exception as e:
        print(f"[extractor] Audio from video extraction failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Главная функция: обработка загруженного файла
# ══════════════════════════════════════════════════════════════════════════════

def process_upload(
    file_bytes: bytes,
    filename: str,
    content_type: str,
    text: str = "",
) -> dict:
    """
    Обрабатывает загруженный файл и возвращает признаки для модели.

    Returns:
        dict с ключами: text, audio (np.ndarray), visual (np.ndarray)
    """
    audio_features = None
    visual_features = None

    # ── Image ──────────────────────────────────────────────────────────
    if content_type and content_type.startswith("image/"):
        visual_features = extract_visual_features_from_image(file_bytes)

    # ── Video ──────────────────────────────────────────────────────────
    elif content_type and content_type.startswith("video/"):
        # Сохраняем во временный файл
        suffix = os.path.splitext(filename)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            visual_features = extract_visual_features_from_video(tmp_path)

            # Пробуем извлечь аудио из видео
            audio_tmp = extract_audio_from_video(tmp_path)
            if audio_tmp:
                audio_features = extract_audio_features(audio_tmp)
                os.unlink(audio_tmp)
        finally:
            os.unlink(tmp_path)

    # ── Audio ──────────────────────────────────────────────────────────
    elif content_type and content_type.startswith("audio/"):
        suffix = os.path.splitext(filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            audio_features = extract_audio_features(tmp_path)
        finally:
            os.unlink(tmp_path)

    return {
        "text": text,
        "audio": audio_features,
        "visual": visual_features,
    }
