"""
feature_extractor.py
══════════════════════════════════════════════════════════════════════════════
Pipeline: видеофайл → COVAREP (74-dim) + OpenFace AU (35-dim) + BERT текст
→ BottleneckFusionModel → предсказание эмоции.

Установка зависимостей:
    pip install opensmile feat faster-whisper opencv-python

Использование:
    from feature_extractor import VideoEmotionPipeline
    pipe = VideoEmotionPipeline()
    result = pipe.predict("video.mp4")
    print(result)
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional
import os

os.environ["TEMP"] = "C:/Temp"
os.environ["TMP"] = "C:/Temp"
import cv2
import numpy as np
import torch
from transformers import BertTokenizer

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Константы — должны совпадать с config.py
# ─────────────────────────────────────────────────────────────────────────────
AUDIO_SEQ_LEN  = 60
AUDIO_FEAT_DIM = 74   # COVAREP / ComParE совместимый размер
VISION_SEQ_LEN = 60
VISION_FEAT_DIM = 35  # OpenFace AU: 17 intensity + 18 presence = 35
TEXT_SEQ_LEN   = 50


# ─────────────────────────────────────────────────────────────────────────────
# БЛОК 1: Извлечение аудио из видео через ffmpeg
# ─────────────────────────────────────────────────────────────────────────────
def extract_audio_from_video(video_path: str, out_wav: str) -> bool:
    """
    Извлечь аудиодорожку из видео в WAV (16kHz, mono).
    Возвращает True если успешно.

    Требует: ffmpeg в PATH
        Windows: https://www.gyan.dev/ffmpeg/builds/ → ffmpeg-release-essentials.zip
                 распакуй и добавь bin/ в PATH
        или: winget install ffmpeg
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-ar", "16000",   # 16kHz — стандарт для речевых признаков
                "-ac", "1",       # mono
                "-vn",            # без видео
                out_wav,
            ],
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.error("ffmpeg error: %s", result.stderr.decode(errors="replace"))
            return False
        return Path(out_wav).exists() and Path(out_wav).stat().st_size > 0
    except FileNotFoundError:
        logger.error(
            "ffmpeg не найден. Установи: winget install ffmpeg  "
            "или скачай с https://www.gyan.dev/ffmpeg/builds/"
        )
        return False
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg timeout")
        return False
    

def compress_video_for_inference(video_path: str, out_path: str) -> bool:
    """
    Конвертирует любое видео (.mov/.mp4) в маленький mp4 для быстрого vision extraction.
    Audio не нужен, потому что audio извлекается из оригинального видео.
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-t", "6",
                "-vf", "fps=5,scale=360:-2",
                "-an",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                out_path,
            ],
            capture_output=True,
            timeout=120,
        )

        if result.returncode != 0:
            logger.error("ffmpeg compress error: %s", result.stderr.decode(errors="replace"))
            return False

        ok = Path(out_path).exists() and Path(out_path).stat().st_size > 0
        if ok:
            logger.info("✅ Fast video created for vision: %s", out_path)
        return ok

    except Exception as e:
        logger.error("Ошибка сжатия видео: %s", e)
        return False



# ─────────────────────────────────────────────────────────────────────────────
# БЛОК 2: COVAREP признаки через opensmile
# ─────────────────────────────────────────────────────────────────────────────
def extract_covarep_features(wav_path: str) -> Optional[np.ndarray]:
    """
    Извлечь акустические признаки через opensmile (ComParE_2016 LLD).

    ComParE_2016 LLD = 65 низкоуровневых дескрипторов:
      MFCCs (1-14), log Mel-энергия, F0, jitter, shimmer, HNR,
      spectral flux, centroid, rolloff — акустически близко к COVAREP.

    Возвращает np.ndarray shape [AUDIO_SEQ_LEN, AUDIO_FEAT_DIM] или None.

    pip install opensmile
    """
    # ── ФИКС: opensmile не умеет работать с путями содержащими кириллицу ────
    # Копируем WAV во временную папку с ASCII-путём перед передачей в opensmile
    safe_wav = None
    try:
        import opensmile

        tmp_dir = Path(tempfile.gettempdir())
        safe_wav = str(tmp_dir / f"smile_{uuid.uuid4().hex}.wav")
        shutil.copy2(wav_path, safe_wav)
        logger.info("opensmile: используем временный ASCII-путь: %s", safe_wav)

        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
            num_workers=1,
        )
        df = smile.process_file(safe_wav)

        if df.empty:
            logger.error("opensmile вернул пустой DataFrame")
            return None

        arr = df.values.astype(np.float32)  # [T, 65]
        logger.info("opensmile: извлечено %d фреймов x %d признаков", *arr.shape)

        # ── Z-нормализация по фрейму (убираем DC-offset и масштаб) ───────────
        mean = arr.mean(axis=0, keepdims=True)
        std  = arr.std(axis=0, keepdims=True) + 1e-8
        arr  = (arr - mean) / std

        # ── Ресемплинг до AUDIO_SEQ_LEN фреймов ──────────────────────────────
        T = arr.shape[0]
        if T >= AUDIO_SEQ_LEN:
            indices = np.linspace(0, T - 1, AUDIO_SEQ_LEN, dtype=int)
            arr = arr[indices]                                 # [60, 65]
        else:
            pad = np.zeros((AUDIO_SEQ_LEN - T, arr.shape[1]), dtype=np.float32)
            arr = np.vstack([arr, pad])                        # [60, 65]

        # ── Привести к AUDIO_FEAT_DIM = 74 (добавить нулевые столбцы) ────────
        F = arr.shape[1]
        if F < AUDIO_FEAT_DIM:
            pad_f = np.zeros((AUDIO_SEQ_LEN, AUDIO_FEAT_DIM - F), dtype=np.float32)
            arr = np.hstack([arr, pad_f])                      # [60, 74]
        elif F > AUDIO_FEAT_DIM:
            arr = arr[:, :AUDIO_FEAT_DIM]                      # [60, 74]

        return arr  # [60, 74]

    except ImportError:
        logger.error("opensmile не установлен: pip install opensmile")
        return None
    except Exception as e:
        logger.error("Ошибка извлечения COVAREP: %s", e, exc_info=True)
        return None
    finally:
        # Удаляем временный ASCII-файл в любом случае
        if safe_wav and Path(safe_wav).exists():
            try:
                Path(safe_wav).unlink()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# БЛОК 3: OpenFace AU признаки через py-feat
# ─────────────────────────────────────────────────────────────────────────────

# OpenFace 35-dim структура (должна совпадать с тренировочными данными):
# AU intensity (17): AU01, AU02, AU04, AU05, AU06, AU07, AU09, AU10,
#                    AU12, AU14, AU15, AU17, AU20, AU23, AU25, AU26, AU45
# AU presence (18):  те же AU + AU28 — бинарные {0, 1}
OPENFACE_AU_INTENSITY = [
    "AU01_r","AU02_r","AU04_r","AU05_r","AU06_r","AU07_r","AU09_r","AU10_r",
    "AU12_r","AU14_r","AU15_r","AU17_r","AU20_r","AU23_r","AU25_r","AU26_r","AU45_r",
]
OPENFACE_AU_PRESENCE = [
    "AU01_c","AU02_c","AU04_c","AU05_c","AU06_c","AU07_c","AU09_c","AU10_c",
    "AU12_c","AU14_c","AU15_c","AU17_c","AU20_c","AU23_c","AU25_c","AU26_c","AU28_c","AU45_c",
]
OPENFACE_COLS = OPENFACE_AU_INTENSITY + OPENFACE_AU_PRESENCE  # итого 35

_detector = None

def extract_openface_features(video_path: str) -> Optional[np.ndarray]:
    global _detector

    try:
        from feat import Detector

        if _detector is None:
            _detector = Detector(
                face_model="retinaface",
                landmark_model="mobilefacenet",
                au_model="xgb",
                facepose_model="img2pose",
                emotion_model="resmasknet",
                device="cpu",
            )

        # Берём 8 равномерных кадров из видео
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            logger.error("Не удалось прочитать видео")
            return None

        frame_indices = np.linspace(0, total_frames - 1, 3, dtype=int)
        tmp_dir = Path(tempfile.gettempdir())
        frame_paths = []

        for i, idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.resize(frame, (400, 400))
            frame_path = tmp_dir / f"feat_frame_{uuid.uuid4().hex}_{i}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(str(frame_path))

        cap.release()

        if not frame_paths:
            logger.error("Не удалось извлечь кадры из видео")
            return None

        result = _detector.detect_image(frame_paths)

        for p in frame_paths:
            Path(p).unlink(missing_ok=True)

        au_cols = [c for c in result.columns if "AU" in c]

        if not au_cols:
            logger.error("py-feat не вернул AU колонки. Доступные: %s", list(result.columns)[:50])
            return None

        arr = result[au_cols].fillna(0).values.astype(np.float32)
        logger.info("py-feat fast: извлечено %d кадров x %d AU признаков", *arr.shape)

        if arr.shape[0] > 1:
            mean = arr.mean(axis=0, keepdims=True)
            std = arr.std(axis=0, keepdims=True) + 1e-8
            arr = (arr - mean) / std

        T, F = arr.shape

        if T >= VISION_SEQ_LEN:
            indices = np.linspace(0, T - 1, VISION_SEQ_LEN, dtype=int)
            arr = arr[indices]
        else:
            pad = np.zeros((VISION_SEQ_LEN - T, F), dtype=np.float32)
            arr = np.vstack([arr, pad])

        if F < VISION_FEAT_DIM:
            pad_f = np.zeros((VISION_SEQ_LEN, VISION_FEAT_DIM - F), dtype=np.float32)
            arr = np.hstack([arr, pad_f])
        elif F > VISION_FEAT_DIM:
            arr = arr[:, :VISION_FEAT_DIM]

        return arr

    except ImportError:
        logger.error("py-feat не установлен: pip install feat")
        return None
    except Exception as e:
        logger.error("Ошибка извлечения OpenFace AU: %s", e, exc_info=True)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# БЛОК 4: Транскрипция речи через faster-whisper (опционально)
# ─────────────────────────────────────────────────────────────────────────────
def transcribe_audio(wav_path: str) -> Optional[str]:
    """
    Транскрибировать речь из WAV через faster-whisper (tiny модель, быстро).

    pip install faster-whisper

    Если faster-whisper не установлен — вернёт None,
    и модель будет работать без текстовой ветки (только audio + vision).
    """
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, info = model.transcribe(wav_path, language="en")
        text = " ".join(seg.text.strip() for seg in segments)
        logger.info("Whisper транскрипция: '%s'", text[:100])
        return text if text.strip() else None

    except ImportError:
        logger.warning(
            "faster-whisper не установлен (pip install faster-whisper). "
            "Text branch будет пустым."
        )
        return None
    except Exception as e:
        logger.error("Ошибка транскрипции: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# БЛОК 5: Токенизация текста через BERT
# ─────────────────────────────────────────────────────────────────────────────
_tokenizer: Optional[BertTokenizer] = None


def tokenize_text(
    text: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Токенизировать текст → (input_ids [1,50], attention_mask [1,50]).
    """
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    enc = _tokenizer(
        text,
        max_length=TEXT_SEQ_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


# ─────────────────────────────────────────────────────────────────────────────
# БЛОК 6: Главный Pipeline
# ─────────────────────────────────────────────────────────────────────────────
class VideoEmotionPipeline:
    """
    Полный pipeline: видеофайл → предсказание эмоции.

    Порядок работы:
    1. ffmpeg: видео → WAV (16kHz mono)
    2. opensmile: WAV → COVAREP признаки [60, 74]
    3. py-feat: видео → OpenFace AU признаки [60, 35]
    4. faster-whisper (опц.): WAV → текст → BERT токены [1, 50]
    5. BottleneckFusionModel: всё вместе → предсказание эмоции

    Пример:
        pipe = VideoEmotionPipeline()
        result = pipe.predict("my_video.mp4")
        # {'emotion': 'happy', 'confidence': 78.3, 'probabilities': {...},
        #  'modalities_used': {'text': True, 'audio': True, 'vision': True}}
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        use_whisper: bool = True,
        text_override: Optional[str] = None,
    ):
        """
        device        — torch device (auto-detect если None)
        use_whisper   — транскрибировать речь (требует faster-whisper)
        text_override — если у тебя уже есть транскрипция, передай её сюда
        """
        import model_loader
        from config import DEVICE

        self.device = device or DEVICE
        self.use_whisper = use_whisper
        self.text_override = text_override
        self._model_loader = model_loader

        # Загружаем модель если ещё не загружена
        if model_loader._model is None:
            model_loader.load_model()

    def predict(self, video_path: str) -> dict:
        """
        Предсказать эмоцию из видеофайла.

        Возвращает dict:
        {
            'emotion': str,
            'confidence': float (0-100),
            'probabilities': {emotion: float},
            'modalities_used': {'text': bool, 'audio': bool, 'vision': bool},
            'transcript': str | None,
        }
        """
        video_path = str(Path(video_path).resolve())
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Видеофайл не найден: {video_path}")

        modalities_used = {"text": False, "audio": False, "vision": False}
        transcript = None

        tmp_dir = Path(tempfile.gettempdir())
        uid = uuid.uuid4().hex
        tmp_wav = str(tmp_dir / f"emotion_{uid}.wav")
        tmp_video_fast = str(tmp_dir / f"emotion_fast_{uid}.mp4")

        try:
            # ── Шаг 1: извлечь аудио ─────────────────────────────────────────
            audio_ok = extract_audio_from_video(video_path, tmp_wav)
            video_fast_ok = compress_video_for_inference(video_path, tmp_video_fast)
            vision_source = tmp_video_fast if video_fast_ok else video_path

            # ── Шаг 2: COVAREP признаки ───────────────────────────────────────
            audio_tensor = None
            if audio_ok:
                audio_arr = extract_covarep_features(tmp_wav)
                if audio_arr is not None:
                    audio_tensor = torch.from_numpy(audio_arr).unsqueeze(0).to(self.device)
                    modalities_used["audio"] = True
                    logger.info("✅ Audio: COVAREP [%s]", tuple(audio_tensor.shape))
                else:
                    logger.warning("⚠️  COVAREP извлечение не удалось — audio=zeros")
            else:
                logger.warning("⚠️  Аудио не извлечено из видео — audio=zeros")

            # ── Шаг 3: OpenFace AU признаки ───────────────────────────────────
            vision_arr = extract_openface_features(vision_source)
            vision_tensor = None
            if vision_arr is not None:
                vision_tensor = torch.from_numpy(vision_arr).unsqueeze(0).to(self.device)
                modalities_used["vision"] = True
                logger.info("✅ Vision: OpenFace AU [%s]", tuple(vision_tensor.shape))
            else:
                logger.warning("⚠️  OpenFace AU извлечение не удалось — vision=zeros")

            # ── Шаг 4: Текст ──────────────────────────────────────────────────
            input_ids, attention_mask = None, None

            if self.text_override:
                # Используем переданный текст
                transcript = self.text_override
                logger.info("Текст (override): '%s'", transcript[:80])
            elif self.use_whisper and audio_ok:
                transcript = transcribe_audio(tmp_wav)

            if transcript and transcript.strip():
                input_ids, attention_mask = tokenize_text(transcript, self.device)
                modalities_used["text"] = True
                logger.info("✅ Text: BERT токенизирован ('%s...')", transcript[:50])
            else:
                logger.warning("⚠️  Текст отсутствует — text branch=zeros")

            # ── Шаг 5: Инференс ───────────────────────────────────────────────
            active = [k for k, v in modalities_used.items() if v]
            logger.info("Запуск инференса | активные модальности: %s", active)

            result = self._model_loader.run_inference(
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio=audio_tensor,
                vision_feats=vision_tensor,
            )

            result["modalities_used"] = modalities_used
            result["transcript"] = transcript

            logger.info(
                "Результат: %s (%.1f%%) | модальности: %s",
                result["emotion"], result["confidence"], active,
            )
            return result

        finally:
            # Удаляем временный WAV
            Path(tmp_wav).unlink(missing_ok=True)
            Path(tmp_video_fast).unlink(missing_ok=True)

    def predict_with_text(self, video_path: str, text: str) -> dict:
        """
        Удобный метод если у тебя уже есть транскрипция.

        Например, если речь на русском и Whisper плохо справляется,
        можно передать текст вручную или через другой ASR.
        """
        old = self.text_override
        self.text_override = text
        try:
            return self.predict(video_path)
        finally:
            self.text_override = old


# ─────────────────────────────────────────────────────────────────────────────
# Интеграция с FastAPI: функции для использования в main.py
# ─────────────────────────────────────────────────────────────────────────────
def extract_all_features(
    video_bytes: bytes,
    suffix: str = ".mp4",
    text_override: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Принять байты видеофайла, вернуть словарь с тензорами для run_inference().

    Использование в main.py:
        features = extract_all_features(video_bytes, suffix=".mp4")
        result = model_loader.run_inference(**features["inference_kwargs"])

    Возвращает:
    {
        "inference_kwargs": {
            "input_ids": Tensor | None,
            "attention_mask": Tensor | None,
            "audio": Tensor | None,
            "vision_feats": Tensor | None,
        },
        "modalities_used": {"text": bool, "audio": bool, "vision": bool},
        "transcript": str | None,
    }
    """
    if device is None:
        from config import DEVICE
        device = DEVICE

    tmp_dir = Path(tempfile.gettempdir())
    uid = uuid.uuid4().hex
    tmp_video = tmp_dir / f"emotion_video_{uid}{suffix}"
    tmp_wav   = tmp_dir / f"emotion_audio_{uid}.wav"
    tmp_video_fast = tmp_dir / f"emotion_fast_{uid}.mp4"

    modalities_used = {"text": False, "audio": False, "vision": False}
    transcript = None
    input_ids = attention_mask = audio_tensor = vision_tensor = None

    try:
        tmp_video.write_bytes(video_bytes)

        # Аудио
        audio_ok = extract_audio_from_video(str(tmp_video), str(tmp_wav))
        video_fast_ok = compress_video_for_inference(str(tmp_video), str(tmp_video_fast))
        vision_source = str(tmp_video_fast) if video_fast_ok else str(tmp_video)
        if audio_ok:
            audio_arr = extract_covarep_features(str(tmp_wav))
            if audio_arr is not None:
                audio_tensor = torch.from_numpy(audio_arr).unsqueeze(0).to(device)
                modalities_used["audio"] = True

        # Vision
        vision_arr = extract_openface_features(vision_source)
        if vision_arr is not None:
            vision_tensor = torch.from_numpy(vision_arr).unsqueeze(0).to(device)
            modalities_used["vision"] = True

        # Текст
        if text_override:
            transcript = text_override
        elif audio_ok:
            transcript = transcribe_audio(str(tmp_wav))

        if transcript and transcript.strip():
            input_ids, attention_mask = tokenize_text(transcript, device)
            modalities_used["text"] = True

    finally:
        tmp_video.unlink(missing_ok=True)
        tmp_wav.unlink(missing_ok=True)
        tmp_video_fast.unlink(missing_ok=True)

    return {
        "inference_kwargs": {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "audio": audio_tensor,
            "vision_feats": vision_tensor,
        },
        "modalities_used": modalities_used,
        "transcript": transcript,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Быстрая проверка (запусти отдельно для теста)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if len(sys.argv) < 2:
        print("Использование: python feature_extractor.py video.mp4")
        print("\nПроверка установки зависимостей:")

        deps = {
            "ffmpeg": ("subprocess", lambda: subprocess.run(
                ["ffmpeg", "-version"], capture_output=True).returncode == 0),
            "opensmile": ("pip install opensmile", lambda: __import__("opensmile") and True),
            "py-feat":   ("pip install feat",      lambda: __import__("feat") and True),
            "faster-whisper": ("pip install faster-whisper",
                               lambda: __import__("faster_whisper") and True),
            "transformers":   ("pip install transformers",
                               lambda: __import__("transformers") and True),
        }

        print()
        for name, (install_cmd, check) in deps.items():
            try:
                ok = check()
                status = "✅ установлен" if ok else "❌ не найден"
            except Exception:
                status = f"❌ не установлен → {install_cmd}"
            print(f"  {name:<20s} {status}")

        print("\nДля установки всех зависимостей:")
        print("  pip install opensmile feat faster-whisper")
        print("  + ffmpeg: winget install ffmpeg")
        sys.exit(0)

    video = sys.argv[1]
    text  = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"\nОбработка: {video}")
    print(f"Текст override: {text or '(нет — используем Whisper)'}\n")

    pipe = VideoEmotionPipeline(use_whisper=(text is None))

    if text:
        result = pipe.predict_with_text(video, text)
    else:
        result = pipe.predict(video)

    print("\n" + "═" * 50)
    print(f"  Эмоция     : {result['emotion']}")
    print(f"  Уверенность: {result['confidence']:.1f}%")
    print(f"  Транскрипция: {result.get('transcript', 'нет')}")
    print(f"  Модальности : {result['modalities_used']}")
    print("\n  Вероятности:")
    probs = sorted(result["probabilities"].items(), key=lambda x: -x[1])
    for emotion, prob in probs:
        bar = "█" * int(prob / 5)
        print(f"    {emotion:<10s} {prob:>6.2f}%  {bar}")
    print("═" * 50)