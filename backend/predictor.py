"""
predictor.py — загрузка модели и inference.

Загружает best_model.pt один раз при старте сервера.
Принимает готовые признаки, возвращает вероятности по 6 эмоциям.
"""

import os
import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from bottleneck_fusion import BottleneckFusion

EMOTION_NAMES = ["happy", "sad", "anger", "surprise", "disgust", "fear"]

# Параметры модели (совпадают с training)
MODEL_PARAMS = dict(
    num_classes=6,
    hidden_dim=256,
    num_bottleneck_tokens=16,
    num_bottleneck_layers=2,
    num_heads=4,
    dropout=0.3,
    freeze_bert=True,
    use_audio=True,
    use_visual=True,
)

MAX_TEXT_LEN = 128
MAX_AUDIO_LEN = 100
MAX_VISUAL_LEN = 100


class EmotionPredictor:
    def __init__(self, model_path: str = None, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "best_model.pt")

        print(f"[Predictor] Loading model from {model_path}")
        print(f"[Predictor] Device: {self.device}")

        # Модель
        self.model = BottleneckFusion(**MODEL_PARAMS)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print("[Predictor] Model loaded successfully")

        # Токенизатор
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        print("[Predictor] Tokenizer ready")

    def predict(
        self,
        text: str = "",
        audio: "np.ndarray | None" = None,
        visual: "np.ndarray | None" = None,
    ) -> dict:
        """
        Основной метод предсказания.

        Args:
            text:   строка текста (может быть пустой)
            audio:  numpy array (T, 74) или None
            visual: numpy array (T, 713) или None

        Returns:
            dict с emotion, confidence, probabilities
        """
        import numpy as np

        # ── Text: BERT tokenization ────────────────────────────────────
        if not text or not text.strip():
            text = "[UNK]"  # пустой текст → один токен

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_TEXT_LEN,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)               # (1, Lt)
        attention_mask = encoding["attention_mask"].to(self.device)     # (1, Lt)

        # ── Audio ──────────────────────────────────────────────────────
        if audio is None or audio.size == 0:
            audio_tensor = torch.zeros(1, 1, 74, device=self.device)
            audio_mask = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        else:
            audio = audio[:MAX_AUDIO_LEN]
            T_a = audio.shape[0]
            audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.device)
            audio_mask = torch.ones(1, T_a, dtype=torch.long, device=self.device)

        # ── Visual ─────────────────────────────────────────────────────
        if visual is None or visual.size == 0:
            visual_tensor = torch.zeros(1, 1, 713, device=self.device)
            visual_mask = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        else:
            visual = visual[:MAX_VISUAL_LEN]
            T_v = visual.shape[0]
            visual_tensor = torch.tensor(visual, dtype=torch.float32).unsqueeze(0).to(self.device)
            visual_mask = torch.ones(1, T_v, dtype=torch.long, device=self.device)

        # ── Inference ──────────────────────────────────────────────────
        with torch.no_grad():
            logits = self.model(
                input_ids, attention_mask,
                audio_tensor, audio_mask,
                visual_tensor, visual_mask,
            )  # (1, 6)

        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        # ── Формируем ответ ────────────────────────────────────────────
        top_idx = int(probs.argmax())
        probabilities = {
            name: round(float(probs[i]) * 100, 1)
            for i, name in enumerate(EMOTION_NAMES)
        }

        return {
            "emotion": EMOTION_NAMES[top_idx],
            "confidence": round(float(probs[top_idx]) * 100, 1),
            "probabilities": probabilities,
        }
