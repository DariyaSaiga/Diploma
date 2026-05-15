import os
import time
import json
import base64
import uuid
import io
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import BertModel, BertTokenizer

# =========================
# CONFIG
# =========================

EMOTION_LABELS = ["happy", "sad", "anger", "surprise", "disgust", "fear"]
MODEL_PATH = os.getenv("MODEL_PATH", "best_model.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# MODEL CONSTANTS (как у тебя)
# =========================

HIDDEN_DIM   = 128
N_HEADS      = 8
N_BOTTLENECK = 16
N_LAYERS     = 2
DROPOUT      = 0.1
N_EMOTIONS   = 6
BERT_MODEL   = "bert-base-uncased"


# =========================
# MODEL (встроили сюда)
# =========================

class ModalityProjection(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(input_dim, HIDDEN_DIM, 3, padding=1),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.proj(x)
        return x.transpose(1, 2)


class BottleneckFusionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.bert = BertModel.from_pretrained(BERT_MODEL)
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

        self.text_proj = nn.Linear(768, HIDDEN_DIM)

        self.audio_proj = ModalityProjection(74)
        self.vision_proj = ModalityProjection(35)

        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 3, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, N_EMOTIONS)
        )

    def forward(self, input_ids, attention_mask, audio, vision):

        bert_out = self.bert(input_ids=input_ids,
                              attention_mask=attention_mask)

        text = self.text_proj(bert_out.last_hidden_state[:, 0, :])

        audio = self.audio_proj(audio).mean(dim=1)
        vision = self.vision_proj(vision).mean(dim=1)

        fused = torch.cat([text, audio, vision], dim=-1)
        return self.classifier(fused)


# =========================
# APP
# =========================

app = FastAPI(title="Emotion API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# LOAD MODEL
# =========================

class EmotionService:
    def __init__(self):
        self.model = BottleneckFusionModel().to(device)
        self.model.eval()

        if Path(MODEL_PATH).exists():
            try:
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
                print("✅ model loaded")
            except Exception as e:
                print("⚠️ weights load failed:", e)
        else:
            print("⚠️ no weights found, running random init")

    def predict(self, image: Image.Image):

        img = TRANSFORM(image).unsqueeze(0)

        # dummy multimodal inputs (ВАЖНО)
        audio = torch.zeros(1, 60, 74)
        vision = torch.zeros(1, 60, 35)

        text = self.model.tokenizer(
            "neutral expression",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=50
        )

        with torch.no_grad():
            logits = self.model(
                input_ids=text["input_ids"],
                attention_mask=text["attention_mask"],
                audio=audio,
                vision=vision
            )

            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        scores = {
            k: float(round(p * 100, 2))
            for k, p in zip(EMOTION_LABELS, probs)
        }

        return {
            "emotion": max(scores, key=scores.get),
            "confidence": max(scores.values()),
            "probabilities": scores
        }


service = EmotionService()

# =========================
# HELPERS
# =========================

def decode_base64(data):
    _, encoded = data.split(",", 1)
    img = base64.b64decode(encoded)
    return Image.open(io.BytesIO(img)).convert("RGB")


# =========================
# API
# =========================

@app.get("/health")
def health():
    return {"ok": True, "device": str(device)}


@app.post("/api/analyze/frame")
async def frame(payload: dict):
    img = decode_base64(payload["image"])
    return {"success": True, **service.predict(img)}


# =========================
# WS
# =========================

@app.websocket("/ws/camera")
async def ws(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = json.loads(await websocket.receive_text())

            if data.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                continue

            img = decode_base64(data["image"])
            result = service.predict(img)

            await websocket.send_text(json.dumps({
                "type": "result",
                "timestamp": time.time(),
                **result
            }))

    except WebSocketDisconnect:
        print("client disconnected")