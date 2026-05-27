#!/usr/bin/env python3
"""FastAPI inference service for melanoma checkpoint."""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = Path(
    os.getenv("MODEL_CHECKPOINT_PATH", str(ROOT / "checkpoints" / "melanoma_best.keras"))
)
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))
CLASS_NAMES: tuple[str, str] = ("benign", "malignant")


def _parse_allowed_origins() -> list[str]:
    raw_origins = os.getenv("ALLOWED_ORIGINS", "*").strip()
    if raw_origins == "*":
        return ["*"]
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]

app = FastAPI(title="Melanoma Inference API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_allowed_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: tf.keras.Model | None = None


def _load_model() -> tf.keras.Model:
    global _model
    if _model is not None:
        return _model
    if not DEFAULT_CHECKPOINT.is_file():
        raise RuntimeError(f"Checkpoint not found: {DEFAULT_CHECKPOINT}")
    _model = tf.keras.models.load_model(str(DEFAULT_CHECKPOINT))
    return _model


def _preprocess(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "checkpoint_exists": DEFAULT_CHECKPOINT.is_file(),
        "model_loaded": _model is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty upload.")

    model = _load_model()
    batch = _preprocess(image_bytes)
    proba = model.predict(batch, verbose=0)[0]
    scores = {name: float(proba[i]) for i, name in enumerate(CLASS_NAMES)}
    label = max(scores, key=scores.get)

    return {
        "label": label,
        "scores": scores,
    }

