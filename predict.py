#!/usr/bin/env python3
"""Run inference on a single image with the trained melanoma CNN."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from datasets.load import CLASS_NAMES

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "melanoma_best.keras"
IMAGE_SIZE = 224


def preprocess(image_path: Path) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(image_path: Path, checkpoint: Path) -> dict[str, float]:
    model = tf.keras.models.load_model(str(checkpoint))
    batch = preprocess(image_path)
    proba = model.predict(batch, verbose=0)[0]
    return {name: float(proba[i]) for i, name in enumerate(CLASS_NAMES)}


def main() -> None:
    p = argparse.ArgumentParser(description="Predict benign vs malignant from one image")
    p.add_argument("image", type=Path, help="Path to skin lesion image")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = p.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    scores = predict(args.image, args.checkpoint)
    label = max(scores, key=scores.get)
    print(f"Prediction: {label}")
    for name, prob in scores.items():
        print(f"  {name}: {prob:.2%}")


if __name__ == "__main__":
    main()
