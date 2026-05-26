#!/usr/bin/env python3
"""Export melanoma_best.keras to TensorFlow.js for the web demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf
import tensorflowjs as tfjs

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "melanoma_best.keras"
DEFAULT_OUTPUT = ROOT / "web" / "public" / "model"


def main() -> None:
    p = argparse.ArgumentParser(description="Export Keras checkpoint to TensorFlow.js")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to .keras checkpoint",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory for TF.js model",
    )
    args = p.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Train first: python train_melanoma.py"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(str(args.checkpoint))
    tfjs.converters.save_keras_model(model, str(args.output))
    print(f"Exported TF.js model to {args.output.resolve()}")


if __name__ == "__main__":
    main()
