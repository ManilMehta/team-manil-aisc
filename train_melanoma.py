#!/usr/bin/env python3
"""Train melanoma CNN on train + train_augmented; evaluate on test; save checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from datasets.load import DEFAULT_DATASET_ROOT, load_benign_malignant_datasets
from models.cnn import DEFAULT_EPOCHS, create_melanoma_detector, train_melanoma_detector

ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_DIR = DEFAULT_DATASET_ROOT / "train"
DEFAULT_TRAIN_AUGMENTED_DIR = DEFAULT_DATASET_ROOT / "train_augmented"
DEFAULT_VAL_DIR = DEFAULT_DATASET_ROOT / "test"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "melanoma_best.keras"


def evaluate(model, val_ds, class_names: list[str]) -> None:
    y_true = np.concatenate([y.numpy() for _, y in val_ds], axis=0)
    y_proba = model.predict(val_ds, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)

    print("\n--- Classification report ---")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("--- Confusion matrix ---")
    print(confusion_matrix(y_true, y_pred))

    malignant_idx = class_names.index("malignant")
    auc = roc_auc_score(y_true == malignant_idx, y_proba[:, malignant_idx])
    print(f"ROC AUC (malignant): {auc:.5f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train CNN on benign/malignant folders.")
    p.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    p.add_argument(
        "--train-augmented-dir",
        type=Path,
        default=DEFAULT_TRAIN_AUGMENTED_DIR,
        help="Augmented images (default: data/melanoma_cancer_dataset/train_augmented)",
    )
    p.add_argument(
        "--no-augmented",
        action="store_true",
        help="Use only --train-dir, skip train_augmented",
    )
    p.add_argument("--val-dir", type=Path, default=DEFAULT_VAL_DIR)
    p.add_argument(
        "--split-from-train",
        action="store_true",
        help="Ignore test folder; 20%% split from train only",
    )
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--validation-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Save best model here (.keras)",
    )
    args = p.parse_args()

    val_dir = None if args.split_from_train else args.val_dir
    if val_dir is not None and not val_dir.is_dir():
        print(f"Warning: {val_dir} not found; using split-from-train.")
        val_dir = None

    train_augmented_dir = None
    if not args.no_augmented:
        if args.train_augmented_dir.is_dir():
            train_augmented_dir = args.train_augmented_dir
        else:
            print(
                f"Warning: {args.train_augmented_dir} not found — training on originals only.\n"
                "  Get it from GitHub: git fetch origin main && "
                "git checkout origin/main -- data/melanoma_cancer_dataset/train_augmented"
            )

    size = (args.img_size, args.img_size)
    train_ds, val_ds, class_names = load_benign_malignant_datasets(
        args.train_dir,
        train_augmented_dir=train_augmented_dir,
        val_dir=val_dir,
        image_size=size,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        seed=args.seed,
    )

    model = create_melanoma_detector((args.img_size, args.img_size, 3))
    print(f"Train (original):  {args.train_dir.resolve()}")
    if train_augmented_dir:
        print(f"Train (augmented): {train_augmented_dir.resolve()}")
    else:
        print("Train (augmented): skipped")
    if val_dir:
        print(f"Validation (test): {val_dir.resolve()}")
    print(f"Checkpoint:        {args.checkpoint.resolve()}")
    print(f"Classes:           {class_names}")
    model.summary()

    train_melanoma_detector(
        model,
        train_ds,
        val_ds,
        epochs=args.epochs,
        checkpoint_path=str(args.checkpoint),
    )
    evaluate(model, val_ds, class_names)

    if args.checkpoint.exists():
        print(f"\nSaved model: {args.checkpoint.resolve()}")
    else:
        print(f"\nNo checkpoint written at {args.checkpoint}")


if __name__ == "__main__":
    main()
