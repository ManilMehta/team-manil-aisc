"""Load benign/malignant image folders into tf.data."""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf

CLASS_NAMES = ("benign", "malignant")

DEFAULT_DATASET_ROOT = Path(__file__).resolve().parent.parent / "data" / "melanoma_cancer_dataset"


def _check_class_folders(root: Path) -> None:
    for name in CLASS_NAMES:
        if not (root / name).is_dir():
            raise FileNotFoundError(
                f"Missing {root / name} — expected benign/ and malignant/ subfolders."
            )


def _directory_dataset(
    directory: Path,
    *,
    image_size: tuple[int, int],
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> tf.data.Dataset:
    return tf.keras.utils.image_dataset_from_directory(
        str(directory),
        labels="inferred",
        label_mode="int",
        class_names=list(CLASS_NAMES),
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def load_benign_malignant_datasets(
    train_dir: str | Path,
    *,
    train_augmented_dir: str | Path | None = None,
    val_dir: str | Path | None = None,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    validation_split: float = 0.2,
    seed: int = 42,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    """
    Load train and validation datasets.

    If train_augmented_dir is set, training uses both train_dir and
    train_augmented_dir (original + augmented images). Validation uses val_dir only.
    """
    train_dir = Path(train_dir)
    _check_class_folders(train_dir)

    def rescale(images, labels):
        return tf.cast(images, tf.float32) / 255.0, labels

    opts = tf.data.AUTOTUNE

    if val_dir is not None:
        val_dir = Path(val_dir)
        _check_class_folders(val_dir)
        train_ds = _directory_dataset(
            train_dir,
            image_size=image_size,
            batch_size=batch_size,
            seed=seed,
            shuffle=True,
        )
        if train_augmented_dir is not None:
            train_augmented_dir = Path(train_augmented_dir)
            _check_class_folders(train_augmented_dir)
            aug_ds = _directory_dataset(
                train_augmented_dir,
                image_size=image_size,
                batch_size=batch_size,
                seed=seed,
                shuffle=True,
            )
            train_ds = (
                train_ds.unbatch()
                .concatenate(aug_ds.unbatch())
                .shuffle(20_000, seed=seed)
                .batch(batch_size)
            )
        val_ds = _directory_dataset(
            val_dir,
            image_size=image_size,
            batch_size=batch_size,
            seed=seed,
            shuffle=False,
        )
    else:
        common = dict(
            directory=str(train_dir),
            labels="inferred",
            label_mode="int",
            class_names=list(CLASS_NAMES),
            seed=seed,
            image_size=image_size,
            batch_size=batch_size,
        )
        train_ds = tf.keras.utils.image_dataset_from_directory(
            validation_split=validation_split,
            subset="training",
            **common,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            validation_split=validation_split,
            subset="validation",
            **common,
        )

    train_ds = train_ds.map(rescale, num_parallel_calls=opts).prefetch(opts)
    val_ds = val_ds.map(rescale, num_parallel_calls=opts).prefetch(opts)
    return train_ds, val_ds, list(CLASS_NAMES)
