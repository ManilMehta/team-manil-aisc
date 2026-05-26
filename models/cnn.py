"""Sequential CNN for benign vs malignant classification."""

from __future__ import annotations

from typing import Any

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Activation,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)

DEFAULT_EPOCHS = 20


def _conv_block(
    model: Sequential,
    filters: int,
    activation: str,
    dropout: float,
    *,
    input_shape: tuple[int, int, int] | None = None,
) -> None:
    kwargs: dict[str, Any] = {}
    if input_shape is not None:
        kwargs["input_shape"] = input_shape
    model.add(Conv2D(filters, (3, 3), padding="same", **kwargs))
    model.add(Activation(activation))
    model.add(Conv2D(filters, (3, 3)))
    model.add(Activation(activation))
    model.add(Conv2D(filters, (3, 3)))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout / 2.0))


def build_melanoma_cnn(
    input_shape: tuple[int, int, int],
    num_classes: int = 2,
    *,
    filters: int = 10,
    dropout: float = 0.5,
    activation: str = "relu",
    dense_units: int = 512,
) -> Sequential:
    """Three conv blocks (3x3), pool, dropout, then dense head + softmax."""
    model = Sequential(name="melanoma_cnn")
    _conv_block(model, filters, activation, dropout, input_shape=input_shape)
    _conv_block(model, filters, activation, dropout)
    _conv_block(model, filters, activation, dropout)
    model.add(Flatten())
    model.add(Dense(dense_units))
    model.add(Activation(activation))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))
    return model


def compile_melanoma_cnn(
    model: Sequential,
    *,
    learning_rate: float = 1e-4,
) -> Sequential:
    opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, decay=1e-6)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=opt,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def create_melanoma_detector(
    input_shape: tuple[int, int, int],
    num_classes: int = 2,
    **build_kwargs: Any,
) -> Sequential:
    model = build_melanoma_cnn(input_shape, num_classes, **build_kwargs)
    return compile_melanoma_cnn(model)


def train_melanoma_detector(
    model: Sequential,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    *,
    epochs: int = DEFAULT_EPOCHS,
    checkpoint_path: str | None = "checkpoints/melanoma_best.keras",
    **fit_kwargs: Any,
) -> tf.keras.callbacks.History:
    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
        ),
    ]
    if checkpoint_path:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_accuracy",
                save_best_only=True,
            )
        )
    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        **fit_kwargs,
    )
