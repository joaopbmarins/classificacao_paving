from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.config.paths import MODEL_METADATA_PATH, MODEL_PATH, TEST_DIR, TRAIN_DIR
from src.config.settings import (
    AUGMENTATION,
    BATCH_SIZE,
    IMAGE_SIZE,
    SEED,
    VALIDATION_SPLIT,
)
from src.data.preprocessing import build_preprocessing_function
from src.models.factory import build_model


class BestValidationCheckpoint(Callback):
    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.best_loss = float("inf")

    def on_epoch_end(self, _epoch, logs=None):
        val_loss = (logs or {}).get("val_loss")
        if val_loss is None:
            raise RuntimeError("Training requires validation data and a val_loss metric.")
        if val_loss < self.best_loss:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.path)
            self.best_loss = float(val_loss)


def create_generators(
    train_dir: str | Path,
    test_dir: str | Path,
    model_name: str = "mobilenetv2",
    roi: bool = False,
):
    preprocessing = build_preprocessing_function(model_name, roi=roi)
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing,
        validation_split=VALIDATION_SPLIT,
        **AUGMENTATION,
    )
    validation_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing,
        validation_split=VALIDATION_SPLIT,
    )
    test_datagen = ImageDataGenerator(preprocessing_function=preprocessing)

    train_data = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        seed=SEED,
    )
    validation_data = validation_datagen.flow_from_directory(
        str(train_dir),
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=SEED,
    )
    test_data = test_datagen.flow_from_directory(
        str(test_dir),
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        classes=list(train_data.class_indices),
    )

    if train_data.class_indices != validation_data.class_indices:
        raise ValueError("Training and validation class mappings do not match.")
    if train_data.class_indices != test_data.class_indices:
        raise ValueError("Training and test class mappings do not match.")

    return train_data, validation_data, test_data


def calculate_class_weights(train_data) -> dict[int, float]:
    labels = np.unique(train_data.classes)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=labels,
        y=train_data.classes,
    )
    return {int(label): float(weight) for label, weight in zip(labels, weights)}


def train_model(
    model_name: str = "mobilenetv2",
    train_dir: str | Path = TRAIN_DIR,
    test_dir: str | Path = TEST_DIR,
    model_path: str | Path = MODEL_PATH,
    metadata_path: str | Path | None = None,
    epochs: int = 10,
    fine_tune_epochs: int = 10,
    roi: bool = False,
):
    tf.keras.utils.set_random_seed(SEED)
    model_path = Path(model_path)
    metadata_path = (
        Path(metadata_path)
        if metadata_path is not None
        else model_path.with_name(MODEL_METADATA_PATH.name)
    )

    train_data, validation_data, _ = create_generators(
        train_dir, test_dir, model_name=model_name, roi=roi
    )
    model, backbone = build_model(
        model_name=model_name,
        num_classes=train_data.num_classes,
    )
    class_weights = calculate_class_weights(train_data)
    checkpoint = BestValidationCheckpoint(model_path)
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=[checkpoint, early_stopping],
    )

    if fine_tune_epochs > 0:
        backbone.trainable = True
        for layer in backbone.layers[:-20]:
            layer.trainable = False
        for layer in backbone.layers[-20:]:
            layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        fine_tune_history = model.fit(
            train_data,
            validation_data=validation_data,
            epochs=fine_tune_epochs,
            class_weight=class_weights,
            callbacks=[checkpoint, early_stopping],
        )
        history.history["fine_tune"] = fine_tune_history.history

    metadata = {
        "model_name": model_name.lower(),
        "image_size": IMAGE_SIZE,
        "roi": roi,
        "class_indices": train_data.class_indices,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return tf.keras.models.load_model(model_path), history


def main():
    parser = argparse.ArgumentParser(description="Train a pavement-classification model.")
    parser.add_argument("--model", choices=["mobilenetv2", "resnet50"], default="mobilenetv2")
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    parser.add_argument("--roi", action="store_true")
    args = parser.parse_args()

    train_model(
        model_name=args.model,
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        model_path=args.output,
        metadata_path=args.output.with_name("model_metadata.json"),
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        roi=args.roi,
    )
    print(f"Best model saved to {args.output}")


if __name__ == "__main__":
    main()
