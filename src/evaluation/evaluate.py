from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.config.paths import (
    METRICS_DIR,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    PLOTS_DIR,
    TEST_DIR,
)
from src.config.settings import BATCH_SIZE
from src.data.preprocessing import build_preprocessing_function


def evaluate_model(
    model_path: str | Path = MODEL_PATH,
    test_dir: str | Path = TEST_DIR,
    metadata_path: str | Path | None = None,
    metrics_path: str | Path = METRICS_DIR / "classification_report.json",
    confusion_matrix_path: str | Path = PLOTS_DIR / "confusion_matrix.png",
):
    model_path = Path(model_path)
    metadata_path = (
        Path(metadata_path)
        if metadata_path is not None
        else model_path.with_name(MODEL_METADATA_PATH.name)
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    class_indices = metadata["class_indices"]
    class_names = [
        name for name, _ in sorted(class_indices.items(), key=lambda item: item[1])
    ]

    test_data = ImageDataGenerator(
        preprocessing_function=build_preprocessing_function(
            metadata["model_name"], roi=metadata["roi"]
        )
    ).flow_from_directory(
        str(test_dir),
        target_size=(metadata["image_size"], metadata["image_size"]),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        classes=class_names,
    )
    if test_data.class_indices != class_indices:
        raise ValueError("Test dataset class mapping does not match model metadata.")

    model = load_model(model_path)
    probabilities = model.predict(test_data, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = test_data.classes
    labels = list(range(len(class_names)))

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    metrics = {
        "model": model_path.name,
        "test_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "class_names": class_names,
    }

    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    )
    confusion_matrix_path = Path(confusion_matrix_path)
    confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(normalized_matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Normalized confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for row in range(len(class_names)):
        for column in range(len(class_names)):
            axis.text(
                column,
                row,
                f"{normalized_matrix[row, column]:.2f}",
                ha="center",
                va="center",
                color="white" if normalized_matrix[row, column] > 0.5 else "black",
            )
    figure.tight_layout()
    figure.savefig(confusion_matrix_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained pavement classifier.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR)
    parser.add_argument("--metadata", type=Path, default=None)
    args = parser.parse_args()

    metrics = evaluate_model(
        model_path=args.model,
        test_dir=args.test_dir,
        metadata_path=args.metadata,
    )
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Metrics saved to {METRICS_DIR / 'classification_report.json'}")
    print(f"Confusion matrix saved to {PLOTS_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
