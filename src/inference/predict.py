from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from src.config.paths import MODEL_PATH, MODEL_METADATA_PATH
from src.data.preprocessing import build_preprocessing_function


def predict_single_image(
    model_path: str | Path = MODEL_PATH,
    image_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> tuple[str, dict[str, float]]:
    if image_path is None:
        raise ValueError("An image_path is required.")

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

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed = build_preprocessing_function(
        metadata["model_name"], roi=metadata["roi"]
    )(image)
    probabilities = load_model(model_path).predict(
        np.expand_dims(processed, axis=0), verbose=0
    )[0]
    predicted_index = int(np.argmax(probabilities))
    scores = {
        class_names[index]: float(probability)
        for index, probability in enumerate(probabilities)
    }
    return class_names[predicted_index], scores


def main():
    parser = argparse.ArgumentParser(description="Predict pavement class for one image.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()

    label, scores = predict_single_image(
        model_path=args.model,
        image_path=args.image,
        metadata_path=args.metadata,
    )
    print(f"Predicted class: {label}")
    print("Probabilities:")
    for class_name, probability in scores.items():
        print(f"  {class_name}: {probability:.4f}")


if __name__ == "__main__":
    main()
