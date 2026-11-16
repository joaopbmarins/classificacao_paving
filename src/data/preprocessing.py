from __future__ import annotations

from typing import Callable

import tensorflow as tf

from src.config.settings import IMAGE_SIZE

PREPROCESSORS = {
    "mobilenetv2": tf.keras.applications.mobilenet_v2.preprocess_input,
    "resnet50": tf.keras.applications.resnet50.preprocess_input,
    "efficientnetb3": tf.keras.applications.efficientnet.preprocess_input,
}


def build_preprocessing_function(
    model_name: str = "mobilenetv2", roi: bool = False
) -> Callable:
    try:
        preprocess_input = PREPROCESSORS[model_name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported model name: {model_name}") from exc

    def preprocess_image(image):
        image = tf.cast(image, tf.float32)
        if roi:
            start = tf.cast(tf.cast(tf.shape(image)[0], tf.float32) * 0.4, tf.int32)
            image = image[start:, :]
        image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        return preprocess_input(image)

    return preprocess_image
