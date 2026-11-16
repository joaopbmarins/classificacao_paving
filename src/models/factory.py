from __future__ import annotations

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB3

from src.config.settings import CLASS_NAMES, IMAGE_SIZE


def build_model(model_name: str = "mobilenetv2", num_classes: int | None = None):
    num_classes = len(CLASS_NAMES) if num_classes is None else num_classes

    if model_name.lower() == "mobilenetv2":
        backbone = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        )
    elif model_name.lower() == "resnet50":
        backbone = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        )
    elif model_name.lower() == "efficientnetb3":
        backbone = EfficientNetB3(
            weights='imagenet',
            include_top=False,
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    backbone.trainable = False
    x = backbone.output
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=backbone.input, outputs=outputs)
    return model, backbone
