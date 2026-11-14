IMAGE_SIZE = 224
BATCH_SIZE = 16
SEED = 42
VALIDATION_SPLIT = 0.2
CLASS_NAMES = ["asphalt", "belgian_blocks", "offroad"]
AUGMENTATION = {
    "rotation_range": 5,
    "zoom_range": 0.1,
    "width_shift_range": 0.05,
    "height_shift_range": 0.05,
    "brightness_range": [0.8, 1.2],
}
