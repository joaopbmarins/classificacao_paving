from pathlib import Path

from src.config.paths import DATASET_DIR, TEST_DIR, TRAIN_DIR


def list_class_folders(dataset_dir: Path | str | None = None):
    base = Path(dataset_dir) if dataset_dir is not None else DATASET_DIR
    train_path = base / "train"
    test_path = base / "test"
    return {
        "train": sorted([p.name for p in train_path.iterdir() if p.is_dir()]),
        "test": sorted([p.name for p in test_path.iterdir() if p.is_dir()]),
    }


def get_dataset_directories():
    return {
        "train": TRAIN_DIR,
        "test": TEST_DIR,
        "root": DATASET_DIR,
    }
