from src.config.paths import DATASET_DIR, TEST_DIR, TRAIN_DIR
from src.data.dataset_loader import list_class_folders


def test_dataset_directories_exist():
    assert DATASET_DIR.exists()
    assert TRAIN_DIR.exists()
    assert TEST_DIR.exists()


def test_class_folders_are_discovered():
    folders = list_class_folders(DATASET_DIR)
    assert "train" in folders
    assert "test" in folders
    assert isinstance(folders["train"], list)
    assert isinstance(folders["test"], list)
