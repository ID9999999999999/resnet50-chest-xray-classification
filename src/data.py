"""Dataset utilities for chest X-ray classification."""

from pathlib import Path

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def validate_dataset_root(dataset_root: str) -> Path:
    """Validate expected dataset folder structure."""
    root = Path(dataset_root)

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    for split in ["train", "val", "test"]:
        for class_name in CLASS_NAMES:
            expected = root / split / class_name
            if not expected.exists():
                raise FileNotFoundError(f"Missing folder: {expected}")

    return root
