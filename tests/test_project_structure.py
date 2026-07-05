from pathlib import Path


def test_required_files_exist():
    required_files = [
        "README.md",
        ".gitignore",
        "requirements.txt",
        "configs/resnet50_baseline.yaml",
        "docs/model_card.md",
        "docs/dataset_card.md",
        "results/recommended_test_metrics.json",
    ]

    for file_path in required_files:
        assert Path(file_path).exists(), f"Missing required file: {file_path}"


def test_required_directories_exist():
    required_dirs = [
        "src",
        "docs",
        "results",
        "notebooks",
        "configs",
        "tests",
        "examples",
    ]

    for dir_path in required_dirs:
        assert Path(dir_path).is_dir(), f"Missing required directory: {dir_path}"
