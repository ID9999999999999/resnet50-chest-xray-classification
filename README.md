# ResNet50 Chest X-ray Classification

[![Python CI](https://github.com/ID9999999999999/resnet50-chest-xray-classification/actions/workflows/python-ci.yml/badge.svg)](https://github.com/ID9999999999999/resnet50-chest-xray-classification/actions/workflows/python-ci.yml)

Educational PyTorch pipeline for ResNet50-based chest X-ray image classification.

This project is for machine-learning experimentation and education only.
It is not a medical diagnostic system.

## Status

Portfolio-ready educational project.

## Project Goal

Build a reproducible image-classification pipeline using transfer learning with ResNet50.

## Main Results

Final test metrics using the selected validation-based threshold:

| Metric | Value |
|---|---:|
| Decision threshold | 0.98 |
| Accuracy | 90.87% |
| ROC-AUC | 95.99% |
| Pneumonia sensitivity | 97.69% |
| Normal specificity | 79.49% |
| Pneumonia F1-score | 93.04% |
| Macro F1 | 89.88% |

Confusion matrix:

| | Predicted NORMAL | Predicted PNEUMONIA |
|---|---:|---:|
| True NORMAL | 186 | 48 |
| True PNEUMONIA | 9 | 381 |

## Repository Structure

- src/
- docs/
- results/
- notebooks/
- configs/
- tests/
- examples/

## Dataset

The dataset is not included in this repository.

## Documentation

- [Model Card](docs/model_card.md)
- [Dataset Card](docs/dataset_card.md)
- [Sanitized Technical Report](docs/term_paper_sanitized.pdf)

## Results Files

- [Test metrics](results/recommended_test_metrics.json)
- [Confusion matrix](results/confusion_matrix_test_recommended_threshold.png)
- [ROC curve](results/roc_curve_test_threshold_solutions.png)
