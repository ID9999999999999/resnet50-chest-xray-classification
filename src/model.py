"""Model definition for ResNet50 chest X-ray classification."""

from torchvision import models
from torch import nn


def build_resnet50(num_classes: int = 2, dropout: float = 0.2):
    """Build a ResNet50 transfer-learning classifier."""
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )

    return model
