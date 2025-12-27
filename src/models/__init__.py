"""Models module for multi-modal classification."""

from src.models.fusion import MultiModalFusionClassifier
from src.models.multitask import MultiTaskClassifier

__all__ = ["MultiModalFusionClassifier", "MultiTaskClassifier"]
