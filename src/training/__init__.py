"""Training module for multi-modal classification."""

from src.training.trainer import (
    train_model,
    WeightedSamplerTrainer,
    group_params_for_optimizer,
    build_multilabel_sample_weights,
)
from src.training.losses import FocalWithLogitsLoss
from src.training.metrics import (
    make_compute_metrics_multi,
    make_compute_metrics_mtl,
    calibrate_thresholds,
)

__all__ = [
    "train_model",
    "WeightedSamplerTrainer",
    "group_params_for_optimizer",
    "build_multilabel_sample_weights",
    "FocalWithLogitsLoss",
    "make_compute_metrics_multi",
    "make_compute_metrics_mtl",
    "calibrate_thresholds",
]
