"""
SageMaker Module for Multi-Modal Content Classification.

This module provides SageMaker-compatible training, inference, and deployment
utilities for the multi-modal hateful content classifier.

Components:
  - train_sagemaker.py: Training script for SageMaker training jobs
  - inference.py: Inference handler for SageMaker endpoints
  - sagemaker_utils.py: Deployment utilities and helpers
"""

from .sagemaker_utils import (
    create_model_package,
    upload_data_to_s3,
    prepare_training_data,
    SageMakerTrainingJob,
    SageMakerEndpoint,
    SageMakerBatchTransform,
)

__all__ = [
    "create_model_package",
    "upload_data_to_s3",
    "prepare_training_data",
    "SageMakerTrainingJob",
    "SageMakerEndpoint",
    "SageMakerBatchTransform",
]
