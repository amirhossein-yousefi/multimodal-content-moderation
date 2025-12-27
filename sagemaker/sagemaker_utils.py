"""
SageMaker Deployment Utilities for Multi-Modal Content Classification.

This module provides helper classes and functions for deploying the
multi-modal classifier on AWS SageMaker.

Features:
  - Training job creation and management
  - Model packaging for deployment
  - Endpoint creation and invocation
  - Batch transform job management
"""

import os
import json
import time
import tarfile
import tempfile
from typing import Any, Dict, List, Optional
from pathlib import Path


def create_model_package(
    model_dir: str,
    output_path: str,
    source_dir: Optional[str] = None,
    requirements_file: Optional[str] = None
) -> str:
    """
    Create a model.tar.gz package for SageMaker deployment.
    
    The package includes:
      - Model weights and config from model_dir
      - Source code from source_dir (if provided)
      - requirements.txt for dependencies
    
    Args:
        model_dir: Directory containing trained model artifacts.
        output_path: Path to write the tar.gz file.
        source_dir: Optional source code directory to include.
        requirements_file: Optional requirements.txt path.
        
    Returns:
        Path to the created tar.gz file.
    """
    with tarfile.open(output_path, "w:gz") as tar:
        # Add model artifacts
        for item in os.listdir(model_dir):
            item_path = os.path.join(model_dir, item)
            tar.add(item_path, arcname=item)
        
        # Add source code if specified
        if source_dir and os.path.isdir(source_dir):
            tar.add(source_dir, arcname="code")
        
        # Add requirements if specified
        if requirements_file and os.path.exists(requirements_file):
            tar.add(requirements_file, arcname="requirements.txt")
    
    return output_path


class SageMakerTrainingJob:
    """
    Helper class for creating and managing SageMaker training jobs.
    
    Example:
        job = SageMakerTrainingJob(
            role="arn:aws:iam::123456789:role/SageMakerRole",
            instance_type="ml.g4dn.xlarge"
        )
        
        job.fit(
            train_s3="s3://bucket/train",
            val_s3="s3://bucket/val",
            hyperparameters={"epochs": 10, "batch-size": 32}
        )
    """
    
    def __init__(
        self,
        role: str,
        instance_type: str = "ml.g4dn.xlarge",
        instance_count: int = 1,
        framework_version: str = "2.1.0",
        py_version: str = "py310",
        source_dir: Optional[str] = None,
        base_job_name: str = "multimodal-classifier",
        region: Optional[str] = None,
    ):
        """
        Initialize training job configuration.
        
        Args:
            role: IAM role ARN for SageMaker.
            instance_type: EC2 instance type for training.
            instance_count: Number of training instances.
            framework_version: PyTorch framework version.
            py_version: Python version.
            source_dir: Local source directory (defaults to project root).
            base_job_name: Base name for training jobs.
            region: AWS region (auto-detected if None).
        """
        self.role = role
        self.instance_type = instance_type
        self.instance_count = instance_count
        self.framework_version = framework_version
        self.py_version = py_version
        self.base_job_name = base_job_name
        self.region = region
        
        # Default source directory to project sagemaker folder
        if source_dir is None:
            self.source_dir = str(Path(__file__).parent)
        else:
            self.source_dir = source_dir
        
        self._estimator = None
        self._job_name = None
    
    def fit(
        self,
        train_s3: str,
        val_s3: str,
        test_s3: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        wait: bool = True,
        logs: bool = True,
    ) -> "SageMakerTrainingJob":
        """
        Launch a SageMaker training job.
        
        Args:
            train_s3: S3 URI for training data.
            val_s3: S3 URI for validation data.
            test_s3: Optional S3 URI for test data.
            hyperparameters: Training hyperparameters.
            wait: Whether to wait for job completion.
            logs: Whether to show training logs.
            
        Returns:
            Self for method chaining.
        """
        try:
            from sagemaker.pytorch import PyTorch
        except ImportError:
            raise ImportError(
                "sagemaker package not installed. "
                "Install with: pip install sagemaker"
            )
        
        # Default hyperparameters
        default_hp = {
            "epochs": 8,
            "batch-size": 32,
            "backend": "clip",
            "encoder-name": "openai/clip-vit-base-patch32",
            "fusion-dim": 512,
            "lr-head": 5e-4,
            "lr-encoder": 1e-5,
            "class-names": "hateful",
        }
        
        if hyperparameters:
            default_hp.update(hyperparameters)
        
        # Create estimator
        self._estimator = PyTorch(
            entry_point="train_sagemaker.py",
            source_dir=self.source_dir,
            role=self.role,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            framework_version=self.framework_version,
            py_version=self.py_version,
            hyperparameters=default_hp,
            base_job_name=self.base_job_name,
            # Include project source code
            dependencies=["../src", "../requirements.txt"],
        )
        
        # Build input channels
        inputs = {
            "train": train_s3,
            "validation": val_s3,
        }
        if test_s3:
            inputs["test"] = test_s3
        
        # Start training
        self._estimator.fit(inputs, wait=wait, logs=logs)
        self._job_name = self._estimator.latest_training_job.name
        
        return self
    
    @property
    def model_data(self) -> Optional[str]:
        """Get S3 URI of trained model artifacts."""
        if self._estimator is None:
            return None
        return self._estimator.model_data
    
    @property
    def job_name(self) -> Optional[str]:
        """Get training job name."""
        return self._job_name


class SageMakerEndpoint:
    """
    Helper class for deploying and managing SageMaker endpoints.
    
    Example:
        endpoint = SageMakerEndpoint(
            model_data="s3://bucket/model.tar.gz",
            role="arn:aws:iam::123456789:role/SageMakerRole"
        )
        
        endpoint.deploy(instance_type="ml.g4dn.xlarge")
        
        result = endpoint.predict(
            text="sample text",
            image_base64="..."
        )
        
        endpoint.delete()
    """
    
    def __init__(
        self,
        model_data: str,
        role: str,
        endpoint_name: Optional[str] = None,
        framework_version: str = "2.1.0",
        py_version: str = "py310",
        region: Optional[str] = None,
    ):
        """
        Initialize endpoint configuration.
        
        Args:
            model_data: S3 URI to model.tar.gz.
            role: IAM role ARN for SageMaker.
            endpoint_name: Optional endpoint name (auto-generated if None).
            framework_version: PyTorch framework version.
            py_version: Python version.
            region: AWS region.
        """
        self.model_data = model_data
        self.role = role
        self.endpoint_name = endpoint_name
        self.framework_version = framework_version
        self.py_version = py_version
        self.region = region
        
        self._predictor = None
    
    def deploy(
        self,
        instance_type: str = "ml.g4dn.xlarge",
        instance_count: int = 1,
        wait: bool = True,
    ) -> "SageMakerEndpoint":
        """
        Deploy model to a SageMaker endpoint.
        
        Args:
            instance_type: EC2 instance type for inference.
            instance_count: Number of instances.
            wait: Whether to wait for endpoint creation.
            
        Returns:
            Self for method chaining.
        """
        try:
            from sagemaker.pytorch import PyTorchModel
            from sagemaker.serializers import JSONSerializer
            from sagemaker.deserializers import JSONDeserializer
        except ImportError:
            raise ImportError(
                "sagemaker package not installed. "
                "Install with: pip install sagemaker"
            )
        
        model = PyTorchModel(
            model_data=self.model_data,
            role=self.role,
            entry_point="inference.py",
            source_dir=str(Path(__file__).parent),
            framework_version=self.framework_version,
            py_version=self.py_version,
        )
        
        self._predictor = model.deploy(
            instance_type=instance_type,
            initial_instance_count=instance_count,
            endpoint_name=self.endpoint_name,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer(),
            wait=wait,
        )
        
        self.endpoint_name = self._predictor.endpoint_name
        
        return self
    
    def predict(
        self,
        text: Optional[str] = None,
        image_base64: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Make a single prediction.
        
        Args:
            text: Text content.
            image_base64: Base64 encoded image.
            image_path: Local path to image file (will be encoded).
            
        Returns:
            Prediction result dictionary.
        """
        import base64
        
        if self._predictor is None:
            raise RuntimeError("Endpoint not deployed. Call deploy() first.")
        
        # Prepare input
        instance = {"text": text or ""}
        
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                instance["image_base64"] = base64.b64encode(f.read()).decode("utf-8")
        elif image_base64:
            instance["image_base64"] = image_base64
        
        # Make prediction
        response = self._predictor.predict({"instances": [instance]})
        
        if "predictions" in response and len(response["predictions"]) > 0:
            return response["predictions"][0]
        return response
    
    def predict_batch(
        self,
        instances: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Make batch predictions.
        
        Args:
            instances: List of input dictionaries with text/image_base64.
            
        Returns:
            List of prediction results.
        """
        if self._predictor is None:
            raise RuntimeError("Endpoint not deployed. Call deploy() first.")
        
        response = self._predictor.predict({"instances": instances})
        return response.get("predictions", [])
    
    def delete(self):
        """Delete the SageMaker endpoint."""
        if self._predictor is not None:
            self._predictor.delete_endpoint()
            self._predictor = None


class SageMakerBatchTransform:
    """
    Helper class for SageMaker Batch Transform jobs.
    
    Use for processing large datasets asynchronously.
    
    Example:
        transform = SageMakerBatchTransform(
            model_data="s3://bucket/model.tar.gz",
            role="arn:aws:iam::123456789:role/SageMakerRole"
        )
        
        transform.run(
            input_s3="s3://bucket/input",
            output_s3="s3://bucket/output"
        )
    """
    
    def __init__(
        self,
        model_data: str,
        role: str,
        instance_type: str = "ml.g4dn.xlarge",
        instance_count: int = 1,
        framework_version: str = "2.1.0",
        py_version: str = "py310",
    ):
        """
        Initialize batch transform configuration.
        
        Args:
            model_data: S3 URI to model.tar.gz.
            role: IAM role ARN.
            instance_type: EC2 instance type.
            instance_count: Number of instances.
            framework_version: PyTorch version.
            py_version: Python version.
        """
        self.model_data = model_data
        self.role = role
        self.instance_type = instance_type
        self.instance_count = instance_count
        self.framework_version = framework_version
        self.py_version = py_version
    
    def run(
        self,
        input_s3: str,
        output_s3: str,
        content_type: str = "application/jsonlines",
        accept: str = "application/jsonlines",
        wait: bool = True,
        logs: bool = True,
    ) -> str:
        """
        Run a batch transform job.
        
        Input format: JSON Lines (one JSON object per line)
        
        Args:
            input_s3: S3 URI for input data.
            output_s3: S3 URI for output data.
            content_type: Input content type.
            accept: Output content type.
            wait: Whether to wait for completion.
            logs: Whether to show logs.
            
        Returns:
            S3 URI of output data.
        """
        try:
            from sagemaker.pytorch import PyTorchModel
        except ImportError:
            raise ImportError(
                "sagemaker package not installed. "
                "Install with: pip install sagemaker"
            )
        
        model = PyTorchModel(
            model_data=self.model_data,
            role=self.role,
            entry_point="inference.py",
            source_dir=str(Path(__file__).parent),
            framework_version=self.framework_version,
            py_version=self.py_version,
        )
        
        transformer = model.transformer(
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            output_path=output_s3,
            accept=accept,
        )
        
        transformer.transform(
            data=input_s3,
            content_type=content_type,
            wait=wait,
            logs=logs,
        )
        
        return output_s3


# ============================================================================
# Utility Functions
# ============================================================================

def upload_data_to_s3(
    local_dir: str,
    bucket: str,
    prefix: str,
    include_patterns: Optional[List[str]] = None,
) -> str:
    """
    Upload local data directory to S3.
    
    Args:
        local_dir: Local directory path.
        bucket: S3 bucket name.
        prefix: S3 key prefix.
        include_patterns: Optional glob patterns to include.
        
    Returns:
        S3 URI of uploaded data.
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        raise ImportError("boto3 not installed. Install with: pip install boto3")
    
    s3 = boto3.client("s3")
    
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_dir)
            s3_key = f"{prefix}/{relative_path}"
            
            # Check include patterns
            if include_patterns:
                import fnmatch
                if not any(fnmatch.fnmatch(file, p) for p in include_patterns):
                    continue
            
            s3.upload_file(local_path, bucket, s3_key)
    
    return f"s3://{bucket}/{prefix}"


def prepare_training_data(
    train_csv: str,
    val_csv: str,
    image_root: str,
    output_dir: str,
    test_csv: Optional[str] = None,
) -> Dict[str, str]:
    """
    Prepare training data for SageMaker by organizing into channel directories.
    
    Creates the following structure:
        output_dir/
          train/
            train.csv
            images/
          validation/
            val.csv
            images/
          test/  (if provided)
            test.csv
            images/
    
    Args:
        train_csv: Path to training CSV.
        val_csv: Path to validation CSV.
        image_root: Root directory containing images.
        output_dir: Output directory for organized data.
        test_csv: Optional path to test CSV.
        
    Returns:
        Dictionary mapping channel names to paths.
    """
    import shutil
    import pandas as pd
    
    channels = {}
    
    for split_name, csv_path in [("train", train_csv), ("validation", val_csv), ("test", test_csv)]:
        if csv_path is None:
            continue
        
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        # Copy CSV
        out_csv = os.path.join(split_dir, f"{split_name}.csv")
        shutil.copy(csv_path, out_csv)
        
        # Create images subdirectory
        img_dir = os.path.join(split_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        
        # Copy referenced images
        df = pd.read_csv(csv_path)
        if "image_path" in df.columns:
            for img_path in df["image_path"].dropna().unique():
                src = os.path.join(image_root, img_path)
                if os.path.exists(src):
                    dst = os.path.join(img_dir, os.path.basename(img_path))
                    shutil.copy(src, dst)
        
        channels[split_name] = split_dir
    
    return channels


# ============================================================================
# CLI for utility functions
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SageMaker deployment utilities")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Package model command
    pkg_parser = subparsers.add_parser("package", help="Create model.tar.gz package")
    pkg_parser.add_argument("--model-dir", type=str, required=True,
                            help="Model directory")
    pkg_parser.add_argument("--output", type=str, required=True,
                            help="Output tar.gz path")
    pkg_parser.add_argument("--source-dir", type=str, default=None,
                            help="Source code directory")
    
    # Prepare data command
    data_parser = subparsers.add_parser("prepare-data", help="Prepare data for SageMaker")
    data_parser.add_argument("--train-csv", type=str, required=True)
    data_parser.add_argument("--val-csv", type=str, required=True)
    data_parser.add_argument("--test-csv", type=str, default=None)
    data_parser.add_argument("--image-root", type=str, required=True)
    data_parser.add_argument("--output-dir", type=str, required=True)
    
    args = parser.parse_args()
    
    if args.command == "package":
        output = create_model_package(
            model_dir=args.model_dir,
            output_path=args.output,
            source_dir=args.source_dir,
        )
        print(f"Created model package: {output}")
    
    elif args.command == "prepare-data":
        channels = prepare_training_data(
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            test_csv=args.test_csv,
            image_root=args.image_root,
            output_dir=args.output_dir,
        )
        print(f"Prepared data channels: {channels}")
    
    else:
        parser.print_help()
