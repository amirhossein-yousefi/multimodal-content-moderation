# SageMaker Deployment Guide

This directory contains scripts and utilities for deploying the multi-modal content classifier on AWS SageMaker.

## 📁 Contents

| File | Description |
|------|-------------|
| `train_sagemaker.py` | Training script for SageMaker training jobs |
| `inference.py` | Inference handler for SageMaker endpoints |
| `sagemaker_utils.py` | Deployment utilities and helper classes |

## 🚀 Quick Start

### Prerequisites

```bash
# Install SageMaker SDK
pip install sagemaker boto3

# Configure AWS credentials
aws configure
```

### Training on SageMaker

#### Option 1: Using Python SDK

```python
from sagemaker.pytorch import PyTorch
import sagemaker

# Get SageMaker session and role
session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Create estimator
estimator = PyTorch(
    entry_point='train_sagemaker.py',
    source_dir='sagemaker',
    role=role,
    instance_type='ml.g4dn.xlarge',
    instance_count=1,
    framework_version='2.1.0',
    py_version='py310',
    hyperparameters={
        'epochs': 8,
        'batch-size': 32,
        'backend': 'clip',
        'encoder-name': 'openai/clip-vit-base-patch32',
        'fusion-dim': 512,
        'lr-head': 5e-4,
        'lr-encoder': 1e-5,
        'class-names': 'hateful',
    },
    # Include project source code
    dependencies=['src', 'requirements.txt'],
)

# Start training
estimator.fit({
    'train': 's3://your-bucket/data/train',
    'validation': 's3://your-bucket/data/val',
    'test': 's3://your-bucket/data/test',  # optional
})
```

#### Option 2: Using Helper Class

```python
from sagemaker.sagemaker_utils import SageMakerTrainingJob

job = SageMakerTrainingJob(
    role="arn:aws:iam::123456789:role/SageMakerRole",
    instance_type="ml.g4dn.xlarge",
)

job.fit(
    train_s3="s3://your-bucket/data/train",
    val_s3="s3://your-bucket/data/val",
    hyperparameters={
        "epochs": 10,
        "batch-size": 32,
        "backend": "clip",
    }
)

print(f"Model artifacts: {job.model_data}")
```

### Deploying an Endpoint

```python
from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# Create model from training artifacts
model = PyTorchModel(
    model_data='s3://your-bucket/model/model.tar.gz',
    role=role,
    entry_point='inference.py',
    source_dir='sagemaker',
    framework_version='2.1.0',
    py_version='py310',
)

# Deploy to endpoint
predictor = model.deploy(
    instance_type='ml.g4dn.xlarge',
    initial_instance_count=1,
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer(),
)

# Make prediction
import base64

with open('image.jpg', 'rb') as f:
    image_b64 = base64.b64encode(f.read()).decode('utf-8')

response = predictor.predict({
    'instances': [
        {
            'text': 'Sample text content',
            'image_base64': image_b64,
        }
    ]
})

print(response)
# {
#     "predictions": [
#         {
#             "class_predictions": {"hateful": false},
#             "probabilities": {"hateful": 0.12},
#             "any_harmful": false
#         }
#     ]
# }

# Clean up
predictor.delete_endpoint()
```

### Using Helper Class for Deployment

```python
from sagemaker.sagemaker_utils import SageMakerEndpoint

endpoint = SageMakerEndpoint(
    model_data="s3://your-bucket/model/model.tar.gz",
    role="arn:aws:iam::123456789:role/SageMakerRole",
)

endpoint.deploy(instance_type="ml.g4dn.xlarge")

# Make prediction
result = endpoint.predict(
    text="Sample text",
    image_path="path/to/image.jpg"
)
print(result)

# Clean up
endpoint.delete()
```

### Batch Transform

For processing large datasets asynchronously:

```python
from sagemaker.sagemaker_utils import SageMakerBatchTransform

transform = SageMakerBatchTransform(
    model_data="s3://your-bucket/model/model.tar.gz",
    role="arn:aws:iam::123456789:role/SageMakerRole",
)

# Input should be JSON Lines format
transform.run(
    input_s3="s3://your-bucket/input/data.jsonl",
    output_s3="s3://your-bucket/output/",
)
```

## 📦 Data Format

### Training Data Structure

SageMaker expects data organized in channel directories:

```
s3://bucket/data/
├── train/
│   ├── train.csv
│   └── images/
│       ├── img001.jpg
│       └── ...
├── validation/
│   ├── val.csv
│   └── images/
│       └── ...
└── test/  (optional)
    ├── test.csv
    └── images/
        └── ...
```

### CSV Format

```csv
text,image_path,labels
"Sample text 1",images/img001.jpg,hateful
"Sample text 2",images/img002.jpg,
"Sample text 3",images/img003.jpg,hateful
```

### Preparing Data for S3

Use the utility function to prepare local data:

```python
from sagemaker.sagemaker_utils import prepare_training_data, upload_data_to_s3

# Organize data into channel directories
channels = prepare_training_data(
    train_csv="data/train.csv",
    val_csv="data/val.csv",
    test_csv="data/test.csv",
    image_root="data/images",
    output_dir="./sagemaker_data",
)

# Upload to S3
import boto3
s3 = boto3.client('s3')

for channel, local_path in channels.items():
    s3_uri = upload_data_to_s3(
        local_dir=local_path,
        bucket="your-bucket",
        prefix=f"data/{channel}",
    )
    print(f"{channel}: {s3_uri}")
```

## 🔧 Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 8 | Number of training epochs |
| `batch-size` | 32 | Training batch size per device |
| `backend` | clip | Vision-language backend (clip/siglip) |
| `head` | fusion | Classification head type (fusion/mtl) |
| `encoder-name` | openai/clip-vit-base-patch32 | HuggingFace encoder |
| `fusion-dim` | 512 | Fusion layer dimension |
| `lr-head` | 5e-4 | Learning rate for classification head |
| `lr-encoder` | 1e-5 | Learning rate for encoder |
| `weight-decay` | 0.02 | Weight decay |
| `precision` | fp16 | Training precision (fp16/bf16/fp32) |
| `loss-type` | bce | Loss function (bce/focal) |
| `class-names` | hateful | Comma-separated class names |
| `augment` | false | Enable data augmentation |
| `weighted-sampling` | false | Use weighted sampling |

## 📋 API Reference

### Inference Request Format

```json
{
    "instances": [
        {
            "text": "Text content to classify",
            "image_base64": "base64_encoded_image_string"
        }
    ]
}
```

### Inference Response Format

```json
{
    "predictions": [
        {
            "class_predictions": {
                "hateful": false
            },
            "probabilities": {
                "hateful": 0.123
            },
            "any_harmful": false
        }
    ]
}
```

## 🔒 IAM Permissions

Your SageMaker execution role needs the following permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::your-bucket/*",
                "arn:aws:s3:::your-bucket"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage",
                "ecr:BatchCheckLayerAvailability"
            ],
            "Resource": "*"
        }
    ]
}
```

## 💰 Cost Optimization

1. **Use Spot Instances** for training:
   ```python
   estimator = PyTorch(
       ...,
       use_spot_instances=True,
       max_wait=3600,  # Maximum wait time
   )
   ```

2. **Use Serverless Inference** for low-traffic endpoints:
   ```python
   from sagemaker.serverless import ServerlessInferenceConfig
   
   serverless_config = ServerlessInferenceConfig(
       memory_size_in_mb=4096,
       max_concurrency=10,
   )
   
   predictor = model.deploy(
       serverless_inference_config=serverless_config,
   )
   ```

3. **Enable Auto Scaling** for production endpoints:
   ```python
   client = boto3.client('application-autoscaling')
   
   client.register_scalable_target(
       ServiceNamespace='sagemaker',
       ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
       ScalableDimension='sagemaker:variant:DesiredInstanceCount',
       MinCapacity=1,
       MaxCapacity=4,
   )
   ```

## 🧪 Local Testing

Test training script locally before deploying:

```bash
# Test training locally
python sagemaker/train_sagemaker.py \
    --train data/mmhs150k \
    --val data/mmhs150k \
    --model-dir ./test_model \
    --epochs 1 \
    --batch-size 4

# Test inference locally
python sagemaker/inference.py \
    --model-dir ./test_model \
    --text "Test content" \
    --image path/to/image.jpg
```

## 📊 Monitoring

Enable CloudWatch metrics for your endpoint:

```python
from sagemaker.model_monitor import DataCaptureConfig

data_capture = DataCaptureConfig(
    enable_capture=True,
    sampling_percentage=100,
    destination_s3_uri='s3://your-bucket/data-capture/',
)

predictor = model.deploy(
    ...,
    data_capture_config=data_capture,
)
```
