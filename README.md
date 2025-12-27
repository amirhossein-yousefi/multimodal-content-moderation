# Multi-Modal Hateful Content Classification

A PyTorch-based framework for multi-modal (image + text) hateful content classification using CLIP and SigLIP encoders with late fusion architectures.

## 🎯 Project Overview

This project implements multi-modal classification models for detecting hateful content in social media memes and posts. It combines visual and textual features using state-of-the-art vision-language models.

### Supported Models

| Model | Architecture | Description |
|-------|--------------|-------------|
| **CLIP Fusion** | `openai/clip-vit-base-patch32` | Late fusion with gated attention mechanism |
| **SigLIP Fusion** | `google/siglip2-base-patch16-224` | SigLIP2 encoder with fusion head |
| **Multi-Task (MTL)** | CLIP-based | Shared backbone with task-specific heads |

### Trained Models Performance (MMHS150K Dataset - Test Set)

| Model | F1 Macro | F1 Micro | ROC-AUC Macro | Throughput (samples/sec) |
|-------|----------|----------|---------------|--------------------------|
| CLIP Fusion | 0.566 | 0.635 | 0.783 | 381.5 |
| CLIP MTL | **0.569** | **0.644** | **0.783** | 390.9 |
| SigLIP Fusion | 0.507 | 0.610 | 0.774 | 236.3 |
| CLIP Fusion (Weighted) | 0.557 | 0.636 | 0.772 | 266.4 |
| CLIP Fusion (Bigger Batch) | 0.515 | 0.517 | **0.804** | **400.9** |

#### Per-Class F1 Scores (CLIP MTL - Best Model)

| Class | F1 Score | ROC-AUC |
|-------|----------|---------|
| Racist | 0.672 | 0.765 |
| Sexist | 0.589 | 0.810 |
| Homophobe | 0.745 | 0.882 |
| Religion | 0.223 | 0.618 |
| Otherhate | 0.616 | 0.842 |

## 🤗 Models on Hugging Face

<!-- Add your model links here -->
| Model | Link |
|-------|------|
| CLIP Fusion | [![Hugging Face](https://img.shields.io/badge/🤗%20Model-yellow.svg)](https://huggingface.co/YOUR_USERNAME/MODEL_NAME) |
| CLIP MTL | [![Hugging Face](https://img.shields.io/badge/🤗%20Model-yellow.svg)](https://huggingface.co/YOUR_USERNAME/MODEL_NAME) |
| SigLIP Fusion | [![Hugging Face](https://img.shields.io/badge/🤗%20Model-yellow.svg)](https://huggingface.co/YOUR_USERNAME/MODEL_NAME) |

## 📁 Project Structure

```
content_multi_modal/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
├── config/
│   └── default.yaml             # Default training configuration
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py           # SocialHarmDataset implementation
│   │   └── preprocessing.py     # Data preparation utilities
│   ├── models/
│   │   ├── __init__.py
│   │   ├── fusion.py            # Multi-modal fusion classifier
│   │   └── multitask.py         # Multi-task learning classifier
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py           # Training logic and Trainer class
│   │   ├── losses.py            # Loss functions (BCE, Focal)
│   │   └── metrics.py           # Evaluation metrics
│   └── utils/
│       ├── __init__.py
│       └── helpers.py           # Utility functions
├── scripts/
│   ├── train.py                 # Main training script
│   ├── evaluate.py              # Evaluation script
│   ├── inference.py             # Inference script
│   └── prepare_data.py          # Data preparation script
├── sagemaker/                   # AWS SageMaker deployment
│   ├── README.md                # SageMaker documentation
│   ├── __init__.py
│   ├── train_sagemaker.py       # SageMaker training script
│   ├── inference.py             # SageMaker inference handler
│   └── sagemaker_utils.py       # Deployment utilities
├── notebooks/
│   └── exploration.ipynb        # Data exploration notebook
├── data/
│   └── mmhs150k/                # MMHS150K dataset
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       ├── class_names.txt
│       └── images/
└── runs/                        # Training outputs and checkpoints
    ├── clip_fusion_mmhshateful/
    ├── clip_mtl_mmhshateful/
    └── siglip_fusion_mmhshateful/
```

## � Dataset

This project uses the **MMHS150K** (Multi-Modal Hate Speech) dataset for training and evaluation.

### About MMHS150K

MMHS150K is a large-scale multi-modal hate speech dataset collected from Twitter, containing 150,000 tweet-image pairs annotated for hate speech detection.

**Paper:** ["Exploring Hate Speech Detection in Multimodal Publications"](https://gombru.github.io/2019/10/09/MMHS/) (WACV 2020)  
**Authors:** Raul Gomez, Jaume Gibert, Lluis Gomez, Dimosthenis Karatzas

### Dataset Statistics

| Split | Samples |
|-------|---------|
| Train | ~112,500 |
| Validation | ~15,000 |
| Test | ~22,500 |

### Label Categories

| Label ID | Category | Description |
|----------|----------|-------------|
| 0 | NotHate | Non-hateful content |
| 1 | Racist | Racist content |
| 2 | Sexist | Sexist content |
| 3 | Homophobe | Homophobic content |
| 4 | Religion | Religion-based hate |
| 5 | OtherHate | Other types of hate speech |

### Obtaining the Dataset

1. **Download MMHS150K** from the official source:
   - Visit: https://gombru.github.io/2019/10/09/MMHS/
   - Request access and download the dataset

2. **Extract the raw data** to `data/_raw_mmhs/MMHS150K/`:
   ```
   data/_raw_mmhs/MMHS150K/
   ├── MMHS150K_GT.json          # Ground truth annotations
   ├── MMHS150K_readme.txt       # Dataset documentation
   ├── hatespeech_keywords.txt   # Keywords used for data collection
   ├── img_resized/              # Images (500px shortest side)
   ├── img_txt/                  # OCR-extracted text from images
   └── splits/
       ├── train_ids.txt
       ├── val_ids.txt
       └── test_ids.txt
   ```

### Data Preparation

Run the data preparation script to convert raw MMHS150K data into the training format:

```bash
python scripts/prepare_data.py \
    --dataset mmhs150k \
    --raw_dir data/_raw_mmhs/MMHS150K \
    --output data/mmhs150k
```

This will create the processed dataset:

```
data/mmhs150k/
├── train.csv           # Training split
├── val.csv             # Validation split  
├── test.csv            # Test split
├── class_names.txt     # Label names
└── images/             # Processed images (copy from raw)
```

**Note:** After running the script, copy images from `data/_raw_mmhs/MMHS150K/img_resized/` to `data/mmhs150k/images/`:

```bash
# Linux/Mac
cp -r data/_raw_mmhs/MMHS150K/img_resized/* data/mmhs150k/images/

# Windows
xcopy data\_raw_mmhs\MMHS150K\img_resized\* data\mmhs150k\images\ /E /I
```

### CSV Format

The prepared CSV files have the following format:

| Column | Description |
|--------|-------------|
| `text` | Tweet text content |
| `image_path` | Image filename (e.g., `12345.jpg`) |
| `labels` | Comma-separated hate categories (e.g., `racist,sexist`) |

### Alternative Dataset: Hateful Memes

You can also use the Facebook Hateful Memes dataset:

```bash
python scripts/prepare_data.py --dataset hateful_memes --output data/
```

This downloads and prepares the dataset automatically from Hugging Face.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/amirhossein-yousefi/multimodal-content-moderation.git
cd multimodal-content-moderation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Training

```bash
# Train CLIP Fusion model
python scripts/train.py \
    --config config/default.yaml \
    --model clip_fusion \
    --train_csv data/mmhs150k/train.csv \
    --val_csv data/mmhs150k/val.csv \
    --test_csv data/mmhs150k/test.csv \
    --image_root data/mmhs150k/images

# Train with custom parameters
python scripts/train.py \
    --backend clip \
    --head fusion \
    --epochs 8 \
    --batch_size 32 \
    --lr_encoder 1e-5 \
    --lr_head 5e-4
```

### Inference

```python
from src.models import MultiModalFusionClassifier
from src.utils import load_model_for_inference

# Load trained model
model, processor, tokenizer = load_model_for_inference(
    checkpoint_path="runs/clip_fusion_mmhshateful/checkpoint-33708"
)

# Predict
prediction = model.predict(image_path="path/to/image.jpg", text="sample text")
```

## ☁️ AWS SageMaker Deployment

This project includes full SageMaker compatibility for cloud-based training and inference.

### SageMaker Training

```python
from sagemaker.pytorch import PyTorch
import sagemaker

session = sagemaker.Session()
role = sagemaker.get_execution_role()

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
        'class-names': 'hateful',
    },
    dependencies=['src', 'requirements.txt'],
)

estimator.fit({
    'train': 's3://your-bucket/data/train',
    'validation': 's3://your-bucket/data/val',
})
```

### SageMaker Endpoint Deployment

```python
from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

model = PyTorchModel(
    model_data=estimator.model_data,
    role=role,
    entry_point='inference.py',
    source_dir='sagemaker',
    framework_version='2.1.0',
    py_version='py310',
)

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
    'instances': [{
        'text': 'Sample text content',
        'image_base64': image_b64,
    }]
})
print(response)
# {"predictions": [{"class_predictions": {"hateful": false}, "probabilities": {"hateful": 0.12}}]}

# Clean up
predictor.delete_endpoint()
```

### Using Helper Classes

```python
from sagemaker.sagemaker_utils import SageMakerTrainingJob, SageMakerEndpoint

# Training
job = SageMakerTrainingJob(
    role="arn:aws:iam::123456789:role/SageMakerRole",
    instance_type="ml.g4dn.xlarge",
)
job.fit(
    train_s3="s3://bucket/train",
    val_s3="s3://bucket/val",
    hyperparameters={"epochs": 10}
)

# Deployment
endpoint = SageMakerEndpoint(
    model_data=job.model_data,
    role="arn:aws:iam::123456789:role/SageMakerRole",
)
endpoint.deploy(instance_type="ml.g4dn.xlarge")

# Inference
result = endpoint.predict(text="Sample text", image_path="image.jpg")
endpoint.delete()
```

📖 **Full SageMaker documentation**: See [sagemaker/README.md](sagemaker/README.md)

## 📊 Dataset

### MMHS150K (Multi-Modal Hate Speech)

- **Classes**: `racist`, `sexist`, `homophobe`, `religion`, `otherhate`
- **Train**: ~110K samples
- **Val**: ~5K samples  
- **Test**: ~10K samples

### Data Format

CSV files with columns:
- `text`: Text content from the meme/post
- `image_path`: Relative path to the image
- `labels`: Comma-separated labels (e.g., "racist,sexist")

## 🔧 Configuration

Training configurations are managed via YAML files. Key parameters:

```yaml
# Model
backend: clip                    # clip, siglip, auto
head: fusion                     # fusion, mtl
encoder_name: openai/clip-vit-base-patch32
fusion_dim: 512

# Training
epochs: 8
batch_size: 32
lr_encoder: 1e-5
lr_head: 5e-4
weight_decay: 0.02
warmup_ratio: 0.05

# Loss
loss_type: bce                   # bce, focal
focal_gamma: 1.5

# Data Augmentation
augment: true
aug_scale_min: 0.8
aug_scale_max: 1.0
```

## 📈 Model Architectures

### Fusion Classifier

Late fusion architecture with gated attention and interaction features.

```
┌─────────────┐     ┌─────────────┐
│   Image     │     │    Text     │
│   Encoder   │     │   Encoder   │
│ (CLIP/SigLIP)     │ (CLIP/SigLIP)
└──────┬──────┘     └──────┬──────┘
       │                   │
       ▼                   ▼
┌─────────────┐     ┌─────────────┐
│  Projection │     │  Projection │
└──────┬──────┘     └──────┬──────┘
       │                   │
       └─────────┬─────────┘
                 │
                 ▼
         ┌─────────────┐
         │ Gated Fusion│◄── Modality presence flags
         └──────┬──────┘
                │
                ▼
    ┌───────────────────────┐
    │  Interaction Features │
    │ [fused, t, v, |t-v|,  │
    │       t*v]            │
    └───────────┬───────────┘
                │
                ▼
         ┌─────────────┐
         │Classification│
         │  Head (MLP) │
         └─────────────┘
```

### Multi-Task Learning (MTL) Classifier

Shared representation with task-specific binary heads for each hate category.

```
┌─────────────┐     ┌─────────────┐
│   Image     │     │    Text     │
│   Encoder   │     │   Encoder   │
└──────┬──────┘     └──────┬──────┘
       │                   │
       ▼                   ▼
┌─────────────┐     ┌─────────────┐
│  Projection │     │  Projection │
└──────┬──────┘     └──────┬──────┘
       │                   │
       └─────────┬─────────┘
                 │
                 ▼
         ┌─────────────┐
         │ Gated Fusion│
         └──────┬──────┘
                │
                ▼
         ┌─────────────┐
         │ Shared Head │
         │    (MLP)    │
         └──────┬──────┘
                │
       ┌────────┼────────┬────────┬────────┐
       ▼        ▼        ▼        ▼        ▼
   ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
   │Racist│ │Sexist│ │Homo- │ │Relig-│ │Other │
   │ Head │ │ Head │ │phobe │ │ ion  │ │ Hate │
   └──────┘ └──────┘ └──────┘ └──────┘ └──────┘

* Supports learnable uncertainty-based task weighting
* Per-task pos_weight for class imbalance
```

## 🧪 Experiments

### Trained Models

Models are saved in `runs/` directory:

1. **clip_fusion_mmhshateful**: CLIP with fusion head (baseline)
2. **clip_fusion_bigger_batch_mmhshateful**: CLIP with larger batch size
3. **clip_fusion_weighted_sampling_mmhshateful**: CLIP with class-balanced sampling
4. **clip_mtl_mmhshateful**: CLIP with multi-task learning heads
5. **siglip_fusion_mmhshateful**: SigLIP2 with fusion head

## 📝 Citation

If you use this code, please cite:

```bibtex
@misc{multimodal_hate_detection,
  title={Multi-Modal Hateful Content Classification},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/content_multi_modal}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
