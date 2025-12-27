# Scripts

This directory contains executable scripts for training, evaluation, and inference.

## Available Scripts

### train.py
Main training script for multi-modal classification models.

```bash
python scripts/train.py --config config/clip_fusion.yaml
```

### evaluate.py
Evaluate a trained model on a test set.

```bash
python scripts/evaluate.py \
    --checkpoint runs/clip_fusion_mmhshateful/checkpoint-33708 \
    --test_csv data/mmhs150k/test.csv \
    --image_root data/mmhs150k/images
```

### inference.py
Run inference on new samples.

```bash
# Single prediction
python scripts/inference.py \
    --checkpoint runs/clip_fusion_mmhshateful/checkpoint-33708 \
    --image path/to/image.jpg \
    --text "Sample text"

# Batch prediction
python scripts/inference.py \
    --checkpoint runs/clip_fusion_mmhshateful/checkpoint-33708 \
    --input_csv samples.csv \
    --output_csv predictions.csv
```

### prepare_data.py
Download and prepare datasets.

```bash
python scripts/prepare_data.py --dataset hateful_memes --output data/
```
