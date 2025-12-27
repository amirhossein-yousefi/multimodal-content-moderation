#!/usr/bin/env python
"""
SageMaker Training Script for Multi-Modal Hateful Content Classification.

This script is designed to run on AWS SageMaker training jobs. It follows
SageMaker's training script conventions:
  - Reads hyperparameters from environment variables or command-line args
  - Reads input data from /opt/ml/input/data/<channel>
  - Writes model artifacts to /opt/ml/model
  - Writes output to /opt/ml/output

Usage (SageMaker):
    from sagemaker.pytorch import PyTorch
    
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
            'batch_size': 32,
            'backend': 'clip',
            'encoder_name': 'openai/clip-vit-base-patch32',
        },
    )
    
    estimator.fit({'train': s3_train, 'val': s3_val})

Local testing:
    python sagemaker/train_sagemaker.py \
        --train /path/to/train/data \
        --val /path/to/val/data \
        --model-dir ./model_output \
        --epochs 2
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# Add project root to path for local imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)

# Import from project modules
from src.data import SocialHarmDataset, collate_fn
from src.models import MultiModalFusionClassifier, MultiTaskClassifier
from src.training import (
    make_compute_metrics_multi,
    make_compute_metrics_mtl,
    calibrate_thresholds,
    WeightedSamplerTrainer,
    build_multilabel_sample_weights,
)
from src.utils import ensure_dir, save_json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# SageMaker default paths
SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
SM_OUTPUT_DATA_DIR = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data')
SM_CHANNEL_TRAIN = os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train')
SM_CHANNEL_VAL = os.environ.get('SM_CHANNEL_VALIDATION', '/opt/ml/input/data/validation')
SM_CHANNEL_TEST = os.environ.get('SM_CHANNEL_TEST', '/opt/ml/input/data/test')
SM_NUM_GPUS = int(os.environ.get('SM_NUM_GPUS', 1))
SM_NUM_CPUS = int(os.environ.get('SM_NUM_CPUS', 4))


def parse_args():
    """Parse command-line arguments and SageMaker hyperparameters."""
    parser = argparse.ArgumentParser(description='Train multi-modal classifier on SageMaker')
    
    # Data channels (SageMaker passes these)
    parser.add_argument('--train', type=str, default=SM_CHANNEL_TRAIN,
                        help='Path to training data directory')
    parser.add_argument('--val', '--validation', type=str, default=SM_CHANNEL_VAL,
                        dest='val', help='Path to validation data directory')
    parser.add_argument('--test', type=str, default=SM_CHANNEL_TEST,
                        help='Path to test data directory')
    parser.add_argument('--model-dir', type=str, default=SM_MODEL_DIR,
                        help='Path to save trained model')
    parser.add_argument('--output-data-dir', type=str, default=SM_OUTPUT_DATA_DIR,
                        help='Path to save output artifacts')
    
    # Model hyperparameters
    parser.add_argument('--backend', type=str, default='clip',
                        choices=['clip', 'siglip'],
                        help='Vision-language model backend')
    parser.add_argument('--head', type=str, default='fusion',
                        choices=['fusion', 'mtl'],
                        help='Classification head type')
    parser.add_argument('--encoder-name', type=str, 
                        default='openai/clip-vit-base-patch32',
                        help='HuggingFace encoder model name')
    parser.add_argument('--fusion-dim', type=int, default=512,
                        help='Fusion layer dimension')
    parser.add_argument('--max-text-length', type=int, default=77,
                        help='Maximum text sequence length')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=8,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size per device')
    parser.add_argument('--eval-batch-size', type=int, default=64,
                        help='Evaluation batch size per device')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--lr-encoder', type=float, default=1e-5,
                        help='Learning rate for encoder')
    parser.add_argument('--lr-head', type=float, default=5e-4,
                        help='Learning rate for classification head')
    parser.add_argument('--weight-decay', type=float, default=0.02,
                        help='Weight decay')
    parser.add_argument('--warmup-ratio', type=float, default=0.05,
                        help='Warmup ratio')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                        help='Maximum gradient norm')
    parser.add_argument('--precision', type=str, default='fp16',
                        choices=['fp16', 'bf16', 'fp32'],
                        help='Training precision')
    
    # Loss hyperparameters
    parser.add_argument('--loss-type', type=str, default='bce',
                        choices=['bce', 'focal'],
                        help='Loss function type')
    parser.add_argument('--focal-gamma', type=float, default=1.5,
                        help='Focal loss gamma parameter')
    
    # Data augmentation
    parser.add_argument('--augment', action='store_true', default=False,
                        help='Enable data augmentation')
    parser.add_argument('--aug-scale-min', type=float, default=0.8,
                        help='Augmentation minimum scale')
    parser.add_argument('--aug-scale-max', type=float, default=1.0,
                        help='Augmentation maximum scale')
    
    # Sampling
    parser.add_argument('--weighted-sampling', action='store_true', default=False,
                        help='Use weighted sampling for imbalanced classes')
    
    # Early stopping
    parser.add_argument('--early-stopping-patience', type=int, default=3,
                        help='Early stopping patience (0 to disable)')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num-workers', type=int, default=SM_NUM_CPUS,
                        help='Number of data loading workers')
    parser.add_argument('--class-names', type=str, default='hateful',
                        help='Comma-separated class names')
    
    args = parser.parse_args()
    return args


def find_data_file(directory: str, pattern: str = '*.csv') -> str:
    """Find a data file in a directory matching pattern."""
    import glob
    search_path = os.path.join(directory, pattern)
    files = glob.glob(search_path)
    
    # Priority order for finding the right file
    priority = ['train.csv', 'val.csv', 'validation.csv', 'test.csv', 'data.csv']
    
    for pf in priority:
        for f in files:
            if os.path.basename(f) == pf:
                return f
    
    if files:
        return files[0]
    
    raise FileNotFoundError(f"No {pattern} files found in {directory}")


def find_image_root(data_dir: str) -> str:
    """Find the image root directory in the data channel."""
    # Check common patterns
    candidates = [
        os.path.join(data_dir, 'images'),
        os.path.join(data_dir, 'img'),
        os.path.join(data_dir, 'img_resized'),
        data_dir,  # Images might be in the same directory
    ]
    
    for candidate in candidates:
        if os.path.isdir(candidate):
            # Check if there are image files
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.webp']:
                import glob
                if glob.glob(os.path.join(candidate, '**', ext), recursive=True):
                    return candidate
    
    return data_dir


def build_tokenizer_and_processor(encoder_name: str):
    """Build tokenizer and image processor for the encoder."""
    logger.info(f"Loading tokenizer and processor from: {encoder_name}")
    
    tok = AutoTokenizer.from_pretrained(encoder_name, use_fast=True)
    proc = AutoImageProcessor.from_pretrained(encoder_name)
    
    # Ensure pad token exists
    if getattr(tok, "pad_token_id", None) is None:
        if getattr(tok, "eos_token", None) is not None:
            tok.pad_token = tok.eos_token
        elif getattr(tok, "cls_token", None) is not None:
            tok.pad_token = tok.cls_token
        elif getattr(tok, "bos_token", None) is not None:
            tok.pad_token = tok.bos_token
    
    return tok, proc


def train(args):
    """Main training function."""
    logger.info("=" * 60)
    logger.info("SageMaker Multi-Modal Training Job")
    logger.info("=" * 60)
    
    # Set seed
    set_seed(args.seed)
    logger.info(f"Random seed: {args.seed}")
    
    # Parse class names
    class_names = [c.strip() for c in args.class_names.split(',') if c.strip()]
    num_labels = len(class_names)
    logger.info(f"Classes: {class_names}")
    
    # Build tokenizer and processor
    tokenizer, img_processor = build_tokenizer_and_processor(args.encoder_name)
    
    # Find data files
    logger.info("Looking for data files...")
    
    train_csv = find_data_file(args.train)
    val_csv = find_data_file(args.val)
    train_image_root = find_image_root(args.train)
    val_image_root = find_image_root(args.val)
    
    logger.info(f"Train CSV: {train_csv}")
    logger.info(f"Val CSV: {val_csv}")
    logger.info(f"Train image root: {train_image_root}")
    logger.info(f"Val image root: {val_image_root}")
    
    # Build datasets
    logger.info("Loading datasets...")
    
    train_ds = SocialHarmDataset(
        csv_path=train_csv,
        image_root=train_image_root,
        tokenizer=tokenizer,
        img_proc=img_processor,
        max_text_length=args.max_text_length,
        class_names=class_names,
        is_train=True,
        augment=args.augment,
        aug_scale=(args.aug_scale_min, args.aug_scale_max)
    )
    
    val_ds = SocialHarmDataset(
        csv_path=val_csv,
        image_root=val_image_root,
        tokenizer=tokenizer,
        img_proc=img_processor,
        max_text_length=args.max_text_length,
        class_names=class_names,
        is_train=False,
    )
    
    logger.info(f"Train samples: {len(train_ds)}")
    logger.info(f"Val samples: {len(val_ds)}")
    
    # Build test dataset if available
    test_ds = None
    if os.path.isdir(args.test):
        try:
            test_csv = find_data_file(args.test)
            test_image_root = find_image_root(args.test)
            test_ds = SocialHarmDataset(
                csv_path=test_csv,
                image_root=test_image_root,
                tokenizer=tokenizer,
                img_proc=img_processor,
                max_text_length=args.max_text_length,
                class_names=class_names,
                is_train=False,
            )
            logger.info(f"Test samples: {len(test_ds)}")
        except FileNotFoundError:
            logger.warning("No test data found, skipping test evaluation")
    
    # Build model
    logger.info(f"Building model: backend={args.backend}, head={args.head}")
    
    if args.head == 'mtl':
        model = MultiTaskClassifier(
            encoder_name=args.encoder_name,
            task_names=class_names,
            fusion_dim=args.fusion_dim,
            backend=args.backend,
            threshold=0.5,
        )
        compute_metrics = make_compute_metrics_mtl(class_names, 0.5)
    else:
        model = MultiModalFusionClassifier(
            encoder_name=args.encoder_name,
            num_labels=num_labels,
            fusion_dim=args.fusion_dim,
            backend=args.backend,
            loss_type=args.loss_type,
            focal_gamma=args.focal_gamma,
        )
        compute_metrics = make_compute_metrics_multi(num_labels, 0.5)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Determine output directory
    output_dir = args.model_dir
    ensure_dir(output_dir)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr_head,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        fp16=(args.precision == 'fp16' and torch.cuda.is_available()),
        bf16=(args.precision == 'bf16' and torch.cuda.is_available()),
        gradient_checkpointing=False,
        dataloader_num_workers=args.num_workers,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='roc_macro',
        greater_is_better=True,
        logging_steps=50,
        logging_dir=os.path.join(output_dir, 'logs'),
        report_to='tensorboard',
        seed=args.seed,
        # SageMaker-specific: disable Hub push
        push_to_hub=False,
    )
    
    # Callbacks
    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience
        ))
    
    # Build sample weights for weighted sampling
    sample_weights = None
    if args.weighted_sampling:
        sample_weights = build_multilabel_sample_weights(train_ds.labels)
        logger.info("Using weighted sampling for imbalanced classes")
    
    # Choose trainer class
    TrainerClass = WeightedSamplerTrainer if sample_weights is not None else Trainer
    
    # Initialize trainer
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "compute_metrics": compute_metrics,
        "data_collator": collate_fn,
        "callbacks": callbacks,
    }
    if sample_weights is not None:
        trainer_kwargs["sample_weights"] = sample_weights
    
    trainer = TrainerClass(**trainer_kwargs)
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Evaluate on validation
    logger.info("Evaluating on validation set...")
    val_results = trainer.evaluate()
    logger.info(f"Validation results: {val_results}")
    
    # Evaluate on test if available
    test_results = None
    if test_ds:
        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(test_ds, metric_key_prefix="test")
        logger.info(f"Test results: {test_results}")
    
    # Calibrate thresholds on validation set
    logger.info("Calibrating thresholds...")
    val_preds = trainer.predict(val_ds)
    val_probs = 1 / (1 + np.exp(-val_preds.predictions))
    val_labels = val_preds.label_ids
    
    thresholds = calibrate_thresholds(
        val_probs, val_labels,
        t_start=0.05,
        t_end=0.95,
        steps=19
    )
    logger.info(f"Calibrated thresholds: {thresholds}")
    
    # Save model to model_dir (SageMaker will package this)
    logger.info(f"Saving model to: {args.model_dir}")
    
    # Save model weights
    trainer.save_model(args.model_dir)
    
    # Save inference config (needed for SageMaker inference)
    inference_config = {
        "encoder_name": args.encoder_name,
        "backend": args.backend,
        "head": args.head,
        "fusion_dim": args.fusion_dim,
        "thresholds": thresholds,
        "class_names": class_names,
        "max_text_length": args.max_text_length,
    }
    save_json(inference_config, os.path.join(args.model_dir, "inference_config.json"))
    
    # Save label map
    label_map = {i: name for i, name in enumerate(class_names)}
    save_json(label_map, os.path.join(args.model_dir, "label_map.json"))
    
    # Save metrics to output dir
    ensure_dir(args.output_data_dir)
    
    save_json(val_results, os.path.join(args.output_data_dir, "val_metrics.json"))
    if test_results:
        save_json(test_results, os.path.join(args.output_data_dir, "test_metrics.json"))
    
    # Save hyperparameters for reference
    hyperparams = vars(args)
    hyperparams['thresholds'] = thresholds
    save_json(hyperparams, os.path.join(args.output_data_dir, "hyperparameters.json"))
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Model saved to: {args.model_dir}")
    logger.info(f"Metrics saved to: {args.output_data_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    args = parse_args()
    train(args)
