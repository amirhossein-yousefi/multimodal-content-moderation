#!/usr/bin/env python
"""
Main training script for multi-modal hateful content classification.

Usage:
    python scripts/train.py --config config/clip_fusion.yaml
    python scripts/train.py --config config/default.yaml --model.backend siglip
    
Examples:
    # Train CLIP fusion model
    python scripts/train.py \
        --config config/clip_fusion.yaml \
        --training.num_train_epochs 10

    # Train with custom data paths
    python scripts/train.py \
        --config config/default.yaml \
        --data.train_csv path/to/train.csv \
        --data.val_csv path/to/val.csv
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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

from src.data import SocialHarmDataset, collate_fn
from src.models import MultiModalFusionClassifier, MultiTaskClassifier
from src.training import (
    make_compute_metrics_multi,
    make_compute_metrics_mtl,
    calibrate_thresholds,
    WeightedSamplerTrainer,
    group_params_for_optimizer,
    build_multilabel_sample_weights,
)
from src.utils import load_config, ensure_dir, save_json


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train multi-modal hateful content classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/default.yaml",
        help="Path to configuration YAML file"
    )
    
    # Data arguments
    parser.add_argument("--data.train_csv", dest="train_csv", type=str, default=None)
    parser.add_argument("--data.val_csv", dest="val_csv", type=str, default=None)
    parser.add_argument("--data.test_csv", dest="test_csv", type=str, default=None)
    parser.add_argument("--data.image_root", dest="image_root", type=str, default=None)
    
    # Model arguments
    parser.add_argument("--model.backend", dest="backend", type=str, choices=["clip", "siglip", "auto"], default=None)
    parser.add_argument("--model.head", dest="head", type=str, choices=["fusion", "mtl"], default=None)
    parser.add_argument("--model.encoder_name", dest="encoder_name", type=str, default=None)
    parser.add_argument("--model.fusion_dim", dest="fusion_dim", type=int, default=None)
    
    # Training arguments
    parser.add_argument("--training.num_train_epochs", dest="num_train_epochs", type=int, default=None)
    parser.add_argument("--training.max_steps", dest="max_steps", type=int, default=None)
    parser.add_argument("--training.per_device_train_batch_size", dest="batch_size", type=int, default=None)
    parser.add_argument("--training.lr_encoder", dest="lr_encoder", type=float, default=None)
    parser.add_argument("--training.lr_head", dest="lr_head", type=float, default=None)
    
    # Output
    parser.add_argument("--saving.output_dir", dest="output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    
    return parser.parse_args()


def override_config(config: dict, args) -> dict:
    """Override config values with command line arguments."""
    overrides = {
        ("data", "train_csv"): args.train_csv,
        ("data", "val_csv"): args.val_csv,
        ("data", "test_csv"): args.test_csv,
        ("data", "image_root"): args.image_root,
        ("model", "backend"): args.backend,
        ("model", "head"): args.head,
        ("model", "encoder_name"): args.encoder_name,
        ("model", "fusion_dim"): args.fusion_dim,
        ("training", "num_train_epochs"): args.num_train_epochs,
        ("training", "max_steps"): args.max_steps,
        ("training", "per_device_train_batch_size"): args.batch_size,
        ("training", "lr_encoder"): args.lr_encoder,
        ("training", "lr_head"): args.lr_head,
        ("saving", "output_dir"): args.output_dir,
    }
    
    for (section, key), value in overrides.items():
        if value is not None:
            if section not in config:
                config[section] = {}
            config[section][key] = value
    
    if args.seed is not None:
        config["seed"] = args.seed
    
    return config


def build_tokenizer_and_processor(encoder_name: str):
    """Build tokenizer and image processor for the encoder."""
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


def main():
    """Main training function."""
    args = parse_args()
    
    # Load and override config
    config = load_config(args.config)
    config = override_config(config, args)
    
    # Set seed
    seed = config.get("seed", 42)
    set_seed(seed)
    
    # Extract config sections
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    loss_cfg = config.get("loss", {})
    aug_cfg = config.get("augmentation", {})
    eval_cfg = config.get("evaluation", {})
    save_cfg = config.get("saving", {})
    log_cfg = config.get("logging", {})
    early_stop_cfg = config.get("early_stopping", {})
    
    # Prepare output directory
    output_dir = save_cfg.get("output_dir", "runs/experiment")
    ensure_dir(output_dir)
    
    # Save config
    save_json(config, os.path.join(output_dir, "config.json"))
    
    print(f"Output directory: {output_dir}")
    print(f"Config: {json.dumps(config, indent=2)}")
    
    # Build tokenizer and processor
    encoder_name = model_cfg.get("encoder_name", "openai/clip-vit-base-patch32")
    tokenizer, img_processor = build_tokenizer_and_processor(encoder_name)
    
    # Load class names
    class_names = data_cfg.get("class_names", [])
    if isinstance(class_names, str) and class_names:
        class_names = [c.strip() for c in class_names.split(",")]
    
    num_labels = len(class_names) if class_names else 1
    
    # Build datasets
    print("Loading datasets...")
    train_ds = SocialHarmDataset(
        csv_path=data_cfg["train_csv"],
        image_root=data_cfg.get("image_root", ""),
        tokenizer=tokenizer,
        img_proc=img_processor,
        max_text_length=model_cfg.get("max_text_length", 77),
        class_names=class_names if class_names else None,
        is_train=True,
        augment=aug_cfg.get("enabled", False),
        aug_scale=(aug_cfg.get("aug_scale_min", 0.8), aug_cfg.get("aug_scale_max", 1.0))
    )
    
    val_ds = SocialHarmDataset(
        csv_path=data_cfg["val_csv"],
        image_root=data_cfg.get("image_root", ""),
        tokenizer=tokenizer,
        img_proc=img_processor,
        max_text_length=model_cfg.get("max_text_length", 77),
        class_names=class_names if class_names else None,
        is_train=False,
    )
    
    test_ds = None
    if data_cfg.get("test_csv"):
        test_ds = SocialHarmDataset(
            csv_path=data_cfg["test_csv"],
            image_root=data_cfg.get("image_root", ""),
            tokenizer=tokenizer,
            img_proc=img_processor,
            max_text_length=model_cfg.get("max_text_length", 77),
            class_names=class_names if class_names else None,
            is_train=False,
        )
    
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds) if test_ds else 0}")
    
    # Build model
    head_type = model_cfg.get("head", "fusion")
    backend = model_cfg.get("backend", "clip")
    
    if head_type == "mtl":
        task_names = class_names if class_names else ["harmful"]
        model = MultiTaskClassifier(
            encoder_name=encoder_name,
            task_names=task_names,
            fusion_dim=model_cfg.get("fusion_dim", 512),
            backend=backend,
            threshold=eval_cfg.get("threshold", 0.5),
            freeze_text=model_cfg.get("freeze_text", False),
            freeze_image=model_cfg.get("freeze_image", False),
            head_hidden_dim=model_cfg.get("head_hidden_dim", None),
            learnable_task_weights=model_cfg.get("learnable_task_weights", False),
        )
        compute_metrics = make_compute_metrics_mtl(task_names, eval_cfg.get("threshold", 0.5))
    else:
        model = MultiModalFusionClassifier(
            encoder_name=encoder_name,
            num_labels=num_labels,
            fusion_dim=model_cfg.get("fusion_dim", 512),
            backend=backend,
            freeze_text=model_cfg.get("freeze_text", False),
            freeze_image=model_cfg.get("freeze_image", False),
            loss_type=loss_cfg.get("type", "bce"),
            focal_gamma=loss_cfg.get("focal_gamma", 1.5),
        )
        compute_metrics = make_compute_metrics_multi(num_labels, eval_cfg.get("threshold", 0.5))
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Training arguments
    precision = train_cfg.get("precision", "fp16")
    max_steps = train_cfg.get("max_steps", -1)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("num_train_epochs", 8),
        max_steps=max_steps if max_steps > 0 else -1,
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 32),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 64),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        learning_rate=train_cfg.get("lr_head", 5e-4),
        weight_decay=train_cfg.get("weight_decay", 0.02),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        fp16=(precision == "fp16"),
        bf16=(precision == "bf16"),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        dataloader_num_workers=train_cfg.get("num_workers", 4),
        eval_strategy=eval_cfg.get("eval_strategy", "epoch"),
        eval_steps=eval_cfg.get("eval_steps", 500) if eval_cfg.get("eval_strategy") == "steps" else None,
        save_strategy=save_cfg.get("save_strategy", "epoch"),
        save_total_limit=save_cfg.get("save_total_limit", 2),
        load_best_model_at_end=save_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=save_cfg.get("metric_for_best_model", "roc_macro"),
        greater_is_better=save_cfg.get("greater_is_better", True),
        logging_steps=log_cfg.get("logging_steps", 50),
        logging_dir=os.path.join(output_dir, "logs"),
        report_to=log_cfg.get("report_to", "tensorboard"),
        seed=seed,
    )
    
    # Callbacks
    callbacks = []
    if early_stop_cfg.get("enabled", True):
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=early_stop_cfg.get("patience", 3)
        ))
    
    # Build sample weights for weighted sampling
    sample_weights = None
    if train_cfg.get("sampler") == "weighted":
        sample_weights = build_multilabel_sample_weights(train_ds.labels)
        print("Using weighted sampling")
    
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
    print("Starting training...")
    trainer.train()
    
    # Evaluate on validation
    print("Evaluating on validation set...")
    val_results = trainer.evaluate()
    save_json(val_results, os.path.join(output_dir, "val_report.json"))
    print(f"Validation results: {val_results}")
    
    # Evaluate on test if available
    if test_ds:
        print("Evaluating on test set...")
        test_results = trainer.evaluate(test_ds, metric_key_prefix="test")
        save_json(test_results, os.path.join(output_dir, "test_metrics.json"))
        print(f"Test results: {test_results}")
    
    # Calibrate thresholds on validation set
    print("Calibrating thresholds...")
    val_preds = trainer.predict(val_ds)
    val_probs = 1 / (1 + np.exp(-val_preds.predictions))
    val_labels = val_preds.label_ids
    
    cal_cfg = eval_cfg.get("calibration", {})
    thresholds = calibrate_thresholds(
        val_probs, val_labels,
        t_start=cal_cfg.get("grid_start", 0.05),
        t_end=cal_cfg.get("grid_end", 0.95),
        steps=cal_cfg.get("grid_steps", 19)
    )
    
    # Save inference config
    inference_config = {
        "encoder_name": encoder_name,
        "backend": backend,
        "fusion_dim": model_cfg.get("fusion_dim", 512),
        "thresholds": thresholds,
        "class_names": class_names if class_names else ["harmful"],
        "best_checkpoint_dir": trainer.state.best_model_checkpoint,
        "use_logit_adjustment": loss_cfg.get("use_logit_adjustment", False),
    }
    save_json(inference_config, os.path.join(output_dir, "inference_config.json"))
    
    # Save label map
    label_map = {i: name for i, name in enumerate(class_names if class_names else ["harmful"])}
    save_json(label_map, os.path.join(output_dir, "label_map.json"))
    
    print(f"\nTraining complete! Output saved to: {output_dir}")
    print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"Calibrated thresholds: {thresholds}")


if __name__ == "__main__":
    main()
