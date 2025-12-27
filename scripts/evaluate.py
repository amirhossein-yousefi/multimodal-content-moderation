#!/usr/bin/env python
"""
Evaluation script for multi-modal hateful content classification.

Usage:
    python scripts/evaluate.py \
        --checkpoint runs/clip_fusion_mmhshateful/checkpoint-33708 \
        --test_csv data/mmhs150k/test.csv \
        --image_root data/mmhs150k/images
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from transformers import AutoTokenizer, AutoImageProcessor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import SocialHarmDataset, collate_fn
from src.models import MultiModalFusionClassifier, MultiTaskClassifier
from src.training.metrics import compute_detailed_metrics
from src.utils import load_json, save_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate multi-modal classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        required=True,
        help="Path to test CSV file"
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="",
        help="Root directory for images"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results (default: checkpoint_dir/eval_results.json)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (auto-detected if not specified)"
    )
    
    return parser.parse_args()


def get_device(device_arg):
    """Get the best available device."""
    if device_arg:
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_from_checkpoint(checkpoint_dir: str, device: str):
    """Load model and config from checkpoint directory."""
    # Find inference config (in parent or checkpoint dir)
    checkpoint_path = Path(checkpoint_dir)
    
    # Look for config in parent directory (typical for HF checkpoints)
    config_paths = [
        checkpoint_path.parent / "inference_config.json",
        checkpoint_path / "inference_config.json",
        checkpoint_path.parent / "config.json",
    ]
    
    config = None
    for config_path in config_paths:
        if config_path.exists():
            config = load_json(str(config_path))
            break
    
    if config is None:
        raise FileNotFoundError(
            f"Could not find inference_config.json or config.json in {checkpoint_dir} or parent"
        )
    
    # Extract model config
    encoder_name = config.get("encoder_name", "openai/clip-vit-base-patch32")
    backend = config.get("backend", "clip")
    fusion_dim = config.get("fusion_dim", 512)
    class_names = config.get("class_names", ["harmful"])
    thresholds = config.get("thresholds", [0.5] * len(class_names))
    
    # Determine model type
    head_type = config.get("head", "fusion")
    
    # Build model
    if head_type == "mtl":
        model = MultiTaskClassifier(
            encoder_name=encoder_name,
            task_names=class_names,
            fusion_dim=fusion_dim,
            backend=backend,
        )
    else:
        model = MultiModalFusionClassifier(
            encoder_name=encoder_name,
            num_labels=len(class_names),
            fusion_dim=fusion_dim,
            backend=backend,
        )
    
    # Load weights
    weights_path = checkpoint_path / "model.safetensors"
    if weights_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(weights_path))
        model.load_state_dict(state_dict)
    else:
        # Try pytorch format
        weights_path = checkpoint_path / "pytorch_model.bin"
        if weights_path.exists():
            state_dict = torch.load(str(weights_path), map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Could not find model weights in {checkpoint_dir}")
    
    model.to(device)
    model.eval()
    
    # Load tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(encoder_name, use_fast=True)
    img_processor = AutoImageProcessor.from_pretrained(encoder_name)
    
    return model, tokenizer, img_processor, config


def evaluate(model, dataloader, device):
    """Run evaluation and collect predictions."""
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            logits = outputs["logits"]
            
            all_logits.append(logits.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())
    
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return all_logits, all_labels


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model, tokenizer, img_processor, config = load_model_from_checkpoint(args.checkpoint, device)
    
    class_names = config.get("class_names", ["harmful"])
    thresholds = config.get("thresholds", [0.5] * len(class_names))
    
    print(f"Classes: {class_names}")
    print(f"Thresholds: {thresholds}")
    
    # Load test dataset
    print(f"Loading test data from: {args.test_csv}")
    test_ds = SocialHarmDataset(
        csv_path=args.test_csv,
        image_root=args.image_root,
        tokenizer=tokenizer,
        img_proc=img_processor,
        max_text_length=77,
        class_names=class_names if len(class_names) > 1 else None,
        is_train=False,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )
    
    print(f"Test samples: {len(test_ds)}")
    
    # Evaluate
    logits, labels = evaluate(model, test_loader, device)
    probs = 1 / (1 + np.exp(-logits))
    
    # Compute metrics with calibrated thresholds
    # Use mean threshold for overall metrics
    mean_threshold = np.mean(thresholds)
    metrics = compute_detailed_metrics(probs, labels, mean_threshold, class_names)
    
    # Also compute per-class metrics with calibrated thresholds
    for i, (name, thresh) in enumerate(zip(class_names, thresholds)):
        bin_pred = (probs[:, i] >= thresh).astype(int)
        from sklearn.metrics import f1_score, precision_score, recall_score
        metrics["per_class"][name]["f1_calibrated"] = float(
            f1_score(labels[:, i], bin_pred, zero_division=0)
        )
        metrics["per_class"][name]["threshold"] = thresh
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"F1 Macro: {metrics['f1_macro']:.4f}")
    print(f"F1 Micro: {metrics['f1_micro']:.4f}")
    print(f"ROC-AUC Macro: {metrics['roc_auc_macro']:.4f}")
    print(f"Precision Macro: {metrics['precision_macro']:.4f}")
    print(f"Recall Macro: {metrics['recall_macro']:.4f}")
    print()
    print("Per-class results:")
    for name, class_metrics in metrics["per_class"].items():
        print(f"  {name}:")
        print(f"    F1: {class_metrics['f1']:.4f} (calibrated: {class_metrics.get('f1_calibrated', class_metrics['f1']):.4f})")
        print(f"    ROC-AUC: {class_metrics['roc_auc']:.4f}")
        print(f"    Precision: {class_metrics['precision']:.4f}")
        print(f"    Recall: {class_metrics['recall']:.4f}")
        print(f"    Support: {class_metrics['support']}")
    print("=" * 60)
    
    # Save results
    output_path = args.output or os.path.join(args.checkpoint, "eval_results.json")
    save_json(metrics, output_path)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
