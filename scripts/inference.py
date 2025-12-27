#!/usr/bin/env python
"""
Inference script for multi-modal hateful content classification.

Usage:
    # Single prediction
    python scripts/inference.py \
        --checkpoint runs/clip_fusion_mmhshateful/checkpoint-33708 \
        --image path/to/image.jpg \
        --text "Sample text content"
    
    # Batch prediction from CSV
    python scripts/inference.py \
        --checkpoint runs/clip_fusion_mmhshateful/checkpoint-33708 \
        --input_csv data/test_samples.csv \
        --output_csv predictions.csv
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor
from tqdm import tqdm

from src.models import MultiModalFusionClassifier, MultiTaskClassifier
from src.utils import load_json


class MultiModalClassifier:
    """High-level inference wrapper for multi-modal classification."""
    
    def __init__(
        self,
        checkpoint_dir: str,
        device: Optional[str] = None
    ):
        """
        Initialize classifier from checkpoint.
        
        Args:
            checkpoint_dir: Path to model checkpoint directory.
            device: Device to use (auto-detected if None).
        """
        self.device = self._get_device(device)
        self._load_model(checkpoint_dir)
    
    def _get_device(self, device_arg: Optional[str]) -> str:
        if device_arg:
            return device_arg
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _load_model(self, checkpoint_dir: str):
        """Load model from checkpoint."""
        checkpoint_path = Path(checkpoint_dir)
        
        # Find config
        config_paths = [
            checkpoint_path.parent / "inference_config.json",
            checkpoint_path / "inference_config.json",
        ]
        
        self.config = None
        for config_path in config_paths:
            if config_path.exists():
                self.config = load_json(str(config_path))
                break
        
        if self.config is None:
            raise FileNotFoundError(f"Could not find inference_config.json in {checkpoint_dir}")
        
        # Model parameters
        encoder_name = self.config.get("encoder_name", "openai/clip-vit-base-patch32")
        backend = self.config.get("backend", "clip")
        fusion_dim = self.config.get("fusion_dim", 512)
        self.class_names = self.config.get("class_names", ["harmful"])
        self.thresholds = self.config.get("thresholds", [0.5] * len(self.class_names))
        
        # Build model
        head_type = self.config.get("head", "fusion")
        if head_type == "mtl":
            self.model = MultiTaskClassifier(
                encoder_name=encoder_name,
                task_names=self.class_names,
                fusion_dim=fusion_dim,
                backend=backend,
            )
        else:
            self.model = MultiModalFusionClassifier(
                encoder_name=encoder_name,
                num_labels=len(self.class_names),
                fusion_dim=fusion_dim,
                backend=backend,
            )
        
        # Load weights
        weights_path = checkpoint_path / "model.safetensors"
        if weights_path.exists():
            from safetensors.torch import load_file
            state_dict = load_file(str(weights_path))
            self.model.load_state_dict(state_dict)
        else:
            weights_path = checkpoint_path / "pytorch_model.bin"
            state_dict = torch.load(str(weights_path), map_location="cpu")
            self.model.load_state_dict(state_dict)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer and processor
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name, use_fast=True)
        self.img_processor = AutoImageProcessor.from_pretrained(encoder_name)
        
        # Get image size
        self.img_size = self._get_img_size()
    
    def _get_img_size(self) -> Tuple[int, int]:
        """Get expected image size."""
        if hasattr(self.img_processor, "size"):
            sz = self.img_processor.size
            if isinstance(sz, dict):
                h = sz.get("height", sz.get("shortest_edge", 224))
                w = sz.get("width", sz.get("shortest_edge", 224))
                return int(h), int(w)
            elif isinstance(sz, int):
                return sz, sz
        return 224, 224
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, float]:
        """Preprocess an image for the model."""
        H, W = self.img_size
        
        if not image_path or not os.path.exists(image_path):
            return torch.zeros(3, H, W), 0.0
        
        try:
            from torchvision import transforms as T
            
            mean = getattr(self.img_processor, "image_mean", [0.5, 0.5, 0.5])
            std = getattr(self.img_processor, "image_std", [0.5, 0.5, 0.5])
            
            transform = T.Compose([
                T.Resize(H, antialias=True),
                T.CenterCrop((H, W)),
                T.ToTensor(),
                T.Normalize(mean, std),
            ])
            
            img = Image.open(image_path).convert("RGB")
            return transform(img), 1.0
        except Exception as e:
            print(f"Warning: Could not load image {image_path}: {e}")
            return torch.zeros(3, H, W), 0.0
    
    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Preprocess text for the model."""
        tok = self.tokenizer(
            text or "",
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        return {
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
        }
    
    def predict(
        self,
        text: Optional[str] = None,
        image_path: Optional[str] = None,
        return_probs: bool = False
    ) -> Dict[str, any]:
        """
        Make a single prediction.
        
        Args:
            text: Text content.
            image_path: Path to image file.
            return_probs: Whether to return probabilities.
            
        Returns:
            Dictionary with predictions.
        """
        # Preprocess
        text_inputs = self.preprocess_text(text or "")
        pixel_values, image_present = self.preprocess_image(image_path or "")
        text_present = 1.0 if text and text.strip() else 0.0
        
        # Build batch
        batch = {
            "input_ids": text_inputs["input_ids"].to(self.device),
            "attention_mask": text_inputs["attention_mask"].to(self.device),
            "pixel_values": pixel_values.unsqueeze(0).to(self.device),
            "text_present": torch.tensor([text_present], device=self.device),
            "image_present": torch.tensor([image_present], device=self.device),
        }
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**batch)
            logits = outputs["logits"].cpu().numpy()[0]
        
        probs = 1 / (1 + np.exp(-logits))
        
        # Apply thresholds
        predictions = {}
        for i, (name, prob, thresh) in enumerate(zip(self.class_names, probs, self.thresholds)):
            predictions[name] = {
                "label": bool(prob >= thresh),
                "probability": float(prob),
                "threshold": float(thresh),
            }
        
        result = {
            "predictions": predictions,
            "any_harmful": any(p["label"] for p in predictions.values()),
        }
        
        if return_probs:
            result["probabilities"] = probs.tolist()
        
        return result
    
    def predict_batch(
        self,
        texts: List[str],
        image_paths: List[str],
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Make batch predictions.
        
        Args:
            texts: List of text strings.
            image_paths: List of image paths.
            batch_size: Processing batch size.
            
        Returns:
            List of prediction dictionaries.
        """
        results = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
            batch_texts = texts[i:i+batch_size]
            batch_images = image_paths[i:i+batch_size]
            
            # Process each item in batch
            batch_results = []
            for text, img_path in zip(batch_texts, batch_images):
                result = self.predict(text, img_path)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with multi-modal classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    
    # Single prediction
    parser.add_argument("--text", type=str, default=None, help="Text content")
    parser.add_argument("--image", type=str, default=None, help="Image path")
    
    # Batch prediction
    parser.add_argument("--input_csv", type=str, default=None, help="Input CSV for batch prediction")
    parser.add_argument("--output_csv", type=str, default=None, help="Output CSV for predictions")
    parser.add_argument("--image_root", type=str, default="", help="Root directory for images")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load classifier
    print(f"Loading model from: {args.checkpoint}")
    classifier = MultiModalClassifier(args.checkpoint, device=args.device)
    print(f"Using device: {classifier.device}")
    print(f"Classes: {classifier.class_names}")
    
    if args.input_csv:
        # Batch prediction
        print(f"Loading data from: {args.input_csv}")
        df = pd.read_csv(args.input_csv)
        
        texts = df["text"].fillna("").tolist()
        image_paths = df["image_path"].fillna("").tolist()
        
        # Prepend image root if specified
        if args.image_root:
            image_paths = [
                os.path.join(args.image_root, p) if p and not os.path.isabs(p) else p
                for p in image_paths
            ]
        
        results = classifier.predict_batch(texts, image_paths, args.batch_size)
        
        # Add predictions to dataframe
        for class_name in classifier.class_names:
            df[f"pred_{class_name}"] = [r["predictions"][class_name]["label"] for r in results]
            df[f"prob_{class_name}"] = [r["predictions"][class_name]["probability"] for r in results]
        
        df["any_harmful"] = [r["any_harmful"] for r in results]
        
        output_path = args.output_csv or "predictions.csv"
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")
        
    elif args.text or args.image:
        # Single prediction
        result = classifier.predict(args.text, args.image, return_probs=True)
        
        print("\n" + "=" * 40)
        print("PREDICTION RESULT")
        print("=" * 40)
        print(f"Text: {args.text[:100]}..." if args.text and len(args.text) > 100 else f"Text: {args.text}")
        print(f"Image: {args.image}")
        print()
        print("Classifications:")
        for name, pred in result["predictions"].items():
            status = "✓ DETECTED" if pred["label"] else "✗ Not detected"
            print(f"  {name}: {status} (prob: {pred['probability']:.3f}, threshold: {pred['threshold']:.2f})")
        print()
        print(f"Any harmful content: {'YES' if result['any_harmful'] else 'NO'}")
        print("=" * 40)
    else:
        print("Error: Specify either --text/--image for single prediction or --input_csv for batch")
        sys.exit(1)


if __name__ == "__main__":
    main()
