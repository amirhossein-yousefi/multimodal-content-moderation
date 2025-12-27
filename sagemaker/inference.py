"""
SageMaker Inference Handler for Multi-Modal Content Classification.

This module implements the model_fn, input_fn, predict_fn, and output_fn
functions required by SageMaker for custom inference endpoints.

The handler supports:
  - Single prediction requests (JSON with text/image_base64)
  - Batch prediction requests (JSON array)
  - Multi-part form data with image files

Request Format (JSON):
    Single:
        {
            "text": "sample text content",
            "image_base64": "base64_encoded_image_string"  # optional
        }
    
    Batch:
        {
            "instances": [
                {"text": "content 1", "image_base64": "..."},
                {"text": "content 2"}
            ]
        }

Response Format:
    {
        "predictions": [
            {
                "class_predictions": {"hateful": false, ...},
                "probabilities": {"hateful": 0.12, ...},
                "any_harmful": false
            }
        ]
    }
"""

import os
import sys
import json
import base64
import logging
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def model_fn(model_dir: str) -> Dict[str, Any]:
    """
    Load model from the model directory.
    
    This function is called once when the SageMaker endpoint is created
    or when a batch transform job starts.
    
    Args:
        model_dir: Path to the directory containing model artifacts.
        
    Returns:
        Dictionary containing the model and all required components.
    """
    logger.info(f"Loading model from: {model_dir}")
    
    # Add project root to path for imports
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from transformers import AutoTokenizer, AutoImageProcessor
    from src.models import MultiModalFusionClassifier, MultiTaskClassifier
    from src.utils import load_json
    
    # Load inference config
    config_path = os.path.join(model_dir, "inference_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"inference_config.json not found in {model_dir}")
    
    config = load_json(config_path)
    
    # Extract model configuration
    encoder_name = config.get("encoder_name", "openai/clip-vit-base-patch32")
    backend = config.get("backend", "clip")
    head_type = config.get("head", "fusion")
    fusion_dim = config.get("fusion_dim", 512)
    class_names = config.get("class_names", ["harmful"])
    thresholds = config.get("thresholds", [0.5] * len(class_names))
    max_text_length = config.get("max_text_length", 77)
    
    logger.info(f"Model config: backend={backend}, head={head_type}, encoder={encoder_name}")
    logger.info(f"Classes: {class_names}")
    
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
    
    # Load model weights
    weights_path = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(weights_path):
        from safetensors.torch import load_file
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded weights from: {weights_path}")
    else:
        weights_path = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
            logger.info(f"Loaded weights from: {weights_path}")
        else:
            raise FileNotFoundError(f"No model weights found in {model_dir}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    logger.info(f"Model loaded to device: {device}")
    
    # Load tokenizer and image processor
    tokenizer = AutoTokenizer.from_pretrained(encoder_name, use_fast=True)
    img_processor = AutoImageProcessor.from_pretrained(encoder_name)
    
    # Get image size
    img_size = _get_img_size(img_processor)
    
    return {
        "model": model,
        "tokenizer": tokenizer,
        "img_processor": img_processor,
        "config": config,
        "class_names": class_names,
        "thresholds": thresholds,
        "max_text_length": max_text_length,
        "img_size": img_size,
        "device": device,
    }


def _get_img_size(img_processor) -> Tuple[int, int]:
    """Get expected image size from processor."""
    if hasattr(img_processor, "size"):
        sz = img_processor.size
        if isinstance(sz, dict):
            h = sz.get("height", sz.get("shortest_edge", 224))
            w = sz.get("width", sz.get("shortest_edge", 224))
            return int(h), int(w)
        elif isinstance(sz, int):
            return sz, sz
    return 224, 224


def input_fn(request_body: bytes, request_content_type: str) -> Dict[str, Any]:
    """
    Deserialize input data for prediction.
    
    Args:
        request_body: Raw request body bytes.
        request_content_type: MIME type of the request.
        
    Returns:
        Parsed input data dictionary.
    """
    logger.info(f"Received request with content type: {request_content_type}")
    
    if request_content_type == "application/json":
        data = json.loads(request_body.decode("utf-8"))
        
        # Handle batch format
        if "instances" in data:
            return {"instances": data["instances"]}
        
        # Handle single request - wrap in list
        return {"instances": [data]}
    
    elif request_content_type.startswith("multipart/form-data"):
        # Handle multipart form data (for direct image upload)
        # This requires additional parsing - using simple approach
        raise ValueError(
            "multipart/form-data not directly supported. "
            "Please encode images as base64 in JSON requests."
        )
    
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data: Dict[str, Any], model_artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make predictions on input data.
    
    Args:
        input_data: Parsed input from input_fn.
        model_artifacts: Model and components from model_fn.
        
    Returns:
        Prediction results dictionary.
    """
    from torchvision import transforms as T
    
    model = model_artifacts["model"]
    tokenizer = model_artifacts["tokenizer"]
    img_processor = model_artifacts["img_processor"]
    class_names = model_artifacts["class_names"]
    thresholds = model_artifacts["thresholds"]
    max_text_length = model_artifacts["max_text_length"]
    img_size = model_artifacts["img_size"]
    device = model_artifacts["device"]
    
    # Get image normalization parameters
    mean = getattr(img_processor, "image_mean", [0.5, 0.5, 0.5])
    std = getattr(img_processor, "image_std", [0.5, 0.5, 0.5])
    
    H, W = img_size
    transform = T.Compose([
        T.Resize(H, antialias=True),
        T.CenterCrop((H, W)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    
    instances = input_data.get("instances", [])
    results = []
    
    for instance in instances:
        # Extract text
        text = instance.get("text", "") or ""
        
        # Extract and decode image
        image_b64 = instance.get("image_base64")
        image_url = instance.get("image_url")  # Optional: URL-based loading
        
        pixel_values, image_present = _process_image(
            image_b64=image_b64,
            image_url=image_url,
            transform=transform,
            img_size=img_size
        )
        
        text_present = 1.0 if text.strip() else 0.0
        
        # Tokenize text
        tok = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_text_length,
            return_tensors="pt",
        )
        
        # Build batch
        batch = {
            "input_ids": tok["input_ids"].to(device),
            "attention_mask": tok["attention_mask"].to(device),
            "pixel_values": pixel_values.unsqueeze(0).to(device),
            "text_present": torch.tensor([text_present], device=device),
            "image_present": torch.tensor([image_present], device=device),
        }
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs["logits"].cpu().numpy()[0]
        
        # Compute probabilities
        probs = 1 / (1 + np.exp(-logits))
        
        # Apply thresholds
        class_predictions = {}
        probabilities = {}
        
        for i, (name, prob, thresh) in enumerate(zip(class_names, probs, thresholds)):
            class_predictions[name] = bool(prob >= thresh)
            probabilities[name] = float(prob)
        
        results.append({
            "class_predictions": class_predictions,
            "probabilities": probabilities,
            "any_harmful": any(class_predictions.values()),
        })
    
    return {"predictions": results}


def _process_image(
    image_b64: Optional[str],
    image_url: Optional[str],
    transform,
    img_size: Tuple[int, int]
) -> Tuple[torch.Tensor, float]:
    """Process image from base64 or URL."""
    H, W = img_size
    
    if image_b64:
        try:
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            return transform(image), 1.0
        except Exception as e:
            logger.warning(f"Failed to decode base64 image: {e}")
            return torch.zeros(3, H, W), 0.0
    
    if image_url:
        try:
            import requests
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            return transform(image), 1.0
        except Exception as e:
            logger.warning(f"Failed to load image from URL: {e}")
            return torch.zeros(3, H, W), 0.0
    
    # No image provided
    return torch.zeros(3, H, W), 0.0


def output_fn(prediction: Dict[str, Any], accept: str) -> Tuple[bytes, str]:
    """
    Serialize prediction results.
    
    Args:
        prediction: Prediction results from predict_fn.
        accept: Requested response MIME type.
        
    Returns:
        Tuple of (serialized response, content type).
    """
    if accept == "application/json" or accept == "*/*":
        return json.dumps(prediction).encode("utf-8"), "application/json"
    else:
        raise ValueError(f"Unsupported accept type: {accept}")


# ============================================================================
# Batch Transform Handler
# ============================================================================

class BatchTransformHandler:
    """
    Handler for SageMaker Batch Transform jobs.
    
    Processes input files line by line (JSON Lines format).
    """
    
    def __init__(self):
        self.model_artifacts = None
    
    def handle(self, input_data: bytes, context) -> bytes:
        """Handle batch transform request."""
        if self.model_artifacts is None:
            model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
            self.model_artifacts = model_fn(model_dir)
        
        # Parse JSON Lines input
        lines = input_data.decode("utf-8").strip().split("\n")
        results = []
        
        for line in lines:
            if not line.strip():
                continue
            
            try:
                instance = json.loads(line)
                input_parsed = {"instances": [instance]}
                prediction = predict_fn(input_parsed, self.model_artifacts)
                results.append(prediction["predictions"][0])
            except Exception as e:
                logger.error(f"Error processing line: {e}")
                results.append({"error": str(e)})
        
        return "\n".join(json.dumps(r) for r in results).encode("utf-8")


# ============================================================================
# Direct invocation support for testing
# ============================================================================

if __name__ == "__main__":
    """Test inference locally."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SageMaker inference locally")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to model directory")
    parser.add_argument("--text", type=str, default="Test content",
                        help="Text to classify")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to image file")
    args = parser.parse_args()
    
    # Load model
    artifacts = model_fn(args.model_dir)
    
    # Prepare input
    instance = {"text": args.text}
    if args.image and os.path.exists(args.image):
        with open(args.image, "rb") as f:
            instance["image_base64"] = base64.b64encode(f.read()).decode("utf-8")
    
    # Make prediction
    input_data = {"instances": [instance]}
    result = predict_fn(input_data, artifacts)
    
    print("\n" + "=" * 50)
    print("PREDICTION RESULT")
    print("=" * 50)
    print(json.dumps(result, indent=2))
