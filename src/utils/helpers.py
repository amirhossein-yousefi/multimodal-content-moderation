"""
Utility functions for the multi-modal classification project.
"""

import os
import ast
import json
from typing import Tuple, Any, List, Dict, Optional
from pathlib import Path

import yaml


def ensure_dir(p: str) -> None:
    """Create directory if it doesn't exist.
    
    Args:
        p: Directory path to create.
    """
    os.makedirs(p, exist_ok=True)


def parse_label_list(v: Any) -> List[str]:
    """
    Parse label values from various formats into a list of strings.
    
    Handles:
      - None -> []
      - List -> [str(x) for x in list]
      - String representation of list -> parsed list
      - Comma-separated string -> split list
    
    Args:
        v: Label value in any supported format.
        
    Returns:
        List of label strings.
    """
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    
    s = str(v).strip()
    if not s:
        return []
    
    # Try parsing as Python literal (e.g., "['a', 'b']")
    try:
        maybe = ast.literal_eval(s)
        if isinstance(maybe, (list, tuple)):
            return [str(x).strip() for x in maybe if str(x).strip()]
    except Exception:
        pass
    
    # Fall back to comma-separated parsing
    return [t.strip() for t in s.split(",") if t.strip()]


def infer_size(proc) -> Tuple[int, int]:
    """
    Robustly infer (H, W) from a HuggingFace image processor.
    
    Works with CLIP, SigLIP, SigLIP2 and similar processors.
    
    Args:
        proc: HuggingFace image processor.
        
    Returns:
        Tuple of (height, width).
    """
    H = W = 224
    
    if hasattr(proc, "size"):
        sz = proc.size
        if isinstance(sz, dict):
            H = int(sz.get("height", sz.get("shortest_edge", H)))
            W = int(sz.get("width", sz.get("shortest_edge", W)))
        elif isinstance(sz, (int, float)):
            H = W = int(sz)
        elif isinstance(sz, (tuple, list)) and len(sz) == 2:
            H, W = int(sz[0]), int(sz[1])
    
    return H, W


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Supports config inheritance via '_base_' key.
    
    Args:
        config_path: Path to YAML configuration file.
        
    Returns:
        Configuration dictionary.
    """
    config_path = Path(config_path)
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Handle config inheritance
    if "_base_" in config:
        base_path = config_path.parent / config.pop("_base_")
        base_config = load_config(str(base_path))
        config = merge_configs(base_config, config)
    
    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two configuration dictionaries.
    
    Override values take precedence over base values.
    Nested dictionaries are merged recursively.
    
    Args:
        base: Base configuration dictionary.
        override: Override configuration dictionary.
        
    Returns:
        Merged configuration dictionary.
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def save_json(data: Any, path: str, indent: int = 2) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save (must be JSON serializable).
        path: Output file path.
        indent: JSON indentation level.
    """
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: str) -> Any:
    """
    Load data from JSON file.
    
    Args:
        path: Input file path.
        
    Returns:
        Loaded data.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_device() -> str:
    """
    Get the best available device (CUDA, MPS, or CPU).
    
    Returns:
        Device string.
    """
    import torch
    
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def count_parameters(model) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Dictionary with total, trainable, and frozen parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": 100 * trainable / total if total > 0 else 0,
    }


def setup_logging(log_dir: str, name: str = "train") -> None:
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory for log files.
        name: Logger name.
    """
    import logging
    
    ensure_dir(log_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"{name}.log")),
            logging.StreamHandler(),
        ],
    )
