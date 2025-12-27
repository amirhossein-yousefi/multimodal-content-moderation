"""
Dataset classes for multi-modal content classification.
"""

from typing import Tuple, List, Optional, Dict, Any
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

from src.utils.helpers import parse_label_list, infer_size


class SocialHarmDataset(Dataset):
    """
    Multi-modal dataset for social media harm detection.
    
    Supports two CSV schemas:
      - Binary: columns [text, image_path, label] (0/1)
      - Multi-label: columns [text, image_path, labels] (comma-separated subset of class_names)
    
    Args:
        csv_path: Path to the CSV file containing data annotations.
        image_root: Root directory containing images.
        tokenizer: HuggingFace tokenizer for text processing.
        img_proc: HuggingFace image processor.
        max_text_length: Maximum sequence length for text tokenization.
        class_names: List of class names for multi-label classification.
        is_train: Whether this is training data (enables augmentation).
        augment: Whether to apply data augmentation.
        aug_scale: Scale range for random resized crop augmentation.
    """
    
    def __init__(
        self,
        csv_path: str,
        image_root: str,
        tokenizer,
        img_proc,
        max_text_length: int,
        class_names: Optional[List[str]] = None,
        is_train: bool = False,
        augment: bool = False,
        aug_scale: Tuple[float, float] = (0.8, 1.0)
    ) -> None:
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.tok = tokenizer
        self.img_proc = img_proc
        self.max_len = max_text_length
        self.is_train = is_train
        self.augment = augment if is_train else False

        # Determine label type
        has_binary = "label" in self.df.columns
        has_multilabel = "labels" in self.df.columns
        if not has_binary and not has_multilabel:
            raise ValueError("CSV must have column 'label' (0/1) or 'labels' (comma-separated).")

        # Process labels
        if has_multilabel:
            if not class_names:
                raise ValueError("Provide class_names for multi-label classification.")
            self.class_names = [c.strip() for c in class_names]
            self.class2id = {c: i for i, c in enumerate(self.class_names)}
            Y = []
            for v in self.df["labels"].fillna(""):
                labs = parse_label_list(v)
                vec = np.zeros(len(self.class_names), dtype=np.float32)
                for name in labs:
                    if name in self.class2id:
                        vec[self.class2id[name]] = 1.0
                Y.append(vec)
            self.labels = torch.tensor(np.stack(Y, axis=0), dtype=torch.float32)
        else:
            self.class_names = ["harmful"]
            self.class2id = {"harmful": 0}
            self.labels = torch.tensor(
                self.df["label"].astype(int).values.reshape(-1, 1).astype(np.float32)
            )

        self.texts = self.df["text"].fillna("").astype(str).tolist()
        self.paths = self.df["image_path"].fillna("").astype(str).tolist()
        self.H, self.W = infer_size(self.img_proc)

        # Initialize image transforms
        self._init_transforms(aug_scale)

    def _init_transforms(self, aug_scale: Tuple[float, float]) -> None:
        """Initialize torchvision transforms for image preprocessing."""
        from torchvision import transforms as T
        
        mean = getattr(self.img_proc, "image_mean", [0.5, 0.5, 0.5])
        std = getattr(self.img_proc, "image_std", [0.5, 0.5, 0.5])

        self.train_tf = T.Compose([
            T.RandomResizedCrop((self.H, self.W), scale=aug_scale, antialias=True),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.1, 0.1, 0.1, 0.05),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        self.eval_tf = T.Compose([
            T.Resize(self.H, antialias=True) if isinstance(self.H, int) else T.Resize((self.H, self.W), antialias=True),
            T.CenterCrop((self.H, self.W)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

    def __len__(self) -> int:
        return len(self.texts)

    def _load_image(self, rel: str) -> Tuple[torch.Tensor, float]:
        """Load and preprocess an image.
        
        Args:
            rel: Relative path to the image.
            
        Returns:
            Tuple of (pixel_values, image_present_flag)
        """
        if not rel:
            # Missing path - return zero tensor
            px = torch.zeros(3, self.H, self.W)
            return px, 0.0
            
        path = rel if os.path.isabs(rel) or not self.image_root else os.path.join(self.image_root, rel)
        try:
            im = Image.open(path).convert("RGB")
            if self.is_train and self.augment:
                px = self.train_tf(im)
            else:
                px = self.eval_tf(im)
            return px, 1.0
        except Exception:
            # Corrupt/unreadable image - return zero tensor
            px = torch.zeros(3, self.H, self.W)
            return px, 0.0

    def __getitem__(self, i: int) -> Dict[str, Any]:
        """Get a single sample from the dataset."""
        text = self.texts[i] or ""
        
        # Tokenize text
        tok = self.tok(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = tok["input_ids"][0]
        
        if "attention_mask" in tok:
            attention_mask = tok["attention_mask"][0]
        else:
            pad_id = getattr(self.tok, "pad_token_id", None)
            if pad_id is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            else:
                attention_mask = (input_ids != pad_id).long()

        # Load image
        pixel, img_present = self._load_image(self.paths[i])
        text_present = 1.0 if text.strip() else 0.0
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel,
            "labels": self.labels[i],
            "text_present": torch.tensor(text_present, dtype=torch.float32),
            "image_present": torch.tensor(img_present, dtype=torch.float32),
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader.
    
    Stacks all tensors in the batch along dimension 0.
    """
    keys = batch[0].keys()
    out = {}
    for k in keys:
        if isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            out[k] = torch.tensor([b[k] for b in batch])
    return out
