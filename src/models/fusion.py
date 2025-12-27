"""
Multi-Modal Fusion Classifier with Gated Fusion Mechanism.

This module implements a late-fusion classifier that combines text and image
features from pre-trained vision-language models (CLIP, SigLIP).
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPModel, AutoModel


class FocalWithLogitsLoss(nn.Module):
    """
    Focal Loss with logits for handling class imbalance.
    
    Focal loss down-weights well-classified examples and focuses on hard examples.
    
    Args:
        alpha: Weighting factor for positive class.
        gamma: Focusing parameter. Higher gamma increases focus on hard examples.
        reduction: Reduction method ('mean', 'sum', 'none').
    """
    
    def __init__(
        self, 
        alpha: Optional[torch.Tensor] = None, 
        gamma: float = 1.5, 
        reduction: str = "mean"
    ):
        super().__init__()
        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce * ((1 - p_t) ** self.gamma)
        
        if self.alpha is not None:
            loss = loss * (self.alpha * targets + (1 - self.alpha) * (1 - targets))
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MultiModalFusionClassifier(nn.Module):
    """
    Multi-modal classifier with gated fusion and interaction features.
    
    Supports two backends:
      - "clip": Uses CLIPModel.from_pretrained()
      - "siglip": Uses AutoModel.from_pretrained() for SigLIP/SigLIP2 checkpoints
    
    Architecture:
        1. Encode text and image using pre-trained encoder
        2. Project embeddings to fusion dimension
        3. Apply gated fusion mechanism with modality presence flags
        4. Combine interaction features (difference, product)
        5. Classify using MLP head
    
    Args:
        encoder_name: HuggingFace model identifier.
        num_labels: Number of output labels (classes).
        fusion_dim: Dimension of fusion layer.
        backend: Encoder backend ('clip' or 'siglip').
        freeze_text: Freeze text encoder weights.
        freeze_image: Freeze image encoder weights.
        loss_type: Loss function type ('bce' or 'focal').
        focal_gamma: Gamma parameter for focal loss.
        pos_weight: Positive class weights for BCE loss.
        alpha_focal: Alpha parameter for focal loss.
    """

    def __init__(
        self,
        encoder_name: str,
        num_labels: int,
        fusion_dim: int = 512,
        backend: str = "clip",
        freeze_text: bool = False,
        freeze_image: bool = False,
        loss_type: str = "bce",
        focal_gamma: float = 1.5,
        pos_weight: Optional[torch.Tensor] = None,
        alpha_focal: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.backend = backend.lower()
        
        # Load backbone encoder
        if self.backend == "clip":
            self.backbone = CLIPModel.from_pretrained(encoder_name)
            d = self.backbone.config.projection_dim
            if freeze_text:
                for p in self.backbone.text_model.parameters():
                    p.requires_grad = False
            if freeze_image:
                for p in self.backbone.vision_model.parameters():
                    p.requires_grad = False
        else:
            # SigLIP/SigLIP2 or any CLIP-like dual encoder via AutoModel
            self.backbone = AutoModel.from_pretrained(encoder_name)
            cfg = self.backbone.config
            d = getattr(cfg, "projection_dim", None)
            if d is None and hasattr(cfg, "text_config"):
                d = getattr(cfg.text_config, "projection_size", None) or \
                    getattr(cfg.text_config, "hidden_size", None)
            if d is None:
                d = getattr(getattr(cfg, "vision_config", None), "hidden_size", None) or \
                    getattr(getattr(cfg, "text_config", None), "hidden_size", None)
            assert d is not None, "Could not infer projection dim for the chosen encoder."
            
            if freeze_text and hasattr(self.backbone, "text_model"):
                for p in self.backbone.text_model.parameters():
                    p.requires_grad = False
            if freeze_image and hasattr(self.backbone, "vision_model"):
                for p in self.backbone.vision_model.parameters():
                    p.requires_grad = False

        # Projection layers
        self.proj_t = nn.Linear(d, fusion_dim)
        self.proj_i = nn.Linear(d, fusion_dim)

        # Gated fusion layers
        self.g_t = nn.Linear(fusion_dim, fusion_dim)
        self.g_i = nn.Linear(fusion_dim, fusion_dim)
        self.gate = nn.Linear(fusion_dim * 2 + 2, fusion_dim)  # +2 for presence flags

        # Interaction-enhanced classifier: [fused, t, v, |t-v|, t*v]
        cls_in = fusion_dim * 5
        self.cls = nn.Sequential(
            nn.LayerNorm(cls_in),
            nn.Linear(cls_in, fusion_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, num_labels),
        )
        self.ln_fused = nn.LayerNorm(fusion_dim)

        # Loss configuration
        self.loss_type = loss_type
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)
        if loss_type == "focal":
            self.criterion = FocalWithLogitsLoss(alpha=alpha_focal, gamma=focal_gamma)
        else:
            self.criterion = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        text_present: torch.Tensor,
        image_present: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Forward pass.
        
        Args:
            input_ids: Tokenized text input [B, seq_len].
            attention_mask: Attention mask for text [B, seq_len].
            pixel_values: Preprocessed image tensor [B, C, H, W].
            text_present: Binary flag indicating text presence [B].
            image_present: Binary flag indicating image presence [B].
            labels: Ground truth labels [B, num_labels].
            
        Returns:
            Dictionary with 'loss' (if labels provided) and 'logits'.
        """
        # Extract features
        t_kwargs = {"input_ids": input_ids}
        if attention_mask is not None:
            t_kwargs["attention_mask"] = attention_mask
        tfeat = self.backbone.get_text_features(**t_kwargs)
        vfeat = self.backbone.get_image_features(pixel_values=pixel_values)

        # Normalize and mask by presence
        tfeat = F.normalize(tfeat, dim=-1) * text_present.unsqueeze(1)
        vfeat = F.normalize(vfeat, dim=-1) * image_present.unsqueeze(1)

        # Project to fusion dimension
        tfeat_p = self.proj_t(tfeat)
        vfeat_p = self.proj_i(vfeat)

        # Gated fusion
        zt = torch.tanh(self.g_t(tfeat_p))
        zi = torch.tanh(self.g_i(vfeat_p))
        presence = torch.stack([text_present, image_present], dim=1)
        g = torch.sigmoid(self.gate(torch.cat([tfeat_p, vfeat_p, presence], dim=1)))

        # Conditional fusion based on modality presence
        fused = torch.where(
            (image_present < 0.5).unsqueeze(1), zt,
            torch.where((text_present < 0.5).unsqueeze(1), zi, g * zt + (1.0 - g) * zi)
        )
        fused = self.ln_fused(fused)

        # Interaction features
        feat = torch.cat([
            fused, 
            tfeat_p, 
            vfeat_p, 
            torch.abs(tfeat_p - vfeat_p), 
            tfeat_p * vfeat_p
        ], dim=1)
        logits = self.cls(feat)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            if self.loss_type == "focal":
                loss = self.criterion(logits, labels)
            else:
                loss = F.binary_cross_entropy_with_logits(
                    logits, labels, 
                    pos_weight=self.pos_weight if self.pos_weight is not None else None
                )
        
        return {"loss": loss, "logits": logits}
