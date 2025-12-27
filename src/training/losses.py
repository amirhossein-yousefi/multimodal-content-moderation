"""
Loss functions for multi-modal classification.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalWithLogitsLoss(nn.Module):
    """
    Focal Loss with logits for handling class imbalance.
    
    Focal loss down-weights well-classified examples and focuses on hard examples.
    This is particularly useful for imbalanced datasets where positive examples
    are rare.
    
    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    
    Args:
        alpha: Weighting factor for positive class. Can be a scalar or tensor.
        gamma: Focusing parameter (>= 0). Higher gamma increases focus on hard examples.
               gamma=0 is equivalent to BCE loss.
        reduction: Reduction method ('mean', 'sum', 'none').
    
    Example:
        >>> criterion = FocalWithLogitsLoss(gamma=2.0)
        >>> logits = torch.randn(32, 5)
        >>> targets = torch.zeros(32, 5)
        >>> targets[:, 0] = 1  # Sparse positive labels
        >>> loss = criterion(logits, targets)
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
        """
        Compute focal loss.
        
        Args:
            logits: Predicted logits [B, C] or [B].
            targets: Ground truth labels (same shape as logits), values in [0, 1].
            
        Returns:
            Focal loss value (scalar if reduction is 'mean' or 'sum').
        """
        prob = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        
        # p_t = p if y=1, (1-p) if y=0
        p_t = prob * targets + (1 - prob) * (1 - targets)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        loss = ce * focal_weight
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = loss * alpha_weight
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.
    
    This loss is designed for datasets with severe class imbalance where
    negative samples significantly outnumber positive samples.
    
    Reference:
        Ridnik et al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021
    
    Args:
        gamma_neg: Focusing parameter for negative samples.
        gamma_pos: Focusing parameter for positive samples.
        clip: Probability margin for asymmetric clipping.
        reduction: Reduction method ('mean', 'sum', 'none').
    """
    
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute asymmetric loss.
        
        Args:
            logits: Predicted logits [B, C].
            targets: Ground truth labels [B, C], values in {0, 1}.
            
        Returns:
            Asymmetric loss value.
        """
        prob = torch.sigmoid(logits)
        
        # Asymmetric clipping for negative samples
        prob_neg = (prob + self.clip).clamp(max=1)
        
        # Compute loss components
        loss_pos = targets * torch.log(prob.clamp(min=1e-8))
        loss_neg = (1 - targets) * torch.log((1 - prob_neg).clamp(min=1e-8))
        
        # Apply asymmetric focusing
        prob_pos = prob
        prob_neg_focal = prob_neg
        
        pt_pos = prob_pos * targets + (1 - prob_pos) * (1 - targets)
        pt_neg = prob_neg_focal * targets + (1 - prob_neg_focal) * (1 - targets)
        
        focal_pos = (1 - pt_pos) ** self.gamma_pos
        focal_neg = (1 - pt_neg) ** self.gamma_neg
        
        loss = -(focal_pos * loss_pos + focal_neg * loss_neg)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
