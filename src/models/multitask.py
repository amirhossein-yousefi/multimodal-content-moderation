"""
Multi-Task Learning Classifier for Hateful Content Detection.

This module implements a multi-task learning classifier with shared representation
and task-specific heads for each hate category.
"""

from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPTextModel, CLIPVisionModel, AutoModel


class MultiTaskClassifier(nn.Module):
    """
    Multi-task classifier with shared projection + fusion, one binary head per task.
    
    This architecture is useful when predicting multiple related binary tasks
    (e.g., racist, sexist, homophobe, religion, otherhate).
    
    Backends:
      - "clip": Uses separate CLIP text & vision towers
      - "auto": Uses AutoModel for SigLIP/CLIP-like models
    
    Args:
        encoder_name: HuggingFace model identifier.
        task_names: List of task names (one head per task).
        fusion_dim: Dimension of fusion layer.
        backend: Encoder backend ('clip' or 'auto').
        threshold: Default prediction threshold.
        freeze_text: Freeze text encoder weights.
        freeze_image: Freeze image encoder weights.
        pos_weight: Positive class weights per task [num_tasks].
        head_hidden_dim: Hidden dimension for task heads (0 = no hidden layer).
        learnable_task_weights: Use learnable uncertainty weights for multi-task loss.
    """
    
    def __init__(
        self,
        encoder_name: str,
        task_names: List[str],
        fusion_dim: int = 512,
        backend: str = "clip",
        threshold: float = 0.5,
        freeze_text: bool = False,
        freeze_image: bool = False,
        pos_weight: Optional[torch.Tensor] = None,
        head_hidden_dim: Optional[int] = None,
        learnable_task_weights: bool = False,
    ):
        super().__init__()
        self.task_names = list(task_names)
        self.num_tasks = len(self.task_names)
        self.threshold = threshold
        self.backend = backend.lower()

        # Load backbone encoder
        if self.backend == "clip":
            self.tower_txt = CLIPTextModel.from_pretrained(encoder_name)
            self.tower_img = CLIPVisionModel.from_pretrained(encoder_name)
            if freeze_text:
                for p in self.tower_txt.parameters():
                    p.requires_grad = False
            if freeze_image:
                for p in self.tower_img.parameters():
                    p.requires_grad = False
            tdim = self.tower_txt.config.hidden_size
            idim = self.tower_img.config.hidden_size
            self.backbone = None
        else:
            # Auto dual-encoder (e.g., SigLIP/SigLIP2)
            self.backbone = AutoModel.from_pretrained(encoder_name)
            if freeze_text and hasattr(self.backbone, "text_model"):
                for p in self.backbone.text_model.parameters():
                    p.requires_grad = False
            if freeze_image and hasattr(self.backbone, "vision_model"):
                for p in self.backbone.vision_model.parameters():
                    p.requires_grad = False
            tdim = getattr(getattr(self.backbone, "text_config", None), "hidden_size", None)
            idim = getattr(getattr(self.backbone, "vision_config", None), "hidden_size", None)
            if tdim is None or idim is None:
                pd = getattr(self.backbone.config, "projection_dim", None)
                tdim = tdim or pd
                idim = idim or pd
            assert tdim is not None and idim is not None, \
                "Could not infer hidden sizes for AutoModel backend."

        # Shared projection + gated fusion
        self.proj_t = nn.Linear(tdim, fusion_dim)
        self.proj_i = nn.Linear(idim, fusion_dim)
        self.g_t = nn.Linear(fusion_dim, fusion_dim)
        self.g_i = nn.Linear(fusion_dim, fusion_dim)
        self.gate = nn.Linear(fusion_dim * 2 + 2, fusion_dim)

        # Shared feature extraction head
        self.shared_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # Task-specific heads
        def make_head():
            if head_hidden_dim and head_hidden_dim > 0:
                return nn.Sequential(
                    nn.Linear(fusion_dim, head_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(head_hidden_dim, 1),
                )
            else:
                return nn.Linear(fusion_dim, 1)
        
        self.heads = nn.ModuleList([make_head() for _ in range(self.num_tasks)])

        # Loss configuration
        if pos_weight is not None:
            assert pos_weight.dim() == 1 and pos_weight.shape[0] == self.num_tasks, \
                "pos_weight must be [num_tasks]"
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

        # Learnable task weights (uncertainty weighting)
        self.log_vars = nn.Parameter(torch.zeros(self.num_tasks)) if learnable_task_weights else None

    def _encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text input to feature vector."""
        if self.backend == "clip":
            out = self.tower_txt(input_ids=input_ids, attention_mask=attention_mask)
            if getattr(out, "pooler_output", None) is not None:
                return out.pooler_output
            return out.last_hidden_state[:, 0]
        else:
            out = self.backbone.text_model(input_ids=input_ids, attention_mask=attention_mask)
            if getattr(out, "pooler_output", None) is not None:
                return out.pooler_output
            return out.last_hidden_state.mean(dim=1)

    def _encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode image input to feature vector."""
        if self.backend == "clip":
            out = self.tower_img(pixel_values=pixel_values)
            if getattr(out, "pooler_output", None) is not None:
                return out.pooler_output
            return out.last_hidden_state.mean(dim=1)
        else:
            out = self.backbone.vision_model(pixel_values=pixel_values)
            if getattr(out, "pooler_output", None) is not None:
                return out.pooler_output
            return out.last_hidden_state.mean(dim=1)

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
            labels: Ground truth labels [B, num_tasks].
            
        Returns:
            Dictionary with 'loss' (if labels provided) and 'logits'.
        """
        # Encode modalities
        tfeat_raw = self._encode_text(input_ids, attention_mask)
        vfeat_raw = self._encode_image(pixel_values)

        # Project to fusion dimension
        tfeat = self.proj_t(tfeat_raw)
        vfeat = self.proj_i(vfeat_raw)

        # Gated fusion
        presence = torch.stack([text_present, image_present], dim=1)
        zt = torch.tanh(self.g_t(tfeat))
        zi = torch.tanh(self.g_i(vfeat))
        g = torch.sigmoid(self.gate(torch.cat([tfeat, vfeat, presence], dim=1)))
        
        # Conditional fusion based on modality presence
        fused = torch.where(
            (image_present < 0.5).unsqueeze(1), zt,
            torch.where((text_present < 0.5).unsqueeze(1), zi, g * zt + (1.0 - g) * zi)
        )

        # Shared head
        shared = self.shared_head(fused)

        # Task-specific predictions
        logits_per_task = []
        for head in self.heads:
            logit = head(shared).squeeze(-1)
            logits_per_task.append(logit)
        logits = torch.stack(logits_per_task, dim=1)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            per_task_losses = []
            for j in range(self.num_tasks):
                pw = self.pos_weight[j] if self.pos_weight is not None else None
                lj = F.binary_cross_entropy_with_logits(
                    logits[:, j], labels[:, j], pos_weight=pw, reduction="mean"
                )
                if self.log_vars is not None:
                    # Uncertainty weighting: exp(-log_var) * loss + 0.5 * log_var
                    per_task_losses.append(
                        torch.exp(-self.log_vars[j]) * lj + 0.5 * self.log_vars[j]
                    )
                else:
                    per_task_losses.append(lj)
            loss = torch.stack(per_task_losses, dim=0).mean()

        return {"loss": loss, "logits": logits}
