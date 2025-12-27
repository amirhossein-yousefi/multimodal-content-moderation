"""
Trainer classes and training utilities.
"""

from typing import Optional, List
import torch
from torch.utils.data import WeightedRandomSampler, DataLoader
from transformers import Trainer


class WeightedSamplerTrainer(Trainer):
    """
    Custom Trainer that supports weighted sampling for imbalanced datasets.
    
    This trainer overrides the train dataloader to use WeightedRandomSampler
    instead of the default random sampler.
    
    Args:
        sample_weights: Per-sample weights for WeightedRandomSampler.
        *args: Arguments passed to parent Trainer.
        **kwargs: Keyword arguments passed to parent Trainer.
    """
    
    def __init__(
        self, 
        *args, 
        sample_weights: Optional[torch.Tensor] = None, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.sample_weights = sample_weights

    def get_train_dataloader(self) -> DataLoader:
        """Override to use weighted sampling if weights are provided."""
        if self.sample_weights is None:
            return super().get_train_dataloader()
        
        sampler = WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self.train_dataset),
            replacement=True
        )
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


def group_params_for_optimizer(
    model: torch.nn.Module, 
    lr_encoder: float, 
    lr_head: float, 
    weight_decay: float
) -> List[dict]:
    """
    Group model parameters for differential learning rates.
    
    Separates encoder (backbone) parameters from head parameters to allow
    different learning rates and regularization.
    
    Args:
        model: The model with named parameters.
        lr_encoder: Learning rate for encoder/backbone parameters.
        lr_head: Learning rate for classification head parameters.
        weight_decay: Weight decay for all parameters.
        
    Returns:
        List of parameter groups for optimizer.
    """
    enc_prefixes = ("backbone.", "tower_txt.", "tower_img.", "clip.")
    enc_params, head_params = [], []
    
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith(enc_prefixes):
            enc_params.append(p)
        else:
            head_params.append(p)
    
    return [
        {"params": enc_params, "lr": lr_encoder, "weight_decay": weight_decay},
        {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
    ]


def build_multilabel_sample_weights(
    labels: torch.Tensor, 
    beta: float = 0.999
) -> torch.Tensor:
    """
    Compute sample weights for class-balanced sampling in multi-label setting.
    
    Uses the effective number of samples approach from:
    Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019
    
    Args:
        labels: Multi-label tensor [N, C] with 0/1 values.
        beta: Smoothing parameter (typically 0.9 to 0.9999).
        
    Returns:
        Sample weights tensor [N] for WeightedRandomSampler.
    """
    with torch.no_grad():
        N, C = labels.shape
        pos_counts = labels.sum(dim=0).clamp(min=1.0)
        
        # Effective number of samples
        eff_num = 1.0 - torch.pow(
            torch.tensor(beta, dtype=torch.float32, device=labels.device), 
            pos_counts
        )
        
        # Class-balanced weights (higher for rare classes)
        cls_w = (1.0 - beta) / eff_num
        
        # Per-example weight = sum of its positive-class weights
        w = (labels * cls_w.unsqueeze(0)).sum(dim=1)
        
        # Give baseline weight to all-zero rows
        min_pos = float(w[w > 0].min().item()) if (w > 0).any() else 1.0
        w = torch.where(w > 0, w, torch.full_like(w, min_pos * 0.1))
        
        # Return as double for WeightedRandomSampler
        return w.detach().cpu().to(torch.double)


def train_model(
    model,
    train_dataset,
    val_dataset,
    training_args,
    compute_metrics,
    data_collator=None,
    callbacks=None,
    use_weighted_sampling: bool = False,
):
    """
    High-level training function.
    
    Args:
        model: Model to train.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        training_args: HuggingFace TrainingArguments.
        compute_metrics: Metrics computation function.
        data_collator: Optional data collator.
        callbacks: Optional list of callbacks.
        use_weighted_sampling: Whether to use class-balanced sampling.
        
    Returns:
        Trained model and trainer.
    """
    sample_weights = None
    if use_weighted_sampling and hasattr(train_dataset, 'labels'):
        sample_weights = build_multilabel_sample_weights(train_dataset.labels)
    
    TrainerClass = WeightedSamplerTrainer if sample_weights is not None else Trainer
    
    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=callbacks,
        sample_weights=sample_weights if sample_weights is not None else None,
    )
    
    trainer.train()
    
    return model, trainer
