"""
Evaluation metrics for multi-modal classification.
"""

from typing import Callable, List, Tuple
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score


def make_compute_metrics_multi(num_labels: int, threshold: float = 0.5) -> Callable:
    """
    Create a compute_metrics function for multi-label classification.
    
    Args:
        num_labels: Number of labels/classes.
        threshold: Decision threshold for converting probabilities to predictions.
        
    Returns:
        Callable that computes metrics from (logits, labels) tuple.
    """
    def compute_metrics(eval_pred):
        if isinstance(eval_pred, (tuple, list)):
            logits, labels = eval_pred
        else:
            logits, labels = eval_pred.predictions, eval_pred.label_ids

        logits = np.array(logits)
        labels = np.array(labels)
        probs = 1 / (1 + np.exp(-logits))  # Sigmoid

        # F1 at threshold
        try:
            bin_preds = (probs >= threshold).astype(int)
            f1_macro = float(f1_score(labels, bin_preds, average="macro", zero_division=0))
            f1_micro = float(f1_score(labels, bin_preds, average="micro", zero_division=0))
        except Exception:
            f1_macro = 0.0
            f1_micro = 0.0

        # ROC-AUC macro (threshold-free, primary selection metric)
        try:
            if num_labels == 1:
                roc_macro = float(roc_auc_score(labels, probs))
            else:
                roc_macro = float(roc_auc_score(labels, probs, average="macro"))
        except Exception:
            roc_macro = 0.0

        return {
            "f1_macro": f1_macro, 
            "f1_micro": f1_micro, 
            "roc_macro": roc_macro
        }
    
    return compute_metrics


def make_compute_metrics_mtl(task_names: List[str], threshold: float = 0.5) -> Callable:
    """
    Create a compute_metrics function for multi-task learning.
    
    Returns per-task metrics in addition to aggregated metrics.
    
    Args:
        task_names: List of task names.
        threshold: Decision threshold for converting probabilities to predictions.
        
    Returns:
        Callable that computes metrics from (logits, labels) tuple.
    """
    def compute_metrics(eval_pred):
        if isinstance(eval_pred, tuple):
            logits, labels = eval_pred
        else:
            logits, labels = eval_pred.predictions, eval_pred.label_ids

        probs = 1.0 / (1.0 + np.exp(-logits))
        bin_preds = (probs >= threshold).astype(int)

        # Aggregate metrics
        try:
            f1_macro = f1_score(labels, bin_preds, average="macro", zero_division=0)
            f1_micro = f1_score(labels, bin_preds, average="micro", zero_division=0)
        except Exception:
            f1_macro, f1_micro = 0.0, 0.0

        try:
            roc_macro = roc_auc_score(labels, probs, average="macro")
        except Exception:
            roc_macro = 0.0

        out = {
            "f1_macro": float(f1_macro), 
            "f1_micro": float(f1_micro), 
            "roc_macro": float(roc_macro)
        }
        
        # Per-task metrics
        for j, name in enumerate(task_names):
            try:
                f1j = f1_score(labels[:, j], bin_preds[:, j], average="binary", zero_division=0)
            except Exception:
                f1j = 0.0
            try:
                rocj = roc_auc_score(labels[:, j], probs[:, j])
            except Exception:
                rocj = 0.0
            out[f"f1_{name}"] = float(f1j)
            out[f"roc_{name}"] = float(rocj)
        
        return out
    
    return compute_metrics


def calibrate_thresholds(
    probs: np.ndarray, 
    y_true: np.ndarray,
    t_start: float = 0.05, 
    t_end: float = 0.95, 
    steps: int = 19
) -> List[float]:
    """
    Find optimal per-class thresholds that maximize F1 score.
    
    Performs a grid search over threshold values for each class independently.
    
    Args:
        probs: Predicted probabilities [N, C].
        y_true: Ground truth labels [N, C].
        t_start: Start of threshold grid.
        t_end: End of threshold grid.
        steps: Number of threshold values to try.
        
    Returns:
        List of optimal thresholds per class.
    """
    grid = np.linspace(t_start, t_end, steps)
    C = probs.shape[1]
    best_thresholds = []
    
    for j in range(C):
        yj = y_true[:, j]
        
        # Skip if no positive samples
        if yj.sum() == 0:
            best_thresholds.append(0.5)
            continue
            
        best_t = 0.5
        best_f1 = -1.0
        pj = probs[:, j]
        
        for t in grid:
            f1 = f1_score(yj, (pj >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
                
        best_thresholds.append(float(best_t))
    
    return best_thresholds


def compute_detailed_metrics(
    probs: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.5,
    class_names: List[str] = None
) -> dict:
    """
    Compute detailed classification metrics.
    
    Args:
        probs: Predicted probabilities [N, C].
        y_true: Ground truth labels [N, C].
        threshold: Decision threshold.
        class_names: Optional list of class names.
        
    Returns:
        Dictionary with comprehensive metrics.
    """
    bin_preds = (probs >= threshold).astype(int)
    n_classes = probs.shape[1]
    
    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]
    
    metrics = {
        "f1_macro": float(f1_score(y_true, bin_preds, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, bin_preds, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, bin_preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, bin_preds, average="macro", zero_division=0)),
    }
    
    try:
        metrics["roc_auc_macro"] = float(roc_auc_score(y_true, probs, average="macro"))
    except ValueError:
        metrics["roc_auc_macro"] = 0.0
    
    # Per-class metrics
    metrics["per_class"] = {}
    for j, name in enumerate(class_names):
        class_metrics = {
            "f1": float(f1_score(y_true[:, j], bin_preds[:, j], zero_division=0)),
            "precision": float(precision_score(y_true[:, j], bin_preds[:, j], zero_division=0)),
            "recall": float(recall_score(y_true[:, j], bin_preds[:, j], zero_division=0)),
            "support": int(y_true[:, j].sum()),
        }
        try:
            class_metrics["roc_auc"] = float(roc_auc_score(y_true[:, j], probs[:, j]))
        except ValueError:
            class_metrics["roc_auc"] = 0.0
        metrics["per_class"][name] = class_metrics
    
    return metrics
