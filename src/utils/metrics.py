"""Metrics computation utilities."""

import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from typing import Dict


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute evaluation metrics for multi-label classification.

    Args:
        predictions: Predicted probabilities [N, num_classes]
        targets: Ground truth labels [N, num_classes]
        threshold: Classification threshold

    Returns:
        Dictionary of metrics
    """
    # Convert to numpy
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()

    # Binary predictions
    preds_binary = (preds_np >= threshold).astype(int)

    # Compute metrics
    metrics = {
        'f1_micro': f1_score(targets_np, preds_binary, average='micro', zero_division=0),
        'f1_macro': f1_score(targets_np, preds_binary, average='macro', zero_division=0),
        'f1_samples': f1_score(targets_np, preds_binary, average='samples', zero_division=0),
        'precision_micro': precision_score(targets_np, preds_binary, average='micro', zero_division=0),
        'precision_macro': precision_score(targets_np, preds_binary, average='macro', zero_division=0),
        'recall_micro': recall_score(targets_np, preds_binary, average='micro', zero_division=0),
        'recall_macro': recall_score(targets_np, preds_binary, average='macro', zero_division=0),
    }

    # Average precision (mAP)
    try:
        metrics['map_micro'] = average_precision_score(targets_np, preds_np, average='micro')
        metrics['map_macro'] = average_precision_score(targets_np, preds_np, average='macro')
    except ValueError:
        metrics['map_micro'] = 0.0
        metrics['map_macro'] = 0.0

    return metrics
