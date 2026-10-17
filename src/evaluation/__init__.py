"""
Evaluation and Ablation suite for ASVspoof 2019 LA deepfake detection.
Includes metrics (EER, min t-DCF, confusion matrix) and ablation models.
"""

from .metrics import compute_eer, compute_min_tdcf, compute_classification_metrics
from .ablation import SingleStreamAblationModel, NoAttentionAblationModel

__all__ = [
    "compute_eer",
    "compute_min_tdcf",
    "compute_classification_metrics",
    "SingleStreamAblationModel",
    "NoAttentionAblationModel"
]
