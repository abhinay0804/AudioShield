"""
Models package for Deepfake Audio Detection.
Includes baseline classifiers (XGBoost), dual-stream CNN-BiLSTM, and Attention networks.
"""

from .baseline import XGBoostBaseline
from .components import (
    MultiHeadSelfAttention,
    TemporalAttentionPooling,
    AMSoftmax,
    init_weights
)
from .dual_stream_model import CNNFrontEnd, DualStreamModel
from .losses import AMSoftmaxLoss, SupConLoss, CombinedLoss

__all__ = [
    "XGBoostBaseline",
    "MultiHeadSelfAttention",
    "TemporalAttentionPooling",
    "AMSoftmax",
    "init_weights",
    "CNNFrontEnd",
    "DualStreamModel",
    "AMSoftmaxLoss",
    "SupConLoss",
    "CombinedLoss"
]
