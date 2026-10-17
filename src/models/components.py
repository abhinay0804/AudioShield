"""
Core architectural components for Multi-Granularity Audio Deepfake Detection.
Includes MultiHeadSelfAttention with attention-weight extraction, TemporalAttentionPooling,
AMSoftmax (Angular Margin Softmax), and standardized weight initialization.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def init_weights(m: nn.Module) -> None:
    """
    Standardized weight initialization:
    - Conv2d / Conv1d: Kaiming Normal (He initialization)
    - Linear: Xavier Uniform (Glorot initialization)
    - LSTM: Orthogonal for recurrent weights, Xavier for input weights
    - BatchNorm / LayerNorm: weights = 1.0, bias = 0.0
    """
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0.0)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class MultiHeadSelfAttention(nn.Module):
    """
    Custom Multi-Head Self-Attention (MHSA) module.
    Specifically designed to capture temporal relationships across speech frames/segments
    and retain attention weight tensors for Phase 5 explainability.
    
    Formula:
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    """
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        # Stored attention weights for explainability inspection (Phase 5)
        self.attention_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape [Batch, Seq_Len, d_model]
            mask: Optional boolean mask [Batch, Seq_Len] or [Batch, 1, 1, Seq_Len]
            
        Returns:
            out: Contextualized output [Batch, Seq_Len, d_model]
            attn_weights: Attention matrix [Batch, n_heads, Seq_Len, Seq_Len]
        """
        B, T, C = x.shape
        
        # 1. Project Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # [B, n_heads, T, d_k]
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # [B, n_heads, T, d_k]
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # [B, n_heads, T, d_k]
        
        # 2. Scaled dot-product scores: [B, n_heads, T, T]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        
        # Save detached copy for explainability saliency maps
        self.attention_weights = attn_weights.detach()
        
        attn_dropped = self.dropout(attn_weights)
        
        # 3. Apply attention weights to values
        context = torch.matmul(attn_dropped, v)  # [B, n_heads, T, d_k]
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)  # [B, T, d_model]
        
        out = self.out_proj(context)
        return out, attn_weights


class TemporalAttentionPooling(nn.Module):
    """
    Learned attention-based temporal aggregation:
    Condenses a temporal sequence of frame representations [B, T, D]
    into a single summary context vector [B, D].
    
    Formula:
        e_t = w^T tanh(W x_t + b)
        alpha = softmax(e)
        c = sum_t alpha_t x_t
    """
    def __init__(self, d_model: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        self.stored_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Sequence tensor [B, T, D]
            
        Returns:
            context: Aggregated vector [B, D]
            weights: Normalized attention weights [B, T]
        """
        # Calculate raw scores [B, T, 1]
        scores = self.scorer(x).squeeze(-1)  # [B, T]
        weights = F.softmax(scores, dim=-1)  # [B, T]
        
        self.stored_weights = weights.detach()
        
        # Weighted sum across time: [B, 1, T] @ [B, T, D] -> [B, 1, D] -> [B, D]
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return context, weights


class AMSoftmax(nn.Module):
    """
    Additive Angular Margin Softmax (AM-Softmax) Layer.
    Forces representations of genuine speech and spoofed speech to be separated
    by an explicit angular decision margin m, greatly increasing feature discriminability.
    
    Formula:
        cos(theta_{y_i}) = (W_{y_i}^T x_i) / (||W_{y_i}|| ||x_i||)
        L_i = -log( exp(s * (cos(theta_{y_i}) - m)) / (exp(s * (cos(theta_{y_i}) - m)) + sum_{j != y_i} exp(s * cos(theta_j))) )
        
    Args:
        in_features: Dimension of input embedding (e.g., 128 or 256)
        num_classes: Number of classes (2 for bonafide vs. spoof)
        margin: Angular margin m (default: 0.35)
        scale: Feature scale factor s (default: 30.0)
    """
    def __init__(
        self,
        in_features: int = 128,
        num_classes: int = 2,
        margin: float = 0.35,
        scale: float = 30.0
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        
        # Weight vectors for each class: [num_classes, in_features]
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input embeddings [Batch, in_features]
            labels: Ground-truth class indices [Batch] (0 or 1), required during training
            
        Returns:
            logits: Scaled cosine logits [Batch, num_classes]
        """
        # 1. Normalize features and weights
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        
        # 2. Cosine similarity: [B, num_classes]
        cosine = F.linear(x_norm, w_norm)
        
        # If in evaluation mode or no labels provided, return scaled cosines
        if labels is None or not self.training:
            return cosine * self.scale
            
        # 3. Add angular margin to target class logits during training
        # Clamp cosine to numerical stability interval [-1 + eps, 1 - eps]
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # Create one-hot mask for ground-truth classes
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        
        # Subtract margin only from target class: (cos(theta) - m)
        output = self.scale * (cosine - one_hot * self.margin)
        return output
