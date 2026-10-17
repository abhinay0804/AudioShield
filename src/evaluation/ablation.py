"""
Ablation Study Framework for Phase 6.
Enables systematic benchmarking of architectural components:
1. Full Proposed Dual-Stream Model
2. Fine-Stream Only (ablate coarse stream)
3. Coarse-Stream Only (ablate fine stream)
4. No-Attention (Mean pooling ablation)
5. Loss Ablation (AM-Softmax vs AM-Softmax + SupCon)
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.config import config, ModelConfig
from src.models.components import AMSoftmax, init_weights
from src.models.dual_stream_model import CNNFrontEnd, DualStreamModel
from src.evaluation.metrics import compute_eer, compute_min_tdcf


class SingleStreamAblationModel(nn.Module):
    """
    Ablated model running ONLY the fine-grained stream or ONLY the coarse-grained stream.
    Demonstrates the complementary necessity of dual-granularity modeling.
    """
    def __init__(self, mode: str = "fine", model_cfg: ModelConfig = config.model):
        super().__init__()
        assert mode in ("fine", "coarse"), f"Invalid mode {mode}, must be 'fine' or 'coarse'"
        self.mode = mode
        self.cfg = model_cfg
        
        self.cnn = CNNFrontEnd(in_channels=model_cfg.in_channels, out_dim=128)
        
        if mode == "fine":
            self.lstm = nn.LSTM(128, 64, num_layers=2, bidirectional=True, batch_first=True, dropout=0.25)
            self.norm = nn.LayerNorm(128)
            from src.models.components import MultiHeadSelfAttention, TemporalAttentionPooling
            self.mhsa = MultiHeadSelfAttention(d_model=128, n_heads=4, dropout=0.1)
            self.pool = TemporalAttentionPooling(d_model=128, hidden_dim=64)
        else:  # coarse
            self.coarse_window = 12
            self.lstm = nn.LSTM(128, 64, num_layers=1, bidirectional=True, batch_first=True)
            self.norm = nn.LayerNorm(128)
            from src.models.components import MultiHeadSelfAttention, TemporalAttentionPooling
            self.mhsa = MultiHeadSelfAttention(d_model=128, n_heads=2, dropout=0.1)
            self.pool = TemporalAttentionPooling(d_model=128, hidden_dim=64)
            
        self.classifier = AMSoftmax(in_features=128, num_classes=2, margin=0.35, scale=30.0)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 3:
            x = x.unsqueeze(1)
            
        cnn_feats = self.cnn(x)  # [B, 100, 128]
        
        if self.mode == "fine":
            h, _ = self.lstm(cnn_feats)
            h = self.norm(h)
            attn_out, _ = self.mhsa(h)
            context, _ = self.pool(attn_out)
        else:
            h_perm = cnn_feats.transpose(1, 2)
            c_in = F.avg_pool1d(h_perm, kernel_size=self.coarse_window, stride=self.coarse_window).transpose(1, 2)
            h, _ = self.lstm(c_in)
            h = self.norm(h)
            attn_out, _ = self.mhsa(h)
            context, _ = self.pool(attn_out)
            
        logits = self.classifier(context, labels=labels)
        return logits, context


class NoAttentionAblationModel(nn.Module):
    """
    Ablated model replacing Multi-Head Self-Attention and Attention Pooling
    with standard global temporal average pooling.
    Proves that the attention mechanism is what drives detection accuracy.
    """
    def __init__(self, model_cfg: ModelConfig = config.model):
        super().__init__()
        self.cnn = CNNFrontEnd(in_channels=model_cfg.in_channels, out_dim=128)
        self.fine_lstm = nn.LSTM(128, 64, num_layers=2, bidirectional=True, batch_first=True, dropout=0.25)
        self.coarse_lstm = nn.LSTM(128, 64, num_layers=1, bidirectional=True, batch_first=True)
        self.coarse_window = 12
        
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.classifier = AMSoftmax(in_features=128, num_classes=2, margin=0.35, scale=30.0)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 3:
            x = x.unsqueeze(1)
            
        cnn_feats = self.cnn(x)  # [B, 100, 128]
        
        # Fine stream without attention: global average pool over time
        fine_h, _ = self.fine_lstm(cnn_feats)
        fine_context = torch.mean(fine_h, dim=1)  # [B, 128]
        
        # Coarse stream without attention: global average pool over segments
        h_perm = cnn_feats.transpose(1, 2)
        c_in = F.avg_pool1d(h_perm, kernel_size=self.coarse_window, stride=self.coarse_window).transpose(1, 2)
        coarse_h, _ = self.coarse_lstm(c_in)
        coarse_context = torch.mean(coarse_h, dim=1)  # [B, 128]
        
        fused = torch.cat([fine_context, coarse_context], dim=1)
        embedding = self.fusion(fused)
        logits = self.classifier(embedding, labels=labels)
        return logits, embedding
