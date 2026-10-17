"""
Dual-Stream Multi-Granularity CNN-BiLSTM + Multi-Head Self-Attention Architecture.
Novel Model for Explainable Audio Deepfake Detection on ASVspoof 2019 LA.
Combines fine-grained frame analysis (~10ms) with coarse-grained segment analysis (~500ms)
and stores dual-level temporal attention maps for explainability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, Union

from src.config import config, ModelConfig
from src.features.extractor import LFCCExtractor
from src.models.components import (
    MultiHeadSelfAttention,
    TemporalAttentionPooling,
    AMSoftmax,
    init_weights
)


class CNNFrontEnd(nn.Module):
    """
    2D Spectrogram Convolutional Front-End.
    Extracts spectro-temporal features from LFCC representations.
    
    Architecture:
        Input: [B, 1, Freq=60, Time=401]
        Conv2D(1 -> 32, 3x3) + BN + LeakyReLU(0.2) + MaxPool(2x2) -> [B, 32, 30, 200]
        Conv2D(32 -> 64, 3x3) + BN + LeakyReLU(0.2) + MaxPool(2x2) -> [B, 64, 15, 100]
        Flatten channels & freq -> [B, 100, 64*15=960]
        Linear(960 -> 128) + LayerNorm + LeakyReLU(0.2) -> [B, 100, 128]
    """
    def __init__(self, in_channels: int = 1, out_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # [B, 32, 30, 200]
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # [B, 64, 15, 100]
        )
        self.proj = nn.Sequential(
            nn.Linear(64 * 15, out_dim),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Spectrogram tensor [B, 1, 60, T]
        Returns:
            features: [B, T', 128] where T' = T // 4
        """
        h = self.conv1(x)
        h = self.conv2(h)  # [B, 64, F', T']
        
        B, C, F_dim, T_dim = h.shape
        # Permute to [B, T', C * F']
        h = h.permute(0, 3, 1, 2).contiguous().view(B, T_dim, C * F_dim)
        out = self.proj(h)  # [B, T', 128]
        return out


class DualStreamModel(nn.Module):
    """
    Multi-Granularity Dual-Stream CNN-BiLSTM with Multi-Head Self-Attention.
    
    Features:
    1. Shared 2D CNN front-end preserving frequency-time geometry.
    2. Stream A (Fine-Grained): 2-layer BiLSTM + 4-Head MHSA + Temporal Attention Pooling.
    3. Stream B (Coarse-Grained): Segment Average Pooling (window=12, ~480ms) + 1-layer BiLSTM + 2-Head MHSA.
    4. Multi-Granularity Fusion: 256-dim concatenated context projected into 128-dim embedding.
    5. AM-Softmax Classifier: Enforces strict angular margin separation between genuine and spoofed speech.
    6. Explainability Support: Retains both fine frame attention and coarse segment attention weights.
    """
    def __init__(
        self,
        model_cfg: ModelConfig = config.model,
        has_internal_lfcc: bool = False
    ):
        super().__init__()
        self.cfg = model_cfg
        self.has_internal_lfcc = has_internal_lfcc
        
        if self.has_internal_lfcc:
            self.lfcc_extractor = LFCCExtractor(config.features, config.audio)
            
        # 1. CNN Front-End: maps [B, 1, 60, T] -> [B, T'≈100, 128]
        self.cnn = CNNFrontEnd(in_channels=model_cfg.in_channels, out_dim=128)
        
        # 2. Fine-Grained Stream (Frame-level ~10ms resolution)
        self.fine_lstm = nn.LSTM(
            input_size=128,
            hidden_size=model_cfg.fine_lstm_hidden // 2,  # 64 per direction -> 128 bidirectional
            num_layers=model_cfg.fine_lstm_layers,        # 2 layers
            bidirectional=True,
            batch_first=True,
            dropout=model_cfg.dropout if model_cfg.fine_lstm_layers > 1 else 0.0
        )
        self.fine_norm = nn.LayerNorm(model_cfg.fine_lstm_hidden)
        self.fine_mhsa = MultiHeadSelfAttention(
            d_model=model_cfg.fine_lstm_hidden,
            n_heads=model_cfg.fine_heads,
            dropout=0.1
        )
        self.fine_pool = TemporalAttentionPooling(
            d_model=model_cfg.fine_lstm_hidden,
            hidden_dim=64
        )
        
        # 3. Coarse-Grained Stream (Segment-level ~480ms resolution)
        # Window size 12 on T'=100 frames yields ~8 segments of ~480ms each
        self.coarse_pool_window = 12
        self.coarse_lstm = nn.LSTM(
            input_size=128,
            hidden_size=model_cfg.coarse_lstm_hidden // 2,  # 64 per direction -> 128 bidirectional
            num_layers=model_cfg.coarse_lstm_layers,       # 1 layer
            bidirectional=True,
            batch_first=True
        )
        self.coarse_norm = nn.LayerNorm(model_cfg.coarse_lstm_hidden)
        self.coarse_mhsa = MultiHeadSelfAttention(
            d_model=model_cfg.coarse_lstm_hidden,
            n_heads=model_cfg.coarse_heads,
            dropout=0.1
        )
        self.coarse_pool = TemporalAttentionPooling(
            d_model=model_cfg.coarse_lstm_hidden,
            hidden_dim=64
        )
        
        # 4. Multi-Granularity Fusion MLP (128 fine + 128 coarse = 256)
        fusion_dim = model_cfg.fine_lstm_hidden + model_cfg.coarse_lstm_hidden  # 256
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, model_cfg.embedding_dim // 2),  # 256 -> 128
            nn.LayerNorm(model_cfg.embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(model_cfg.dropout)  # 0.3-0.4 dropout
        )
        
        # 5. Classifier (AM-Softmax with scale s=30 and margin m=0.35)
        self.classifier = AMSoftmax(
            in_features=model_cfg.embedding_dim // 2,  # 128
            num_classes=model_cfg.num_classes,         # 2
            margin=model_cfg.am_softmax_margin,
            scale=model_cfg.am_softmax_scale
        )
        
        # Initialize all weights with standardized distributions
        self.apply(init_weights)

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training and inference.
        
        Args:
            x: Input tensor. Can be:
               - Raw waveform [B, 1, 64000] (if has_internal_lfcc is True)
               - LFCC spectrogram [B, 60, T]
               - 4D spectrogram [B, 1, 60, T]
            labels: Ground truth labels [B] (0 for spoof, 1 for bonafide).
                    Used by AMSoftmax during training.
                    
        Returns:
            logits: Classification logits [B, 2]
            embedding: Normalized feature embedding [B, 128] (for SupCon / t-SNE)
            fine_attn_weights: Fine frame attention scores [B, T_fine]
            coarse_attn_weights: Coarse segment attention scores [B, T_coarse]
        """
        # Ensure 4D shape [B, 1, 60, T]
        if self.has_internal_lfcc and x.ndim == 3 and x.shape[1] == 1:
            x = self.lfcc_extractor(x)
            
        if x.ndim == 3:
            x = x.unsqueeze(1)  # [B, 1, 60, T]
            
        # 1. CNN Feature Representation: [B, 100, 128]
        cnn_feats = self.cnn(x)
        
        # 2. Fine-Grained Stream
        fine_h, _ = self.fine_lstm(cnn_feats)  # [B, 100, 128]
        fine_h = self.fine_norm(fine_h)
        fine_mhsa_out, _ = self.fine_mhsa(fine_h)  # [B, 100, 128]
        fine_context, fine_attn_weights = self.fine_pool(fine_mhsa_out)  # [B, 128], [B, 100]
        
        # 3. Coarse-Grained Stream
        # Temporal Average Pooling: [B, 100, 128] -> [B, 128, 100] -> [B, 128, 8] -> [B, 8, 128]
        h_permuted = cnn_feats.transpose(1, 2)
        coarse_input = F.avg_pool1d(h_permuted, kernel_size=self.coarse_pool_window, stride=self.coarse_pool_window)
        coarse_input = coarse_input.transpose(1, 2)  # [B, ~8, 128]
        
        coarse_h, _ = self.coarse_lstm(coarse_input)  # [B, 8, 128]
        coarse_h = self.coarse_norm(coarse_h)
        coarse_mhsa_out, _ = self.coarse_mhsa(coarse_h)  # [B, 8, 128]
        coarse_context, coarse_attn_weights = self.coarse_pool(coarse_mhsa_out)  # [B, 128], [B, 8]
        
        # 4. Multi-Granularity Fusion
        fused = torch.cat([fine_context, coarse_context], dim=1)  # [B, 256]
        embedding = self.fusion(fused)  # [B, 128]
        
        # 5. AM-Softmax Classification
        logits = self.classifier(embedding, labels=labels)  # [B, 2]
        
        return logits, embedding, fine_attn_weights, coarse_attn_weights

    def get_attention_maps(self) -> Dict[str, torch.Tensor]:
        """
        Returns full multi-head attention matrices stored from the most recent forward pass.
        Useful for Phase 5 head specialization and per-attack signature analysis.
        """
        return {
            "fine_heads_matrix": self.fine_mhsa.attention_weights,     # [B, 4, 100, 100]
            "fine_pooling_weights": self.fine_pool.stored_weights,     # [B, 100]
            "coarse_heads_matrix": self.coarse_mhsa.attention_weights, # [B, 2, 8, 8]
            "coarse_pooling_weights": self.coarse_pool.stored_weights  # [B, 8]
        }
