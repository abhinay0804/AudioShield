"""
Unit tests for Phase 4: Core Deep Learning Architecture.
Tests MultiHeadSelfAttention, TemporalAttentionPooling, AMSoftmax, CNNFrontEnd,
DualStreamModel, SupConLoss, CombinedLoss, and backward gradient flow.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import config
from src.models.components import (
    MultiHeadSelfAttention,
    TemporalAttentionPooling,
    AMSoftmax,
    init_weights
)
from src.models.dual_stream_model import CNNFrontEnd, DualStreamModel
from src.models.losses import AMSoftmaxLoss, SupConLoss, CombinedLoss


def test_multi_head_self_attention():
    """Verify MHSA output dimensions and attention matrix storage."""
    B, T, D = 4, 100, 128
    n_heads = 4
    mhsa = MultiHeadSelfAttention(d_model=D, n_heads=n_heads, dropout=0.0)
    
    x = torch.randn(B, T, D)
    out, attn_weights = mhsa(x)
    
    assert out.shape == (B, T, D)
    assert attn_weights.shape == (B, n_heads, T, T)
    assert mhsa.attention_weights is not None
    assert mhsa.attention_weights.shape == (B, n_heads, T, T)
    
    # Verify probabilities sum to 1.0 along key dimension
    row_sums = attn_weights.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums), rtol=1e-4, atol=1e-4)


def test_temporal_attention_pooling():
    """Verify learned temporal pooling produces context vector and normalized weights."""
    B, T, D = 4, 100, 128
    pooler = TemporalAttentionPooling(d_model=D, hidden_dim=64)
    
    x = torch.randn(B, T, D)
    context, weights = pooler(x)
    
    assert context.shape == (B, D)
    assert weights.shape == (B, T)
    assert pooler.stored_weights is not None
    
    # Weights should sum to 1 along sequence dimension
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(B), rtol=1e-4, atol=1e-4)


def test_am_softmax_margin():
    """Verify AMSoftmax margin subtraction during training and scaling during eval."""
    in_features, num_classes = 64, 2
    margin, scale = 0.35, 30.0
    layer = AMSoftmax(in_features=in_features, num_classes=num_classes, margin=margin, scale=scale)
    
    x = torch.randn(4, in_features)
    labels = torch.tensor([0, 1, 0, 1])
    
    # 1. Training mode with labels
    layer.train()
    logits_train = layer(x, labels=labels)
    assert logits_train.shape == (4, num_classes)
    
    # 2. Eval mode without labels
    layer.eval()
    logits_eval = layer(x)
    assert logits_eval.shape == (4, num_classes)
    
    # In training mode, target class cosine should be strictly smaller than eval mode due to -m
    with torch.no_grad():
        diff = (logits_eval - logits_train).gather(1, labels.unsqueeze(1))
        # diff should equal scale * margin = 30 * 0.35 = 10.5
        expected_diff = scale * margin
        torch.testing.assert_close(diff.squeeze(1), torch.full((4,), expected_diff), rtol=1e-3, atol=1e-3)


def test_cnn_front_end():
    """Verify 2D spectrogram CNN transforms [B, 1, 60, 401] -> [B, 100, 128]."""
    B = 2
    cnn = CNNFrontEnd(in_channels=1, out_dim=128)
    
    spectrogram = torch.randn(B, 1, 60, 401)
    out = cnn(spectrogram)
    
    assert out.shape == (B, 100, 128)
    assert not torch.isnan(out).any()


def test_dual_stream_model_forward():
    """Verify end-to-end forward pass of DualStreamModel with fine and coarse attention."""
    B = 4
    model = DualStreamModel()
    model.eval()
    
    # Input spectrogram [B, 1, 60, 401]
    x = torch.randn(B, 1, 60, 401)
    
    logits, embedding, fine_attn, coarse_attn = model(x)
    
    assert logits.shape == (B, 2)
    assert embedding.shape == (B, 128)
    assert fine_attn.shape == (B, 100)
    assert coarse_attn.shape == (B, 8)
    
    # Check stored attention maps
    maps = model.get_attention_maps()
    assert maps["fine_heads_matrix"].shape == (B, 4, 100, 100)
    assert maps["coarse_heads_matrix"].shape == (B, 2, 8, 8)
    assert maps["fine_pooling_weights"].shape == (B, 100)
    assert maps["coarse_pooling_weights"].shape == (B, 8)


def test_loss_functions():
    """Verify SupConLoss and CombinedLoss computation."""
    B, D = 8, 128
    embeddings = torch.randn(B, D)
    labels = torch.tensor([0, 1, 0, 1, 0, 0, 1, 1])
    logits = torch.randn(B, 2)
    
    # SupConLoss
    supcon = SupConLoss(temperature=0.1)
    sc_loss = supcon(embeddings, labels)
    assert sc_loss.ndim == 0
    assert sc_loss.item() > 0
    
    # CombinedLoss
    combined = CombinedLoss(supcon_lambda=0.1)
    tot_loss, metrics = combined(logits, embeddings, labels)
    assert tot_loss.ndim == 0
    assert "loss_total" in metrics
    assert "loss_am" in metrics
    assert "loss_supcon" in metrics
    assert metrics["loss_total"] > 0


def test_gradient_flow_and_backward():
    """Verify full gradient propagation through all layers in DualStreamModel."""
    model = DualStreamModel()
    model.train()
    
    loss_fn = CombinedLoss(supcon_lambda=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    x = torch.randn(4, 1, 60, 401, requires_grad=False)
    labels = torch.tensor([0, 1, 0, 1])
    
    optimizer.zero_grad()
    logits, embedding, _, _ = model(x, labels=labels)
    loss, _ = loss_fn(logits, embedding, labels)
    loss.backward()
    
    # Verify gradients exist and are non-zero across components
    assert model.cnn.conv1[0].weight.grad is not None
    assert (model.cnn.conv1[0].weight.grad.abs().sum() > 0).item()
    
    assert model.fine_lstm.weight_ih_l0.grad is not None
    assert (model.fine_lstm.weight_ih_l0.grad.abs().sum() > 0).item()
    
    assert model.coarse_lstm.weight_ih_l0.grad is not None
    assert (model.coarse_lstm.weight_ih_l0.grad.abs().sum() > 0).item()
    
    assert model.classifier.weight.grad is not None
    assert (model.classifier.weight.grad.abs().sum() > 0).item()
    
    optimizer.step()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
