"""
Unit tests for Phase 6: Final Evaluation Suite & Ablation Models.
Tests EER, min t-DCF, classification metrics, SingleStreamAblationModel,
and NoAttentionAblationModel.
"""

import pytest
import numpy as np
import torch

from src.evaluation.metrics import compute_eer, compute_min_tdcf, compute_classification_metrics
from src.evaluation.ablation import SingleStreamAblationModel, NoAttentionAblationModel


def test_compute_eer():
    """Verify EER handles separated, overlapping, and random distributions."""
    # 1. Perfectly separated distributions -> 0% EER
    bonafide = np.random.uniform(0.8, 1.0, 100)
    spoof = np.random.uniform(0.0, 0.2, 100)
    eer, thresh, _, _, _ = compute_eer(bonafide, spoof)
    assert eer < 0.1
    assert 0.1 <= thresh <= 0.9

    # 2. Fully overlapping distributions -> ~50% EER
    np.random.seed(42)
    shared = np.random.normal(0, 1, 500)
    eer_overlap, _, _, _, _ = compute_eer(shared[:250], shared[250:])
    assert 40.0 <= eer_overlap <= 60.0


def test_compute_min_tdcf():
    """Verify min t-DCF returns valid non-negative float and threshold."""
    bonafide = np.array([0.9, 0.85, 0.8, 0.75, 0.7])
    spoof = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
    min_tdcf, thresh = compute_min_tdcf(bonafide, spoof)
    
    assert isinstance(min_tdcf, float)
    assert min_tdcf >= 0.0
    assert isinstance(thresh, float)


def test_compute_classification_metrics():
    """Verify accuracy, precision, recall, F1, and confusion matrix computation."""
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    threshold = 0.5
    
    metrics = compute_classification_metrics(scores, labels, threshold)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert np.isclose(metrics["specificity"], 1.0)
    assert metrics["confusion_matrix"]["TP"] == 2
    assert metrics["confusion_matrix"]["TN"] == 2
    assert metrics["confusion_matrix"]["FP"] == 0
    assert metrics["confusion_matrix"]["FN"] == 0


def test_single_stream_ablation_models():
    """Verify SingleStreamAblationModel forward pass for fine and coarse modes."""
    B, C, F_dim, T = 2, 1, 60, 401
    x = torch.randn(B, C, F_dim, T)
    
    # 1. Fine stream only
    fine_model = SingleStreamAblationModel(mode="fine")
    fine_model.eval()
    logits_fine, ctx_fine = fine_model(x)
    assert logits_fine.shape == (B, 2)
    assert ctx_fine.shape == (B, 128)
    
    # 2. Coarse stream only
    coarse_model = SingleStreamAblationModel(mode="coarse")
    coarse_model.eval()
    logits_coarse, ctx_coarse = coarse_model(x)
    assert logits_coarse.shape == (B, 2)
    assert ctx_coarse.shape == (B, 128)


def test_no_attention_ablation_model():
    """Verify NoAttentionAblationModel forward pass without attention modules."""
    B, C, F_dim, T = 2, 1, 60, 401
    x = torch.randn(B, C, F_dim, T)
    
    model = NoAttentionAblationModel()
    model.eval()
    logits, emb = model(x)
    assert logits.shape == (B, 2)
    assert emb.shape == (B, 128)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
