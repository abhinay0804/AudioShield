"""
Unit tests for Phase 3: XGBoost Baseline Classifier & EER Evaluation Metric.
Tests feature aggregation (240-dim), model fitting, prediction, EER calculation, and per-attack evaluation.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.models.baseline import XGBoostBaseline, compute_eer


def test_compute_eer_perfect_separation():
    """Test compute_eer with perfectly separable bonafide vs spoof scores."""
    bonafide = np.array([0.9, 0.95, 0.88, 0.92, 0.99])
    spoof = np.array([0.1, 0.05, 0.12, 0.08, 0.02])
    
    eer, threshold, fpr, tpr, thresholds = compute_eer(bonafide, spoof)
    assert eer == 0.0, f"Expected 0.0% EER for perfect separation, got {eer}"
    assert 0.12 <= threshold <= 0.88


def test_compute_eer_overlapping():
    """Test compute_eer with overlapping distributions."""
    np.random.seed(42)
    bonafide = np.random.normal(loc=0.6, scale=0.15, size=100)
    spoof = np.random.normal(loc=0.4, scale=0.15, size=100)
    
    eer, threshold, fpr, tpr, thresholds = compute_eer(bonafide, spoof)
    assert 0.0 < eer < 50.0
    assert 0.0 <= threshold <= 1.0


def test_feature_aggregation_shape():
    """Test extract_utterance_features shape transformation [B, 60, T] -> [B, 240]."""
    baseline = XGBoostBaseline()
    
    # Batch = 4, 60 LFCC coeffs, 401 time frames
    dummy_lfcc = torch.randn(4, 60, 401)
    feats = baseline.extract_utterance_features(dummy_lfcc)
    
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (4, 240), f"Expected shape (4, 240), got {feats.shape}"
    assert not np.isnan(feats).any(), "Features contain NaN values"


def test_baseline_fit_predict_eval(tmp_path):
    """Test complete fit, predict, evaluate, save, and load cycle for XGBoostBaseline."""
    np.random.seed(42)
    
    # 50 training samples (25 bonafide, 25 spoof)
    X_train = np.random.randn(50, 240)
    # Make bonafide samples have slightly higher mean for class 1
    X_train[:25] += 0.5
    y_train = np.array([1]*25 + [0]*25)
    
    baseline = XGBoostBaseline(n_estimators=10, max_depth=3)
    baseline.fit(X_train, y_train)
    
    assert baseline.is_fitted
    
    # Predict
    probs = baseline.predict_proba(X_train)
    assert probs.shape == (50,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    
    # Evaluate with attack types
    attack_types = ["bonafide"]*25 + ["A07"]*15 + ["A08"]*10
    eval_metrics = baseline.evaluate(X_train, y_train, attack_types)
    
    assert "overall_eer" in eval_metrics
    assert "overall_auc" in eval_metrics
    assert "per_attack_eer" in eval_metrics
    assert "A07" in eval_metrics["per_attack_eer"]
    assert "A08" in eval_metrics["per_attack_eer"]
    
    # Save & Load
    baseline.save(tmp_path)
    assert (tmp_path / "xgboost_baseline.json").exists()
    assert (tmp_path / "xgboost_scaler.json").exists()
    
    new_baseline = XGBoostBaseline()
    new_baseline.load(tmp_path)
    new_probs = new_baseline.predict_proba(X_train)
    
    np.testing.assert_allclose(probs, new_probs, rtol=1e-4)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
