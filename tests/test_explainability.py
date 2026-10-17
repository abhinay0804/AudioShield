"""
Unit tests for Phase 5: Explainability & Attention Analysis.
Tests AttentionExtractor, temporal saliency interpolation, AttackSignatureAnalyzer,
and HeadSpecializationAnalyzer.
"""

import pytest
import numpy as np
import torch

from src.config import config
from src.models.dual_stream_model import DualStreamModel
from src.explainability.attention_extractor import AttentionExtractor
from src.explainability.signature_analysis import AttackSignatureAnalyzer
from src.explainability.head_specialization import HeadSpecializationAnalyzer


def test_attention_extractor_shapes():
    """Verify AttentionExtractor extracts expected attention tensors and probabilities."""
    model = DualStreamModel(config.model)
    extractor = AttentionExtractor(model=model, cfg=config)
    
    waveform = torch.randn(2, 1, 64000)
    res = extractor.extract_from_waveform(waveform)
    
    assert res["probs"].shape == (2, 2)
    assert res["pred_label"].shape == (2,)
    assert res["fine_frame_weights"].shape == (2, 100)
    assert res["fine_heads_matrix"].shape == (2, 4, 100, 100)
    assert res["coarse_segment_weights"].shape == (2, 8)
    assert res["coarse_heads_matrix"].shape == (2, 2, 8, 8)
    assert res["lfcc"].shape == (2, 60, 401)
    
    # Check probabilities sum to 1
    np.testing.assert_allclose(res["probs"].sum(axis=1), np.ones(2), rtol=1e-4)


def test_temporal_saliency_interpolation():
    """Verify interpolation maps 100-step pooling weights to 401 spectrogram frames."""
    extractor = AttentionExtractor(model=DualStreamModel(config.model), cfg=config)
    fine_weights = np.random.uniform(0, 1, (100,))
    fine_weights /= fine_weights.sum()
    
    saliency = extractor.compute_temporal_saliency(fine_weights, num_frames=401)
    
    assert saliency.shape == (401,)
    assert not np.isnan(saliency).any()
    assert np.isclose(saliency.sum(), 1.0, rtol=1e-3)


def test_attack_signature_analyzer():
    """Verify AttackSignatureAnalyzer aggregates attention profiles and metrics."""
    analyzer = AttackSignatureAnalyzer()
    
    for _ in range(5):
        w_bonafide = np.random.uniform(0.01, 0.05, 100)
        w_bonafide /= w_bonafide.sum()
        analyzer.add_sample("bonafide", w_bonafide)
        
        # Spoof with sharp peak in center
        w_spoof = np.zeros(100)
        w_spoof[45:55] = 1.0
        w_spoof /= w_spoof.sum()
        analyzer.add_sample("A07", w_spoof)
        
    all_sigs = analyzer.compute_all_signatures()
    assert "bonafide" in all_sigs
    assert "A07" in all_sigs
    
    sig_bonafide = all_sigs["bonafide"]
    sig_spoof = all_sigs["A07"]
    
    assert len(sig_bonafide["mean_profile"]) == 100
    assert sig_spoof["peak_top10_mass"] > sig_bonafide["peak_top10_mass"]
    assert sig_spoof["nucleus_mass"] > sig_spoof["onset_mass"]
    assert sig_bonafide["entropy"] > sig_spoof["entropy"]  # Uniform has higher entropy


def test_head_specialization_analyzer():
    """Verify head diversity matrix has 1.0 on diagonal and locality index computes properly."""
    analyzer = HeadSpecializationAnalyzer(num_fine_heads=4, num_coarse_heads=2)
    
    for _ in range(4):
        # 4 heads x 100 x 100
        fine_mat = np.random.uniform(0, 1, (4, 100, 100))
        fine_mat /= fine_mat.sum(axis=-1, keepdims=True)
        
        # 2 heads x 8 x 8
        coarse_mat = np.random.uniform(0, 1, (2, 8, 8))
        coarse_mat /= coarse_mat.sum(axis=-1, keepdims=True)
        
        analyzer.add_sample(fine_mat, coarse_mat)
        
    summary = analyzer.get_summary()
    div_fine = np.array(summary["fine_heads_diversity"])
    assert div_fine.shape == (4, 4)
    # Diagonals should be 1.0
    for i in range(4):
        assert np.isclose(div_fine[i, i], 1.0, atol=1e-3)
        
    loc_fine = summary["fine_heads_locality"]
    assert len(loc_fine) == 4
    for val in loc_fine:
        assert val >= 0.0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
