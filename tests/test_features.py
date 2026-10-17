"""
Unit tests for Phase 2: Feature Extraction.
Tests LFCC, LogMel, MFCC extractors, TemporalPooler, CMVN, and the factory function.
Run with: PYTHONPATH=. venv/bin/pytest tests/test_features.py -v
"""

import pytest
import torch
import numpy as np

from src.config import config
from src.features.extractor import (
    LFCCExtractor,
    LogMelExtractor,
    MFCCExtractor,
    TemporalPooler,
    get_extractor,
)

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dummy_waveform():
    """Standard 4-second batch of waveforms, shape [B=4, 1, 64000]."""
    torch.manual_seed(42)
    return torch.randn(4, 1, 64000)


@pytest.fixture(scope="module")
def lfcc_extractor():
    ext = LFCCExtractor()
    ext.eval()
    return ext


@pytest.fixture(scope="module")
def logmel_extractor():
    ext = LogMelExtractor()
    ext.eval()
    return ext


@pytest.fixture(scope="module")
def mfcc_extractor():
    ext = MFCCExtractor()
    ext.eval()
    return ext


# --------------------------------------------------------------------------
# LFCC Tests
# --------------------------------------------------------------------------

class TestLFCCExtractor:

    def test_output_shape_standard(self, lfcc_extractor, dummy_waveform):
        """LFCC output must be [B, 60, T] for 4s 16kHz audio."""
        with torch.no_grad():
            out = lfcc_extractor(dummy_waveform)
        assert out.shape[0] == 4, f"Batch dim mismatch: {out.shape}"
        assert out.shape[1] == 60, f"Feature dim must be 60 (20 static + 20Δ + 20ΔΔ): got {out.shape[1]}"
        # T should be around 401 for hop=160, n_fft=512, 64000 samples
        assert 390 <= out.shape[2] <= 420, f"Time dim out of expected range: {out.shape[2]}"

    def test_output_shape_2d_input(self, lfcc_extractor):
        """Should handle [B, T] input (no channel dim)."""
        x = torch.randn(2, 64000)
        with torch.no_grad():
            out = lfcc_extractor(x)
        assert out.shape == torch.Size([2, 60, out.shape[2]])

    def test_output_shape_1d_input(self, lfcc_extractor):
        """Should handle single [T] input (no batch or channel)."""
        x = torch.randn(64000)
        with torch.no_grad():
            out = lfcc_extractor(x)
        assert out.shape[0] == 1
        assert out.shape[1] == 60

    def test_cmvn_zero_mean(self, lfcc_extractor, dummy_waveform):
        """After CMVN, each feature dimension should have ~zero mean over time."""
        with torch.no_grad():
            out = lfcc_extractor(dummy_waveform)
        # Mean over time axis (dim=-1)
        mean = out.mean(dim=-1)  # [B, 60]
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4), \
            f"CMVN mean not ~0. Max |mean|={mean.abs().max():.8f}"

    def test_cmvn_unit_variance(self, lfcc_extractor, dummy_waveform):
        """After CMVN, each feature dimension should have ~unit std over time."""
        with torch.no_grad():
            out = lfcc_extractor(dummy_waveform)
        std = out.std(dim=-1)  # [B, 60]
        assert torch.allclose(std, torch.ones_like(std), atol=1e-4), \
            f"CMVN std not ~1. Max |std - 1|={(std - 1).abs().max():.6f}"

    def test_no_nan_or_inf(self, lfcc_extractor, dummy_waveform):
        """Output must contain no NaN or Inf values."""
        with torch.no_grad():
            out = lfcc_extractor(dummy_waveform)
        assert not torch.isnan(out).any(), "NaN detected in LFCC output"
        assert not torch.isinf(out).any(), "Inf detected in LFCC output"

    def test_output_dim_property(self, lfcc_extractor):
        assert lfcc_extractor.output_dim == 60

    def test_different_samples_different_features(self, lfcc_extractor):
        """Two different waveforms should produce different features."""
        x1 = torch.randn(1, 1, 64000)
        x2 = torch.randn(1, 1, 64000)
        with torch.no_grad():
            f1 = lfcc_extractor(x1)
            f2 = lfcc_extractor(x2)
        assert not torch.allclose(f1, f2), "Different inputs produced identical LFCC features"

    def test_eval_mode_deterministic(self, lfcc_extractor, dummy_waveform):
        """LFCC should be deterministic in eval mode."""
        with torch.no_grad():
            out1 = lfcc_extractor(dummy_waveform)
            out2 = lfcc_extractor(dummy_waveform)
        assert torch.allclose(out1, out2), "LFCC not deterministic in eval mode"


# --------------------------------------------------------------------------
# Log-Mel Tests
# --------------------------------------------------------------------------

class TestLogMelExtractor:

    def test_output_shape(self, logmel_extractor, dummy_waveform):
        """LogMel output must be [B, 80, T]."""
        with torch.no_grad():
            out = logmel_extractor(dummy_waveform)
        assert out.shape[0] == 4
        assert out.shape[1] == 80, f"Expected 80 mel bands, got {out.shape[1]}"
        assert out.shape[2] > 0

    def test_no_nan_or_inf(self, logmel_extractor, dummy_waveform):
        with torch.no_grad():
            out = logmel_extractor(dummy_waveform)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_output_dim_property(self, logmel_extractor):
        assert logmel_extractor.output_dim == 80

    def test_same_time_frames_as_lfcc(self, lfcc_extractor, logmel_extractor, dummy_waveform):
        """LFCC and LogMel should produce same number of time frames (same hop_length)."""
        with torch.no_grad():
            lfcc_out = lfcc_extractor(dummy_waveform)
            mel_out = logmel_extractor(dummy_waveform)
        assert lfcc_out.shape[2] == mel_out.shape[2], \
            f"Time dim mismatch: LFCC={lfcc_out.shape[2]}, LogMel={mel_out.shape[2]}"


# --------------------------------------------------------------------------
# MFCC Tests
# --------------------------------------------------------------------------

class TestMFCCExtractor:

    def test_output_shape(self, mfcc_extractor, dummy_waveform):
        """MFCC output must be [B, 120, T] (40 static + 40Δ + 40ΔΔ)."""
        with torch.no_grad():
            out = mfcc_extractor(dummy_waveform)
        assert out.shape[0] == 4
        assert out.shape[1] == 120, f"Expected 120 dims (40*3), got {out.shape[1]}"

    def test_no_nan_or_inf(self, mfcc_extractor, dummy_waveform):
        with torch.no_grad():
            out = mfcc_extractor(dummy_waveform)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_output_dim_property(self, mfcc_extractor):
        assert mfcc_extractor.output_dim == 120


# --------------------------------------------------------------------------
# TemporalPooler Tests
# --------------------------------------------------------------------------

class TestTemporalPooler:

    def test_avg_pool_output_shape(self):
        """Average pooling should reduce T by window_size."""
        pooler = TemporalPooler(window_size=12, pool_mode="avg")
        x = torch.randn(4, 96, 128)   # T=96, divisible by 12
        out = pooler(x)
        assert out.shape == (4, 8, 128), f"Expected [4, 8, 128], got {list(out.shape)}"

    def test_max_pool_output_shape(self):
        """Max pooling should produce same shape as average pooling."""
        pooler = TemporalPooler(window_size=12, pool_mode="max")
        x = torch.randn(4, 96, 128)
        out = pooler(x)
        assert out.shape == (4, 8, 128)

    def test_handles_non_divisible_T(self):
        """When T is not divisible by window_size, output should floor-divide."""
        pooler = TemporalPooler(window_size=12, pool_mode="avg")
        x = torch.randn(4, 100, 128)  # 100 // 12 = 8 segments, 4 frames discarded
        out = pooler(x)
        assert out.shape[1] == 8

    def test_constant_input_constant_output(self):
        """Pooling a constant tensor should return the same constant."""
        pooler = TemporalPooler(window_size=4, pool_mode="avg")
        x = torch.ones(2, 8, 16) * 3.14
        out = pooler(x)
        assert torch.allclose(out, torch.ones_like(out) * 3.14)

    def test_coarse_window_from_config(self):
        """Validate coarse window (50 frames → from config) produces ~8 segments from T=401."""
        # After CNN, T≈401 → T'≈100 (after 2×2 maxpool twice = /4)
        # Then TemporalPooler(window_size=12) gives ~8 segments
        pooler = TemporalPooler(window_size=12)
        x = torch.randn(4, 100, 128)
        out = pooler(x)
        assert 6 <= out.shape[1] <= 10, f"Expected ~8 coarse segments, got {out.shape[1]}"


# --------------------------------------------------------------------------
# Factory Tests
# --------------------------------------------------------------------------

class TestGetExtractor:

    def test_returns_lfcc(self):
        ext = get_extractor("lfcc")
        assert isinstance(ext, LFCCExtractor)
        assert ext.output_dim == 60

    def test_returns_logmel(self):
        ext = get_extractor("logmel")
        assert isinstance(ext, LogMelExtractor)
        assert ext.output_dim == 80

    def test_returns_mfcc(self):
        ext = get_extractor("mfcc")
        assert isinstance(ext, MFCCExtractor)
        assert ext.output_dim == 120

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Unknown feature_type"):
            get_extractor("wavelet")


# --------------------------------------------------------------------------
# Integration: Full pipeline from waveform to features
# --------------------------------------------------------------------------

class TestEndToEndPipeline:

    def test_lfcc_on_silent_audio(self, lfcc_extractor):
        """Silent (zero) audio should not crash, CMVN may produce zeros or constant."""
        x = torch.zeros(1, 1, 64000)
        with torch.no_grad():
            out = lfcc_extractor(x)
        assert not torch.isnan(out).any(), "NaN on silent audio"
        assert not torch.isinf(out).any(), "Inf on silent audio"

    def test_lfcc_on_pure_tone(self, lfcc_extractor):
        """Pure 440Hz sine wave should produce valid features."""
        t = torch.linspace(0, 4.0, 64000)
        x = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0).unsqueeze(0)  # [1, 1, 64000]
        with torch.no_grad():
            out = lfcc_extractor(x)
        assert out.shape[1] == 60
        assert not torch.isnan(out).any()

    def test_gradient_flows_through_lfcc(self):
        """LFCC features should be differentiable (for end-to-end training compatibility)."""
        extractor = LFCCExtractor()
        x = torch.randn(2, 1, 64000, requires_grad=True)
        out = extractor(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "Gradient did not flow through LFCCExtractor"
        assert not torch.isnan(x.grad).any(), "NaN gradient through LFCCExtractor"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
