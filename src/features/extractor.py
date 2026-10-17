"""
Feature Extraction Module for Audio Deepfake Detection.

Implements three extractors:
  - LFCCExtractor  (PRIMARY): Linear Frequency Cepstral Coefficients
                               20 static + 20 Delta + 20 DeltaDelta = 60 dims
                               Gold standard for anti-spoofing (1.8–3.5% EER vs 6–12% for Mel-based)
  - LogMelExtractor (ABLATION): 80 mel-filterbank log energies
  - MFCCExtractor   (BASELINE): 40 MFCCs + Delta + DeltaDelta = 120 dims

All extractors produce shape: [batch, n_coeffs, T_frames] (frequency-first / spectrogram-convention)
TemporalPooler collapses time dimension to fixed coarse segments (for coarse-grained stream).

Design rationale:
- LFCC uses a LINEAR frequency scale (equal resolution 0–8 kHz) which preserves high-frequency
  phase artifacts introduced by vocoders and neural TTS systems. The Mel scale compresses 4–8 kHz,
  exactly where spoofing artifacts concentrate, losing discriminative information.
- N_FFT=512 (32ms window), HOP_LENGTH=160 (10ms step), WIN_LENGTH=320 (20ms) follows
  ASVspoof 2019 standard. This yields T≈401 frames for 4-second input at 16kHz.
- Delta and Delta-Delta computed via ETSI EN 301 912 standard 9-point regression coefficients,
  capturing temporal dynamics (rate of change of spectral features across frames).
- CMVN (Cepstral Mean-Variance Normalization) applied per utterance — prevents recording-condition
  mismatch and removes channel effects. Critical for cross-condition generalization.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F_audio
from typing import Optional, Literal

from src.config import config, FeatureConfig, AudioConfig


# ---------------------------------------------------------------------------
# LFCC Extractor — Primary Feature (DO NOT CHANGE without updating novelty claim)
# ---------------------------------------------------------------------------

class LFCCExtractor(nn.Module):
    """
    Linear Frequency Cepstral Coefficient extractor.

    Pipeline:
        waveform [B, 1, 64000]
        → torchaudio LFCC (linear filterbank DCT-based cepstrum)
        → static LFCCs [B, n_lfcc, T]
        → ComputeDeltas for Δ and ΔΔ coefficients
        → concatenate [B, 3*n_lfcc, T]   (= [B, 60, T] by default)
        → CMVN (utterance-level mean-variance normalization)
        → output [B, 60, T]

    Args:
        feat_cfg: FeatureConfig dataclass instance
        audio_cfg: AudioConfig dataclass instance
        apply_cmvn: Whether to apply Cepstral Mean-Variance Normalization (recommended: True)
    """

    def __init__(
        self,
        feat_cfg: FeatureConfig = config.features,
        audio_cfg: AudioConfig = config.audio,
        apply_cmvn: bool = True,
    ) -> None:
        super().__init__()
        self.feat_cfg = feat_cfg
        self.apply_cmvn = apply_cmvn
        self.n_static = feat_cfg.n_lfcc           # 20
        self.out_dim = feat_cfg.n_lfcc * 3        # 60 (static + Δ + ΔΔ)

        # torchaudio LFCC uses DCT on top of a LINEAR filterbank
        # speckwargs are passed to torchaudio.transforms.Spectrogram internally
        self._lfcc = torchaudio.transforms.LFCC(
            sample_rate=audio_cfg.sample_rate,     # 16000 Hz
            n_filter=feat_cfg.n_filter,            # 60 linear filters
            f_min=feat_cfg.f_min,                  # 0 Hz
            f_max=feat_cfg.f_max,                  # 8000 Hz (Nyquist)
            n_lfcc=feat_cfg.n_lfcc,                # 20 cepstral coefficients
            dct_type=2,                            # Standard DCT-II (SciPy/HTK convention)
            norm="ortho",                          # Orthonormal DCT
            log_lf=True,                           # Apply log to filterbank energies before DCT
            speckwargs={
                "n_fft": feat_cfg.n_fft,           # 512 → 32ms FFT window at 16kHz
                "win_length": feat_cfg.win_length, # 400 samples → 25ms
                "hop_length": feat_cfg.hop_length, # 160 samples → 10ms shift
                "window_fn": torch.hamming_window, # Standard ASR window
            },
        )

        # ComputeDeltas: 9-point ETSI regression coefficients (width=9 is the default)
        self._delta = torchaudio.transforms.ComputeDeltas(win_length=9, mode="replicate")

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: Audio tensor. Accepted shapes:
                - [B, 1, T_samples]  (standard dataset output)
                - [B, T_samples]     (2-D, channel squeezed)
                - [1, T_samples]     (single sample with channel)
                - [T_samples]        (raw 1-D audio)

        Returns:
            lfcc_features: [B, 60, T_frames]
                           T_frames ≈ 401 for 64000-sample input with hop=160, n_fft=512
        """
        waveform = self._ensure_batch_channel(waveform)

        # [B, 1, n_lfcc, T] — torchaudio keeps the channel dim
        static = self._lfcc(waveform).squeeze(1)   # → [B, n_lfcc=20, T]

        # Delta and Delta-Delta
        delta = self._delta(static)                # [B, 20, T]
        delta2 = self._delta(delta)                # [B, 20, T]

        # Concatenate along feature dim: [B, 60, T]
        features = torch.cat([static, delta, delta2], dim=1)

        # Utterance-level CMVN: normalize each utterance independently
        if self.apply_cmvn:
            features = self._cmvn(features)

        return features

    @staticmethod
    def _cmvn(features: torch.Tensor) -> torch.Tensor:
        """
        Utterance-level Cepstral Mean-Variance Normalization.
        Applied over time axis (dim=-1): zero mean, unit variance per feature dimension.
        
        Args:
            features: [B, n_coeffs, T]
        Returns:
            normalized: [B, n_coeffs, T]
        """
        # Compute stats over time axis (dim=-1)
        mean = features.mean(dim=-1, keepdim=True)    # [B, n_coeffs, 1]
        std = features.std(dim=-1, keepdim=True) + 1e-8
        return (features - mean) / std

    @staticmethod
    def _ensure_batch_channel(waveform: torch.Tensor) -> torch.Tensor:
        """Normalize input to [B, 1, T] shape."""
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)  # [T] → [1, 1, T]
        elif waveform.ndim == 2:
            # Could be [B, T] or [1, T]
            waveform = waveform.unsqueeze(1)               # → [B, 1, T]
        # waveform.ndim == 3: already [B, 1, T] or [B, C, T]
        return waveform

    @property
    def output_dim(self) -> int:
        """Number of output feature dimensions (60 = 20 static + 20 Δ + 20 ΔΔ)."""
        return self.out_dim


# ---------------------------------------------------------------------------
# Log-Mel Extractor — Ablation Study
# ---------------------------------------------------------------------------

class LogMelExtractor(nn.Module):
    """
    Log Mel Spectrogram extractor for ablation comparison against LFCC.

    Pipeline:
        waveform [B, 1, T_samples]
        → MelSpectrogram (80 bands, linear→mel→log)
        → AmplitudeToDB (convert power to dB)
        → CMVN
        → output [B, 80, T_frames]

    This uses Mel-scale which compresses 4–8 kHz compared to LFCC's linear scale,
    demonstrating performance degradation in ablation studies.
    """

    def __init__(
        self,
        feat_cfg: FeatureConfig = config.features,
        audio_cfg: AudioConfig = config.audio,
        apply_cmvn: bool = True,
    ) -> None:
        super().__init__()
        self.apply_cmvn = apply_cmvn
        self.out_dim = feat_cfg.n_mels  # 80

        self._mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=audio_cfg.sample_rate,
            n_fft=feat_cfg.n_fft,
            win_length=feat_cfg.win_length,
            hop_length=feat_cfg.hop_length,
            f_min=feat_cfg.f_min,
            f_max=feat_cfg.f_max,
            n_mels=feat_cfg.n_mels,
            window_fn=torch.hamming_window,
            power=2.0,  # Power spectrogram (energy)
        )
        self._to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [B, 1, T_samples] or compatible
        Returns:
            log_mel: [B, n_mels=80, T_frames]
        """
        waveform = LFCCExtractor._ensure_batch_channel(waveform)
        features = self._mel(waveform).squeeze(1)   # [B, n_mels, T]
        features = self._to_db(features)            # dB scale
        if self.apply_cmvn:
            features = LFCCExtractor._cmvn(features)
        return features

    @property
    def output_dim(self) -> int:
        return self.out_dim


# ---------------------------------------------------------------------------
# MFCC Extractor — Traditional Baseline
# ---------------------------------------------------------------------------

class MFCCExtractor(nn.Module):
    """
    MFCC extractor for traditional baseline comparison.

    Pipeline:
        waveform [B, 1, T_samples]
        → torchaudio MFCC (mel filterbank + log + DCT)
        → ComputeDeltas for Δ and ΔΔ
        → concatenate: [B, 3*n_mfcc, T]   (= [B, 120, T] by default)
        → CMVN
        → output [B, 120, T_frames]

    MFCC compresses high frequencies (Mel scale) AND applies DCT compression,
    making it the worst performer on ASVspoof 2019 LA (expected EER: 8.5–12%).
    """

    def __init__(
        self,
        feat_cfg: FeatureConfig = config.features,
        audio_cfg: AudioConfig = config.audio,
        apply_cmvn: bool = True,
    ) -> None:
        super().__init__()
        self.apply_cmvn = apply_cmvn
        self.n_static = feat_cfg.n_mfcc    # 40
        self.out_dim = feat_cfg.n_mfcc * 3  # 120

        self._mfcc = torchaudio.transforms.MFCC(
            sample_rate=audio_cfg.sample_rate,
            n_mfcc=feat_cfg.n_mfcc,
            melkwargs={
                "n_fft": 1024,             # Larger FFT → 513 bins → no empty mel filters with n_mels=128
                "win_length": feat_cfg.win_length,
                "hop_length": feat_cfg.hop_length,
                "f_min": feat_cfg.f_min,
                "f_max": feat_cfg.f_max,
                "n_mels": 128,  # Standard MFCC uses more mel bins before DCT
                "window_fn": torch.hamming_window,
            },
        )
        self._delta = torchaudio.transforms.ComputeDeltas(win_length=9, mode="replicate")

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [B, 1, T_samples] or compatible
        Returns:
            mfcc_features: [B, 120, T_frames]
        """
        waveform = LFCCExtractor._ensure_batch_channel(waveform)
        static = self._mfcc(waveform).squeeze(1)   # [B, n_mfcc=40, T]
        delta = self._delta(static)                # [B, 40, T]
        delta2 = self._delta(delta)                # [B, 40, T]
        features = torch.cat([static, delta, delta2], dim=1)  # [B, 120, T]
        if self.apply_cmvn:
            features = LFCCExtractor._cmvn(features)
        return features

    @property
    def output_dim(self) -> int:
        return self.out_dim


# ---------------------------------------------------------------------------
# Temporal Pooler — for Coarse-Grained Stream
# ---------------------------------------------------------------------------

class TemporalPooler(nn.Module):
    """
    Reduces temporal resolution via average pooling over fixed-size windows.

    Used in the coarse-grained stream to aggregate fine-grained CNN features
    into segment-level (~500ms) representations before the coarse BiLSTM.

    Input:  [B, T_fine, d_model]    (e.g., [B, ~100, 128] post-CNN)
    Output: [B, T_coarse, d_model]  (e.g., [B, ~8, 128])

    Args:
        window_size: Number of fine frames to pool over (default=12 → ~480ms at 40ms/frame)
        pool_mode: 'avg' (default) or 'max'
    """

    def __init__(self, window_size: int = 12, pool_mode: Literal["avg", "max"] = "avg") -> None:
        super().__init__()
        self.window_size = window_size
        self.pool_mode = pool_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model] — time-major sequence tensor
        Returns:
            pooled: [B, T_coarse, d_model]
        """
        B, T, D = x.shape

        # Trim T so it's divisible by window_size
        T_trim = (T // self.window_size) * self.window_size
        x_trim = x[:, :T_trim, :]  # [B, T_trim, D]

        # Reshape into segments: [B, n_segments, window_size, D]
        n_segments = T_trim // self.window_size
        x_seg = x_trim.reshape(B, n_segments, self.window_size, D)

        if self.pool_mode == "avg":
            pooled = x_seg.mean(dim=2)   # [B, n_segments, D]
        else:
            pooled = x_seg.max(dim=2).values

        return pooled


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------

def get_extractor(
    feature_type: Literal["lfcc", "logmel", "mfcc"] = "lfcc",
    feat_cfg: FeatureConfig = config.features,
    audio_cfg: AudioConfig = config.audio,
    apply_cmvn: bool = True,
) -> nn.Module:
    """
    Factory: returns the appropriate feature extractor by name.

    Args:
        feature_type: One of 'lfcc' (default/primary), 'logmel' (ablation), 'mfcc' (baseline)
        feat_cfg: FeatureConfig instance
        audio_cfg: AudioConfig instance
        apply_cmvn: Whether to normalize features (recommended: True)

    Returns:
        nn.Module feature extractor with .output_dim property
    """
    if feature_type == "lfcc":
        return LFCCExtractor(feat_cfg, audio_cfg, apply_cmvn)
    elif feature_type == "logmel":
        return LogMelExtractor(feat_cfg, audio_cfg, apply_cmvn)
    elif feature_type == "mfcc":
        return MFCCExtractor(feat_cfg, audio_cfg, apply_cmvn)
    else:
        raise ValueError(f"Unknown feature_type '{feature_type}'. Choose from: 'lfcc', 'logmel', 'mfcc'")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    print("=" * 60)
    print("Feature Extractor Shape Verification")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dummy_waveform = torch.randn(4, 1, 64000).to(device)  # [B=4, C=1, T=64000]

    # --- LFCC ---
    lfcc_extractor = LFCCExtractor().to(device)
    lfcc_extractor.eval()
    with torch.no_grad():
        lfcc_out = lfcc_extractor(dummy_waveform)
    print(f"\nLFCCExtractor:")
    print(f"  Input:  {list(dummy_waveform.shape)}")
    print(f"  Output: {list(lfcc_out.shape)}  (expected: [4, 60, ~401])")
    assert lfcc_out.shape[1] == 60, f"LFCC dim mismatch: got {lfcc_out.shape[1]}, expected 60"

    # --- Log-Mel ---
    mel_extractor = LogMelExtractor().to(device)
    mel_extractor.eval()
    with torch.no_grad():
        mel_out = mel_extractor(dummy_waveform)
    print(f"\nLogMelExtractor:")
    print(f"  Input:  {list(dummy_waveform.shape)}")
    print(f"  Output: {list(mel_out.shape)}  (expected: [4, 80, ~401])")
    assert mel_out.shape[1] == 80, f"LogMel dim mismatch: got {mel_out.shape[1]}, expected 80"

    # --- MFCC ---
    mfcc_extractor = MFCCExtractor().to(device)
    mfcc_extractor.eval()
    with torch.no_grad():
        mfcc_out = mfcc_extractor(dummy_waveform)
    print(f"\nMFCCExtractor:")
    print(f"  Input:  {list(dummy_waveform.shape)}")
    print(f"  Output: {list(mfcc_out.shape)}  (expected: [4, 120, ~401])")
    assert mfcc_out.shape[1] == 120, f"MFCC dim mismatch: got {mfcc_out.shape[1]}, expected 120"

    # --- TemporalPooler ---
    pooler = TemporalPooler(window_size=12)
    # Simulate CNN output after reshape: [B, T≈100, 128]
    cnn_out = torch.randn(4, 100, 128)
    pooled = pooler(cnn_out)
    print(f"\nTemporalPooler:")
    print(f"  Input:  {list(cnn_out.shape)}")
    print(f"  Output: {list(pooled.shape)}  (expected: [4, 8, 128])")
    assert pooled.shape[1] == 8, f"Pooler segment count mismatch: got {pooled.shape[1]}"

    # --- CMVN verification (zero mean, unit std per feature dimension) ---
    mean_check = lfcc_out.mean(dim=-1)
    std_check = lfcc_out.std(dim=-1)
    print(f"\nCMVN Verification (LFCC):")
    print(f"  Mean (should be ~0): min={mean_check.min():.4f}, max={mean_check.max():.4f}")
    print(f"  Std  (should be ~1): min={std_check.min():.4f}, max={std_check.max():.4f}")

    # --- Factory ---
    for ft in ["lfcc", "logmel", "mfcc"]:
        ext = get_extractor(ft)
        print(f"\nFactory '{ft}': output_dim={ext.output_dim}")

    print("\n✅ All shape checks passed.")
