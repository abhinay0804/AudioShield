"""
Data Augmentation module for audio waveforms and 2D spectrogram/LFCC features.
Includes SpecAugment (frequency and time masking) and Waveform noise injection.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import numpy as np

from src.config import config, AugmentationConfig


class SpecAugment(nn.Module):
    """
    SpecAugment implementation for 2D feature representations (LFCC, Mel Spectrograms).
    Applies frequency masking and time masking.
    
    Accepts tensors of shape:
        - [batch, channels, freq, time]
        - [batch, freq, time]
        - [freq, time]
    """
    def __init__(self, cfg: AugmentationConfig = config.augmentation):
        super().__init__()
        self.cfg = cfg
        self.freq_mask_param = cfg.freq_mask_param
        self.time_mask_param = cfg.time_mask_param
        self.num_freq_masks = cfg.num_freq_masks
        self.num_time_masks = cfg.num_time_masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature tensor
        Returns:
            Masked tensor of same shape
        """
        if not self.training or not self.cfg.apply_spec_augment:
            return x

        # Ensure x is at least 3D [batch, freq, time]
        orig_ndim = x.ndim
        if orig_ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif orig_ndim == 3:
            x = x.unsqueeze(1)
            
        b, c, f, t = x.shape
        masked_x = x.clone()

        # Apply Frequency Masking
        for _ in range(self.num_freq_masks):
            if f > self.freq_mask_param and self.freq_mask_param > 0:
                f_len = torch.randint(0, self.freq_mask_param + 1, (1,)).item()
                if f_len > 0:
                    f0 = torch.randint(0, f - f_len + 1, (1,)).item()
                    masked_x[:, :, f0:f0 + f_len, :] = 0.0

        # Apply Time Masking
        for _ in range(self.num_time_masks):
            if t > self.time_mask_param and self.time_mask_param > 0:
                t_len = torch.randint(0, self.time_mask_param + 1, (1,)).item()
                if t_len > 0:
                    t0 = torch.randint(0, t - t_len + 1, (1,)).item()
                    masked_x[:, :, :, t0:t0 + t_len] = 0.0

        # Restore original dimension shape
        if orig_ndim == 2:
            masked_x = masked_x.squeeze(0).squeeze(0)
        elif orig_ndim == 3:
            masked_x = masked_x.squeeze(1)

        return masked_x


class WaveformAugment(nn.Module):
    """
    Waveform-level data augmentation.
    Applies Gaussian noise injection with target SNR (in dB) and random time scaling.
    """
    def __init__(self, cfg: AugmentationConfig = config.augmentation):
        super().__init__()
        self.cfg = cfg

    def add_noise(self, waveform: torch.Tensor, snr_db: float = 20.0) -> torch.Tensor:
        """Adds zero-mean Gaussian noise to achieve target Signal-to-Noise Ratio (SNR)."""
        signal_power = torch.mean(waveform ** 2, dim=-1, keepdim=True)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / (snr_linear + 1e-8)
        
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power + 1e-8)
        return waveform + noise

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.training or not self.cfg.apply_noise:
            return waveform
            
        return self.add_noise(waveform, snr_db=self.cfg.noise_snr_db)


if __name__ == "__main__":
    print("Testing Augmentation Module...")
    spec_aug = SpecAugment()
    spec_aug.train()
    
    # Test on a dummy feature tensor [Batch=4, Freq=60, Time=400]
    dummy_feat = torch.ones(4, 60, 400)
    augmented = spec_aug(dummy_feat)
    
    print(f"Original shape: {dummy_feat.shape}, Augmented shape: {augmented.shape}")
    num_zeroed = (augmented == 0.0).sum().item()
    total_elements = augmented.numel()
    print(f"SpecAugment zeroed out {num_zeroed}/{total_elements} elements ({100.0 * num_zeroed / total_elements:.2f}%)")
