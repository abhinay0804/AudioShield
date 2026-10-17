"""
Centralized Configuration Module for Audio Deepfake Detection Project.
Defines paths, audio hyperparameters, feature parameters, model parameters, and training settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
import os
import torch

# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

@dataclass
class PathConfig:
    """Paths configuration for datasets, protocol files, and outputs."""
    root_dir: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    
    # ASVspoof 2019 LA Dataset paths
    asvspoof_root: Path = PROJECT_ROOT / "data" / "LA"
    
    # Protocol file paths (official train protocol is .trn.txt, dev and eval are .trl.txt)
    train_protocol: Path = PROJECT_ROOT / "data" / "LA" / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"
    dev_protocol: Path = PROJECT_ROOT / "data" / "LA" / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt"
    eval_protocol: Path = PROJECT_ROOT / "data" / "LA" / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt"
    
    # Audio directories
    train_audio_dir: Path = PROJECT_ROOT / "data" / "LA" / "ASVspoof2019_LA_train" / "flac"
    dev_audio_dir: Path = PROJECT_ROOT / "data" / "LA" / "ASVspoof2019_LA_dev" / "flac"
    eval_audio_dir: Path = PROJECT_ROOT / "data" / "LA" / "ASVspoof2019_LA_eval" / "flac"
    
    # Synthetic/dummy dataset directory (fallback when real dataset is not present)
    dummy_data_dir: Path = PROJECT_ROOT / "data" / "dummy"
    
    # Checkpoints & Outputs
    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints"
    models_dir: Path = PROJECT_ROOT / "models"
    results_dir: Path = PROJECT_ROOT / "results"
    output_dir: Path = PROJECT_ROOT / "outputs"
    log_dir: Path = PROJECT_ROOT / "outputs" / "logs"

    def __post_init__(self):
        """Create output directories if they do not exist."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.dummy_data_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class AudioConfig:
    """Audio preprocessing configuration."""
    sample_rate: int = 16000          # Target sample rate (16 kHz)
    duration_sec: float = 4.0          # Target fixed duration (4 seconds)
    target_samples: int = 64000        # sample_rate * duration_sec (16000 * 4)
    trim_silence: bool = True          # Trim leading/trailing silence
    top_db: float = 30.0               # Silence threshold in dB
    normalize: bool = True             # Peak amplitude normalization


@dataclass
class FeatureConfig:
    """Feature extraction parameters (LFCC, Log-Mel, MFCC)."""
    # LFCC Parameters (Primary Feature Set)
    n_lfcc: int = 20                   # Static LFCC coefficients
    n_filter: int = 60                 # Number of linear filterbank filters
    compute_deltas: bool = True        # Compute Delta + Delta-Delta (20 + 20 + 20 = 60 total)
    
    # Shared STFT Parameters
    n_fft: int = 512                   # FFT window size (512 = 32ms at 16kHz — correct for anti-spoofing)
    win_length: int = 400              # Frame window length (~25ms at 16kHz)
    hop_length: int = 160              # Frame hop length (~10ms at 16kHz)
    f_min: float = 0.0                 # Minimum frequency
    f_max: float = 8000.0              # Maximum frequency (Nyquist at 16kHz)
    
    # Log-Mel Parameters (Ablation)
    n_mels: int = 80
    
    # MFCC Parameters (Ablation)
    n_mfcc: int = 40


@dataclass
class AugmentationConfig:
    """Data augmentation parameters (SpecAugment & Noise)."""
    apply_spec_augment: bool = True
    freq_mask_param: int = 10          # Max frequency channels to mask
    time_mask_param: int = 25          # Max time frames to mask
    num_freq_masks: int = 2            # Number of freq masks
    num_time_masks: int = 2            # Number of time masks
    
    apply_noise: bool = False
    noise_snr_db: float = 20.0         # Signal to noise ratio in dB


@dataclass
class ModelConfig:
    """Architecture configuration for Dual-Stream CNN-BiLSTM + Attention."""
    in_channels: int = 1               # Input feature channels (1 for spectrogram image)
    num_classes: int = 2               # Bonafide (1) vs Spoof (0)
    
    # CNN Front-end
    cnn_out_channels: int = 64
    
    # Fine-grained Stream (Frame-level ~10ms)
    fine_lstm_hidden: int = 128
    fine_lstm_layers: int = 2
    fine_heads: int = 4
    
    # Coarse-grained Stream (Segment-level ~500ms)
    coarse_window_frames: int = 50     # 50 frames * 10ms = 500ms
    coarse_lstm_hidden: int = 128
    coarse_lstm_layers: int = 1
    coarse_heads: int = 2
    
    # Embedding & Classification
    embedding_dim: int = 256           # Combined context embedding size
    dropout: float = 0.3
    
    # Loss Function Parameters
    am_softmax_margin: float = 0.35    # Angular margin (m)
    am_softmax_scale: float = 30.0     # Feature scaling factor (s)
    supcon_lambda: float = 0.1         # Supervised contrastive loss weight


@dataclass
class TrainingConfig:
    """Hyperparameters for model training."""
    seed: int = 42
    batch_size: int = 32
    num_workers: int = 4
    num_epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler_t_max: int = 30          # Cosine Annealing T_max
    scheduler_eta_min: float = 1e-6
    early_stopping_patience: int = 6
    grad_clip: float = 1.0             # Max gradient norm
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class MasterConfig:
    """Master configuration aggregating all sub-configs."""
    paths: PathConfig = field(default_factory=PathConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# Default global configuration instance
config = MasterConfig()
