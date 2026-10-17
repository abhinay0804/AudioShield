"""
Unit and integration tests for Phase 1: Data Pipeline & Preprocessing.
Tests dataset parsing, dummy generation, audio loading, dataloaders, class weights, and augmentations.
"""

import pytest
import torch
from pathlib import Path

from src.config import config
from src.data.dataset import ASVspoofDataset, generate_dummy_dataset
from src.data.dataloader import create_dataloaders, compute_class_weights
from src.data.augmentation import SpecAugment, WaveformAugment


def test_dummy_dataset_generation(tmp_path):
    """Test synthetic dummy dataset creation."""
    created_paths = generate_dummy_dataset(
        output_dir=tmp_path,
        num_train=10,
        num_dev=5,
        num_eval=5
    )
    assert "train" in created_paths
    assert "dev" in created_paths
    assert "eval" in created_paths
    assert created_paths["train"].exists()


def test_dataset_loading(tmp_path):
    """Test ASVspoofDataset parsing and item fetching."""
    generate_dummy_dataset(output_dir=tmp_path, num_train=10, num_dev=5, num_eval=5)
    proto_path = tmp_path / "LA" / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trl.txt"
    audio_dir = tmp_path / "LA" / "ASVspoof2019_LA_train" / "flac"
    
    ds = ASVspoofDataset(
        protocol_path=proto_path,
        audio_dir=audio_dir,
        is_train=True,
        auto_generate_dummy=False
    )
    
    assert len(ds) == 10
    wf, label, attack_type, utt_id = ds[0]
    
    # Check shape: [1, 64000] for 4s at 16kHz
    assert wf.ndim == 2
    assert wf.shape[0] == 1
    assert wf.shape[1] == config.audio.target_samples
    assert label.item() in (0, 1)
    assert isinstance(attack_type, str)
    assert isinstance(utt_id, str)


def test_dataloaders(tmp_path):
    """Test DataLoader creation and batch iteration."""
    generate_dummy_dataset(output_dir=tmp_path, num_train=16, num_dev=8, num_eval=8)
    
    # Override config paths temporarily
    test_cfg = config
    test_cfg.paths.train_protocol = tmp_path / "LA" / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trl.txt"
    test_cfg.paths.dev_protocol = tmp_path / "LA" / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt"
    test_cfg.paths.eval_protocol = tmp_path / "LA" / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt"
    test_cfg.paths.train_audio_dir = tmp_path / "LA" / "ASVspoof2019_LA_train" / "flac"
    test_cfg.paths.dev_audio_dir = tmp_path / "LA" / "ASVspoof2019_LA_dev" / "flac"
    test_cfg.paths.eval_audio_dir = tmp_path / "LA" / "ASVspoof2019_LA_eval" / "flac"
    test_cfg.training.num_workers = 0
    test_cfg.training.batch_size = 4
    
    train_loader, dev_loader, eval_loader, class_weights = create_dataloaders(test_cfg)
    
    assert len(train_loader) == 4
    assert class_weights.shape[0] == 2
    assert (class_weights > 0).all()
    
    for wf_batch, lbl_batch, attack_batch, utt_batch in train_loader:
        assert wf_batch.shape == (4, 1, test_cfg.audio.target_samples)
        assert lbl_batch.shape == (4,)
        break


def test_spec_augment():
    """Test SpecAugment frequency and time masking."""
    aug = SpecAugment()
    aug.train()
    
    x = torch.ones(2, 60, 400)  # [batch=2, freq=60, time=400]
    out = aug(x)
    
    assert out.shape == x.shape
    assert (out == 0.0).any(), "SpecAugment should mask out some values"


def test_waveform_augment():
    """Test Waveform noise injection."""
    aug = WaveformAugment()
    aug.train()
    
    wf = torch.zeros(1, 16000)  # Pure silence
    out = aug(wf)
    
    assert out.shape == wf.shape
    if config.augmentation.apply_noise:
        assert (out != 0.0).any()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
