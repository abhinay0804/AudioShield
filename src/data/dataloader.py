"""
DataLoader module for ASVspoof 2019 dataset.
Provides dataloader factories, class weight computation, and custom collation functions.
"""

from typing import Tuple, Dict, Optional, Union
from pathlib import Path
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.config import config, MasterConfig
from src.data.dataset import ASVspoofDataset


def compute_class_weights(dataset: ASVspoofDataset) -> torch.Tensor:
    """
    Computes inverse class frequency weights for loss balancing:
    weight[class] = Total / (2 * count[class])
    
    Returns:
        Tensor of shape [2] containing weights for [Spoof (0), Bonafide (1)]
    """
    labels = [sample["label"] for sample in dataset.samples]
    num_spoof = sum(1 for l in labels if l == 0)
    num_bonafide = sum(1 for l in labels if l == 1)
    total = len(labels)
    
    if num_spoof == 0 or num_bonafide == 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float32)
        
    weight_spoof = total / (2.0 * num_spoof)
    weight_bonafide = total / (2.0 * num_bonafide)
    
    weights = torch.tensor([weight_spoof, weight_bonafide], dtype=torch.float32)
    print(f"[ClassWeights] Total: {total} | Spoof (0): {num_spoof} (weight={weight_spoof:.3f}) | Bonafide (1): {num_bonafide} (weight={weight_bonafide:.3f})")
    return weights


def get_weighted_sampler(dataset: ASVspoofDataset) -> WeightedRandomSampler:
    """Creates a WeightedRandomSampler to handle class imbalance during training."""
    labels = [sample["label"] for sample in dataset.samples]
    weights = compute_class_weights(dataset)
    sample_weights = [weights[l].item() for l in labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler


def create_dataloaders(
    cfg: MasterConfig = config,
    use_dummy_if_missing: bool = False,
    use_weighted_sampler: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Creates DataLoaders for train, dev, and eval datasets.
    
    Returns:
        (train_loader, dev_loader, eval_loader, class_weights)
    """
    # Initialize Datasets
    train_dataset = ASVspoofDataset(
        protocol_path=cfg.paths.train_protocol,
        audio_dir=cfg.paths.train_audio_dir,
        audio_cfg=cfg.audio,
        is_train=True,
        auto_generate_dummy=use_dummy_if_missing
    )
    
    dev_dataset = ASVspoofDataset(
        protocol_path=cfg.paths.dev_protocol,
        audio_dir=cfg.paths.dev_audio_dir,
        audio_cfg=cfg.audio,
        is_train=False,
        auto_generate_dummy=use_dummy_if_missing
    )
    
    eval_dataset = ASVspoofDataset(
        protocol_path=cfg.paths.eval_protocol,
        audio_dir=cfg.paths.eval_audio_dir,
        audio_cfg=cfg.audio,
        is_train=False,
        auto_generate_dummy=use_dummy_if_missing
    )
    
    # Compute class weights on training dataset
    class_weights = compute_class_weights(train_dataset)
    
    # Setup sampler / shuffle
    sampler = get_weighted_sampler(train_dataset) if use_weighted_sampler else None
    shuffle = (sampler is None)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.training.num_workers,
        pin_memory=True if cfg.training.device == "cuda" else False
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True if cfg.training.device == "cuda" else False
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True if cfg.training.device == "cuda" else False
    )
    
    return train_loader, dev_loader, eval_loader, class_weights


if __name__ == "__main__":
    print("Testing DataLoader Module...")
    # Reduce num_workers to 0 for quick inline script testing
    config.training.num_workers = 0
    config.training.batch_size = 8
    
    train_loader, dev_loader, eval_loader, class_weights = create_dataloaders(config)
    print(f"Train batches: {len(train_loader)}, Dev batches: {len(dev_loader)}, Eval batches: {len(eval_loader)}")
    
    for waveforms, labels, attack_types, utt_ids in train_loader:
        print(f"Batch Waveform shape: {waveforms.shape}, Labels shape: {labels.shape}")
        print(f"Sample Attack Types in batch: {attack_types[:4]}")
        break
