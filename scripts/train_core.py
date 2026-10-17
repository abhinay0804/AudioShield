"""
Training Script for Phase 4: Dual-Stream CNN-BiLSTM + Multi-Head Self-Attention.
Trains the proposed neural architecture on the official ASVspoof 2019 LA dataset,
validates with early stopping on Dev EER, and evaluates on the 71k evaluation trials.
"""

import sys
import argparse
import torch

from src.config import config
from src.data.dataloader import create_dataloaders
from src.models.dual_stream_model import DualStreamModel
from src.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Dual-Stream Audio Deepfake Detection Model")
    parser.add_argument("--epochs", type=int, default=config.training.num_epochs, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.training.batch_size, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=config.training.learning_rate, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=config.training.early_stopping_patience, help="Early stopping patience")
    parser.add_argument("--num-workers", type=int, default=config.training.num_workers, help="DataLoader workers")
    return parser.parse_args()


def main():
    # Ensure immediate unbuffered terminal output like Google Colab
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
        
    args = parse_args()
    
    print("==========================================================================")
    print(" Phase 4: Dual-Stream CNN-BiLSTM + Multi-Head Attention Training Pipeline  ")
    print("==========================================================================")
    print(f"Device: {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    print(f"Dataset root: {config.paths.asvspoof_root}")
    print(f"Hyperparameters: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, patience={args.patience}")
    
    # Update config overrides
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.early_stopping_patience = args.patience
    config.training.num_workers = args.num_workers
    
    # 1. Create DataLoaders for real ASVspoof 2019 LA dataset
    print("\n[1/3] Loading official ASVspoof 2019 LA DataLoaders...")
    train_loader, dev_loader, eval_loader, class_weights = create_dataloaders(
        cfg=config,
        use_dummy_if_missing=False,
        use_weighted_sampler=True
    )
    print(f"  Train samples: {len(train_loader.dataset)} | Batches: {len(train_loader)}")
    print(f"  Dev samples:   {len(dev_loader.dataset)}   | Batches: {len(dev_loader)}")
    print(f"  Eval samples:  {len(eval_loader.dataset)}  | Batches: {len(eval_loader)}")
    
    # 2. Instantiate DualStreamModel
    print("\n[2/3] Instantiating DualStreamModel...")
    model = DualStreamModel(config.model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # 3. Initialize Trainer and run training
    print("\n[3/3] Launching Trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        eval_loader=eval_loader,
        cfg=config,
        class_weights=class_weights
    )
    
    trainer.fit(num_epochs=args.epochs)
    print("\nDual-Stream model training and evaluation finished successfully!")


if __name__ == "__main__":
    main()
