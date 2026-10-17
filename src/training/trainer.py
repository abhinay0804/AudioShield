"""
Trainer module for Phase 4: Dual-Stream CNN-BiLSTM + Multi-Head Attention.
Implements mixed precision training (AMP), Cosine Annealing scheduler,
gradient clipping, early stopping on Dev EER, checkpointing, and metric logging.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

from src.config import config, MasterConfig
from src.features.extractor import LFCCExtractor
from src.data.augmentation import SpecAugment
from src.models.dual_stream_model import DualStreamModel
from src.models.losses import CombinedLoss
from src.models.baseline import compute_eer


class Trainer:
    """
    Orchestrates training, validation, early stopping, and evaluation
    for the Dual-Stream Multi-Granularity model.
    """
    def __init__(
        self,
        model: Optional[DualStreamModel] = None,
        train_loader: Optional[DataLoader] = None,
        dev_loader: Optional[DataLoader] = None,
        eval_loader: Optional[DataLoader] = None,
        cfg: MasterConfig = config,
        class_weights: Optional[torch.Tensor] = None
    ):
        self.cfg = cfg
        self.device = torch.device(cfg.training.device)
        
        # 1. Model & Feature Extractor
        self.model = model if model is not None else DualStreamModel(cfg.model)
        self.model.to(self.device)
        
        self.lfcc_extractor = LFCCExtractor(cfg.features, cfg.audio).to(self.device).eval()
        self.spec_augment = SpecAugment(cfg.augmentation).to(self.device)
        
        # 2. DataLoaders
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.eval_loader = eval_loader
        
        # 3. Loss & Optimization
        self.criterion = CombinedLoss(
            supcon_lambda=cfg.model.supcon_lambda,
            temperature=0.1,
            class_weights=class_weights.to(self.device) if class_weights is not None else None
        ).to(self.device)
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.training.scheduler_t_max,
            eta_min=cfg.training.scheduler_eta_min
        )
        
        # 4. Mixed Precision Scaler
        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        self.device_type = device_type
        self.scaler = torch.amp.GradScaler("cuda", enabled=(device_type == "cuda"))
        
        # 5. Tracking & Directories
        self.checkpoint_dir = Path(cfg.paths.checkpoint_dir)
        self.results_dir = Path(cfg.paths.results_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_dev_eer = float("inf")
        self.epochs_without_improvement = 0
        self.history: List[Dict[str, Union[int, float]]] = []

    def train_epoch(self, epoch: int, total_epochs: int) -> Dict[str, float]:
        """
        Executes one training epoch with mixed precision and gradient clipping.
        Displays detailed Colab-style real-time progress bar.
        """
        self.model.train()
        self.spec_augment.train()
        
        total_loss = 0.0
        total_am_loss = 0.0
        total_supcon_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(
            enumerate(self.train_loader),
            total=num_batches,
            desc=f"  Train [{epoch:02d}/{total_epochs:02d}]",
            file=sys.stdout,
            dynamic_ncols=True,
            leave=True,
            bar_format="{l_bar}{bar:25}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )
        
        for batch_idx, (waveforms, labels, _, _) in pbar:
            waveforms = waveforms.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=self.device_type, enabled=(self.device_type == "cuda")):
                # 1. On-the-fly LFCC extraction & SpecAugment
                with torch.no_grad():
                    lfcc = self.lfcc_extractor(waveforms)  # [B, 60, T]
                lfcc_aug = self.spec_augment(lfcc)         # [B, 60, T]
                
                # 2. Forward pass with labels for AMSoftmax
                logits, embedding, _, _ = self.model(lfcc_aug, labels=labels)
                
                # 3. Loss calculation
                loss, loss_dict = self.criterion(logits, embedding, labels)
                
            # 4. Backward with AMP Scaler
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.training.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss_dict["loss_total"]
            total_am_loss += loss_dict["loss_am"]
            total_supcon_loss += loss_dict["loss_supcon"]
            
            curr_avg_loss = total_loss / (batch_idx + 1)
            curr_avg_am = total_am_loss / (batch_idx + 1)
            curr_avg_sc = total_supcon_loss / (batch_idx + 1)
            
            gpu_mem = f"{torch.cuda.memory_allocated() / (1024**2):.0f}MB" if torch.cuda.is_available() else "CPU"
            
            pbar.set_postfix({
                "loss": f"{curr_avg_loss:.4f}",
                "am": f"{curr_avg_am:.4f}",
                "sc": f"{curr_avg_sc:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.1e}",
                "mem": gpu_mem
            })
            
        self.scheduler.step()
        sys.stdout.flush()
        
        return {
            "train_loss": total_loss / num_batches,
            "train_am_loss": total_am_loss / num_batches,
            "train_supcon_loss": total_supcon_loss / num_batches,
            "lr": self.optimizer.param_groups[0]["lr"]
        }

    def evaluate(
        self,
        loader: DataLoader,
        split_name: str = "Dev",
        epoch_str: str = ""
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Evaluates model on validation or test DataLoader, computing EER and AUC.
        Displays detailed Colab-style real-time progress bar.
        """
        self.model.eval()
        
        scores_list = []
        labels_list = []
        attacks_list = []
        num_batches = len(loader)
        
        desc_label = f"  {split_name} {epoch_str}".strip()
        pbar = tqdm(
            loader,
            total=num_batches,
            desc=desc_label,
            file=sys.stdout,
            dynamic_ncols=True,
            leave=True,
            bar_format="{l_bar}{bar:25}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        with torch.no_grad():
            for waveforms, labels, attack_types, _ in pbar:
                waveforms = waveforms.to(self.device, non_blocking=True)
                
                with torch.amp.autocast(device_type=self.device_type, enabled=(self.device_type == "cuda")):
                    lfcc = self.lfcc_extractor(waveforms)
                    logits, _, _, _ = self.model(lfcc)  # eval mode (no margin subtracted)
                    
                # Class 1 (bonafide) probability score
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                
                scores_list.extend(probs)
                labels_list.extend(labels.numpy())
                attacks_list.extend(attack_types)
                
        scores = np.array(scores_list)
        labels_np = np.array(labels_list)
        attacks_np = np.array(attacks_list)
        
        bonafide_scores = scores[labels_np == 1]
        spoof_scores = scores[labels_np == 0]
        
        overall_eer, threshold, _, _, _ = compute_eer(bonafide_scores, spoof_scores)
        
        # Per-attack breakdown
        per_attack_eer = {}
        unique_attacks = sorted(list(set(attacks_np[labels_np == 0])))
        for att in unique_attacks:
            if att in ("-", "bonafide"):
                continue
            att_scores = scores[(attacks_np == att) & (labels_np == 0)]
            if len(att_scores) > 0:
                att_eer, _, _, _, _ = compute_eer(bonafide_scores, att_scores)
                per_attack_eer[att] = round(att_eer, 4)
                
        sys.stdout.flush()
        return {
            "overall_eer": round(overall_eer, 4),
            "threshold": round(threshold, 6),
            "per_attack_eer": per_attack_eer,
            "scores": scores,
            "labels": labels_np,
            "attacks": attacks_np
        }

    def save_checkpoint(self, path: Union[str, Path], is_best: bool = False, extra_info: Optional[dict] = None):
        """Saves model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_dev_eer": self.best_dev_eer,
            "config": self.cfg,
            "extra_info": extra_info or {}
        }
        torch.save(checkpoint, str(path))
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, str(best_path))

    def fit(self, num_epochs: Optional[int] = None) -> List[Dict]:
        """
        Runs the full training loop across epochs with early stopping on Dev EER.
        Prints clear, detailed progress blocks identical to Google Colab.
        """
        epochs = num_epochs or self.cfg.training.num_epochs
        patience = self.cfg.training.early_stopping_patience
        
        print("\n" + "=" * 80)
        print("      Dual-Stream CNN-BiLSTM + Multi-Head Attention Training Pipeline      ")
        print("=" * 80)
        print(f"  • Hardware Device:       {self.device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
        print(f"  • Total Training Epochs: {epochs}")
        print(f"  • Early Stopping:        {patience} epochs patience on Dev EER")
        print(f"  • Mixed Precision (AMP): {'Enabled' if self.device_type == 'cuda' else 'Disabled'}")
        print(f"  • Initial Learning Rate: {self.cfg.training.learning_rate:.1e} (Cosine Annealing to {self.cfg.training.scheduler_eta_min:.1e})")
        print("=" * 80 + "\n")
        
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            print(f"Epoch {epoch:02d}/{epochs:02d}")
            print("-" * 80)
            
            # 1. Training Phase
            train_metrics = self.train_epoch(epoch, epochs)
            
            # 2. Validation Phase
            dev_results = self.evaluate(self.dev_loader, split_name="Dev", epoch_str=f"[{epoch:02d}/{epochs:02d}]")
            epoch_time = time.time() - t0
            
            dev_eer = dev_results["overall_eer"]
            is_best = dev_eer < self.best_dev_eer
            
            if is_best:
                prev_best = self.best_dev_eer
                self.best_dev_eer = dev_eer
                self.epochs_without_improvement = 0
                self.save_checkpoint(
                    self.checkpoint_dir / f"checkpoint_epoch_{epoch:02d}.pt",
                    is_best=True,
                    extra_info={"dev_eer": dev_eer, "epoch": epoch}
                )
                status_msg = f"★ NEW BEST MODEL! (Dev EER: {dev_eer:.2f}% | Prev: {prev_best if prev_best != float('inf') else 'N/A'}) -> Saved"
            else:
                self.epochs_without_improvement += 1
                status_msg = f"No improvement ({self.epochs_without_improvement}/{patience} patience epochs)"
                
            epoch_log = {
                "epoch": epoch,
                "train_loss": train_metrics["train_loss"],
                "train_am_loss": train_metrics["train_am_loss"],
                "train_supcon_loss": train_metrics["train_supcon_loss"],
                "dev_eer": dev_eer,
                "lr": train_metrics["lr"],
                "time_sec": epoch_time
            }
            self.history.append(epoch_log)
            
            # Colab-style summary box per epoch
            gpu_mem = f"{torch.cuda.memory_allocated() / (1024**2):.0f}MB" if torch.cuda.is_available() else "N/A"
            print("-" * 80)
            print(f"Epoch {epoch:02d}/{epochs:02d} Summary  [{epoch_time:.1f}s elapsed]")
            print(f"  • Train Loss:        {train_metrics['train_loss']:.4f} (AM-Softmax: {train_metrics['train_am_loss']:.4f} | SupCon: {train_metrics['train_supcon_loss']:.4f})")
            print(f"  • Dev Set EER:       {dev_eer:.2f}% (Baseline: 18.37% | Target: 2-5%)")
            print(f"  • Optimal Threshold: {dev_results['threshold']:.6f}")
            print(f"  • Learning Rate:     {train_metrics['lr']:.2e}")
            print(f"  • GPU VRAM:          {gpu_mem}")
            print(f"  • Status:            {status_msg}")
            print("=" * 80 + "\n")
            sys.stdout.flush()
            
            # Save history incrementally
            with open(self.results_dir / "training_history.json", "w") as f:
                json.dump(self.history, f, indent=2)
                
            # Early stopping check
            if self.epochs_without_improvement >= patience:
                print(f"[Early Stopping] No improvement in Dev EER for {patience} consecutive epochs. Stopping early.\n")
                break
                
        # Final Evaluation on Eval Set with Best Model
        print("\n==========================================================================")
        print(" Loading best model checkpoint for final evaluation on Eval Set...")
        best_ckpt = torch.load(self.checkpoint_dir / "best_model.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(best_ckpt["model_state_dict"])
        
        eval_results = self.evaluate(self.eval_loader, split_name="Eval")
        print(f"\nFINAL EVALUATION SET OVERALL EER: {eval_results['overall_eer']:.2f}%")
        print("Per-Attack Breakdown (Eval Set):")
        for att, eer in eval_results["per_attack_eer"].items():
            print(f"  Attack {att:5s}: EER = {eer:6.2f}%")
        print("==========================================================================")
        
        # Save final metrics & scores
        final_summary = {
            "best_dev_eer": self.best_dev_eer,
            "eval_eer": eval_results["overall_eer"],
            "eval_threshold": eval_results["threshold"],
            "eval_per_attack": eval_results["per_attack_eer"]
        }
        with open(self.results_dir / "core_model_metrics.json", "w") as f:
            json.dump(final_summary, f, indent=2)
            
        np.savez_compressed(
            self.results_dir / "core_model_scores.npz",
            eval_scores=eval_results["scores"],
            eval_labels=eval_results["labels"],
            eval_attacks=eval_results["attacks"]
        )
        print(f"Saved evaluation metrics to {self.results_dir / 'core_model_metrics.json'}")
        return self.history
