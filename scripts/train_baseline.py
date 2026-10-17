"""
Training and Evaluation Script for Phase 3: XGBoost Baseline Classifier.
Extracts LFCC statistical features (240-dim) from real ASVspoof 2019 LA dataset,
trains XGBoost, evaluates overall & per-attack EER, and saves outputs.
"""

import sys
import json
import time
import numpy as np
import torch
from pathlib import Path

from src.config import config
from src.data.dataloader import create_dataloaders
from src.models.baseline import XGBoostBaseline


def main():
    print("==========================================================================")
    print("      Phase 3: XGBoost Baseline Training & Evaluation (Real Dataset)       ")
    print("==========================================================================")
    print(f"Device available: {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    print(f"Dataset root: {config.paths.asvspoof_root}")
    
    # 1. Verify real dataset existence
    if not config.paths.train_protocol.exists() or not config.paths.train_audio_dir.exists():
        print("\n[ERROR] Real ASVspoof 2019 LA dataset files not found!")
        print(f"  Expected protocol path: {config.paths.train_protocol}")
        print(f"  Expected audio dir:     {config.paths.train_audio_dir}")
        print("\nSynthetic data fallback is strictly disabled for training per project policy.")
        print("Please ensure LA.zip download finishes and extracts to data/LA/.")
        sys.exit(1)
        
    print("\n[1/4] Creating DataLoaders for real ASVspoof 2019 LA dataset...")
    t0 = time.time()
    train_loader, dev_loader, eval_loader, class_weights = create_dataloaders(
        cfg=config,
        use_dummy_if_missing=False,
        use_weighted_sampler=False
    )
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Dev samples:   {len(dev_loader.dataset)}")
    print(f"  Eval samples:  {len(eval_loader.dataset)}")
    print(f"  DataLoader init time: {time.time() - t0:.2f}s")

    # 2. Extract aggregated LFCC features
    baseline = XGBoostBaseline(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    features_cache_path = config.paths.results_dir / "baseline_features.npz"
    if features_cache_path.exists():
        print(f"\n[2/4] Found cached LFCC features at {features_cache_path}. Loading...")
        cached = np.load(features_cache_path, allow_pickle=True)
        X_train, y_train, train_attacks, train_utts = cached["X_train"], cached["y_train"], cached["train_attacks"].tolist(), cached["train_utts"].tolist()
        X_dev, y_dev, dev_attacks, dev_utts = cached["X_dev"], cached["y_dev"], cached["dev_attacks"].tolist(), cached["dev_utts"].tolist()
        X_eval, y_eval, eval_attacks, eval_utts = cached["X_eval"], cached["y_eval"], cached["eval_attacks"].tolist(), cached["eval_utts"].tolist()
        print(f"  Loaded X_train {X_train.shape}, X_dev {X_dev.shape}, X_eval {X_eval.shape}")
    else:
        print("\n[2/4] Extracting 240-dim aggregated LFCC features (GPU accelerated)...")
        t0 = time.time()
        X_train, y_train, train_attacks, train_utts = baseline.extract_features_from_loader(train_loader, desc="[Train Set LFCC]")
        print(f"    X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        
        X_dev, y_dev, dev_attacks, dev_utts = baseline.extract_features_from_loader(dev_loader, desc="[Dev Set LFCC]")
        print(f"    X_dev shape: {X_dev.shape}, y_dev shape: {y_dev.shape}")
        
        X_eval, y_eval, eval_attacks, eval_utts = baseline.extract_features_from_loader(eval_loader, desc="[Eval Set LFCC]")
        print(f"    X_eval shape: {X_eval.shape}, y_eval shape: {y_eval.shape}")
        print(f"  Total feature extraction time: {time.time() - t0:.2f}s")
        
        # Cache extracted features to disk
        np.savez_compressed(
            features_cache_path,
            X_train=X_train, y_train=y_train, train_attacks=np.array(train_attacks), train_utts=np.array(train_utts),
            X_dev=X_dev, y_dev=y_dev, dev_attacks=np.array(dev_attacks), dev_utts=np.array(dev_utts),
            X_eval=X_eval, y_eval=y_eval, eval_attacks=np.array(eval_attacks), eval_utts=np.array(eval_utts)
        )
        print(f"  Cached extracted features to {features_cache_path}")

    # 3. Fit XGBoost classifier
    print("\n[3/4] Fitting XGBoost Baseline Classifier...")
    t0 = time.time()
    baseline.fit(X_train, y_train)
    print(f"  Training time: {time.time() - t0:.2f}s")

    # 4. Evaluate performance
    print("\n[4/4] Evaluating baseline performance on Dev and Eval sets...")
    dev_metrics = baseline.evaluate(X_dev, y_dev, dev_attacks)
    eval_metrics = baseline.evaluate(X_eval, y_eval, eval_attacks)

    print("\n==========================================================================")
    print(f"  DEV SET  --> Overall EER: {dev_metrics['overall_eer']:.2f}% | AUC: {dev_metrics['overall_auc']:.4f}")
    print(f"  EVAL SET --> Overall EER: {eval_metrics['overall_eer']:.2f}% | AUC: {eval_metrics['overall_auc']:.4f}")
    print("--------------------------------------------------------------------------")
    print("  Per-Attack EER Breakdown (Eval Set):")
    for att, eer in eval_metrics['per_attack_eer'].items():
        print(f"    Attack {att:5s}: EER = {eer:6.2f}%")
    print("==========================================================================")

    # Save model and results
    baseline.save(config.paths.models_dir)
    
    results = {
        "dev_metrics": dev_metrics,
        "eval_metrics": eval_metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    results_path = config.paths.results_dir / "baseline_metrics.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved baseline metric results to {results_path}")

    # Persist real per-utterance scores for ROC/DET plotting in notebook 03
    scores_path = config.paths.results_dir / "baseline_scores.npz"
    np.savez(
        scores_path,
        dev_scores=baseline.predict_proba(X_dev), dev_labels=y_dev, dev_attacks=np.array(dev_attacks),
        eval_scores=baseline.predict_proba(X_eval), eval_labels=y_eval, eval_attacks=np.array(eval_attacks),
        eval_utts=np.array(eval_utts),
    )
    print(f"Saved per-utterance scores to {scores_path}")


if __name__ == "__main__":
    main()
