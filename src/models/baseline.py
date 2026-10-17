"""
XGBoost Baseline Model for Explainable Audio Deepfake Detection.
Extracts LFCC statistical aggregates (mean, std, skewness, kurtosis) per utterance (240-dim vector)
and classifies bonafide vs. spoof using XGBoost.
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from scipy.stats import skew, kurtosis
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.config import config
from src.features.extractor import LFCCExtractor


def compute_eer(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Equal Error Rate (EER) and threshold from bonafide (target=1) and spoof (target=0) decision scores.
    
    Args:
        bonafide_scores: 1D array of predicted probability scores for genuine speech (target=1).
        spoof_scores: 1D array of predicted probability scores for spoofed speech (target=0).
        
    Returns:
        eer (float): Equal Error Rate in percentage (0 to 100).
        threshold (float): Decision threshold where FAR == FRR.
        fpr (np.ndarray): False Positive Rate array.
        tpr (np.ndarray): True Positive Rate array.
        thresholds (np.ndarray): Threshold values corresponding to FPR/TPR.
    """
    y_true = np.concatenate([np.ones_like(bonafide_scores), np.zeros_like(spoof_scores)])
    y_scores = np.concatenate([bonafide_scores, spoof_scores])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    
    # EER is the point where FPR == FNR
    eer_threshold = thresholds[np.nanargmin(np.abs(fpr - fnr))]
    eer = float(fpr[np.nanargmin(np.abs(fpr - fnr))] * 100.0)
    
    # Precise interpolation for EER
    try:
        eer = float(brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0) * 100.0)
    except Exception:
        pass  # Fallback to discrete min point if interpolation fails
        
    return eer, float(eer_threshold), fpr, tpr, thresholds


class XGBoostBaseline:
    """
    Baseline Audio Anti-Spoofing Classifier based on LFCC Feature Summaries + XGBoost.
    """
    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            eval_metric="logloss",
            n_jobs=-1
        )
        self.lfcc_extractor = LFCCExtractor(config.features, config.audio)
        self.is_fitted = False

    def extract_utterance_features(self, lfcc: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Aggregates frame-level LFCC features [B, 60, T] into utterance-level stats [B, 240].
        Computes Mean, Std, Skewness, Kurtosis along time dimension T for each of 60 coefficients.
        Runs directly on GPU if a torch.Tensor is provided.
        
        Args:
            lfcc: Tensor of shape [B, 60, T] or [60, T], or numpy ndarray.
            
        Returns:
            features: 2D numpy array of shape [B, 240].
        """
        if isinstance(lfcc, torch.Tensor):
            if lfcc.ndim == 2:
                lfcc = lfcc.unsqueeze(0)  # [1, 60, T]
            m = torch.mean(lfcc, dim=2)
            diff = lfcc - m.unsqueeze(2)
            var = torch.mean(diff ** 2, dim=2)
            s = torch.sqrt(torch.clamp(var, min=1e-12))
            sk = torch.mean(diff ** 3, dim=2) / (s ** 3 + 1e-12)
            kt = torch.mean(diff ** 4, dim=2) / (var ** 2 + 1e-12) - 3.0
            feats = torch.cat([m, s, sk, kt], dim=1)
            return torch.nan_to_num(feats).detach().cpu().numpy()
            
        # Fallback for raw numpy array
        if lfcc.ndim == 2:
            lfcc = np.expand_dims(lfcc, axis=0)  # [1, 60, T]
        mean_feat = np.mean(lfcc, axis=2)
        std_feat = np.std(lfcc, axis=2)
        skew_feat = skew(lfcc, axis=2, bias=False)
        kurt_feat = kurtosis(lfcc, axis=2, bias=False)
        utterance_features = np.hstack([mean_feat, std_feat, skew_feat, kurt_feat])  # [B, 240]
        return np.nan_to_num(utterance_features)

    def extract_features_from_loader(
        self,
        dataloader,
        desc: str = "Extracting LFCC features"
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Extracts aggregated LFCC features for all samples in a PyTorch DataLoader.
        
        Returns:
            X: Feature matrix [N, 240].
            y: Label array [N].
            attack_types: List of attack type tags [N].
            utt_ids: List of utterance IDs [N].
        """
        from tqdm import tqdm
        X_list, y_list, attack_list, utt_list = [], [], [], []
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lfcc_extractor.to(device)
        self.lfcc_extractor.eval()
        
        with torch.no_grad():
            for waveforms, labels, attack_types, utt_ids in tqdm(dataloader, desc=desc, dynamic_ncols=True):
                waveforms = waveforms.to(device)
                lfcc_batch = self.lfcc_extractor(waveforms)  # [B, 60, T]
                feats = self.extract_utterance_features(lfcc_batch)  # [B, 240]
                
                X_list.append(feats)
                y_list.append(labels.numpy())
                attack_list.extend(attack_types)
                utt_list.extend(utt_ids)
                
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        return X, y, attack_list, utt_list

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fits StandardScaler and XGBoost classifier on training features.
        """
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts bonafide probability (class 1) for features X.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict_proba().")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def evaluate(self, X_eval: np.ndarray, y_eval: np.ndarray, attack_types: List[str]) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Evaluates baseline performance, returning overall EER, AUC, and per-attack-type EER.
        """
        scores = self.predict_proba(X_eval)
        
        bonafide_mask = (y_eval == 1)
        spoof_mask = (y_eval == 0)
        
        bonafide_scores = scores[bonafide_mask]
        spoof_scores = scores[spoof_mask]
        
        overall_eer, threshold, fpr, tpr, _ = compute_eer(bonafide_scores, spoof_scores)
        overall_auc = float(roc_auc_score(y_eval, scores))
        
        # Per-attack EER breakdown
        attack_types_np = np.array(attack_types)
        per_attack_eer = {}
        
        unique_attacks = sorted(list(set(attack_types_np[spoof_mask])))
        for att in unique_attacks:
            if att in ("-", "bonafide"):
                continue
            att_spoof_mask = (attack_types_np == att) & spoof_mask
            att_spoof_scores = scores[att_spoof_mask]
            
            if len(att_spoof_scores) > 0:
                att_eer, _, _, _, _ = compute_eer(bonafide_scores, att_spoof_scores)
                per_attack_eer[att] = round(att_eer, 4)
                
        return {
            "overall_eer": round(overall_eer, 4),
            "overall_auc": round(overall_auc, 4),
            "threshold": round(threshold, 6),
            "per_attack_eer": per_attack_eer
        }

    def save(self, model_dir: Union[str, Path] = config.paths.models_dir):
        """
        Saves XGBoost model and StandardScaler metadata to file.
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "xgboost_baseline.json"
        scaler_path = model_dir / "xgboost_scaler.json"
        
        self.model.save_model(str(model_path))
        
        scaler_data = {
            "mean_": self.scaler.mean_.tolist() if hasattr(self.scaler, "mean_") else None,
            "scale_": self.scaler.scale_.tolist() if hasattr(self.scaler, "scale_") else None,
            "var_": self.scaler.var_.tolist() if hasattr(self.scaler, "var_") else None,
            "n_samples_seen_": int(self.scaler.n_samples_seen_) if hasattr(self.scaler, "n_samples_seen_") else 0
        }
        with open(scaler_path, "w") as f:
            json.dump(scaler_data, f, indent=2)
            
        print(f"[XGBoostBaseline] Saved model to {model_path} and scaler to {scaler_path}")

    def load(self, model_dir: Union[str, Path] = config.paths.models_dir):
        """
        Loads trained XGBoost model and StandardScaler metadata from file.
        """
        model_dir = Path(model_dir)
        model_path = model_dir / "xgboost_baseline.json" if model_dir.is_dir() else model_dir
        scaler_path = model_dir / "xgboost_scaler.json" if model_dir.is_dir() else model_dir.parent / "xgboost_scaler.json"
        
        self.model.load_model(str(model_path))
        
        if scaler_path.exists():
            with open(scaler_path, "r") as f:
                scaler_data = json.load(f)
            if scaler_data.get("mean_") is not None:
                self.scaler.mean_ = np.array(scaler_data["mean_"])
                self.scaler.scale_ = np.array(scaler_data["scale_"])
                self.scaler.var_ = np.array(scaler_data["var_"])
                self.scaler.n_samples_seen_ = scaler_data["n_samples_seen_"]
                
        self.is_fitted = True
        print(f"[XGBoostBaseline] Loaded model from {model_path} and scaler from {scaler_path}")
