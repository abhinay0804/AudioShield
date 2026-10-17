"""
Evaluation metrics for Audio Deepfake Detection on ASVspoof 2019 LA.
Includes EER computation, min t-DCF (tandem Detection Cost Function),
ROC/DET curve generators, and standard classification metrics.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_recall_fscore_support


def compute_eer(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Equal Error Rate (EER) and operating threshold.
    
    Args:
        bonafide_scores: 1D array of scores for genuine speech (higher = more genuine)
        spoof_scores: 1D array of scores for deepfake speech
        
    Returns:
        eer: Equal Error Rate in percentage (0 to 100)
        threshold: Optimal decision threshold at EER
        fpr: False positive rate array (FAR)
        fnr: False negative rate array (FRR)
        thresholds: Candidate decision thresholds
    """
    labels = np.concatenate([np.ones_like(bonafide_scores), np.zeros_like(spoof_scores)])
    scores = np.concatenate([bonafide_scores, spoof_scores])
    
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    
    try:
        # Find threshold where FPR == FNR
        eer_threshold = brentq(lambda t: interp1d(thresholds, fpr)(t) - interp1d(thresholds, fnr)(t),
                               thresholds.min(), thresholds.max())
        eer = float(interp1d(thresholds, fpr)(eer_threshold)) * 100.0
    except (ValueError, RuntimeError):
        # Fallback to closest point
        diff = np.abs(fpr - fnr)
        idx = np.argmin(diff)
        eer = float((fpr[idx] + fnr[idx]) / 2.0) * 100.0
        eer_threshold = float(thresholds[idx])
        
    return eer, eer_threshold, fpr, fnr, thresholds


def compute_min_tdcf(
    bonafide_scores: np.ndarray,
    spoof_scores: np.ndarray,
    p_spoof: float = 0.05,
    p_tar: float = 0.9405,
    p_non: float = 0.0095,
    c_miss: float = 1.0,
    c_fa: float = 10.0
) -> Tuple[float, float]:
    """
    Computes the minimum tandem Detection Cost Function (min t-DCF) proxy
    normalized by the default cost of no CM (Kinnunen et al., ASVspoof 2019).
    
    Returns:
        min_tdcf: Normalized minimum t-DCF value (lower is better, 0 is perfect)
        threshold: Decision threshold achieving min t-DCF
    """
    labels = np.concatenate([np.ones_like(bonafide_scores), np.zeros_like(spoof_scores)])
    scores = np.concatenate([bonafide_scores, spoof_scores])
    
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    
    # Cost weights
    c_miss_cm = c_miss * p_tar
    c_fa_cm = c_fa * p_spoof
    c_default = min(c_miss_cm, c_fa_cm)
    
    # t-DCF cost vector across thresholds
    t_dcf_costs = (c_miss_cm * fnr + c_fa_cm * fpr) / (c_default + 1e-12)
    
    min_idx = np.argmin(t_dcf_costs)
    min_tdcf = float(t_dcf_costs[min_idx])
    opt_threshold = float(thresholds[min_idx])
    
    return min_tdcf, opt_threshold


def compute_classification_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float
) -> Dict[str, Union[float, Dict[str, int]]]:
    """
    Computes comprehensive binary classification metrics given a decision threshold.
    
    Args:
        scores: Predicted probabilities of bonafide (class 1)
        labels: Ground truth labels (1 = bonafide, 0 = spoof)
        threshold: Decision boundary
        
    Returns:
        Dictionary of accuracy, precision, recall, f1, specificity, and confusion matrix counts.
    """
    preds = (scores >= threshold).astype(int)
    
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp + 1e-12)
    
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        }
    }
