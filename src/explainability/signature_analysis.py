"""
Attack Signature Analysis for Phase 5.
Quantifies temporal attention patterns across different speech synthesis and voice conversion attacks.
Calculates temporal entropy, peak concentration, and attack-specific attention signatures.
"""

from typing import Dict, List, Tuple
import numpy as np


class AttackSignatureAnalyzer:
    """
    Analyzes and compares temporal attention signatures across bonafide speech
    and 13 deepfake attack algorithms (A07-A19).
    """
    def __init__(self):
        # Stores collected frame weights per attack category
        # Key: attack_type (e.g., 'bonafide', 'A07', 'A10', etc.) -> List of [100] arrays
        self.attack_profiles: Dict[str, List[np.ndarray]] = {}

    def add_sample(self, attack_type: str, fine_weights: np.ndarray):
        """
        Registers an extracted attention weight vector for an utterance.
        
        Args:
            attack_type: Attack tag string ('bonafide', 'A07', etc.)
            fine_weights: 1D array of shape [100]
        """
        if attack_type not in self.attack_profiles:
            self.attack_profiles[attack_type] = []
        self.attack_profiles[attack_type].append(fine_weights.astype(np.float32))

    def compute_signature(self, attack_type: str) -> Dict[str, float]:
        """
        Computes summary signature statistics for a specific attack type:
        - mean_profile: Average temporal curve across 100 frames
        - entropy: Shannon entropy of the attention distribution (lower = more localized spikes)
        - peak_ratio: Ratio of top-10% frames to total attention (higher = sharper artifact detection)
        - onset_ratio: Proportion of attention in the first third of utterance (0-33%)
        - nucleus_ratio: Proportion of attention in middle third (33-66%)
        - coda_ratio: Proportion of attention in final third (66-100%)
        """
        if attack_type not in self.attack_profiles or len(self.attack_profiles[attack_type]) == 0:
            return {}
            
        weights_matrix = np.array(self.attack_profiles[attack_type])  # [N, 100]
        mean_profile = np.mean(weights_matrix, axis=0)
        mean_profile = mean_profile / (mean_profile.sum() + 1e-8)  # normalize
        
        # 1. Shannon Entropy: H = - sum(p * log(p))
        eps = 1e-12
        entropy = -float(np.sum(mean_profile * np.log(mean_profile + eps)))
        
        # 2. Peak concentration: fraction of attention mass in top 10 frames
        sorted_weights = np.sort(mean_profile)[::-1]
        peak_top10 = float(np.sum(sorted_weights[:10]))
        
        # 3. Positional distribution
        T = len(mean_profile)
        t1, t2 = T // 3, 2 * (T // 3)
        onset_mass = float(np.sum(mean_profile[:t1]))
        nucleus_mass = float(np.sum(mean_profile[t1:t2]))
        coda_mass = float(np.sum(mean_profile[t2:]))
        
        return {
            "num_samples": len(weights_matrix),
            "entropy": round(entropy, 4),
            "peak_top10_mass": round(peak_top10, 4),
            "onset_mass": round(onset_mass, 4),
            "nucleus_mass": round(nucleus_mass, 4),
            "coda_mass": round(coda_mass, 4),
            "mean_profile": mean_profile.tolist()
        }

    def compute_all_signatures(self) -> Dict[str, Dict]:
        """
        Computes signatures for all tracked attack types and returns complete summary.
        """
        signatures = {}
        for attack in sorted(self.attack_profiles.keys()):
            sig = self.compute_signature(attack)
            if sig:
                signatures[attack] = sig
        return signatures
