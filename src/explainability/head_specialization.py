"""
Head Specialization Analysis for Phase 5.
Quantifies the distinct roles and acoustic specialization of each attention head
in the Dual-Stream Multi-Head Self-Attention module.
"""

from typing import Dict, List, Tuple
import numpy as np


class HeadSpecializationAnalyzer:
    """
    Quantifies the diversity and specialization of attention heads.
    Measures head diversity (pairwise cosine similarity), temporal locality index
    (bandwidth), and attention entropy across heads.
    """
    def __init__(self, num_fine_heads: int = 4, num_coarse_heads: int = 2):
        self.num_fine_heads = num_fine_heads
        self.num_coarse_heads = num_coarse_heads
        
        self.fine_matrices: List[np.ndarray] = []  # List of [4, 100, 100]
        self.coarse_matrices: List[np.ndarray] = [] # List of [2, 8, 8]

    def add_sample(self, fine_heads_matrix: np.ndarray, coarse_heads_matrix: np.ndarray):
        """
        Args:
            fine_heads_matrix: Array [4, 100, 100]
            coarse_heads_matrix: Array [2, 8, 8]
        """
        self.fine_matrices.append(fine_heads_matrix.astype(np.float32))
        self.coarse_matrices.append(coarse_heads_matrix.astype(np.float32))

    def compute_head_diversity_matrix(self, is_coarse: bool = False) -> np.ndarray:
        """
        Computes pairwise cosine similarity between attention heads averaged across samples.
        Low off-diagonal values indicate strong functional specialization.
        
        Returns:
            sim_matrix: [n_heads, n_heads] array where entry (i, j) is the cosine similarity.
        """
        matrices = self.coarse_matrices if is_coarse else self.fine_matrices
        n_heads = self.num_coarse_heads if is_coarse else self.num_fine_heads
        
        if len(matrices) == 0:
            return np.eye(n_heads)
            
        all_samples = np.stack(matrices, axis=0)  # [N, n_heads, T, T]
        N, H, T, _ = all_samples.shape
        
        # Flatten each head's attention matrix into a vector: [N, H, T*T]
        flat_heads = all_samples.reshape(N, H, -1)
        
        # Compute cosine similarity per sample, then average
        sim_matrix = np.zeros((H, H), dtype=np.float32)
        for i in range(H):
            for j in range(H):
                # Cosine similarity between head i and head j
                dot = np.sum(flat_heads[:, i, :] * flat_heads[:, j, :], axis=1)
                norm_i = np.linalg.norm(flat_heads[:, i, :], axis=1) + 1e-8
                norm_j = np.linalg.norm(flat_heads[:, j, :], axis=1) + 1e-8
                sim_matrix[i, j] = float(np.mean(dot / (norm_i * norm_j)))
                
        return np.round(sim_matrix, 4)

    def compute_head_locality_index(self, is_coarse: bool = False) -> List[float]:
        """
        Computes the average temporal dispersion (distance from diagonal) for each head:
            Locality = (1/T) sum_{t, tau} A(t, tau) * |t - tau|
            
        Lower value = Local head (focuses on adjacent speech frames/boundaries).
        Higher value = Global head (focuses on long-range dependencies across the sentence).
        """
        matrices = self.coarse_matrices if is_coarse else self.fine_matrices
        n_heads = self.num_coarse_heads if is_coarse else self.num_fine_heads
        
        if len(matrices) == 0:
            return [0.0] * n_heads
            
        all_samples = np.stack(matrices, axis=0)  # [N, H, T, T]
        N, H, T, _ = all_samples.shape
        
        # Create distance matrix |t - tau|: [T, T]
        t_indices = np.arange(T)
        dist_matrix = np.abs(t_indices[:, None] - t_indices[None, :])
        
        # Weighted distance: sum(A * dist) / T
        locality_per_head = []
        for h in range(H):
            head_att = all_samples[:, h, :, :]  # [N, T, T]
            dispersion = np.sum(head_att * dist_matrix, axis=(1, 2)) / T
            locality_per_head.append(round(float(np.mean(dispersion)), 3))
            
        return locality_per_head

    def get_summary(self) -> Dict:
        """
        Returns full head specialization report.
        """
        return {
            "fine_heads_diversity": self.compute_head_diversity_matrix(is_coarse=False).tolist(),
            "fine_heads_locality": self.compute_head_locality_index(is_coarse=False),
            "coarse_heads_diversity": self.compute_head_diversity_matrix(is_coarse=True).tolist(),
            "coarse_heads_locality": self.compute_head_locality_index(is_coarse=True)
        }
