"""
Loss functions for Multi-Granularity Deepfake Detection.
Includes:
- AMSoftmaxLoss: CrossEntropy on Angular Margin Scaled Logits with Class Weighting.
- SupConLoss: Supervised Contrastive Loss (Khosla et al., 2020).
- CombinedLoss: Multi-task loss combining AM-Softmax and SupCon.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss (Khosla et al., NeurIPS 2020).
    Pulls embeddings belonging to the same class (e.g. all bonafide voices) together
    while pushing apart embeddings of different classes (bonafide vs. spoof).
    
    Formula:
        L_i = - 1/|P(i)| sum_{p in P(i)} log( exp(z_i * z_p / tau) / sum_{a != i} exp(z_i * z_a / tau) )
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings: Tensor [B, D]
            labels: Tensor [B] of ground truth class indices
        Returns:
            loss: Scalar supervised contrastive loss
        """
        device = embeddings.device
        B = embeddings.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        # 1. Normalize embeddings to unit hypersphere
        z = F.normalize(embeddings, p=2, dim=1)
        
        # 2. Similarity matrix: [B, B]
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        
        # Mask out self-contrast (diagonal entries)
        diag_mask = torch.eye(B, dtype=torch.bool, device=device)
        
        # 3. Positive pairs mask: y_i == y_j and i != j
        labels = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.T) & (~diag_mask)
        
        # For numerical stability: subtract max similarity per row
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim = sim_matrix - sim_max.detach()
        
        # Denominator: sum over all a != i
        exp_sim = torch.exp(sim) * (~diag_mask)
        denom = exp_sim.sum(dim=1, keepdim=True) + 1e-8
        
        # Log-probabilities: log(exp(sim) / denom) = sim - log(denom)
        log_prob = sim - torch.log(denom)
        
        # Mean log-probability over positive pairs for each anchor i
        num_pos_per_anchor = pos_mask.sum(dim=1)
        valid_anchors = num_pos_per_anchor > 0
        
        if not valid_anchors.any():
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1)[valid_anchors] / num_pos_per_anchor[valid_anchors]
        loss = -mean_log_prob_pos.mean()
        return loss


class AMSoftmaxLoss(nn.Module):
    """
    Loss function for AM-Softmax logits.
    Computes weighted Cross-Entropy over angular margin scaled cosine logits.
    """
    def __init__(self, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Output from AMSoftmax layer [B, num_classes]
            labels: Ground truth labels [B]
        """
        return self.ce(logits, labels.long())


class CombinedLoss(nn.Module):
    """
    Composite Multi-Task Objective:
        L_total = L_AMSoftmax + lambda * L_SupCon
    
    Args:
        supcon_lambda: Scaling factor lambda for supervised contrastive loss (default: 0.1)
        temperature: Temperature parameter for SupCon (default: 0.1)
        class_weights: Optional class balancing weights [2]
    """
    def __init__(
        self,
        supcon_lambda: float = 0.1,
        temperature: float = 0.1,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.supcon_lambda = supcon_lambda
        self.am_loss = AMSoftmaxLoss(weight=class_weights)
        self.supcon_loss = SupConLoss(temperature=temperature)

    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            logits: AM-Softmax logits [B, 2]
            embeddings: Normalized representations [B, D]
            labels: Ground-truth class labels [B]
            
        Returns:
            total_loss: Scalar loss tensor for backpropagation
            metrics: Dictionary of detached scalar loss values for logging
        """
        l_am = self.am_loss(logits, labels)
        l_supcon = self.supcon_loss(embeddings, labels)
        
        total_loss = l_am + self.supcon_lambda * l_supcon
        
        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_am": float(l_am.item()),
            "loss_supcon": float(l_supcon.item())
        }
        return total_loss, metrics
