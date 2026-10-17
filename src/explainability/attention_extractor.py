"""
Attention Extractor for Phase 5: Explainability & Attention Signature Analysis.
Extracts fine-grained frame attention (4 heads x 100 frames) and coarse-grained segment
attention (2 heads x 8 segments) from the trained DualStreamModel.
"""

from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import numpy as np

from src.config import config, MasterConfig
from src.features.extractor import LFCCExtractor
from src.models.dual_stream_model import DualStreamModel


class AttentionExtractor:
    """
    Extracts and formats multi-granularity temporal attention representations
    from a trained DualStreamModel for explainability analysis.
    """
    def __init__(
        self,
        model: Optional[DualStreamModel] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        cfg: MasterConfig = config
    ):
        self.cfg = cfg
        self.device = torch.device(cfg.training.device)
        
        # 1. Initialize Model
        if model is not None:
            self.model = model
        else:
            self.model = DualStreamModel(cfg.model)
            ckpt_path = Path(checkpoint_path) if checkpoint_path else Path(cfg.paths.checkpoint_dir) / "best_model.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Ensure Phase 4 training finished.")
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            
        self.model.to(self.device).eval()
        self.lfcc_extractor = LFCCExtractor(cfg.features, cfg.audio).to(self.device).eval()

    def extract_from_waveform(self, waveform: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Extracts multi-granularity attention weights for a single audio waveform or batch.
        
        Args:
            waveform: Tensor of shape [1, 64000] or [B, 1, 64000]
            
        Returns:
            Dictionary containing:
                - 'probs': Softmax probability array [B, 2] (class 1 is bonafide)
                - 'pred_label': Predicted label [B] (0=spoof, 1=bonafide)
                - 'fine_frame_weights': Saliency scores per frame [B, 100]
                - 'fine_heads_matrix': Pairwise self-attention matrices [B, 4, 100, 100]
                - 'coarse_segment_weights': Saliency scores per segment [B, 8]
                - 'coarse_heads_matrix': Pairwise segment attention matrices [B, 2, 8, 8]
                - 'lfcc': Extracted LFCC spectrogram [B, 60, 401]
        """
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)  # [1, 1, 64000]
            
        waveform = waveform.to(self.device)
        
        with torch.no_grad():
            lfcc = self.lfcc_extractor(waveform)  # [B, 60, 401]
            logits, embedding, fine_weights, coarse_weights = self.model(lfcc)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred_label = np.argmax(probs, axis=1)
            
            maps = self.model.get_attention_maps()
            
            return {
                "probs": probs,
                "pred_label": pred_label,
                "fine_frame_weights": fine_weights.cpu().numpy(),
                "fine_heads_matrix": maps["fine_heads_matrix"].cpu().numpy(),
                "coarse_segment_weights": coarse_weights.cpu().numpy(),
                "coarse_heads_matrix": maps["coarse_heads_matrix"].cpu().numpy(),
                "lfcc": lfcc.cpu().numpy()
            }

    def compute_temporal_saliency(self, fine_weights: np.ndarray, num_frames: int = 401) -> np.ndarray:
        """
        Interpolates the 100-step fine-grained pooling weights back to the original
        401-frame LFCC time axis for direct overlay on spectrograms.
        
        Args:
            fine_weights: Array of shape [B, 100] or [100]
            num_frames: Original spectrogram frame count (401)
            
        Returns:
            interpolated_saliency: Array of shape [B, 401] or [401]
        """
        single = (fine_weights.ndim == 1)
        if single:
            fine_weights = np.expand_dims(fine_weights, 0)
            
        B, T = fine_weights.shape
        x_orig = np.linspace(0, 1, T)
        x_target = np.linspace(0, 1, num_frames)
        
        saliency = np.zeros((B, num_frames), dtype=np.float32)
        for b in range(B):
            saliency[b] = np.interp(x_target, x_orig, fine_weights[b])
            saliency[b] = saliency[b] / (saliency[b].sum() + 1e-8)  # re-normalize
            
        return saliency[0] if single else saliency
