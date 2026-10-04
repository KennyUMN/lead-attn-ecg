"""
Asymmetric Loss for Multi-Label Learning with Extreme Imbalance.

Reference:
    Ridnik et al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021.
    Addresses extreme class imbalance (e.g. 18:1 ratio of NORM to CLBBB)
    by discounting the gradient from easy negative background samples.
"""
import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for multi-label classification.
    
    Args:
        gamma_neg: Focusing parameter for negative samples (default: 4).
        gamma_pos: Focusing parameter for positive samples (default: 1).
        clip: Probability margin clipping for negative samples (default: 0.05).
        eps: Small numerical constant for log stability (default: 1e-8).
    """
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward computation of ASL.
        
        Args:
            x: Raw model logits [Batch, num_classes].
            y: Ground-truth binary targets [Batch, num_classes].
        
        Returns:
            Scalar loss averaged over the batch.
        """
        p = torch.sigmoid(x)
        p_neg = (1.0 - p + self.clip).clamp(max=1.0) if self.clip > 0 else (1.0 - p)
        loss_pos = y * torch.log(p.clamp(min=self.eps)) * ((1.0 - p) ** self.gamma_pos)
        loss_neg = (1.0 - y) * torch.log(p_neg.clamp(min=self.eps)) * (p ** self.gamma_neg)
        return -(loss_pos + loss_neg).sum(dim=-1).mean()
