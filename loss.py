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


class NoiseConsistencyLoss(nn.Module):
    """
    Dual-Branch Consistency Loss as formulated in the original NC-GAT proposal:
    L_total = L_classification(clean, y) + lambda_c * L_consistency(clean, noisy)

    Enables true noise-consistency regularization without VRAM OOM on compact architectures.
    """
    def __init__(self, base_loss: nn.Module = None, lambda_consistency: float = 1.0, mode: str = "mse"):
        super().__init__()
        self.base_loss = base_loss if base_loss is not None else AsymmetricLoss()
        self.lambda_c = lambda_consistency
        self.mode = mode

    def forward(self, logits_clean: torch.Tensor, logits_noisy: torch.Tensor, y: torch.Tensor):
        loss_cls = self.base_loss(logits_clean, y)
        p_clean = torch.sigmoid(logits_clean)
        p_noisy = torch.sigmoid(logits_noisy)

        if self.mode == "mse":
            loss_cons = F.mse_loss(p_noisy, p_clean)
        elif self.mode == "kl":
            loss_cons = F.kl_div(torch.log(p_noisy.clamp(min=1e-8)), p_clean, reduction="batchmean")
        else:
            loss_cons = F.mse_loss(p_noisy, p_clean)

        total_loss = loss_cls + self.lambda_c * loss_cons
        return total_loss, loss_cls, loss_cons

