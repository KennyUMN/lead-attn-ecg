"""
LeadAttnResNet: 12-Lead Electrocardiogram Neural Network with Inter-Lead Attention.

Replaces broken 12-node GAT with native PyTorch Multi-Head Self-Attention:
- Stem: 1D-ResNet with shared weights across leads via batch folding [B * 12, 1, T] -> [B, 12, d_model]
- Spatial Modeling: Native inter-lead Multi-Head Attention [B, 12, d_model] -> [B, 12, 12] attention map (Native XAI)
- Classifier: Global lead-pooling + MLP for multi-label arrhythmia prediction
- Total Parameters: 163,622 (~640 KB FP32)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    """1D Residual convolutional block with optional downsampling shortcut."""
    def __init__(self, channels: int, kernel_size: int = 7, stride: int = 2):
        super().__init__()
        p = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, stride=stride, padding=p)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, stride=1, padding=p)
        self.bn2 = nn.BatchNorm1d(channels)
        self.shortcut = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, stride=stride),
            nn.BatchNorm1d(channels)
        ) if stride != 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + res)


class LeadAttnResNet(nn.Module):
    """
    Lead-Attention 1D-ResNet for 12-lead ECG representation learning.
    
    Args:
        num_classes: Number of multi-label diagnostic targets (default: 6).
        d_model: Channel dimension for inter-lead tokens (default: 128).
        nhead: Number of attention heads in multi-head attention (default: 4).
    """
    def __init__(self, num_classes: int = 6, d_model: int = 128, nhead: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        # Shared 1D-ResNet temporal feature extractor per lead
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            ResidualBlock1D(32, stride=2),
            nn.Conv1d(32, 64, kernel_size=1),
            ResidualBlock1D(64, stride=2),
            nn.Conv1d(64, d_model, kernel_size=1),
            nn.AdaptiveAvgPool1d(1)
        )

        # Inter-lead spatial attention: models electrical dipole projection across leads
        self.lead_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

        # Multi-label classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [Batch, 12, Time_steps] (e.g. [B, 12, 1000]).
            return_attn: If True, returns (logits, attention_weights).
        
        Returns:
            logits: Unnormalized logits [Batch, num_classes].
            attn_weights: Inter-lead attention weights [Batch, 12, 12] (if return_attn=True).
        """
        B, N, T = x.shape
        # Batch folding: process all 12 leads through shared 1D convs in a single GPU kernel
        x_folded = x.view(B * N, 1, T)
        lead_tokens = self.stem(x_folded).view(B, N, -1)  # [B, 12, d_model]

        # Inter-lead attention
        attn_out, attn_weights = self.lead_attention(
            lead_tokens, lead_tokens, lead_tokens, need_weights=True, average_attn_weights=True
        )
        tokens = self.norm(lead_tokens + attn_out)

        # Global average pooling over lead dimension
        pooled = tokens.mean(dim=1)  # [B, d_model]
        logits = self.classifier(pooled)

        if return_attn:
            return logits, attn_weights
        return logits
