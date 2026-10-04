"""
Fast In-Memory ECG Dataset with Electrophysiological Data Augmentations.
"""
import torch
import numpy as np
from torch.utils.data import Dataset


class FastECGDataset(Dataset):
    """
    High-performance in-memory dataset for 12-lead ECG signals.
    
    Supports physiological augmentations:
    1. Gaussian EMG noise injection
    2. Baseline wander (respiratory drift simulation)
    3. Random lead masking (spatial electrode detachment simulation)
    """
    def __init__(self, X: np.ndarray, y: np.ndarray = None, augment: bool = False):
        self.X = torch.from_numpy(X).float() if isinstance(X, np.ndarray) else X
        self.y = torch.from_numpy(y).float() if y is not None else None
        self.augment = augment

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx].clone()
        if self.augment:
            # 1. Gaussian noise
            if torch.rand(1).item() > 0.5:
                x = x + torch.randn_like(x) * 0.03
            # 2. Baseline wander (low frequency sine wave 0.2 - 0.3 Hz)
            if torch.rand(1).item() > 0.5:
                t = torch.linspace(0, 10, x.shape[-1])
                f = 0.2 + 0.1 * torch.rand(1).item()
                wander = 0.10 * torch.sin(2 * np.pi * f * t)
                x = x + wander
            # 3. Spatial Lead Dropout: simulate single electrode detachment
            if torch.rand(1).item() > 0.7:
                drop_lead = torch.randint(0, 12, (1,)).item()
                x[drop_lead, :] = 0.0

        if self.y is not None:
            return x, self.y[idx]
        return x
