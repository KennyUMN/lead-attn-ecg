"""
Evaluation & Clinical Stress-Testing Suite for LeadAttnResNet.
"""
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from dataset import FastECGDataset


TARGET_CLASSES = ["NORM", "AFIB", "CLBBB", "1AVB", "SBRAD", "STACH"]
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def evaluate_benchmark(model: torch.nn.Module, loader: DataLoader, device: str = "cuda") -> Dict:
    """Computes Macro and Per-Class AUROC and F1 score."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = torch.sigmoid(model(x)).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(y.numpy())

    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    bin_preds = (preds >= 0.5).astype(int)

    f1s, aurocs = [], []
    for c in range(targets.shape[1]):
        if len(np.unique(targets[:, c])) > 1:
            auroc = roc_auc_score(targets[:, c], preds[:, c])
            f1 = f1_score(targets[:, c], bin_preds[:, c], zero_division=0)
            aurocs.append(auroc)
            f1s.append(f1)
        else:
            aurocs.append(0.5)
            f1s.append(0.0)

    return {
        "macro_auroc": float(np.mean(aurocs)),
        "macro_f1": float(np.mean(f1s)),
        "class_auroc": {TARGET_CLASSES[i]: round(aurocs[i], 4) for i in range(len(TARGET_CLASSES))},
        "class_f1": {TARGET_CLASSES[i]: round(f1s[i], 4) for i in range(len(TARGET_CLASSES))}
    }


def run_stress_tests(model: torch.nn.Module, X_test: np.ndarray, y_test: np.ndarray, device: str = "cuda") -> Dict:
    """
    Evaluates robustness under simulated clinical degradation:
    1. Gaussian EMG noise (sigma=0.10)
    2. Severe baseline wander (0.25 Hz respiration)
    3. Precordial lead dropout (V1, V5, V6 detached)
    """
    clean_loader = DataLoader(FastECGDataset(X_test, y_test), batch_size=128, shuffle=False)
    clean_metrics = evaluate_benchmark(model, clean_loader, device=device)
    clean_auroc = clean_metrics["macro_auroc"]

    # 1. Gaussian Noise
    X_noise = X_test + np.random.randn(*X_test.shape).astype(np.float32) * 0.10
    noise_res = evaluate_benchmark(model, DataLoader(FastECGDataset(X_noise, y_test), batch_size=128, shuffle=False), device=device)

    # 2. Baseline Wander
    t = np.linspace(0, 10, X_test.shape[-1]).astype(np.float32)
    wander = (0.20 * np.sin(2 * np.pi * 0.25 * t)).reshape(1, 1, -1)
    wander_res = evaluate_benchmark(model, DataLoader(FastECGDataset(X_test + wander, y_test), batch_size=128, shuffle=False), device=device)

    # 3. Missing Precordial Leads
    X_masked = X_test.copy()
    X_masked[:, [6, 10, 11], :] = 0.0  # V1, V5, V6
    mask_res = evaluate_benchmark(model, DataLoader(FastECGDataset(X_masked, y_test), batch_size=128, shuffle=False), device=device)

    return {
        "clean_auroc": clean_auroc,
        "gaussian_noise": {
            "macro_auroc": noise_res["macro_auroc"],
            "delta": round(noise_res["macro_auroc"] - clean_auroc, 4)
        },
        "baseline_wander": {
            "macro_auroc": wander_res["macro_auroc"],
            "delta": round(wander_res["macro_auroc"] - clean_auroc, 4)
        },
        "missing_precordial": {
            "macro_auroc": mask_res["macro_auroc"],
            "delta": round(mask_res["macro_auroc"] - clean_auroc, 4)
        }
    }


def extract_inter_lead_attention(model: torch.nn.Module, X: np.ndarray, y: np.ndarray, target_class_idx: int = 2, device: str = "cuda") -> List[Tuple]:
    """Extracts top inter-lead spatial attention couplings for a given class."""
    model.eval()
    indices = np.where(y[:, target_class_idx] == 1.0)[0][:30]
    if len(indices) == 0:
        return []

    with torch.no_grad():
        x_sub = torch.from_numpy(X[indices]).float().to(device)
        _, attn = model(x_sub, return_attn=True)  # [B, 12, 12]
        mean_attn = attn.mean(dim=0).cpu().numpy()

    couplings = []
    for i in range(12):
        for j in range(12):
            if i != j:
                couplings.append(((LEAD_NAMES[i], LEAD_NAMES[j]), float(mean_attn[i, j])))
    couplings.sort(key=lambda x: x[1], reverse=True)
    return couplings[:10]
