"""
Training script for LeadAttnResNet on PTB-XL ECG dataset.
"""
import os
import sys
import time
import ast
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import LeadAttnResNet
from loss import AsymmetricLoss
from dataset import FastECGDataset
from evaluate import evaluate_benchmark, run_stress_tests, extract_inter_lead_attention, TARGET_CLASSES


def train(data_dir: str, csv_path: str, epochs: int = 15, batch_size: int = 64, lr: float = 1e-3, out_dir: str = "."):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Using compute device: {device}")

    # Load metadata
    df = pd.read_csv(csv_path)
    # Check matching npy files
    df = df[df["ecg_id"].apply(lambda eid: os.path.exists(os.path.join(data_dir, f"{eid:05d}.npy")))].reset_index(drop=True)
    df["scp_dict"] = df["scp_codes"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else {})

    labels = np.zeros((len(df), len(TARGET_CLASSES)), dtype=np.float32)
    for c_idx, code in enumerate(TARGET_CLASSES):
        labels[:, c_idx] = df["scp_dict"].apply(lambda d: 1.0 if code in d else 0.0).values

    print(f"Total records: {len(df)} across {len(TARGET_CLASSES)} classes")

    # In-memory pre-loading
    print("⏳ Pre-loading signals into RAM...")
    t0 = time.time()
    signals = np.zeros((len(df), 12, 1000), dtype=np.float32)
    for i, eid in enumerate(df["ecg_id"]):
        arr = np.load(os.path.join(data_dir, f"{eid:05d}.npy"))
        signals[i] = arr.T
    print(f"✅ Loaded in {time.time()-t0:.2f}s! ({signals.nbytes / 1e6:.1f} MB)")

    # Stratified Splits (PTB-XL standard: 1-8 train, 9 val, 10 test)
    train_mask = df["strat_fold"].between(1, 8).values
    val_mask = (df["strat_fold"] == 9).values
    test_mask = (df["strat_fold"] == 10).values

    X_train, y_train = signals[train_mask], labels[train_mask]
    X_val, y_val = signals[val_mask], labels[val_mask]
    X_test, y_test = signals[test_mask], labels[test_mask]

    print(f"Splits -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    train_loader = DataLoader(FastECGDataset(X_train, y_train, augment=True), batch_size=batch_size, shuffle=True, pin_memory=(device=="cuda"))
    val_loader = DataLoader(FastECGDataset(X_val, y_val, augment=False), batch_size=batch_size * 2, shuffle=False)
    test_loader = DataLoader(FastECGDataset(X_test, y_test, augment=False), batch_size=batch_size * 2, shuffle=False)

    model = LeadAttnResNet(num_classes=len(TARGET_CLASSES), d_model=128, nhead=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)

    best_ckpt_path = os.path.join(out_dir, "best_lead_attn_model.pt")
    best_val_auroc = 0.0

    print(f"\n🏋️ Starting {epochs}-Epoch Training Loop...")
    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += loss.item() * len(x)
        scheduler.step()

        val_metrics = evaluate_benchmark(model, val_loader, device=device)
        ep_time = time.time() - t_ep
        print(f"Epoch {epoch:02d} ({ep_time:.1f}s) | Loss: {total_loss/len(X_train):.4f} | "
              f"Val AUROC: {val_metrics['macro_auroc']:.4f} | Val F1: {val_metrics['macro_f1']:.4f}", flush=True)

        if val_metrics["macro_auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["macro_auroc"]
            torch.save(model.state_dict(), best_ckpt_path)

    # Final Benchmark
    print("\n================ FINAL TEST SET BENCHMARK (FOLD 10) ================")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    test_res = evaluate_benchmark(model, test_loader, device=device)
    print(f"Macro-AUROC : {test_res['macro_auroc']:.4f}")
    print(f"Macro-F1    : {test_res['macro_f1']:.4f}")
    print("Class-wise Breakdown:")
    for code in TARGET_CLASSES:
        print(f"  {code:6s} -> AUROC: {test_res['class_auroc'][code]:.4f} | F1: {test_res['class_f1'][code]:.4f}")

    # Stress Test
    print("\n================ NOISE & STRESS TESTS ================")
    stress_res = run_stress_tests(model, X_test, y_test, device=device)
    for test_name, metrics in stress_res.items():
        if test_name != "clean_auroc":
            print(f"  {test_name:20s}: AUROC = {metrics['macro_auroc']:.4f} (Delta: {metrics['delta']:+.4f})")

    # XAI
    print("\n================ INTER-LEAD ATTENTION COUPLING (LBBB) ================")
    couplings = extract_inter_lead_attention(model, X_test, y_test, target_class_idx=2, device=device)
    for (l1, l2), weight in couplings[:5]:
        print(f"  {l1} <---> {l2} : weight = {weight:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/kaggle/working/ptbxl_100hz/ptbxl_data_signals100")
    parser.add_argument("--csv_path", type=str, default="/kaggle/working/ptbxl_database.csv")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", type=str, default=".")
    args = parser.parse_args()

    train(args.data_dir, args.csv_path, args.epochs, args.batch_size, args.lr, args.out_dir)
