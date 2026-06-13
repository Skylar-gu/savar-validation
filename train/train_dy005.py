"""
CNN training for D_y = 0.05 experiment.
Same architecture as cnn_forecaster.py; uses data/splits_dy005/ and checkpoints_dy005/.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os, time

sys.path.insert(0, os.path.dirname(__file__))
from cnn_forecaster import (
    MultiRealisationDataset, SpatioTemporalCNN,
    run_epoch, K, BASE_CH, BATCH_SIZE, LR, EPOCHS, DEVICE,
)

SPLIT_DIR = os.path.join("data", "splits_dy005")
CKPT_DIR  = "checkpoints_dy005"
os.makedirs(CKPT_DIR, exist_ok=True)

# ── datasets ─────────────────────────────────────────────────────────────────
train_ds = MultiRealisationDataset(os.path.join(SPLIT_DIR, "train"), K)
val_ds   = MultiRealisationDataset(os.path.join(SPLIT_DIR, "val"),   K)

ny = train_ds.segs[0].shape[2]
nx = train_ds.segs[0].shape[3]

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)

# ── model ────────────────────────────────────────────────────────────────────
model     = SpatioTemporalCNN(ny=ny, nx=nx, k=K, base_ch=BASE_CH).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR / 20)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nSpatioTemporalCNN  [D_y = 0.05 × I_L experiment]")
print(f"  Device     : {DEVICE}")
print(f"  Parameters : {n_params:,}  ({n_params/1e6:.2f}M)")
print(f"  Train set  : {len(train_ds):,} windows")
print(f"  Val set    : {len(val_ds):,} windows")
print(f"  Epochs     : {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LR}\n")

print(f"{'Epoch':>6} {'Train MSE':>10} {'Train RMSE':>11} "
      f"{'Val MSE':>9} {'Val RMSE':>10} {'Val Corr':>9}  {'LR':>8}  Time")
print("-" * 85)

best_val_rmse = float("inf")
history = []

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    tr = run_epoch(model, train_loader, optimizer)
    vl = run_epoch(model, val_loader)
    scheduler.step()

    elapsed = time.time() - t0
    lr_now  = scheduler.get_last_lr()[0]

    history.append({
        "epoch": epoch,
        "train_mse": tr["mse"], "train_rmse": tr["rmse"],
        "val_mse":   vl["mse"], "val_rmse":   vl["rmse"],
        "val_corr":  vl["corr"], "lr": lr_now,
    })

    print(f"{epoch:>6}   {tr['mse']:>9.4f}   {tr['rmse']:>10.4f}   "
          f"{vl['mse']:>8.4f}   {vl['rmse']:>9.4f}   "
          f"{vl['corr']:>8.4f}  {lr_now:>8.2e}  {elapsed:.1f}s")

    if vl["rmse"] < best_val_rmse:
        best_val_rmse = vl["rmse"]
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_rmse": best_val_rmse,
        }, os.path.join(CKPT_DIR, "best.pt"))
        print(f"  *** new best  val RMSE={best_val_rmse:.4f} ***")

    if epoch % 10 == 0:
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "history": history,
        }, os.path.join(CKPT_DIR, f"epoch_{epoch:03d}.pt"))
        print(f"  Checkpoint: epoch_{epoch:03d}.pt")

np.save(os.path.join(CKPT_DIR, "history.npy"), history)
print(f"\nTraining complete.")
print(f"Best val RMSE : {best_val_rmse:.4f}")
print(f"Checkpoints   → {CKPT_DIR}/")
