"""
Resume CNN training from a checkpoint.
Usage: python3 -u resume_training.py [checkpoint_path]
Default: checkpoints/epoch_030.pt  → trains epochs 31-50
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os, glob, time

# ── import model classes from cnn_forecaster ─────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from cnn_forecaster import (
    MultiRealisationDataset, SpatioTemporalCNN,
    run_epoch, forecast_corr,
    K, BASE_CH, BATCH_SIZE, LR, EPOCHS, DEVICE, CKPT_DIR, SPLIT_DIR,
)

# ── config ────────────────────────────────────────────────────────────────────
RESUME_CKPT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(CKPT_DIR, "epoch_030.pt")

# ── datasets ─────────────────────────────────────────────────────────────────
train_ds = MultiRealisationDataset(os.path.join(SPLIT_DIR, "train"), K)
val_ds   = MultiRealisationDataset(os.path.join(SPLIT_DIR, "val"),   K)

ny = train_ds.segs[0].shape[2]
nx = train_ds.segs[0].shape[3]

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)

# ── model + optimizer ────────────────────────────────────────────────────────
model = SpatioTemporalCNN(ny=ny, nx=nx, k=K, base_ch=BASE_CH).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR / 20)

# ── load checkpoint ──────────────────────────────────────────────────────────
print(f"Loading checkpoint: {RESUME_CKPT}")
ckpt = torch.load(RESUME_CKPT, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
start_epoch = ckpt["epoch"]          # e.g. 30
history = ckpt.get("history", [])   # may or may not be present in epoch_NNN.pt

# advance the scheduler to match epoch 30 state
for _ in range(start_epoch):
    scheduler.step()

# load best val RMSE from best.pt so we can keep updating it correctly
best_ckpt_path = os.path.join(CKPT_DIR, "best.pt")
if os.path.exists(best_ckpt_path):
    best_ckpt = torch.load(best_ckpt_path, map_location="cpu")
    best_val_rmse = float(best_ckpt["val_rmse"])
else:
    best_val_rmse = float("inf")

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nResuming SpatioTemporalCNN from epoch {start_epoch}")
print(f"  Device       : {DEVICE}")
print(f"  Parameters   : {n_params:,}  ({n_params/1e6:.2f}M)")
print(f"  Best val RMSE so far: {best_val_rmse:.4f}")
print(f"  Training epochs {start_epoch + 1} → {EPOCHS}\n")

print(f"{'Epoch':>6} {'Train MSE':>10} {'Train RMSE':>11} "
      f"{'Val MSE':>9} {'Val RMSE':>10} {'Val Corr':>9}  {'LR':>8}  Time")
print("-" * 85)

for epoch in range(start_epoch + 1, EPOCHS + 1):
    t0 = time.time()

    tr = run_epoch(model, train_loader, optimizer)
    vl = run_epoch(model, val_loader)
    scheduler.step()

    elapsed = time.time() - t0
    lr_now  = scheduler.get_last_lr()[0]

    row = {
        "epoch": epoch,
        "train_mse": tr["mse"], "train_rmse": tr["rmse"],
        "val_mse":   vl["mse"], "val_rmse":   vl["rmse"],
        "val_corr":  vl["corr"], "lr": lr_now,
    }
    history.append(row)

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
        print(f"  *** new best  val RMSE={best_val_rmse:.4f}  saved best.pt ***")

    if epoch % 10 == 0:
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "history": history,
        }, os.path.join(CKPT_DIR, f"epoch_{epoch:03d}.pt"))
        print(f"  Checkpoint saved: epoch_{epoch:03d}.pt")

np.save(os.path.join(CKPT_DIR, "history.npy"), history)
print(f"\nTraining complete.")
print(f"Best val RMSE : {best_val_rmse:.4f}")
print(f"Checkpoints   → {CKPT_DIR}/")
