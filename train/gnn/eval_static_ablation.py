"""
R2 static-inputs rung, readout (a): ablation check (litext plan Step 2).

Loads the static-inputs checkpoint and evaluates val corr/rmse twice:
  1. intact static channels (sin/cos coords + hub flag)
  2. static channels zeroed at inference

If the val-corr delta < 0.002 the network ignores the static inputs — record
and stop the rung (that IS the finding). Baseline (no-static checkpoint) corr
printed for reference.

Env: R2_CKPT (checkpoints/hetdynamics_eqvar_static/best.pt),
     R2_BASE_CKPT (checkpoints/hetdynamics_eqvar/best.pt),
     R2_SPLIT (data/splits_hetdynamics_eqvar/val)
"""
import sys, os
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("GNN_STATIC_INPUTS", "1")
from gnn_forecaster import MeshGNN, MultiRealisationDataset, run_epoch, K, DEVICE
from torch.utils.data import DataLoader

CKPT      = os.environ.get("R2_CKPT", "checkpoints/hetdynamics_eqvar_static/best.pt")
BASE_CKPT = os.environ.get("R2_BASE_CKPT", "checkpoints/hetdynamics_eqvar/best.pt")
SPLIT     = os.environ.get("R2_SPLIT", "data/splits_hetdynamics_eqvar/val")

ds = MultiRealisationDataset(SPLIT, K)
loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

model = MeshGNN(50, 50, static_in=1).to(DEVICE)
ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
model.load_state_dict(ck["model_state"])
model.eval()

m_on = run_epoch(model, loader)
print(f"static ON : corr={m_on['corr']:.4f} rmse={m_on['rmse']:.4f}")

saved = model.static_feats.clone()
model.static_feats.zero_()
m_off = run_epoch(model, loader)
model.static_feats.copy_(saved)
print(f"static OFF: corr={m_off['corr']:.4f} rmse={m_off['rmse']:.4f}")
print(f"delta corr = {m_on['corr'] - m_off['corr']:+.4f}  "
      f"({'IGNORED (<0.002) — rung stops here' if abs(m_on['corr']-m_off['corr']) < 0.002 else 'USED — continue rung readouts'})")

base = MeshGNN(50, 50, static_in=0).to(DEVICE)
ckb = torch.load(BASE_CKPT, map_location=DEVICE, weights_only=False)
base.load_state_dict(ckb["model_state"])
base.eval()
m_base = run_epoch(base, loader)
print(f"baseline (no-static ckpt): corr={m_base['corr']:.4f} rmse={m_base['rmse']:.4f}")

np.save("results/litext_r2_static_ablation.npy",
        dict(on=m_on, off=m_off, base=m_base,
             delta_corr=float(m_on['corr'] - m_off['corr'])),
        allow_pickle=True)
print("saved -> results/litext_r2_static_ablation.npy")
