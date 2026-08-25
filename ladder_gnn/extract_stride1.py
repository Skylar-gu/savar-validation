"""Stride-1 W-pooled activations of the MeshGNN (last MP layer H) for ALL 100 eqvar
realisations. Mirrors pcmci/explain_activation_collapse.py (stride-1 is required:
stride-5 extraction made every true lag unmatchable, Follow-up 1). Z aligned as in
sae_data/*/Z_full.npy: Z[r,j,t] = latent[r,j,t+K] (forecast target of window t).
Writes OUT/activations_stride1_all.npy (100, 8, 2397, 256) and Z_stride1_all.npy."""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, torch
from common import *
sys.path.insert(0, str(ROOT / "train" / "gnn"))
from gnn_forecaster import MeshGNN, K, HIDDEN

paths = sorted(DATA_DIR.glob("realisation_*.npz"))
assert len(paths) == 100
d0 = np.load(paths[0]); T_TOTAL = d0["observations"].shape[1]; T_EFF = T_TOTAL - K
device = torch.device("cuda")
model = MeshGNN(ny=NY, nx=NX, k=K).to(device)
ck = torch.load(CKPT, map_location=device); model.load_state_dict(ck["model_state"]); model.eval()
print(f"MeshGNN {CKPT}  val_rmse={ck.get('val_rmse', float('nan')):.4f}  T_EFF={T_EFF}")
cap = {}
model.layers[-1].register_forward_hook(lambda m, i, o: cap.update(act=o))
acts = np.lib.format.open_memmap(ACTS_CACHE, mode="w+", dtype=np.float32,
                                 shape=(100, N_MODES, T_EFF, HIDDEN))
Z = np.empty((100, N_MODES, T_EFF), np.float32)
BS = 128; t0 = time.time()
with torch.no_grad():
    for ri, p in enumerate(paths):
        d = np.load(p)
        obs = torch.from_numpy(d["observations"].astype(np.float32).T.reshape(T_TOTAL, NY, NX))
        W_t = torch.from_numpy(d["W"].astype(np.float32)).to(device)
        win = torch.stack([obs[t:t + K] for t in range(T_EFF)])
        out = []
        for i in range(0, T_EFF, BS):
            model(win[i:i + BS].to(device))
            out.append(torch.einsum("jl,blc->bjc", W_t, cap["act"]).cpu().numpy())
        acts[ri] = np.concatenate(out, 0).transpose(1, 0, 2)
        Z[ri] = d["latent_states"][:, K:K + T_EFF]
        if (ri + 1) % 10 == 0:
            print(f"  [{ri+1}/100] {time.time()-t0:.0f}s")
acts.flush(); np.save(Z_CACHE, Z)
# gate: the 44 cached stride-1 realisations from Follow-up 1 must be reproduced
old = np.load(SAE_DIR / "activations_stride1_sel.npy", mmap_mode="r")
idx = np.load(SAE_DIR / "activations_stride1_sel_idx.npy")
err = max(float(np.abs(np.asarray(old[k]) - acts[ri]).max()) for k, ri in enumerate(idx[:3]))
print(f"GATE vs Follow-up-1 cache (3 reals): max|diff| = {err:.2e}")
print(f"finite: {np.isfinite(acts).all()}  saved {ACTS_CACHE} {acts.shape}  [{time.time()-t0:.0f}s]")
