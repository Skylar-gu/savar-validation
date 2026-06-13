"""
Decompose each RAW-diurnal SAE best-feature's alignment into what it actually
tracks: diurnal cycle vs annual cycle vs dynamics residual. Answers "are the
'strong' features tracking the diurnal or the annual cycle, or real dynamics?"

For each mode j:
  - load SAE_j, encode mode-j activations, take its best feature
  - partition Z_j into intercept + diurnal(1 harm) + annual(2 harm) + residual
    via joint harmonic regression on absolute time (t = K + window index)
  - report Pearson r of the best feature with Z_total / diurnal / annual / residual
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

_ap = argparse.ArgumentParser()
_ap.add_argument("--deseason", action="store_true",
                 help="Decompose deseasonalized features (expect diurnal/annual≈0)")
_args = _ap.parse_args()
DATA = Path("sae_data_diurnal_deseason" if _args.deseason else "sae_data_diurnal")
print(f"Decomposing features from {DATA}/\n")
N_MODES, INPUT_DIM, N_FEATURES, K_TOPK = 8, 256, 512, 25
K, P_D, P_A = 3, 4, 1461
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TopKSAE(nn.Module):
    def __init__(s, d, n, k):
        super().__init__(); s.k = k
        s.encoder = nn.Linear(d, n); s.decoder = nn.Linear(n, d)
    def encode(s, x):
        pre = s.encoder(x); v, i = torch.topk(pre, s.k, dim=-1)
        a = torch.zeros_like(pre); a.scatter_(-1, i, F.relu(v)); return a


def r(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


acts = np.load(DATA / "activations_full.npy")   # (R, M, T, C)
Z    = np.load(DATA / "Z_full.npy")             # (R, M, T)
align = np.load(DATA / "alignment_per_mode.npy", allow_pickle=True).item()
nR, nM, T, d = acts.shape

# joint harmonic design on absolute time, tiled across realisations
t = K + np.arange(T)
cols = [np.ones(T)]
wd = 2*np.pi*t/P_D; cols += [np.cos(wd), np.sin(wd)]                  # diurnal (cols 1:3)
for h in (1, 2):
    wa = 2*np.pi*h*t/P_A; cols += [np.cos(wa), np.sin(wa)]           # annual (cols 3:7)
Dm = np.tile(np.stack(cols, 1), (nR, 1))                             # (R*T, 7)
Dpinv = np.linalg.pinv(Dm)

print(f"  {'Mode':<5} {'best r(Z)':>9} {'r diurnal':>10} {'r annual':>9} {'r dynamics':>11}")
print(f"  {'-'*48}")
for j in range(N_MODES):
    ck = torch.load(DATA / f"sae_mode_{j}.pt", map_location=DEVICE, weights_only=False)
    sae = TopKSAE(INPUT_DIM, N_FEATURES, K_TOPK).to(DEVICE)
    sae.load_state_dict(ck["model_state"]); sae.eval()
    mean_j, std_j = ck["act_mean"], ck["act_std"]
    bf = align[j]["best_feat"]

    Aj = acts[:, j].reshape(-1, d)
    Xn = torch.from_numpy(((Aj - mean_j) / std_j).astype(np.float32)).to(DEVICE)
    with torch.no_grad():
        feat = np.concatenate([sae.encode(Xn[i:i+4096])[:, bf].cpu().numpy()
                               for i in range(0, len(Xn), 4096)])

    zj = Z[:, j].reshape(-1).astype(np.float64)
    beta = Dpinv @ zj
    diurnal = Dm[:, 1:3] @ beta[1:3]
    annual  = Dm[:, 3:7] @ beta[3:7]
    dynamics = zj - Dm @ beta
    print(f"  X{j:<4} {r(feat, zj):>9.3f} {r(feat, diurnal):>10.3f} "
          f"{r(feat, annual):>9.3f} {r(feat, dynamics):>11.3f}")
