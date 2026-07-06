"""
Mode-identity probes: is mode identity decodable from the pooled activations
AT ALL, or do probes fail the same way SAE features do?

For each pooled 256-dim sample (mode-weighted GNN hidden state), predict WHICH
mode j (8-way, chance = 12.5%) it came from. Split by realisation (train first
85, val last 15), matching the SAE scripts.

Probes
------
  linear/raw      softmax probe on globally-standardised activations
  linear/demeaned same, after removing each mode's TRAIN mean (identity info
                  beyond a constant offset — 'shape' rather than 'address')
  mlp/raw         256→256 ReLU →8 (nonlinear decodability)
  linear/sae      softmax probe on the mixed SAE's 512-dim codes (does the SAE
                  bottleneck preserve or destroy whatever identity info exists)

If linear/raw ≫ chance while SAE features show zero specificity, identity IS
present in the activations and the SAE simply doesn't carve features along it.
If all probes ≈ chance, the streams are statistically identical (pure content
code) and no feature method could ever separate them.

Output: probe_mode_identity.npy in --datadir.
"""

import sys, argparse
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

_ap = argparse.ArgumentParser()
_ap.add_argument("--datadir", required=True)
_ap.add_argument("--epochs", type=int, default=8)
_args = _ap.parse_args()
DATA_DIR = Path(_args.datadir)

INPUT_DIM  = 256
N_FEATURES = 512
K_TOPK     = 25
N_MODES    = 8
VAL_FRAC   = 0.15
BATCH      = 4096
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

acts_full = np.load(DATA_DIR / "activations_full.npy")   # (100, 8, T, 256)
n_real, n_modes, t_eff, d = acts_full.shape
n_val   = max(1, int(n_real * VAL_FRAC))
n_train = n_real - n_val

def flatten(a):   # (R, 8, T, d) -> X (R*8*T, d), y (R*8*T,)
    X = a.reshape(-1, d).astype(np.float32)
    y = np.tile(np.repeat(np.arange(n_modes), a.shape[2]), a.shape[0])
    return X, y.astype(np.int64)

X_tr, y_tr = flatten(acts_full[:n_train])
X_va, y_va = flatten(acts_full[n_train:])

# global standardisation (fit on train, no mode label used)
mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-8
X_tr_g, X_va_g = (X_tr - mu) / sd, (X_va - mu) / sd

# per-mode-demeaned variant (train means; removes the constant 'address' offset)
mode_means = np.stack([X_tr[y_tr == j].mean(0) for j in range(N_MODES)])
X_tr_d = ((X_tr - mode_means[y_tr]) / sd).astype(np.float32)
X_va_d = ((X_va - mode_means[y_va]) / sd).astype(np.float32)

# mixed-SAE codes (if checkpoint exists)
sae_feats = None
sae_path = DATA_DIR / "sae_mixed.pt"
if sae_path.exists():
    class TopKSAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.k = K_TOPK
            self.encoder = nn.Linear(INPUT_DIM, N_FEATURES, bias=True)
            self.decoder = nn.Linear(N_FEATURES, INPUT_DIM, bias=True)
        def encode(self, x):
            pre = self.encoder(x)
            v, i = torch.topk(pre, self.k, dim=-1)
            acts = torch.zeros_like(pre)
            acts.scatter_(-1, i, F.relu(v))
            return acts
    ckpt = torch.load(sae_path, map_location=DEVICE, weights_only=False)
    sae = TopKSAE().to(DEVICE)
    sae.encoder.weight.data = ckpt["model_state"]["encoder.weight"]
    sae.encoder.bias.data   = ckpt["model_state"]["encoder.bias"]
    sae.decoder.weight.data = ckpt["model_state"]["decoder.weight"]
    sae.decoder.bias.data   = ckpt["model_state"]["decoder.bias"]
    m_s, s_s = ckpt["act_mean"], ckpt["act_std"]

    def sae_encode(X):
        Xn = torch.from_numpy(((X - m_s) / s_s).astype(np.float32))
        out = []
        with torch.no_grad():
            for i in range(0, len(Xn), BATCH):
                out.append(sae.encode(Xn[i:i+BATCH].to(DEVICE)).cpu().numpy())
        return np.concatenate(out)
    sae_feats = (sae_encode(X_tr), sae_encode(X_va))
else:
    print(f"[note] {sae_path} not found — skipping linear/sae probe")


def train_probe(Xtr, ytr, Xva, yva, hidden=0, epochs=_args.epochs, lr=1e-3):
    din = Xtr.shape[1]
    if hidden:
        net = nn.Sequential(nn.Linear(din, hidden), nn.ReLU(),
                            nn.Linear(hidden, N_MODES)).to(DEVICE)
    else:
        net = nn.Linear(din, N_MODES).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xtr_t = torch.from_numpy(Xtr); ytr_t = torch.from_numpy(ytr)
    Xva_t = torch.from_numpy(Xva).to(DEVICE); yva_t = torch.from_numpy(yva).to(DEVICE)
    best = 0.0
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr_t))
        net.train()
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i+BATCH]
            xb, yb = Xtr_t[idx].to(DEVICE), ytr_t[idx].to(DEVICE)
            loss = F.cross_entropy(net(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pred = torch.cat([net(Xva_t[i:i+BATCH]).argmax(1)
                              for i in range(0, len(Xva_t), BATCH)])
            acc = (pred == yva_t).float().mean().item()
        best = max(best, acc)
    # per-mode accuracy at final epoch
    with torch.no_grad():
        per_mode = [(pred[yva_t == j] == j).float().mean().item() for j in range(N_MODES)]
    return best, per_mode


print(f"\nMode-identity probes — {DATA_DIR}   (chance = {1/N_MODES:.3f})")
print("=" * 74)
results = {}
runs = [("linear/raw",      X_tr_g, X_va_g, 0),
        ("linear/demeaned", X_tr_d, X_va_d, 0),
        ("mlp/raw",         X_tr_g, X_va_g, 256)]
if sae_feats is not None:
    runs.append(("linear/sae-codes", sae_feats[0], sae_feats[1], 0))

for name, xtr, xva, hid in runs:
    acc, per_mode = train_probe(xtr, y_tr, xva, y_va, hidden=hid)
    results[name] = {"acc": acc, "per_mode": per_mode}
    pm = "  ".join(f"{a:.2f}" for a in per_mode)
    print(f"  {name:<18} acc = {acc:.4f}   per-mode: {pm}")

np.save(DATA_DIR / "probe_mode_identity.npy", results)
print(f"\nResults → {DATA_DIR}/probe_mode_identity.npy")
