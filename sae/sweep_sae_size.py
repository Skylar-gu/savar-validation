"""
Dictionary-size sweep for the mixed-mode SAE: does alignment improve with
FEWER features (less feature splitting) or degrade (features forced onto the
high-variance PC0 carrier)?

For each (n_features, k) config: train a mixed SAE (same recipe as
train_sae_mixed.py, shorter schedule), then per mode report the best feature's
|r| with Z_j, specificity, and live-feature count.

Output: sweep_sae_size.npy in --datadir; checkpoints are NOT kept.
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
_ap.add_argument("--epochs", type=int, default=40)
_args = _ap.parse_args()
DATA_DIR = Path(_args.datadir)

INPUT_DIM = 256
N_MODES   = 8
LR        = 1e-3
BATCH     = 256
VAL_FRAC  = 0.15
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIGS = [(32, 6), (64, 8), (128, 12), (256, 25), (512, 25), (1024, 25)]

RESAMPLE_INTERVAL = 1000
DEAD_WINDOW       = 500


class TopKSAE(nn.Module):
    def __init__(self, n_features, k):
        super().__init__()
        self.k = k
        self.n_features = n_features
        self.encoder = nn.Linear(INPUT_DIM, n_features, bias=True)
        self.decoder = nn.Linear(n_features, INPUT_DIM, bias=True)
        nn.init.normal_(self.decoder.weight, std=1.0 / INPUT_DIM ** 0.5)
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)
            self.encoder.weight.data = self.decoder.weight.data.T.clone()
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x):
        pre = self.encoder(x)
        v, i = torch.topk(pre, self.k, dim=-1)
        a = torch.zeros_like(pre)
        a.scatter_(-1, i, F.relu(v))
        return a

    def forward(self, x):
        a = self.encode(x)
        return a, self.decoder(a)

    @torch.no_grad()
    def normalise_decoder(self):
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)


def pearson_col(A, b):
    A = A.astype(np.float64); b = b.astype(np.float64)
    A_c = A - A.mean(0); b_c = b - b.mean()
    norms = np.linalg.norm(A_c, axis=0) + 1e-12
    return (A_c.T @ b_c) / (norms * (np.linalg.norm(b_c) + 1e-12))


acts_full = np.load(DATA_DIR / "activations_full.npy")
Z_full    = np.load(DATA_DIR / "Z_full.npy")
CEILINGS  = list(np.load(DATA_DIR / "ceilings.npy"))
n_real, n_modes, t_eff, d = acts_full.shape
n_val   = max(1, int(n_real * VAL_FRAC))
n_train = n_real - n_val

train_x = acts_full[:n_train].reshape(-1, d).astype(np.float32)
val_x   = acts_full[n_train:].reshape(-1, d).astype(np.float32)
mean_g  = train_x.mean(0)
std_g   = train_x.std(0) + 1e-8
X_train = torch.from_numpy((train_x - mean_g) / std_g).to(DEVICE)
X_val   = torch.from_numpy((val_x   - mean_g) / std_g).to(DEVICE)

# eval streams: full data per mode, normalised once
mode_X = [torch.from_numpy(((acts_full[:, j].reshape(-1, d) - mean_g) / std_g)
                           .astype(np.float32)) for j in range(N_MODES)]
mode_Z = [Z_full[:, j].reshape(-1) for j in range(N_MODES)]

results = {}
print(f"Dictionary-size sweep — {DATA_DIR}   ({_args.epochs} epochs each)")

for n_feat, k in CONFIGS:
    torch.manual_seed(0)
    sae = TopKSAE(n_feat, k).to(DEVICE)
    opt = torch.optim.Adam(sae.parameters(), lr=LR)
    last_fired  = torch.zeros(n_feat, dtype=torch.long, device=DEVICE)
    global_step = 0
    steps = len(X_train) // BATCH
    # eval the best-val model, not the final one: the re-init resample causes
    # transient recon spikes, and an epoch can end mid-spike (worse at large N)
    best_val, best_state = float("inf"), None

    for ep in range(_args.epochs):
        perm = torch.randperm(len(X_train), device=DEVICE)
        for i in range(steps):
            batch = X_train[perm[i*BATCH:(i+1)*BATCH]]
            a, rec = sae(batch)
            loss = F.mse_loss(rec, batch)
            opt.zero_grad(); loss.backward(); opt.step()
            sae.normalise_decoder()
            fired = (a.detach() > 0).any(0)
            last_fired[fired] = global_step
            global_step += 1
            if global_step % RESAMPLE_INTERVAL == 0:
                dead = ((global_step - last_fired) > DEAD_WINDOW)
                if dead.any():   # simple re-init resample (no residual targeting)
                    di = dead.nonzero(as_tuple=True)[0]
                    smp = X_train[torch.randperm(len(X_train), device=DEVICE)[:len(di)]]
                    with torch.no_grad():
                        w = F.normalize(smp, dim=1)
                        sae.encoder.weight.data[di] = w
                        sae.decoder.weight.data[:, di] = w.T
                        sae.encoder.bias.data[di] = 0.0
                    # reset Adam state for resampled features — stale second
                    # moments on re-initialized weights diverge at large N
                    for p, sl in ((sae.encoder.weight, np.s_[di]),
                                  (sae.decoder.weight, np.s_[:, di])):
                        st = opt.state.get(p)
                        if st:
                            st["exp_avg"][sl] = 0.0
                            st["exp_avg_sq"][sl] = 0.0
        with torch.no_grad():
            vmse_ep = sum(F.mse_loss(sae(X_val[i:i+65536])[1], X_val[i:i+65536],
                                     reduction="sum").item()
                          for i in range(0, len(X_val), 65536)) / X_val.numel()
        if vmse_ep < best_val:
            best_val = vmse_ep
            best_state = {k: v.detach().clone() for k, v in sae.state_dict().items()}

    sae.load_state_dict(best_state)
    sae.eval()
    # per-mode alignment of this dictionary
    per_mode = {}
    enc = []
    with torch.no_grad():
        for j in range(N_MODES):
            fa = torch.cat([sae.encode(mode_X[j][i:i+8192].to(DEVICE)).cpu()
                            for i in range(0, len(mode_X[j]), 8192)]).numpy()
            enc.append(fa)
    C = [pearson_col(enc[j], mode_Z[j]) for j in range(N_MODES)]
    for j in range(N_MODES):
        best = int(np.abs(C[j]).argmax())
        max_r = float(np.abs(C[j])[best])
        cross = max(float(np.abs(C[k])[best]) for k in range(N_MODES) if k != j)
        per_mode[j] = {"best": best, "max_r": max_r, "spec": max_r - cross,
                       "frac": max_r / CEILINGS[j]}
    live = int(sum((fa > 0).any(0).sum() for fa in enc[:1]))  # live on mode-0 stream
    live_all = int(np.union1d(*[np.where((enc[j] > 0).any(0))[0] for j in (0, 7)]).size) \
        if N_MODES > 1 else live
    with torch.no_grad():
        vr = torch.cat([sae(mode_X[0][i:i+8192].to(DEVICE))[1].cpu()
                        for i in range(0, len(mode_X[0]), 8192)])
        vmse = F.mse_loss(vr, mode_X[0]).item()

    results[(n_feat, k)] = {"per_mode": per_mode, "val_mse_m0": vmse,
                            "best_val_mse": best_val}
    rs = "  ".join(f"{per_mode[j]['max_r']:.3f}" for j in range(N_MODES))
    sp = "  ".join(f"{per_mode[j]['spec']:+.2f}" for j in range(4, 8))
    print(f"\n  N={n_feat:<5} K={k:<3}  reconMSE(m0)={vmse:.4f}  bestVal={best_val:.4f}")
    print(f"    best|r| X0..X7 : {rs}")
    print(f"    spec  X4..X7   : {sp}")

np.save(DATA_DIR / "sweep_sae_size.npy", {str(k): v for k, v in results.items()})
print(f"\nResults → {DATA_DIR}/sweep_sae_size.npy")
