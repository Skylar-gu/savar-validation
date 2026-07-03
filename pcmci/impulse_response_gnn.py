"""
Block B — Impulse-response dynamical test of the frozen eqvar MeshGNN
(Hakim–Masanam arm: probe the trained emulator with structured perturbations,
no internals needed).

Protocol
--------
1. Load frozen checkpoints_hetdynamics_eqvar/best.pt (MeshGNN, gcn mode, emb=0).
2. From held-out test windows (K=3 frames), add an impulse d*pattern_j to the
   LAST input frame, pattern_j = W_plus[:, j] (latent->pixel map of mode j),
   scaled so the impulse's peak equals IMP_SIGMA x pixel std.
3. Autoregressive rollout N_STEPS steps, perturbed vs unperturbed; response
   dX_t averaged over >= N_WINDOWS windows (projection is linear so we average
   the dX maps, then project).
4. R[i, j, tau] = W[i] . mean_dX[j, tau]  (response of mode i at lag tau to an
   impulse in mode j). Edge j->i if max_tau |R[i,j,tau]| exceeds a
   permutation-null threshold (pixel-permuted projection patterns), i != j.
5. Score directed-edge F1 against the ground-truth cross-edge set (pair level,
   marginalizing lags — the rollout response of a lag-l edge peaks near tau=l,
   which we report as best-lag agreement). Baseline: PCMCI+ on Z F1=0.823.
6. Self-response e-folding time per mode (log-linear fit of |R[j,j,tau]|) vs
   the designed phi timescales tau_j = -1/ln phi_j.

Env: IR_SPLIT (default data/splits_hetdynamics_eqvar/test), IR_CKPT,
     IR_NWIN (default 240), IR_NSTEPS (12), IR_SIGMA (1.0), IR_NPERM (1000),
     IR_ALPHA (0.01)
Output: results/impulse_response.npy
"""

import sys, os, glob
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.gnn_forecaster import MeshGNN, K

SPLIT   = os.environ.get("IR_SPLIT", "data/splits_hetdynamics_eqvar/test")
CKPT    = os.environ.get("IR_CKPT", "checkpoints_hetdynamics_eqvar/best.pt")
N_WIN   = int(os.environ.get("IR_NWIN", 240))
N_STEPS = int(os.environ.get("IR_NSTEPS", 12))
SIGMA   = float(os.environ.get("IR_SIGMA", 1.0))
N_PERM  = int(os.environ.get("IR_NPERM", 1000))
ALPHA   = float(os.environ.get("IR_ALPHA", 0.01))
OUT     = os.environ.get("IR_OUT", "results/impulse_response.npy")
PHI     = np.array([0.15, 0.30, 0.42, 0.55, 0.68, 0.78, 0.86, 0.92])
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rng = np.random.default_rng(0)

# ── model ─────────────────────────────────────────────────────────────────────
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
model = MeshGNN(50, 50).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
model.adjacency()

# ── data + ground truth ───────────────────────────────────────────────────────
paths = sorted(glob.glob(os.path.join(SPLIT, "realisation_*.npz")))
d0 = np.load(paths[0])
W, W_plus = d0["W"].astype(np.float64), d0["W_plus"].astype(np.float64)
N_MODES = W.shape[0]
fine_edges = d0["fine_edges"]
gt_pairs = {(int(c), int(e)) for c, e, l, v in fine_edges if int(c) != int(e)}
gt_lag = {}
for c, e, l, v in fine_edges:
    c, e, l = int(c), int(e), int(l)
    if c != e:
        gt_lag.setdefault((c, e), []).append(l)

# windows: sample realisations round-robin, one random window each
obs_list = [np.load(p)["observations"] for p in paths]
pix_std = float(np.concatenate([o.reshape(len(o), -1) for o in obs_list[:10]]).std())
print(f"model={CKPT}  split={SPLIT}  ({len(paths)} realisations)  pixel std={pix_std:.4f}")
print(f"impulse peak = {SIGMA} x pixel std;  {N_WIN} windows x {N_MODES} impulses, "
      f"{N_STEPS}-step rollout")

# impulse fields: (8, 2500), peak amplitude SIGMA*pix_std
patterns = W_plus / np.abs(W_plus).max(0, keepdims=True)          # unit peak
impulses = (SIGMA * pix_std * patterns.T).astype(np.float32)      # (8, L)

windows = []
ri = 0
while len(windows) < N_WIN:
    o = obs_list[ri % len(obs_list)]
    t0 = rng.integers(0, len(o) - K - N_STEPS)
    windows.append(o[t0:t0 + K])
    ri += 1
windows = np.stack(windows)                                       # (N_WIN, K, 50, 50)


@torch.no_grad()
def rollout(x0):
    """x0: (B, K, 50, 50) tensor -> stacked predictions (B, N_STEPS, 50, 50)."""
    x = x0
    outs = []
    for _ in range(N_STEPS):
        y = model(x)                                              # (B, 1, 50, 50)
        outs.append(y[:, 0])
        x = torch.cat([x[:, 1:], y], dim=1)
    return torch.stack(outs, dim=1)                               # (B, N_STEPS, 50, 50)


# accumulate mean dX per (impulse mode, lag): (8, N_STEPS, 2500)
mean_dX = np.zeros((N_MODES, N_STEPS, 2500))
BS = 24
for b0 in range(0, N_WIN, BS):
    wb = torch.from_numpy(windows[b0:b0 + BS]).float().to(DEVICE)  # (b, K, 50, 50)
    base = rollout(wb).cpu().numpy()                               # (b, S, 50, 50)
    for j in range(N_MODES):
        wp = wb.clone()
        wp[:, -1] += torch.from_numpy(impulses[j].reshape(50, 50)).to(DEVICE)
        pert = rollout(wp).cpu().numpy()
        dX = (pert - base).reshape(len(wb), N_STEPS, -1)           # (b, S, L)
        mean_dX[j] += dX.sum(0)
mean_dX /= N_WIN

# latent response R[i, j, tau]
R = np.einsum("il,jtl->ijt", W, mean_dX)                           # (8, 8, S)

# permutation null: pixel-permute the projection patterns
null_max = np.zeros((N_PERM, N_MODES, N_MODES))
for p in range(N_PERM):
    Wp_ = W[:, rng.permutation(2500)]
    Rp = np.einsum("il,jtl->ijt", Wp_, mean_dX)
    null_max[p] = np.abs(Rp).max(2)
thresh = np.quantile(null_max, 1 - ALPHA, axis=0)                  # (8, 8) per (i,j)

stat = np.abs(R).max(2)                                            # (8, 8) max_tau |R|
best_lag = np.abs(R).argmax(2) + 1                                 # tau in 1..S
det_pairs = {(j, i) for i in range(N_MODES) for j in range(N_MODES)
             if i != j and stat[i, j] > thresh[i, j]}

tp = len(gt_pairs & det_pairs); fp = len(det_pairs - gt_pairs); fn = len(gt_pairs - det_pairs)
P = tp / (tp + fp) if tp + fp else 0.0
Rc = tp / (tp + fn) if tp + fn else 0.0
F1 = 2 * P * Rc / (P + Rc) if P + Rc else 0.0

print(f"\nDirected-edge detection (pair level, alpha={ALPHA}, {N_PERM} perms):")
print(f"  TP={tp} FP={fp} FN={fn}  ->  P={P:.3f}  R={Rc:.3f}  F1={F1:.3f}")
print(f"  (PCMCI+ on true Z baseline: F1=0.823)")
print(f"\n  detected: {sorted(det_pairs)}")
print(f"  missed:   {sorted(gt_pairs - det_pairs)}")
print(f"  false+:   {sorted(det_pairs - gt_pairs)}")

lag_rows = []
print(f"\n  edge (j->i)   resp peak lag vs true fine lag(s):")
for (c, e) in sorted(gt_pairs & det_pairs):
    bl = int(best_lag[e, c]); tl = gt_lag[(c, e)]
    lag_rows.append((c, e, bl, tl))
    print(f"   X{c}->X{e}   peak tau={bl}   true l={tl}   |R|={stat[e, c]:.4f}")

# self-response e-folding time vs designed phi
print(f"\nSelf-response e-folding vs designed timescales:")
print(f"  {'mode':<6} {'tau_design':>10} {'tau_efold':>10} {'R self t=1':>11}")
efold = np.zeros(N_MODES)
for j in range(N_MODES):
    r = np.abs(R[j, j])
    # log-linear fit over lags where response is decaying and positive
    v = r[r > 1e-12]
    n = min(len(v), N_STEPS)
    y = np.log(r[:n] + 1e-15)
    A_ = np.vstack([np.arange(1, n + 1), np.ones(n)]).T
    slope = np.linalg.lstsq(A_, y, rcond=None)[0][0]
    efold[j] = -1.0 / slope if slope < 0 else np.inf
    tau_d = -1.0 / np.log(PHI[j])
    print(f"  X{j:<5} {tau_d:>10.2f} {efold[j]:>10.2f} {r[0]:>11.4f}")

tau_design = -1.0 / np.log(PHI)
fin = np.isfinite(efold)
def _spear(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra*rb).sum() / np.sqrt((ra**2).sum() * (rb**2).sum()))
sp = _spear(tau_design[fin], efold[fin]) if fin.sum() > 2 else np.nan
print(f"\n  Spearman(tau_design, tau_efold) = {sp:.3f}   (n={int(fin.sum())})")

os.makedirs("results", exist_ok=True)
out = dict(R=R, stat=stat, thresh=thresh, best_lag=best_lag,
           det_pairs=sorted(det_pairs), gt_pairs=sorted(gt_pairs),
           P=P, recall=Rc, F1=F1, tp=tp, fp=fp, fn=fn,
           efold=efold, tau_design=tau_design, spearman_timescale=sp,
           lag_rows=lag_rows, sigma=SIGMA, n_win=N_WIN, n_steps=N_STEPS,
           alpha=ALPHA, n_perm=N_PERM, ckpt=CKPT, split=SPLIT)
np.save(OUT, out, allow_pickle=True)
print(f"\nsaved -> {OUT}")
