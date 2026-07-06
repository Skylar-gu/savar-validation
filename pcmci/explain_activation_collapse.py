"""
Follow-up 1 — Explain the Block G collapse (F1 0.853 -> 0.020 on pooled GNN acts).

Block G's series (c) had TWO confounds vs (a)/(b):
  1. CADENCE MISMATCH: sae_data/hetdynamics_eqvar/activations_full.npy was
     extracted at STRIDE 5 (T_eff=480 vs T=2400), so PCMCI+ lags on (c) were in
     5-fine-step units, scored against fine-unit ground-truth lags (1..6).
  2. LAG LUMPING: one ridge readout aligned to the forecast target Z(t+K),
     while the K=3 input window potentially smears fine lags.

This script separates cadence artifact vs lag smearing vs genuine info loss:
  0. Re-extract pooled GNN activations at STRIDE 1 for the realisations Block G
     used (0..23 for PCMCI, 80..99 for ridge fitting); cached to SAE_DIR.
  1. Per-lag ridge readouts: for each mode j and offset delta (target
     Z_j(e + delta), e = window-end frame t+K-1), fit ridge on tail reals,
     report held-out |r| on reals 0..23. Also diff_r = |r| for predicting the
     within-window change Z_j(e) - Z_j(e-2) (lag-resolution test: a smeared
     representation cannot predict the difference).
  2. PCMCI+ (ParCorr, tau_min=0, tau_max=max fine lag, pc_alpha 0.05) on
     stride-1 readout series at offsets delta in PCMCI_OFFSETS; F1 vs GT and
     vs Block G's (a) numbers.
  3. DMD (rank 20, mirrors pcmci/dmd_timescales.py) on the stacked stride-1
     pooled activation streams (8x256 = 2048 channels, z-scored): eigenvalue
     e-folding times matched to modes by block-norm; Spearman vs designed tau.
     Secondary: per-mode-block DMD with Z-matched eigenvalue.

Output: results/activation_collapse_explained.npy
Env: AC_NREAL (24), AC_RIDGE_REALS (20), AC_PCALPHA (0.05), AC_DMD_NREAL (20)
"""

import sys, os
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import Ridge
from tigramite.data_processing import DataFrame
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI
from pydmd import DMD

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from gnn_forecaster import MeshGNN, K, HIDDEN

DATA_DIR   = Path("data/realisations_hetdynamics_eqvar")
SAE_DIR    = Path("sae_data/hetdynamics_eqvar")
CKPT       = Path("checkpoints/hetdynamics_eqvar/best.pt")
CACHE      = SAE_DIR / "activations_stride1_sel.npy"
CACHE_IDX  = SAE_DIR / "activations_stride1_sel_idx.npy"
N_REAL     = int(os.environ.get("AC_NREAL", 24))
RIDGE_TAIL = int(os.environ.get("AC_RIDGE_REALS", 20))
PC_ALPHA   = float(os.environ.get("AC_PCALPHA", 0.05))
DMD_NREAL  = int(os.environ.get("AC_DMD_NREAL", 20))
DMD_RANK   = 20
N_MODES    = 8
NY, NX     = 50, 50
EXTRACT_BS = 64
# offsets delta: target Z_j(e + delta), e = t+K-1 (window-end frame).
# delta=-2,-1,0 are the frames INSIDE the K=3 window; +1 is the forecast target
# (Block G's original object); +2..+4 probe longer horizons.
READOUT_OFFSETS = list(range(-4, 5))
PCMCI_OFFSETS   = [-2, 0, 1]      # first-in-window, last-in-window, forecast tgt
PHI = np.array([0.15, 0.30, 0.42, 0.55, 0.68, 0.78, 0.86, 0.92])
TAU_DESIGN = -1.0 / np.log(PHI)

paths = sorted(DATA_DIR.glob("realisation_*.npz"))
d0 = np.load(paths[0])
fine_edges = d0["fine_edges"]
gt = {(int(c), int(e), int(l)) for c, e, l, _ in fine_edges if int(c) != int(e)}
TAU_MAX = max(l for _, _, l in gt)
T_TOTAL = int(d0["observations"].shape[1])
T_EFF   = T_TOTAL - K                      # stride-1 windows
REAL_SEL = list(range(N_REAL)) + list(range(100 - RIDGE_TAIL, 100))

# ── 0. stride-1 extraction (cached) ──────────────────────────────────────────
if CACHE.exists() and CACHE_IDX.exists() and \
        list(np.load(CACHE_IDX)) == REAL_SEL:
    print(f"[extract] using cache {CACHE}")
    acts = np.load(CACHE, mmap_mode="r")           # (n_sel, 8, T_EFF, 256)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MeshGNN(ny=NY, nx=NX, k=K).to(device)
    ckpt = torch.load(CKPT, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[extract] MeshGNN val RMSE={ckpt.get('val_rmse', float('nan')):.4f} "
          f"device={device}, stride-1, {len(REAL_SEL)} realisations")
    captured = {}
    model.layers[-1].register_forward_hook(lambda m, i, o: captured.update(act=o))
    acts = np.empty((len(REAL_SEL), N_MODES, T_EFF, HIDDEN), dtype=np.float32)
    with torch.no_grad():
        for k_r, ri in enumerate(REAL_SEL):
            d = np.load(paths[ri])
            obs = d["observations"].astype(np.float32)     # (2500, T)
            W_t = torch.from_numpy(d["W"].astype(np.float32)).to(device)
            frames = torch.from_numpy(obs.T.reshape(T_TOTAL, NY, NX))
            windows = torch.stack([frames[t:t + K] for t in range(T_EFF)])
            out = []
            for i in range(0, T_EFF, EXTRACT_BS):
                model(windows[i:i + EXTRACT_BS].to(device))
                H = captured["act"]                          # (B, 2500, 256)
                out.append(torch.einsum("jl,blc->bjc", W_t, H).cpu().numpy())
            acts[k_r] = np.concatenate(out, 0).transpose(1, 0, 2)
            if (k_r + 1) % 8 == 0:
                print(f"  [{k_r+1}/{len(REAL_SEL)}]")
    np.save(CACHE, acts)
    np.save(CACHE_IDX, np.array(REAL_SEL))
    print(f"[extract] saved {CACHE}  {acts.shape}")

Z_all = np.stack([np.load(paths[ri])["latent_states"].astype(np.float64)
                  for ri in REAL_SEL])                       # (n_sel, 8, T)
tail = slice(N_REAL, N_REAL + RIDGE_TAIL)

# ── 1. per-lag ridge readouts ────────────────────────────────────────────────
# activation index t <-> window-end frame e = t + K - 1; target Z_j(e + delta).
# valid t range for delta: e + delta in [0, T_TOTAL-1].
rng = np.random.default_rng(0)
SUB = 40000

def t_range(delta):
    lo = max(0, -(K - 1 + delta))
    hi = min(T_EFF, T_TOTAL - (K - 1 + delta))
    return lo, hi

readout = {}                      # (j, delta) -> Ridge
readout_r = np.zeros((N_MODES, len(READOUT_OFFSETS)))
for di, delta in enumerate(READOUT_OFFSETS):
    lo, hi = t_range(delta)
    for j in range(N_MODES):
        Xf = np.asarray(acts[tail, j, lo:hi]).reshape(-1, HIDDEN)
        yf = np.stack([Z_all[N_REAL + r, j, lo + K - 1 + delta:
                             hi + K - 1 + delta]
                       for r in range(RIDGE_TAIL)]).reshape(-1)
        sel = rng.choice(len(yf), size=min(SUB, len(yf)), replace=False)
        r = Ridge(alpha=10.0).fit(Xf[sel], yf[sel])
        readout[(j, delta)] = r
        # held-out r on reals 0..N_REAL
        Xh = np.asarray(acts[:N_REAL, j, lo:hi]).reshape(-1, HIDDEN)
        yh = np.stack([Z_all[r_, j, lo + K - 1 + delta: hi + K - 1 + delta]
                       for r_ in range(N_REAL)]).reshape(-1)
        readout_r[j, di] = abs(np.corrcoef(r.predict(Xh), yh)[0, 1])

print("\nper-lag readout held-out |r|  (rows=modes, cols=delta rel. window-end)")
print("  delta: " + "  ".join(f"{d:+d}   " for d in READOUT_OFFSETS))
for j in range(N_MODES):
    print(f"  X{j}:   " + "  ".join(f"{readout_r[j, di]:.3f}"
                                    for di in range(len(READOUT_OFFSETS))))

# lag-resolution: predict within-window change Z_j(e) - Z_j(e-2)
diff_r = np.zeros(N_MODES)
for j in range(N_MODES):
    lo, hi = t_range(0)           # e-2 = t >= 0 ok
    Xf = np.asarray(acts[tail, j, lo:hi]).reshape(-1, HIDDEN)
    yf = np.stack([Z_all[N_REAL + r, j, lo + K - 1: hi + K - 1] -
                   Z_all[N_REAL + r, j, lo: hi]
                   for r in range(RIDGE_TAIL)]).reshape(-1)
    sel = rng.choice(len(yf), size=min(SUB, len(yf)), replace=False)
    r = Ridge(alpha=10.0).fit(Xf[sel], yf[sel])
    Xh = np.asarray(acts[:N_REAL, j, lo:hi]).reshape(-1, HIDDEN)
    yh = np.stack([Z_all[r_, j, lo + K - 1: hi + K - 1] - Z_all[r_, j, lo: hi]
                   for r_ in range(N_REAL)]).reshape(-1)
    diff_r[j] = abs(np.corrcoef(r.predict(Xh), yh)[0, 1])
print("\nwithin-window change readout |r| (Z(e)-Z(e-2)): " +
      " ".join(f"X{j}={diff_r[j]:.3f}" for j in range(N_MODES)))

# ── 2. PCMCI+ on stride-1 readout series ─────────────────────────────────────
def detect(graph):
    N, _, T1 = graph.shape
    return {(c, e, tau) for c in range(N) for e in range(N) if c != e
            for tau in range(1, T1) if graph[c, e, tau] == "-->"}

def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)

pcmci_rows = {}
for delta in PCMCI_OFFSETS:
    lo, hi = t_range(delta)
    agg = dict(tp=0, fp=0, fn=0)
    for ri in range(N_REAL):
        Zc = np.stack([readout[(j, delta)].predict(
            np.asarray(acts[ri, j, lo:hi])) for j in range(N_MODES)])
        pc = PCMCI(dataframe=DataFrame(Zc.T), cond_ind_test=ParCorr(),
                   verbosity=0)
        res = pc.run_pcmciplus(tau_min=0, tau_max=TAU_MAX, pc_alpha=PC_ALPHA)
        dv = detect(res["graph"])
        agg["tp"] += len(gt & dv); agg["fp"] += len(dv - gt)
        agg["fn"] += len(gt - dv)
        if (ri + 1) % 6 == 0:
            print(f"  [delta={delta:+d}] {ri+1}/{N_REAL} realisations")
    p, r, f1 = prf(agg["tp"], agg["fp"], agg["fn"])
    pcmci_rows[delta] = dict(P=p, R=r, F1=f1, **agg)
    print(f"PCMCI+ stride-1, readout delta={delta:+d}: "
          f"F1={f1:.3f} P={p:.2f} R={r:.2f}")

# ── 3. DMD on stride-1 activation streams ────────────────────────────────────
def spear(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else np.nan

per_real = []
per_mode_taus = [[] for _ in range(N_MODES)]      # stacked-DMD, block-matched
per_mode_taus_blk = [[] for _ in range(N_MODES)]  # per-block DMD, Z-matched
for ri in range(DMD_NREAL):
    A = np.asarray(acts[ri]).transpose(0, 2, 1).reshape(N_MODES * HIDDEN, T_EFF)
    A = (A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-9)
    dmd = DMD(svd_rank=DMD_RANK)
    dmd.fit(A.astype(np.float64))
    lam = dmd.eigs
    taus = -1.0 / np.log(np.clip(np.abs(lam), 1e-9, 1 - 1e-9))
    modes = dmd.modes                              # (2048, r)
    blk = np.abs(modes).reshape(N_MODES, HIDDEN, -1)
    blk_norm = np.linalg.norm(blk, axis=1)         # (8, r)
    blk_norm = blk_norm / (blk_norm.sum(0, keepdims=True) + 1e-12)
    best_blob = blk_norm.argmax(0)
    best_share = blk_norm.max(0)
    per_real.append(dict(taus=taus, best_blob=best_blob, best_share=best_share))
    for j in range(N_MODES):
        sel = np.where(best_blob == j)[0]
        if len(sel):
            k = sel[np.argmax(best_share[sel])]
            per_mode_taus[j].append(taus[k])
    # secondary: per-mode-block DMD, eigenvalue matched by corr with Z_j
    Zt = Z_all[ri, :, K - 1: K - 1 + T_EFF]        # Z at window-end frames
    for j in range(N_MODES):
        Aj = np.asarray(acts[ri, j]).T             # (256, T_EFF)
        Aj = (Aj - Aj.mean(1, keepdims=True)) / (Aj.std(1, keepdims=True) + 1e-9)
        dj = DMD(svd_rank=10)
        dj.fit(Aj.astype(np.float64))
        dyn = np.real(dj.dynamics)                 # (r, T_EFF)
        cors = [abs(np.corrcoef(dyn[k_], Zt[j, :dyn.shape[1]])[0, 1])
                if dyn[k_].std() > 1e-9 else 0.0 for k_ in range(dyn.shape[0])]
        k_ = int(np.argmax(cors))
        tj = -1.0 / np.log(np.clip(abs(dj.eigs[k_]), 1e-9, 1 - 1e-9))
        per_mode_taus_blk[j].append(tj)

tau_act = np.array([np.median(t) if t else np.nan for t in per_mode_taus])
tau_blk = np.array([np.median(t) if t else np.nan for t in per_mode_taus_blk])
fin = np.isfinite(tau_act)
sp_act = spear(TAU_DESIGN[fin], tau_act[fin]) if fin.sum() > 2 else np.nan
sp_blk = spear(TAU_DESIGN, tau_blk)
print(f"\nDMD on stride-1 pooled activation streams ({DMD_NREAL} reals, rank {DMD_RANK})")
print(f"  {'mode':<6} {'tau_design':>10} {'tau_act(stacked)':>17} {'tau_act(per-blk)':>17}")
for j in range(N_MODES):
    print(f"  X{j:<5} {TAU_DESIGN[j]:>10.2f} {tau_act[j]:>17.2f} {tau_blk[j]:>17.2f}")
print(f"  Spearman(tau_design, tau_act): stacked={sp_act:.3f}  per-block={sp_blk:.3f}"
      f"   (pixel-DMD reference: 0.976)")

os.makedirs("results", exist_ok=True)
np.save("results/activation_collapse_explained.npy",
        dict(readout_offsets=READOUT_OFFSETS, readout_r=readout_r,
             diff_r=diff_r, pcmci_rows=pcmci_rows, pc_alpha=PC_ALPHA,
             tau_max=TAU_MAX, n_real=N_REAL,
             tau_design=TAU_DESIGN, tau_act_stacked=tau_act,
             tau_act_perblock=tau_blk, spearman_stacked=sp_act,
             spearman_perblock=sp_blk, dmd_per_real=per_real,
             note="stride-1 rerun of Block G series (c); original acts were "
                  "stride-5 (cadence mismatch vs fine-lag GT)"),
        allow_pickle=True)
print("saved -> results/activation_collapse_explained.npy")
