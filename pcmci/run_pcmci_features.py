"""
Phase 6 (PDF) — Causal discovery on SAE FEATURE time series.

The existing `run_pcmci.py` runs PCMCI on the ground-truth latent modes Z and is a
*method check*. The PDF's Phase 6 instead runs discovery on the **discovered SAE
features** — the realistic case (no privileged latents). This script:

  1. Builds a feature time series: for each mode j, encode mode-j activations
     through the per-mode SAE_j and take its Z_j-aligned feature (alignment_per_mode
     best_feat[j]). Result per realisation: F (T, 8) — one discovered feature per mode,
     sign-flipped so it is positively aligned with its latent.
  2. Step 1 — Mutual-information screening: report which lagged feature pairs carry
     any dependence (PCMCI's PC step subsumes this; reported for completeness).
  3. Step 2 — PCMCI (ParCorr, tau_max=2): estimate the causal graph.
  4. Step 3 — Orientation: lagged links (tau>=1) are directed by time order.
  5. Evaluate the recovered feature_graph against the SAME ground-truth G used for
     the latent run, so feature-PCMCI F1 is directly comparable to latent-PCMCI F1.

Mirrors run_pcmci.py's index conventions and metrics exactly.
"""

import sys, argparse
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tigramite.data_processing import DataFrame
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI
from sklearn.feature_selection import mutual_info_regression

_ap = argparse.ArgumentParser()
_ap.add_argument("--diurnal", action="store_true")
_ap.add_argument("--dy005", action="store_true")
_ap.add_argument("--n_real", type=int, default=100)
_args = _ap.parse_args()

if _args.diurnal:
    DATA_DIR, SAE_DIR = Path("data/realisations_diurnal"), Path("sae_data_diurnal")
    SAVE_PATH = Path("results/pcmci_features_diurnal.npy")
elif _args.dy005:
    DATA_DIR, SAE_DIR = Path("data/realisations_dy005"), Path("sae_data_dy005")
    SAVE_PATH = Path("results/pcmci_features_dy005.npy")
else:
    DATA_DIR, SAE_DIR = Path("data/realisations"), Path("sae_data")
    SAVE_PATH = Path("results/pcmci_features.npy")

TAU_MAX, ALPHA, PC_ALPHA = 2, 0.05, 0.2
N_MODES, INPUT_DIM, N_FEATURES, K_TOPK = 8, 256, 512, 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TopKSAE(nn.Module):
    def __init__(self, d, n, k):
        super().__init__()
        self.k = k
        self.encoder = nn.Linear(d, n); self.decoder = nn.Linear(n, d)

    def encode(self, x):
        pre = self.encoder(x)
        v, i = torch.topk(pre, self.k, dim=-1)
        a = torch.zeros_like(pre); a.scatter_(-1, i, F.relu(v)); return a


# ── metric helpers (identical conventions to run_pcmci.py) ─────────────────────
def gt_edges(G, cross_only=True):
    N, _, tau_max = G.shape
    out = set()
    for eff in range(N):
        for cause in range(N):
            if cross_only and cause == eff:
                continue
            for tau in range(1, tau_max + 1):
                if G[eff, cause, tau - 1] != 0:
                    out.add((cause, eff, tau))
    return out


def detected_edges(p_matrix, alpha, cross_only=True):
    N = p_matrix.shape[0]; tau_max = p_matrix.shape[2] - 1
    out = set()
    for cause in range(N):
        for eff in range(N):
            if cross_only and cause == eff:
                continue
            for tau in range(1, tau_max + 1):
                if p_matrix[cause, eff, tau] < alpha:
                    out.add((cause, eff, tau))
    return out


def prf(tp, fp, fn):
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return prec, rec, f1


# ── load ground truth + SAE feature construction ──────────────────────────────
paths = sorted(DATA_DIR.glob("realisation_*.npz"))[: _args.n_real]
G = np.load(paths[0])["ground_truth_graph"].astype(np.float64)
gt = gt_edges(G, cross_only=True)

acts_full = np.load(SAE_DIR / "activations_full.npy")        # (R, 8, T, 256)
Z_full = np.load(SAE_DIR / "Z_full.npy")                     # (R, 8, T)
align = np.load(SAE_DIR / "alignment_per_mode.npy", allow_pickle=True).item()
R, _, T_eff, _ = acts_full.shape
R = min(R, _args.n_real)

print("PCMCI Phase 6 — SAE FEATURE Time Series Causal Discovery")
print("=" * 64)
print(f"  dataset={DATA_DIR.name}  realisations={R}  T={T_eff}  modes={N_MODES}")
print(f"  tau_max={TAU_MAX}  alpha={ALPHA}  pc_alpha={PC_ALPHA}")
print(f"  ground-truth cross-mode edges: {len(gt)}\n")

# Encode each mode's aligned feature for all realisations: feats (R, T, 8)
feats = np.zeros((R, T_eff, N_MODES), np.float64)
align_r = np.zeros(N_MODES)   # |r| of each aligned feature with its latent
for j in range(N_MODES):
    ckpt = torch.load(SAE_DIR / f"sae_mode_{j}.pt", map_location=DEVICE, weights_only=False)
    sae = TopKSAE(INPUT_DIM, N_FEATURES, K_TOPK).to(DEVICE)
    sae.load_state_dict(ckpt["model_state"]); sae.eval()
    mean_j = np.asarray(ckpt["act_mean"], np.float32); std_j = np.asarray(ckpt["act_std"], np.float32)
    bf = align[j]["best_feat"]
    sign = float(np.sign(align[j]["C_j"][bf])) or 1.0          # flip to +align with Z_j
    align_r[j] = align[j]["max_r"]
    Xj = acts_full[:R, j].reshape(-1, INPUT_DIM)               # (R*T, 256)
    Xn = torch.from_numpy(((Xj - mean_j) / (std_j + 1e-8)).astype(np.float32))
    col = np.empty(len(Xn), np.float32)
    with torch.no_grad():
        for i in range(0, len(Xn), 16384):
            col[i:i+16384] = sae.encode(Xn[i:i+16384].to(DEVICE))[:, bf].cpu().numpy()
    feats[:, :, j] = sign * col.reshape(R, T_eff)
print("Aligned feature |r| with latents: " +
      "  ".join(f"X{j}:{align_r[j]:.2f}" for j in range(N_MODES)))
print(f"Mean feature activation rate (fraction nonzero): "
      f"{(feats != 0).mean():.3f}\n")

# ── Step 1: MI screening (informational; PCMCI's PC step does the real screening)
print("── Step 1: lagged mutual-information screen (max over tau=1..2, mean over real.) ──")
mi = np.zeros((N_MODES, N_MODES))
rng = np.random.default_rng(0)
sub = rng.choice(R, min(R, 20), replace=False)                # subsample realisations for MI speed
for cause in range(N_MODES):
    for eff in range(N_MODES):
        if cause == eff:
            continue
        vals = []
        for r in sub:
            for tau in (1, 2):
                x = feats[r, :-tau, cause].reshape(-1, 1); y = feats[r, tau:, eff]
                if x.std() < 1e-9 or y.std() < 1e-9:
                    continue
                vals.append(mutual_info_regression(x, y, random_state=0)[0])
        mi[cause, eff] = np.mean(vals) if vals else 0.0
mi_thresh = np.percentile(mi[mi > 0], 50) if (mi > 0).any() else 0.0
n_screen = int((mi > mi_thresh).sum())
print(f"  candidate ordered pairs surviving MI>median ({mi_thresh:.4f}): {n_screen}/{N_MODES*(N_MODES-1)}")
gt_pairs = {(c, e) for (c, e, _) in gt}
mi_kept = {(c, e) for c in range(N_MODES) for e in range(N_MODES)
           if c != e and mi[c, e] > mi_thresh}
print(f"  of {len(gt_pairs)} true cause→eff pairs, MI-screen keeps {len(gt_pairs & mi_kept)}\n")

# ── Step 2+3: PCMCI per realisation ───────────────────────────────────────────
records = []
for k in range(R):
    df = DataFrame(feats[k])                                   # (T, 8)
    pcobj = PCMCI(dataframe=df, cond_ind_test=ParCorr(), verbosity=0)
    res = pcobj.run_pcmci(tau_min=1, tau_max=TAU_MAX, pc_alpha=PC_ALPHA, alpha_level=ALPHA)
    det = detected_edges(res["p_matrix"], ALPHA, cross_only=True)
    tp, fp, fn = len(gt & det), len(det - gt), len(gt - det)
    prec, rec, f1 = prf(tp, fp, fn)
    records.append(dict(tp=tp, fp=fp, fn=fn, prec=prec, rec=rec, f1=f1,
                        detected=det, p_matrix=res["p_matrix"], val_matrix=res["val_matrix"]))
    if (k + 1) % 10 == 0:
        print(f"  [{k+1:3d}/{R}]  running mean F1 = {np.mean([r['f1'] for r in records]):.3f}")

precs = np.array([r["prec"] for r in records]); recs = np.array([r["rec"] for r in records])
f1s = np.array([r["f1"] for r in records]); tps = np.array([r["tp"] for r in records])
fps = np.array([r["fp"] for r in records]); fns = np.array([r["fn"] for r in records])

print(f"\n{'─'*64}\n  Aggregate edge-recovery (feature graph vs ground truth)\n{'─'*64}")
print(f"  Ground truth edges : {len(gt)}")
print(f"  Precision          : {precs.mean():.3f} ± {precs.std():.3f}")
print(f"  Recall             : {recs.mean():.3f} ± {recs.std():.3f}")
print(f"  F1                 : {f1s.mean():.3f} ± {f1s.std():.3f}")
print(f"  Mean TP / FP / FN  : {tps.mean():.1f} / {fps.mean():.1f} / {fns.mean():.1f}")

# ── per-edge recovery ─────────────────────────────────────────────────────────
print(f"\n{'─'*64}\n  Per-edge recovery rate\n{'─'*64}")
edge_recovery = {}
for cause, eff, tau in sorted(gt):
    hits = sum(1 for r in records if (cause, eff, tau) in r["detected"])
    rate = hits / len(records); edge_recovery[(cause, eff, tau)] = rate
    coeff = G[eff, cause, tau - 1]
    flag = "  <-- lag-2 only" if tau == 2 and G[eff, cause, 0] == 0 else ""
    print(f"  X{cause}(t-{tau}) -> X{eff}(t)  coeff={coeff:+.2f}  "
          f"recovery={rate*100:5.1f}%  (aligned |r|: cause {align_r[cause]:.2f}, eff {align_r[eff]:.2f}){flag}")

# ── sign accuracy ─────────────────────────────────────────────────────────────
sc = st = 0
for r in records:
    for cause, eff, tau in gt & r["detected"]:
        st += 1
        sc += int(np.sign(G[eff, cause, tau - 1]) == np.sign(r["val_matrix"][cause, eff, tau]))
print(f"\n  Sign accuracy on TP edges: {sc}/{st} = {(sc/st*100 if st else 0):.1f}%")

# ── false positives ───────────────────────────────────────────────────────────
fp_counts = {}
for r in records:
    for e in r["detected"] - gt:
        fp_counts[e] = fp_counts.get(e, 0) + 1
print(f"\n  Most common false positives (top 8):")
for (c, e, t), n in sorted(fp_counts.items(), key=lambda x: -x[1])[:8]:
    print(f"    X{c}(t-{t}) -> X{e}(t)  in {n}/{R}")

# ── compare to the latent-PCMCI baseline ──────────────────────────────────────
base = Path("results/pcmci_results.npy")
if base.exists() and not (_args.diurnal or _args.dy005):
    b = np.load(base, allow_pickle=True).item()
    print(f"\n{'─'*64}\n  Feature-PCMCI vs latent-PCMCI (same ground truth)\n{'─'*64}")
    print(f"  {'':12}{'latent Z':>12}{'SAE feature':>14}")
    print(f"  {'Precision':12}{b['precision'].mean():>12.3f}{precs.mean():>14.3f}")
    print(f"  {'Recall':12}{b['recall'].mean():>12.3f}{recs.mean():>14.3f}")
    print(f"  {'F1':12}{b['f1'].mean():>12.3f}{f1s.mean():>14.3f}")

np.save(SAVE_PATH, dict(precision=precs, recall=recs, f1=f1s, tp=tps, fp=fps, fn=fns,
                        edge_recovery=edge_recovery, fp_counts=fp_counts,
                        ground_truth=gt, aligned_r=align_r, mi=mi))
print(f"\nResults saved → {SAVE_PATH}")
