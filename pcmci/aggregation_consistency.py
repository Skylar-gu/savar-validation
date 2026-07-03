"""
Block G — Aggregation-consistency check on W-pooling (Adag, arXiv 2505.10476).

Pixel→mode aggregation can create/destroy conditional independencies unless
consistency conditions hold. Our W-pooling is exactly such an aggregation.

On data/realisations_hetdynamics_eqvar:
 1. W row-support overlap: for each pair (i,j), |supp_i ∩ supp_j| / |supp_i ∪
    supp_j| (Jaccard) and mass overlap  Σ_l min(W_i, W_j) (rows L1-normalized).
 2. PCMCI+ (ParCorr, tau_min=0, tau_max=max fine lag, pc_alpha 0.05) on
      (a) true Z          — latent_states
      (b) W-pooled pixels — Zb_j(t) = W[j] · obs(:, t)
      (c) W-pooled GNN activations — per-mode ridge readout Ẑ_j(t) of the
          pooled 256-dim activation (sae_data_hetdynamics_eqvar), ridge fit on
          held-out realisations (the pipeline's decodable-content object;
          NOTE: activations see a K=3 window → lag smearing is expected and is
          part of what (c) measures).
 3. Score each graph against ground-truth Φ (directed cross edges, exact τ)
    and (b),(c) against (a)'s detected edge set (edge-set agreement F1).

Output: results/aggregation_consistency.npy
Env: AG_NREAL (24), AG_PCALPHA (0.05), AG_RIDGE_REALS (20, from the tail)
"""

import sys, os
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from tigramite.data_processing import DataFrame
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

DATA_DIR   = Path("data/realisations_hetdynamics_eqvar")
SAE_DIR    = Path("sae_data_hetdynamics_eqvar")
N_REAL     = int(os.environ.get("AG_NREAL", 24))
PC_ALPHA   = float(os.environ.get("AG_PCALPHA", 0.05))
RIDGE_TAIL = int(os.environ.get("AG_RIDGE_REALS", 20))
K = 3
N_MODES = 8


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


paths = sorted(DATA_DIR.glob("realisation_*.npz"))
assert len(paths) >= N_REAL + RIDGE_TAIL
d0 = np.load(paths[0])
W = d0["W"].astype(np.float64)                      # (8, 2500)
fine_edges = d0["fine_edges"]
gt = {(int(c), int(e), int(l)) for c, e, l, _ in fine_edges if int(c) != int(e)}
TAU_MAX = max(l for _, _, l in gt)

# ── 1. overlap structure of W ─────────────────────────────────────────────────
supp = W > 0
jac = np.zeros((N_MODES, N_MODES)); mass = np.zeros((N_MODES, N_MODES))
for i in range(N_MODES):
    for j in range(N_MODES):
        inter = (supp[i] & supp[j]).sum(); union = (supp[i] | supp[j]).sum()
        jac[i, j] = inter / union if union else 0.0
        mass[i, j] = np.minimum(W[i], W[j]).sum()
off = ~np.eye(N_MODES, dtype=bool)
print(f"W row-support overlap: max off-diag Jaccard = {jac[off].max():.4f}, "
      f"max off-diag mass overlap = {mass[off].max():.4f} "
      f"(rows L1-normalized; 0 = perfectly disjoint)")

# ── ridge readout for (c), fit on tail realisations ───────────────────────────
acts_full = np.load(SAE_DIR / "activations_full.npy")   # (100, 8, T_eff, 256)
Z_full = np.load(SAE_DIR / "Z_full.npy")
ridge = {}
for j in range(N_MODES):
    Xf = acts_full[-RIDGE_TAIL:, j].reshape(-1, acts_full.shape[-1])
    yf = Z_full[-RIDGE_TAIL:, j].reshape(-1)
    r = Ridge(alpha=10.0).fit(Xf, yf)
    ridge[j] = r
    pred = r.predict(Xf)
    rr = np.corrcoef(pred, yf)[0, 1]
    print(f"  ridge readout X{j}: in-sample |r|={abs(rr):.3f}")

# ── 2. PCMCI+ on the three series types ───────────────────────────────────────
def detect(graph):
    N, _, T1 = graph.shape
    return {(c, e, tau) for c in range(N) for e in range(N) if c != e
            for tau in range(1, T1) if graph[c, e, tau] == "-->"}


def run_pcmciplus(Zs):
    pc = PCMCI(dataframe=DataFrame(Zs), cond_ind_test=ParCorr(), verbosity=0)
    res = pc.run_pcmciplus(tau_min=0, tau_max=TAU_MAX, pc_alpha=PC_ALPHA)
    return detect(res["graph"])


agg = {v: dict(tp=0, fp=0, fn=0, agree_tp=0, agree_fp=0, agree_fn=0)
       for v in ("a_trueZ", "b_pooled_pixels", "c_pooled_acts")}

for ri in range(N_REAL):
    d = np.load(paths[ri])
    Z = d["latent_states"].astype(np.float64)            # (8, T)
    obs = d["observations"].astype(np.float64)           # (2500, T)
    Zb = W @ obs                                          # (8, T)
    Zc = np.stack([ridge[j].predict(acts_full[ri, j]) for j in range(N_MODES)])

    det = {"a_trueZ": run_pcmciplus(Z.T),
           "b_pooled_pixels": run_pcmciplus(Zb.T),
           "c_pooled_acts": run_pcmciplus(Zc.T)}
    for v, dv in det.items():
        a = agg[v]
        a["tp"] += len(gt & dv); a["fp"] += len(dv - gt); a["fn"] += len(gt - dv)
        ref = det["a_trueZ"]
        a["agree_tp"] += len(ref & dv); a["agree_fp"] += len(dv - ref)
        a["agree_fn"] += len(ref - dv)
    if (ri + 1) % 6 == 0:
        print(f"  [{ri+1}/{N_REAL}] realisations done")

rows = {}
print(f"\n{'series':<18} {'F1 vs GT':>9} {'P':>6} {'R':>6} {'F1 vs (a)':>10}")
for v, a in agg.items():
    p, r, f1 = prf(a["tp"], a["fp"], a["fn"])
    _, _, f1a = prf(a["agree_tp"], a["agree_fp"], a["agree_fn"])
    rows[v] = dict(P=p, R=r, F1_gt=f1, F1_vs_a=f1a, **a)
    print(f"{v:<18} {f1:>9.3f} {p:>6.2f} {r:>6.2f} {f1a:>10.3f}")

os.makedirs("results", exist_ok=True)
np.save("results/aggregation_consistency.npy",
        dict(jaccard=jac, mass_overlap=mass, rows=rows, n_real=N_REAL,
             pc_alpha=PC_ALPHA, tau_max=TAU_MAX,
             note="c uses ridge readout of pooled GNN acts (K=3 window smearing included)"),
        allow_pickle=True)
print("saved -> results/aggregation_consistency.npy")
