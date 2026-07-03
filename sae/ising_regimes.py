"""
Block D — Dilution diagnosis via inverse-Ising couplings + community structure
(Goodfire concept-manifolds, arXiv 2604.28119).

Concepts land in capture / tiling / dilution regimes. Prediction from
notes/next_steps_plan.md: mode *identity* is in the dilution regime; slow-mode
*content* approaches capture.

Pipeline (mixed eqvar SAE, sae_data_hetdynamics_eqvar/sae_mixed.pt):
 1. Encode all 8 mode streams; binarize (active = selected by TopK, code > 0).
    Restrict to features active in > 0.5% of samples.
 2. Pairwise Ising couplings J by per-feature logistic regression
    (pseudo-likelihood, sklearn, L2). Symmetrize J = (J + J.T)/2.
 3. Communities of the |J| graph (Louvain, python-louvain; |J| edge weights).
 4. Map communities to modes by mean |corr| of member features with each Z_j.
 5. Per-mode regime classification from quantitative signatures (all stored):
      capture  — top matched feature dominates (top1 |r| ≥ CAP_R and
                 top1/top2 ≥ CAP_RATIO)
      tiling   — no single dominating feature, but the mode's top-M features
                 concentrate in one community (share ≥ TILE_SHARE) with
                 within-community mean |J| above the global off-diag mean
      dilution — otherwise (spread across communities / weak couplings)
 6. Identity-vs-content split: same analysis on per-mode demeaned codes
    ("content") vs raw codes; identity signal = the mode-mean code pattern —
    we additionally report each feature's identity strength (R² of its
    activation predicted by mode label alone) aggregated per community.

Output: results/ising_regimes.npy
Env: IS_NSAMP (default 120000 subsample for the Ising fit), IS_MINACT (0.005),
     IS_TOPM (20), IS_CAP_R (0.5), IS_CAP_RATIO (1.5), IS_TILE_SHARE (0.6)
"""

import sys, os
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pathlib import Path
import networkx as nx
import community as community_louvain
from sklearn.linear_model import LogisticRegression

from eval_sae_metrics import INPUT_DIM, N_MODES, load_sae, encode, pearson_cols

DATADIR    = Path("sae_data_hetdynamics_eqvar")
N_SAMP     = int(os.environ.get("IS_NSAMP", 120000))
MIN_ACT    = float(os.environ.get("IS_MINACT", 0.005))
TOP_M      = int(os.environ.get("IS_TOPM", 20))
CAP_R      = float(os.environ.get("IS_CAP_R", 0.5))
CAP_RATIO  = float(os.environ.get("IS_CAP_RATIO", 1.5))
TILE_SHARE = float(os.environ.get("IS_TILE_SHARE", 0.6))

rng = np.random.default_rng(0)

# ── codes ─────────────────────────────────────────────────────────────────────
acts_full = np.load(DATADIR / "activations_full.npy")
Z_full    = np.load(DATADIR / "Z_full.npy")
sae, mu, sd = load_sae(DATADIR / "sae_mixed.pt")
streams = [acts_full[:, j].reshape(-1, INPUT_DIM) for j in range(N_MODES)]
Zs      = [Z_full[:, j].reshape(-1) for j in range(N_MODES)]
codes   = [encode(sae, streams[j], mu, sd) for j in range(N_MODES)]
n_per   = codes[0].shape[0]
print(f"codes: 8 x ({n_per}, {codes[0].shape[1]})")


def ising_communities(code_mat, tag):
    """code_mat: (n, F) continuous codes (already variant-transformed).
    Returns dict with J, communities, and per-mode regime table."""
    A = (code_mat > 0)                                  # binarized activity
    act_rate = A.mean(0)
    keep = np.where(act_rate > MIN_ACT)[0]
    Ak = A[:, keep].astype(np.float64)
    n, F = Ak.shape
    sub = rng.choice(n, size=min(N_SAMP, n), replace=False)
    Asub = Ak[sub]
    print(f"[{tag}] kept {F}/{code_mat.shape[1]} features "
          f"(act>{MIN_ACT:.1%}); Ising fit on {len(sub):,} samples")

    # pseudo-likelihood: logistic regression of each feature on the rest
    J = np.zeros((F, F))
    X = Asub * 2.0 - 1.0                                # spins in {-1,+1}
    for f in range(F):
        y = Asub[:, f].astype(int)
        if y.min() == y.max():
            continue
        Xo = np.delete(X, f, axis=1)
        lr = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=200)
        lr.fit(Xo, y)
        row = np.insert(lr.coef_[0], f, 0.0)
        J[f] = row / 2.0                                # Ising convention
    J = (J + J.T) / 2.0

    # Louvain on |J|
    G = nx.Graph()
    G.add_nodes_from(range(F))
    absJ = np.abs(J)
    thr = np.percentile(absJ[absJ > 0], 50) if (absJ > 0).any() else 0.0
    for i in range(F):
        for k in range(i + 1, F):
            if absJ[i, k] > thr:
                G.add_edge(i, k, weight=absJ[i, k])
    part = community_louvain.best_partition(G, random_state=0)
    labels = np.array([part[i] for i in range(F)])
    n_comm = labels.max() + 1
    within = [np.abs(J[np.ix_(labels == c, labels == c)]).mean()
              for c in range(n_comm)]
    offdiag = np.abs(J[~np.eye(F, dtype=bool)]).mean()
    print(f"[{tag}] {n_comm} communities; global mean|J|={offdiag:.4f}")

    return dict(J=J, keep=keep, labels=labels, n_comm=n_comm,
                within_J=np.array(within), global_meanJ=offdiag)


def classify(codes_pm, isres, tag):
    """codes_pm: list of 8 (n, F_full) code mats per mode stream."""
    keep, labels = isres["keep"], isres["labels"]
    n_comm = isres["n_comm"]
    within, gJ = isres["within_J"], isres["global_meanJ"]
    table = {}
    for j in range(N_MODES):
        r = np.abs(pearson_cols(codes_pm[j][:, keep], Zs[j]))
        order = np.argsort(-r)
        top1, top2 = r[order[0]], r[order[1]]
        topM = order[:TOP_M]
        shares = np.array([(labels[topM] == c).mean() for c in range(n_comm)])
        cbest = int(shares.argmax())
        if top1 >= CAP_R and top1 / (top2 + 1e-12) >= CAP_RATIO:
            regime = "capture"
        elif shares[cbest] >= TILE_SHARE and within[cbest] > gJ:
            regime = "tiling"
        else:
            regime = "dilution"
        table[j] = dict(regime=regime, top1_r=float(top1), top2_r=float(top2),
                        ratio=float(top1 / (top2 + 1e-12)),
                        comm_share=float(shares[cbest]), best_comm=cbest,
                        within_J=float(within[cbest]), global_J=float(gJ),
                        n_top_features=TOP_M)
        print(f"[{tag}] X{j}: {regime:9s} top|r|={top1:.3f}/{top2:.3f} "
              f"(x{top1/(top2+1e-12):.2f})  comm{cbest} share={shares[cbest]:.2f} "
              f"withinJ={within[cbest]:.4f} (global {gJ:.4f})")
    return table


# raw codes (content + identity mixed)
codes_cat = np.concatenate(codes, 0)
is_raw = ising_communities(codes_cat, "raw")
tab_raw = classify(codes, is_raw, "raw")

# per-mode demeaned codes (content only — identity signal removed)
codes_dm = [c - c.mean(0, keepdims=True) for c in codes]
# demeaning breaks the TopK >0 binarization; re-binarize on the RAW active set
# but analyze correlations on demeaned values. For the Ising graph the activity
# pattern is unchanged by demeaning, so reuse is_raw's J/communities and only
# reclassify with demeaned correlations.
tab_dm = classify(codes_dm, is_raw, "demeaned")

# identity strength per feature: R^2 of activation from mode label alone
lab = np.repeat(np.arange(N_MODES), n_per)
cc = codes_cat[:, is_raw["keep"]]
mode_means = np.stack([cc[lab == j].mean(0) for j in range(N_MODES)])
ss_between = (n_per * (mode_means - cc.mean(0)) ** 2).sum(0)
ss_total = ((cc - cc.mean(0)) ** 2).sum(0) + 1e-12
id_r2 = ss_between / ss_total
comm_id = {int(c): float(id_r2[is_raw["labels"] == c].mean())
           for c in range(is_raw["n_comm"])}
print("\nidentity R^2 (feature activation ~ mode label), by community:")
for c, v in sorted(comm_id.items(), key=lambda kv: -kv[1]):
    print(f"  comm {c}: mean id-R2 = {v:.3f}  "
          f"({int((is_raw['labels']==c).sum())} features)")

os.makedirs("results", exist_ok=True)
np.save("results/ising_regimes.npy",
        dict(raw=tab_raw, demeaned=tab_dm, J=is_raw["J"], keep=is_raw["keep"],
             labels=is_raw["labels"], within_J=is_raw["within_J"],
             global_meanJ=is_raw["global_meanJ"], identity_r2=id_r2,
             comm_identity_r2=comm_id,
             params=dict(n_samp=N_SAMP, min_act=MIN_ACT, top_m=TOP_M,
                         cap_r=CAP_R, cap_ratio=CAP_RATIO,
                         tile_share=TILE_SHARE)),
        allow_pickle=True)
print("\nsaved -> results/ising_regimes.npy")
