"""
Phase 8 (PDF) — Main hypothesis test: are causally-central modes also
forecast-important modes?

Two per-mode quantities, then their correlation:

(A) Causal centrality  — computed on the directed summary graph over the 8 modes
    (cross-edges only, auto-loops excluded). Measures from the PDF:
      out_degree, in_degree, total_degree, betweenness, pagerank,
      ancestor_count, descendant_count, and |coeff|-weighted out-strength.
    Computed on BOTH the ground-truth graph (links_coeffs) and the
    PCMCI-recovered graph (edges recovered in ≥50% of realisations,
    results/pcmci_results.npy → edge_recovery).

(B) Forecast importance — per-mode INPUT-SPACE ablation on the diurnal CNN,
    evaluated on the held-out TEST split (out-of-sample; CNN never trained on
    these timesteps). For mode j we remove its reconstructed spatial
    contribution from every input frame,
        obs_ablated = obs − W_plus[:,j] · (W_flat[j] · obs)   (= Z_j set to 0),
    feed the ablated K-frame window to the CNN, and predict the ORIGINAL
    (un-ablated) next frame. forecast_importance[j] = RMSE_ablated(j) − RMSE_base.
    This is SAE-independent: it intervenes directly on the causal nodes.

Hypothesis (PDF): causally-central modes carry more of the forecasting
computation, so centrality and forecast_importance should be positively
correlated. We report Pearson + Spearman, and — because input-space ablation
trivially scales with a mode's variance — a partial correlation controlling for
per-mode variance.
"""

import sys, glob
from pathlib import Path
import numpy as np
import torch
import networkx as nx
from scipy import stats

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_gen"))
from cnn_forecaster import SpatioTemporalCNN, K, BASE_CH

CKPT = "checkpoints_diurnal/best.pt"
TEST = "data/splits_diurnal/test"
NY = NX = 50
L = NY * NX
N_MODES = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_REAL = 40
T_STRIDE = 8
REC_THRESH = 0.5   # PCMCI edge counted if recovered in ≥50% of realisations

# ── ground-truth cross-edges (X_i → X_j, |coeff|), auto-loops excluded ────────
# from data_gen/instantiate_model.py links_coeffs
LINKS = {
    0: [((0, -1), 0.45), ((2, -2), 0.22)],
    1: [((1, -1), 0.50), ((0, -1), 0.35)],
    2: [((2, -1), 0.35), ((1, -1), 0.40)],
    3: [((3, -1), 0.55), ((0, -1), 0.30), ((2, -1), -0.30)],
    4: [((4, -1), 0.40), ((1, -2), 0.25)],
    5: [((5, -1), 0.30), ((4, -1), 0.35), ((0, -2), -0.20)],
    6: [((6, -1), 0.50), ((3, -1), 0.30), ((5, -2), 0.25)],
    7: [((7, -1), 0.45), ((6, -1), 0.20), ((3, -2), -0.15)],
}


def summary_digraph(weighted_edges):
    """weighted_edges: list of (src, dst, weight>0). Build a DiGraph on all 8 modes
    (collapsing lags into a summary graph; multiple lags between a pair are summed)."""
    G = nx.DiGraph()
    G.add_nodes_from(range(N_MODES))
    for s, d, w in weighted_edges:
        if G.has_edge(s, d):
            G[s][d]["weight"] += w
        else:
            G.add_edge(s, d, weight=w)
    return G


def gt_edges():
    edges = []
    for j, parents in LINKS.items():
        for (i, _tau), c in parents:
            if i != j:
                edges.append((i, j, abs(c)))
    return edges


def recovered_edges():
    d = np.load("results/pcmci_results.npy", allow_pickle=True).item()
    er = d["edge_recovery"]          # {(i,j,tau): freq}
    edges = []
    for (i, j, _tau), freq in er.items():
        if i != j and freq >= REC_THRESH:
            edges.append((i, j, float(freq)))
    return edges


def centralities(G):
    """Return dict measure -> np.array over modes 0..7."""
    nodes = list(range(N_MODES))
    out_deg = np.array([G.out_degree(n) for n in nodes], float)
    in_deg = np.array([G.in_degree(n) for n in nodes], float)
    out_str = np.array([G.out_degree(n, weight="weight") for n in nodes], float)
    btw = nx.betweenness_centrality(G, normalized=True)
    pr = nx.pagerank(G, alpha=0.85) if G.number_of_edges() else {n: 1 / N_MODES for n in nodes}
    anc = np.array([len(nx.ancestors(G, n)) for n in nodes], float)
    desc = np.array([len(nx.descendants(G, n)) for n in nodes], float)
    return {
        "out_degree": out_deg,
        "in_degree": in_deg,
        "total_degree": out_deg + in_deg,
        "out_strength": out_str,
        "betweenness": np.array([btw[n] for n in nodes]),
        "pagerank": np.array([pr[n] for n in nodes]),
        "ancestors": anc,
        "descendants": desc,
    }


# ── forecast importance via input-space per-mode ablation (held-out test) ─────
def forecast_importance():
    model = SpatioTemporalCNN(ny=NY, nx=NX, k=K, base_ch=BASE_CH).to(DEVICE)
    ck = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(ck["model_state"]); model.eval()
    print(f"Loaded {CKPT}  val RMSE={ck['val_rmse']:.4f}")

    paths = sorted(glob.glob(str(Path(TEST) / "realisation_*.npz")))[:N_REAL]
    se = np.zeros(N_MODES); se_base = 0.0; tot = 0; nwin = 0
    z2_sum = np.zeros(N_MODES); z_count = 0

    with torch.no_grad():
        for p in paths:
            d = np.load(p)
            obs_sp = d["observations"].astype(np.float32)        # (T,50,50)
            Tn = obs_sp.shape[0]
            obs_flat = obs_sp.reshape(Tn, L).T                   # (L,T)
            Wf = d["W"].astype(np.float32)                       # (8,L)
            Wp = d["W_plus"].astype(np.float32)                  # (L,8)
            Z = Wf @ obs_flat                                    # (8,T)
            z2_sum += (Z ** 2).sum(1); z_count += Tn

            # ablated full-timeline frames per mode (input only)
            abl = []
            for j in range(N_MODES):
                contrib = np.outer(Wp[:, j], Z[j])               # (L,T)
                abl.append((obs_flat - contrib).T.reshape(Tn, NY, NX))

            tlist = list(range(0, Tn - K, T_STRIDE))
            tgt = obs_sp[[t + K for t in tlist]]                 # original targets
            base_w = np.stack([obs_sp[t:t + K] for t in tlist])  # (nw,K,50,50)
            abl_w = [np.stack([a[t:t + K] for t in tlist]) for a in abl]

            for i in range(0, len(base_w), 64):
                yb = torch.from_numpy(tgt[i:i + 64]).to(DEVICE).unsqueeze(1)
                xb = torch.from_numpy(base_w[i:i + 64]).to(DEVICE).unsqueeze(1)  # (B,1,K,H,W)
                pb = model(xb)
                se_base += ((pb - yb) ** 2).sum().item()
                tot += yb.numel(); nwin += xb.shape[0]
                for j in range(N_MODES):
                    xj = torch.from_numpy(abl_w[j][i:i + 64]).to(DEVICE).unsqueeze(1)
                    pj = model(xj)
                    se[j] += ((pj - yb) ** 2).sum().item()
            print(f"  {Path(p).name}: cumulative windows={nwin}")

    base_rmse = (se_base / tot) ** 0.5
    rmse = (se / tot) ** 0.5
    fi = rmse - base_rmse
    var = z2_sum / z_count
    print(f"\ntest windows={nwin}  pixels={tot:,}  baseline RMSE={base_rmse:.4f}")
    return fi, base_rmse, var, nwin


def partial_spearman(x, y, z):
    """Spearman partial correlation of x,y controlling for z (rank-residualised)."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    def resid(a, b):
        b1 = np.vstack([np.ones_like(b), b]).T
        beta, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ beta
    ex, ey = resid(rx, rz), resid(ry, rz)
    r, _ = stats.pearsonr(ex, ey)
    return r


# ── run ───────────────────────────────────────────────────────────────────────
print("Phase 8 — causal centrality vs forecast importance")
print("=" * 70)

C_gt = centralities(summary_digraph(gt_edges()))
C_rec = centralities(summary_digraph(recovered_edges()))
fi, base_rmse, var, nwin = forecast_importance()

print("\nPer-mode summary (ground-truth centrality):")
hdr = f"  {'mode':>4}  {'fcst_imp':>9}  {'Δ%':>6}  {'var(Z)':>7}  " + \
      "  ".join(f"{k[:5]:>5}" for k in C_gt)
print(hdr)
for m in range(N_MODES):
    row = f"  X{m:>3}  {fi[m]:+9.5f}  {100*fi[m]/base_rmse:+5.1f}%  {var[m]:>7.3f}  " + \
          "  ".join(f"{C_gt[k][m]:>5.2f}" for k in C_gt)
    print(row)

print("\nCorrelation of each centrality measure with forecast_importance:")
print(f"  {'measure':>14}  {'GT Pear':>8}  {'GT Spear':>9}  {'GT p|var':>9}  "
      f"{'REC Pear':>9}  {'REC Spear':>10}")
corr_out = {}
for k in C_gt:
    pe_g = stats.pearsonr(C_gt[k], fi)[0]
    sp_g = stats.spearmanr(C_gt[k], fi)[0]
    pv_g = partial_spearman(C_gt[k], fi, var)
    pe_r = stats.pearsonr(C_rec[k], fi)[0]
    sp_r = stats.spearmanr(C_rec[k], fi)[0]
    corr_out[k] = dict(gt_pearson=pe_g, gt_spearman=sp_g, gt_partial_var=pv_g,
                       rec_pearson=pe_r, rec_spearman=sp_r)
    print(f"  {k:>14}  {pe_g:>+8.3f}  {sp_g:>+9.3f}  {pv_g:>+9.3f}  "
          f"{pe_r:>+9.3f}  {sp_r:>+10.3f}")

# variance confound check
v_pe = stats.pearsonr(var, fi)[0]; v_sp = stats.spearmanr(var, fi)[0]
print(f"\nConfound check — corr(var(Z), forecast_importance): "
      f"Pearson {v_pe:+.3f}, Spearman {v_sp:+.3f}")
print("(input-space ablation scales with a mode's amplitude; 'GT p|var' columns "
      "partial that out)")

out = {
    "forecast_importance": fi, "baseline_rmse": base_rmse, "var_Z": var, "nwin": nwin,
    "centrality_gt": C_gt, "centrality_rec": C_rec,
    "correlations": corr_out,
    "var_pearson": v_pe, "var_spearman": v_sp,
}
np.save("results/phase8_centrality_vs_importance.npy", out)
print("\nSaved → results/phase8_centrality_vs_importance.npy")
