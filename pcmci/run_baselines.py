"""
Phase 6 — alternative causal discoverers on the mode time series Z.

Runs PCMCI, DYNOTEARS, or TSCI on the latent modes Z of all realisations and scores
edge recovery against the same ground-truth graph G used by run_pcmci.py. This makes
the three methods directly comparable.

Methods
-------
  pcmci     : tigramite PCMCI+ParCorr (the project baseline). Lag-resolved.
  dynotears : causalnex DYNOTEARS via the isolated .venv_causalnex (shell-out).
              Lag-resolved, signed weights; edges thresholded by |weight|.
  tsci      : vendored Tangent Space Causal Inference (NeurIPS 2024). Pairwise,
              NOT lag-resolved -> SUMMARY graph only (AUROC/AUPRC).

Metrics
-------
  * Summary-graph AUROC / AUPRC (lag collapsed) — computed for ALL methods; matches
    the CausalDynamics leaderboard.
  * Lag-resolved Precision/Recall/F1, per-edge recovery, sign accuracy — only for the
    lag-resolved methods (pcmci, dynotears).

Index conventions identical to run_pcmci.py:  edges stored (cause, eff, tau);
G[eff, cause, tau-1] is the coeff of X_cause(t-tau) in X_eff(t).
"""

import sys, argparse
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_common import (
    gt_edges, prf, detected_from_pmatrix, detected_from_scores,
    summary_gt_matrix, summary_score_matrix, auroc_auprc,
)

TAU_MAX = 2


# ── SAE feature construction (mirrors run_pcmci_features.py exactly) ────────────
def build_sae_features(sae_dir, n_real):
    """Encode each mode's Z-aligned SAE feature into a (R, T, N) series.

    For mode j: load per-mode SAE_j, encode the standardised mode-j activations,
    take the feature aligned with latent Z_j (alignment_per_mode best_feat[j]) and
    sign-flip it to be positively aligned. Returns (feats, align_r)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    INPUT_DIM, N_FEATURES, K_TOPK = 256, 512, 25
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class TopKSAE(nn.Module):
        def __init__(self, d, n, k):
            super().__init__()
            self.k = k
            self.encoder = nn.Linear(d, n); self.decoder = nn.Linear(n, d)

        def encode(self, x):
            pre = self.encoder(x)
            v, i = torch.topk(pre, self.k, dim=-1)
            a = torch.zeros_like(pre); a.scatter_(-1, i, F.relu(v)); return a

    acts_full = np.load(sae_dir / "activations_full.npy", mmap_mode="r")   # (R,8,T,256)
    align = np.load(sae_dir / "alignment_per_mode.npy", allow_pickle=True).item()
    R = min(acts_full.shape[0], n_real); T_eff = acts_full.shape[2]; n_modes = acts_full.shape[1]

    feats = np.zeros((R, T_eff, n_modes), np.float64)
    align_r = np.zeros(n_modes)
    for j in range(n_modes):
        ckpt = torch.load(sae_dir / f"sae_mode_{j}.pt", map_location=device, weights_only=False)
        sae = TopKSAE(INPUT_DIM, N_FEATURES, K_TOPK).to(device)
        sae.load_state_dict(ckpt["model_state"]); sae.eval()
        mean_j = np.asarray(ckpt["act_mean"], np.float32); std_j = np.asarray(ckpt["act_std"], np.float32)
        bf = align[j]["best_feat"]
        sign = float(np.sign(align[j]["C_j"][bf])) or 1.0
        align_r[j] = align[j]["max_r"]
        Xj = np.asarray(acts_full[:R, j]).reshape(-1, INPUT_DIM)
        Xn = torch.from_numpy(((Xj - mean_j) / (std_j + 1e-8)).astype(np.float32))
        col = np.empty(len(Xn), np.float32)
        with torch.no_grad():
            for i in range(0, len(Xn), 16384):
                col[i:i+16384] = sae.encode(Xn[i:i+16384].to(device))[:, bf].cpu().numpy()
        feats[:, :, j] = sign * col.reshape(R, T_eff)
    return feats, align_r


_ap = argparse.ArgumentParser()
_ap.add_argument("--method", choices=["pcmci", "dynotears", "tsci"], required=True)
_ap.add_argument("--dy005", action="store_true", help="Use D_y=0.05·I_L dataset")
_ap.add_argument("--diurnal", action="store_true", help="Use diurnal/annual dataset")
_ap.add_argument("--features", action="store_true",
                 help="Run on the discovered SAE feature series (per-mode aligned "
                      "feature) instead of the ground-truth latent modes Z")
_ap.add_argument("--deseason", action="store_true",
                 help="Subtract the ensemble-mean climatology (cycle removal); "
                      "intended for the diurnal dataset")
_ap.add_argument("--n_real", type=int, default=100)
_ap.add_argument("--max_T", type=int, default=None,
                 help="Truncate each series to the first max_T timesteps (preserves "
                      "lag structure; needed for gpdc_torch, which OOMs above T~1500 "
                      "on long e.g. diurnal data). Truncate rather than stride so τ=1,2 "
                      "edges stay aligned with the ground-truth graph.")
# pcmci
_ap.add_argument("--alpha", type=float, default=0.05)
_ap.add_argument("--pc_alpha", type=float, default=0.2)
_ap.add_argument("--cond_ind_test",
                 choices=["parcorr", "gpdc", "gpdc_torch", "cmiknn"],
                 default="parcorr",
                 help="PCMCI conditional-independence test. parcorr=linear/Gaussian "
                      "(fast, CPU); gpdc=GP+distance-corr (nonlinear, CPU, very slow); "
                      "gpdc_torch=same but gpytorch GP on GPU (nonlinear, CUDA); "
                      "cmiknn=kNN CMI (any dependence, CPU, slowest). All non-parcorr "
                      "tests give UNSIGNED scores.")
# dynotears
_ap.add_argument("--lambda_w", type=float, default=0.05)
_ap.add_argument("--lambda_a", type=float, default=0.05)
_ap.add_argument("--w_threshold", type=float, default=0.05,
                 help="|weight| detection threshold for lag-resolved P/R/F1")
# tsci
_ap.add_argument("--tsci_tau", type=int, default=1)
_ap.add_argument("--tsci_Q", type=int, default=3)
_ap.add_argument("--tsci_auto", action="store_true", help="auto lag/dim selection")
_ap.add_argument("--tsci_thresh", type=float, default=None,
                 help="summary score threshold for TSCI summary P/R/F1 "
                      "(default: median of off-diagonal scores)")
_args = _ap.parse_args()

assert not (_args.dy005 and _args.diurnal), "pick at most one of --dy005 / --diurnal"
if _args.deseason and not _args.features:
    print("[warn] --deseason given without --features; deseasoning the latent Z.")

if _args.diurnal:
    DATA_DIR, SAE_DIR, variant = Path("data/realisations_diurnal"), Path("sae_data_diurnal"), "_diurnal"
elif _args.dy005:
    DATA_DIR, SAE_DIR, variant = Path("data/realisations_dy005"), Path("sae_data_dy005"), "_dy005"
else:
    DATA_DIR, SAE_DIR, variant = Path("data/realisations"), Path("sae_data"), ""

# suffix encodes the full data variant so runs never clobber one another (nor
# run_pcmci.py's canonical results/pcmci_results.npy). The cond-ind test is tagged
# only for the non-default choices so existing parcorr result names are unchanged.
cit_tag = "" if _args.cond_ind_test == "parcorr" else f"_{_args.cond_ind_test}"
suffix = ("_features" if _args.features else "") + variant + ("_deseason" if _args.deseason else "") + cit_tag
SAVE_PATH = Path(f"results/baseline_{_args.method}{suffix}.npy")

LAG_RESOLVED = _args.method in ("pcmci", "dynotears")


# ── ground truth (always from the latent realisations) ─────────────────────────
paths = sorted(DATA_DIR.glob("realisation_*.npz"))[: _args.n_real]
assert paths, f"No realisations found in {DATA_DIR}"
G = np.load(paths[0])["ground_truth_graph"].astype(np.float64)   # (N, N, tau_max)
gt = gt_edges(G, cross_only=True)
N = G.shape[0]


# ── build the (R, T, N) series the chosen method runs on ───────────────────────
if _args.features:
    data, align_r = build_sae_features(SAE_DIR, _args.n_real)     # (R, T, N) feature series
    if _args.deseason:
        data = data - data.mean(axis=0, keepdims=True)            # remove ensemble-mean cycle
    src = f"SAE features ({SAE_DIR.name})" + (" deseasoned" if _args.deseason else "")
else:
    data = np.stack([np.load(p)["latent_states"].astype(np.float64).T for p in paths])  # (R,T,N)
    if _args.deseason:
        data = data - data.mean(axis=0, keepdims=True)
    src = f"latent Z ({DATA_DIR.name})" + (" deseasoned" if _args.deseason else "")
if _args.max_T is not None and data.shape[1] > _args.max_T:
    data = data[:, : _args.max_T, :]
    src += f"  [truncated T→{_args.max_T}]"
R = data.shape[0]

print(f"Phase 6 — Causal discovery ({_args.method.upper()}) on {src}")
print("=" * 64)
print(f"  ground truth={DATA_DIR.name}  realisations={R}  T={data.shape[1]}  modes={N}")
if _args.features:
    print("  aligned feature |r| with latents: " +
          " ".join(f"X{j}:{align_r[j]:.2f}" for j in range(N)))
print(f"  tau_max={TAU_MAX}  ground-truth cross-mode edges={len(gt)}")
print(f"  lag-resolved: {LAG_RESOLVED}")
print()


# ── run the chosen method -> per-realisation (score_matrix, detected, val) ─────
# score_summary[r] : (N, N) summary score for AUROC/AUPRC (all methods)
# For lag-resolved methods also: detected set + signed lag-resolved score for sign acc.

score_summary = np.zeros((R, N, N))
records = []   # one dict per realisation

if _args.method == "pcmci":
    from tigramite.data_processing import DataFrame
    from tigramite.pcmci import PCMCI

    def _make_cit(name):
        if name == "parcorr":
            from tigramite.independence_tests.parcorr import ParCorr
            return ParCorr()
        if name == "gpdc":
            from tigramite.independence_tests.gpdc import GPDC
            return GPDC()                       # GP regression + distance correlation
        if name == "gpdc_torch":
            from tigramite.independence_tests.gpdc_torch import GPDCtorch
            return GPDCtorch()                  # gpytorch GP on GPU (auto-detects CUDA)
        if name == "cmiknn":
            from tigramite.independence_tests.cmiknn import CMIknn
            return CMIknn()                     # kNN CMI + local-permutation shuffle test
        raise ValueError(name)

    print(f"  cond_ind_test = {_args.cond_ind_test}"
          + ("  (UNSIGNED scores — sign accuracy not reported)"
             if _args.cond_ind_test != "parcorr" else ""))
    for k in range(R):
        df = DataFrame(data[k])
        res = PCMCI(dataframe=df, cond_ind_test=_make_cit(_args.cond_ind_test),
                    verbosity=0).run_pcmci(
            tau_min=1, tau_max=TAU_MAX, pc_alpha=_args.pc_alpha, alpha_level=_args.alpha)
        det = detected_from_pmatrix(res["p_matrix"], _args.alpha, cross_only=True)
        score_summary[k] = summary_score_matrix(res["val_matrix"])
        records.append(dict(detected=det, val=res["val_matrix"]))
        if (k + 1) % 10 == 0:
            print(f"  [pcmci] {k+1}/{R}")

elif _args.method == "dynotears":
    from methods import dynotears
    scores = dynotears.discover_all(
        data, tau_max=TAU_MAX, lambda_w=_args.lambda_w, lambda_a=_args.lambda_a,
        w_threshold=0.0)                                       # raw weights; threshold in eval
    for k in range(R):
        det = detected_from_scores(scores[k], _args.w_threshold, cross_only=True)
        score_summary[k] = summary_score_matrix(scores[k])
        records.append(dict(detected=det, val=scores[k]))

elif _args.method == "tsci":
    from methods import tsci_adapter
    summ = tsci_adapter.discover_all(
        data, tau=_args.tsci_tau, Q=_args.tsci_Q, auto_embed=_args.tsci_auto)
    for k in range(R):
        score_summary[k] = np.abs(summ[k]); np.fill_diagonal(score_summary[k], 0.0)
        records.append(dict(summary=summ[k]))


# ── summary-graph AUROC / AUPRC (all methods) ──────────────────────────────────
aurocs, auprcs = [], []
for k in range(R):
    a, p = auroc_auprc(score_summary[k], G, cross_only=True)
    aurocs.append(a); auprcs.append(p)
aurocs = np.array(aurocs); auprcs = np.array(auprcs)

print(f"\n{'─'*64}\n  Summary-graph metrics (lag collapsed; comparable across methods)\n{'─'*64}")
print(f"  AUROC : {np.nanmean(aurocs):.3f} ± {np.nanstd(aurocs):.3f}")
print(f"  AUPRC : {np.nanmean(auprcs):.3f} ± {np.nanstd(auprcs):.3f}")
print(f"  (summary ground-truth edges: {int(summary_gt_matrix(G).sum())}/{N*(N-1)})")


# ── lag-resolved metrics (pcmci, dynotears) ────────────────────────────────────
out = dict(method=_args.method, auroc=aurocs, auprc=auprcs,
           score_summary=score_summary, ground_truth=gt)

if LAG_RESOLVED:
    tps = np.zeros(R); fps = np.zeros(R); fns = np.zeros(R)
    precs = np.zeros(R); recs = np.zeros(R); f1s = np.zeros(R)
    for k, r in enumerate(records):
        det = r["detected"]
        tp, fp, fn = len(gt & det), len(det - gt), len(gt - det)
        tps[k], fps[k], fns[k] = tp, fp, fn
        precs[k], recs[k], f1s[k] = prf(tp, fp, fn)

    print(f"\n{'─'*64}\n  Lag-resolved edge recovery (cross-mode)\n{'─'*64}")
    print(f"  Precision : {precs.mean():.3f} ± {precs.std():.3f}")
    print(f"  Recall    : {recs.mean():.3f} ± {recs.std():.3f}")
    print(f"  F1        : {f1s.mean():.3f} ± {f1s.std():.3f}")
    print(f"  Mean TP/FP/FN : {tps.mean():.1f} / {fps.mean():.1f} / {fns.mean():.1f}")

    print(f"\n  Per-edge recovery rate")
    edge_recovery = {}
    for cause, eff, tau in sorted(gt):
        hits = sum(1 for r in records if (cause, eff, tau) in r["detected"])
        rate = hits / R; edge_recovery[(cause, eff, tau)] = rate
        coeff = G[eff, cause, tau - 1]
        flag = "  <-- lag-2 only" if tau == 2 and G[eff, cause, 0] == 0 else ""
        print(f"    X{cause}(t-{tau}) -> X{eff}(t)  coeff={coeff:+.2f}  recovery={rate*100:5.1f}%{flag}")

    # Sign accuracy is only meaningful for signed scores (parcorr val_matrix,
    # dynotears weights). gpdc/cmiknn return unsigned association strengths.
    signed = not (_args.method == "pcmci" and _args.cond_ind_test != "parcorr")
    sc = st = 0
    if signed:
        for r in records:
            for cause, eff, tau in gt & r["detected"]:
                st += 1
                sc += int(np.sign(G[eff, cause, tau - 1]) == np.sign(r["val"][cause, eff, tau]))
        print(f"\n  Sign accuracy on TP edges: {sc}/{st} = {(sc/st*100 if st else 0):.1f}%")
    else:
        print(f"\n  Sign accuracy: n/a ({_args.cond_ind_test} scores are unsigned)")

    fp_counts = {}
    for r in records:
        for e in r["detected"] - gt:
            fp_counts[e] = fp_counts.get(e, 0) + 1
    print(f"\n  Most common false positives (top 8):")
    for (c, e, t), n in sorted(fp_counts.items(), key=lambda x: -x[1])[:8]:
        print(f"    X{c}(t-{t}) -> X{e}(t)  in {n}/{R}")

    out.update(precision=precs, recall=recs, f1=f1s, tp=tps, fp=fps, fn=fns,
               edge_recovery=edge_recovery, fp_counts=fp_counts,
               cond_ind_test=(_args.cond_ind_test if _args.method == "pcmci" else None),
               sign_acc=((sc / st if st else 0.0) if signed else None))
else:
    # TSCI: summary-level P/R/F1 at a threshold (informational; AUROC is the headline)
    thr = _args.tsci_thresh
    A = summary_gt_matrix(G)
    mask = ~np.eye(N, dtype=bool)
    precs = np.zeros(R); recs = np.zeros(R); f1s = np.zeros(R)
    for k in range(R):
        s = score_summary[k]
        t = np.median(s[mask]) if thr is None else thr
        pred = (s > t) & mask
        tp = int((pred & (A == 1)).sum()); fp = int((pred & (A == 0)).sum())
        fn = int((~pred & (A == 1) & mask).sum())
        precs[k], recs[k], f1s[k] = prf(tp, fp, fn)
    print(f"\n{'─'*64}\n  Summary-level P/R/F1 (threshold="
          f"{'median' if thr is None else thr})\n{'─'*64}")
    print(f"  Precision : {precs.mean():.3f}   Recall : {recs.mean():.3f}   F1 : {f1s.mean():.3f}")
    out.update(summary_precision=precs, summary_recall=recs, summary_f1=f1s)


np.save(SAVE_PATH, out)
print(f"\nResults saved → {SAVE_PATH}")
