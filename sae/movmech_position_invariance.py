"""
P1/P2/P3 — the left-vs-right test (spec v2 §"key test") + controls.

P1 (THE test). For each mechanism k, the pooled activation feat[r,k,:] (256-dim,
W[r,k]-pooled at k's per-realisation location) should predict Z_k(t). Split
realisations by k's blob-centre column: LEFT (x < mid) vs RIGHT (x >= mid). Fit
a ridge readout on LEFT realisations, evaluate on held-out RIGHT realisations
(out-of-region) AND held-out LEFT (in-region); repeat train-on-RIGHT; average.
  * location code  -> out-of-region R^2 collapses (channel profile is
    position-dependent), big in-region gap.
  * mechanism code -> R^2 holds across regions, gap ~ 0.
Report per-mechanism out-region R^2 and gap = in-region - out-region.

P2 (position-shuffle null). Redo the split with RANDOM region labels (ignore
true centre). Both a location and a mechanism code then show gap ~ 0 — confirms
any P1 gap is genuine position structure, not partition variance.

P3 (aliasing vs binding). Shift the input frames by delta pixels, run the model,
pool at the correspondingly-shifted W. If the internal representation shifts
cleanly with the blob (shift-equivariant), the shifted-pooled activation matches
the unshifted one -> binding preserved. If it scrambles -> internal aliasing.
corr(feat_shift, feat_orig) per mechanism, per delta; plain vs blurpool (A0)
separates a signal-processing bug (aliasing) from a concept-forming gap.

Usage:
  python3 sae/movmech_position_invariance.py --datadir sae_data_movmech_place \
      --raw data/realisations_movmech_place --tag place_plain \
      [--ckpt checkpoints_movmech_place/best.pt --variant plain]   # +P3
Output: results/movmech_posinv_<tag>.npy
"""

import sys, argparse, glob
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
from pathlib import Path

INPUT_DIM = 256
N_MODES = 8
FINE_MID = 50.0    # fine grid is 100 wide; centre column x < 50 = LEFT
RIDGE_LAM = 10.0
SUB = 6000         # samples per fit cap


def ridge_fit(X, y, lam=RIDGE_LAM):
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mu) / sd
    A = Xs.T @ Xs + lam * np.eye(Xs.shape[1])
    w = np.linalg.solve(A, Xs.T @ (y - y.mean()))
    return w, mu, sd, y.mean()


def ridge_pred(X, w, mu, sd, b):
    return ((X - mu) / sd) @ w + b


def r2(y, yhat):
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - yhat) ** 2).sum() / (ss + 1e-12))


def gather(acts, Z, reals, k, rng, cap=SUB):
    """Stack (samples, 256) and (samples,) for mechanism k over given realisations."""
    X = acts[reals, k].reshape(-1, INPUT_DIM)
    y = Z[reals, k].reshape(-1)
    if len(y) > cap:
        idx = rng.choice(len(y), cap, replace=False)
        X, y = X[idx], y[idx]
    return X.astype(np.float64), y.astype(np.float64)


def p1_p2(acts, Z, centres_x, tag, rng):
    """centres_x: (R, 8) blob-centre column per realisation/mechanism."""
    R = acts.shape[0]
    rows = {"p1": [], "p2": []}
    for mode in ("p1", "p2"):
        for k in range(N_MODES):
            if mode == "p1":
                left = np.where(centres_x[:, k] < FINE_MID)[0]
                right = np.where(centres_x[:, k] >= FINE_MID)[0]
            else:  # random split, same sizes as p1
                nL = int((centres_x[:, k] < FINE_MID).sum())
                perm = rng.permutation(R)
                left, right = perm[:nL], perm[nL:]
            if len(left) < 4 or len(right) < 4:
                rows[mode].append(dict(k=k, out_r2=np.nan, in_r2=np.nan, gap=np.nan,
                                       nL=len(left), nR=len(right)))
                continue
            out_r2s, in_r2s = [], []
            for A, B in ((left, right), (right, left)):
                # hold out 20% of the train region for in-region eval
                Ap = rng.permutation(A)
                cut = max(1, int(0.8 * len(Ap)))
                tr, inr = Ap[:cut], Ap[cut:]
                Xtr, ytr = gather(acts, Z, tr, k, rng)
                w, mu, sd, b = ridge_fit(Xtr, ytr)
                Xin, yin = gather(acts, Z, inr, k, rng)
                Xout, yout = gather(acts, Z, B, k, rng)
                in_r2s.append(r2(yin, ridge_pred(Xin, w, mu, sd, b)))
                out_r2s.append(r2(yout, ridge_pred(Xout, w, mu, sd, b)))
            in_r2, out_r2 = float(np.mean(in_r2s)), float(np.mean(out_r2s))
            rows[mode].append(dict(k=k, out_r2=out_r2, in_r2=in_r2, gap=in_r2 - out_r2,
                                   nL=len(left), nR=len(right)))
    print(f"\n=== {tag}  P1 (true left/right) vs P2 (shuffle) ===")
    print(f"{'mech':>4} | {'P1 in':>7} {'P1 out':>7} {'P1 gap':>7} | {'P2 in':>7} {'P2 out':>7} {'P2 gap':>7} | nL/nR")
    for k in range(N_MODES):
        a, b = rows["p1"][k], rows["p2"][k]
        print(f"X{k:>3} | {a['in_r2']:>7.3f} {a['out_r2']:>7.3f} {a['gap']:>7.3f} | "
              f"{b['in_r2']:>7.3f} {b['out_r2']:>7.3f} {b['gap']:>7.3f} | {a['nL']}/{a['nR']}")
    m1o = np.nanmean([r["out_r2"] for r in rows["p1"]])
    m1i = np.nanmean([r["in_r2"] for r in rows["p1"]])
    m1g = np.nanmean([r["gap"] for r in rows["p1"]])
    m2g = np.nanmean([r["gap"] for r in rows["p2"]])
    print(f"  MEAN P1: in={m1i:.3f} out={m1o:.3f} gap={m1g:.3f}   |   P2 gap={m2g:.3f}")
    verdict = ("ABSTRACTION (mechanism code)" if m1o > 0.5 and m1g < 0.15
               else "LOCATION code (P1 fails out-of-region)" if m1g > 0.2
               else "partial")
    print(f"  VERDICT: {verdict}")
    return dict(p1=rows["p1"], p2=rows["p2"], mean_p1_out=m1o, mean_p1_in=m1i,
                mean_p1_gap=m1g, mean_p2_gap=m2g, verdict=verdict)


def p3_shift_equivariance(ckpt, variant, raw_dir, deltas=(1, 2, 4, 8), n_real=6):
    """Representational shift-equivariance under an input roll."""
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
    from gnn_forecaster import MeshGNN, K
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if variant == "plain":
        model = MeshGNN(ny=50, nx=50, k=K).to(dev)
    else:
        from mesh_gnn_variants import VariantMeshGNN
        model = VariantMeshGNN(ny=50, nx=50, variant=variant, k=K).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev)["model_state"])
    model.eval()
    cap = {}
    model.decoder[0].register_forward_hook(lambda m, i, o: cap.__setitem__("H", i[0]))

    paths = sorted(glob.glob(f"{raw_dir}/realisation_*.npz"))[:n_real]
    NY = NX = 50
    per_delta = {d: [] for d in deltas}
    with torch.no_grad():
        for p in paths:
            dd = np.load(p)
            obs = dd["observations"].astype(np.float32)   # (2500, T)
            W = dd["W"].astype(np.float32)                 # (8, 2500)
            T = obs.shape[1]
            ts = list(range(200, 260))                     # 60 windows
            frames = obs.T.reshape(T, NY, NX)
            win = torch.from_numpy(np.stack([frames[t:t+K] for t in ts])).to(dev)
            model(win); H0 = cap["H"].clone()              # (B, L, 256)
            feat0 = torch.einsum("jl,blc->bjc", torch.from_numpy(W).to(dev),
                                 H0.reshape(H0.size(0), NY*NX, -1))
            for d in deltas:
                fr_s = np.roll(frames, d, axis=2)          # shift columns by d
                win_s = torch.from_numpy(np.stack([fr_s[t:t+K] for t in ts])).to(dev)
                model(win_s); Hs = cap["H"]
                Ws = np.roll(W.reshape(8, NY, NX), d, axis=2).reshape(8, NY*NX)
                feats = torch.einsum("jl,blc->bjc", torch.from_numpy(Ws).to(dev),
                                     Hs.reshape(Hs.size(0), NY*NX, -1))
                # per-mechanism corr between shifted-pooled and orig-pooled activation
                a = feat0.reshape(feat0.size(0), 8, -1).cpu().numpy()
                b = feats.reshape(feats.size(0), 8, -1).cpu().numpy()
                cr = [float(np.corrcoef(a[:, k].ravel(), b[:, k].ravel())[0, 1])
                      for k in range(8)]
                per_delta[d].append(cr)
    print(f"\n=== P3 shift-equivariance ({variant}) — corr(feat_shift, feat_orig) ===")
    print(f"{'delta':>5} | " + " ".join(f"X{k}" for k in range(8)) + " | mean")
    out = {}
    for d in deltas:
        arr = np.array(per_delta[d])           # (n_real, 8)
        mk = arr.mean(0)
        out[d] = mk
        print(f"{d:>5} | " + " ".join(f"{v:.2f}" for v in mk) + f" | {mk.mean():.3f}")
    return {str(d): out[d] for d in deltas}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", required=True)
    ap.add_argument("--raw", required=True, help="realisation dir (for centres)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--variant", default="plain")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    acts = np.load(Path(a.datadir) / "activations_full.npy")   # (R, 8, T, 256)
    Z = np.load(Path(a.datadir) / "Z_full.npy")                # (R, 8, T)
    paths = sorted(glob.glob(f"{a.raw}/realisation_*.npz"))
    centres_x = np.array([np.load(p)["centres"][:, 1] for p in paths])  # (R, 8) column
    assert centres_x.shape[0] == acts.shape[0], (centres_x.shape, acts.shape)

    res = p1_p2(acts, Z, centres_x, a.tag, rng)
    if a.ckpt:
        res["p3"] = p3_shift_equivariance(a.ckpt, a.variant, a.raw)
    Path("results").mkdir(exist_ok=True)
    np.save(f"results/movmech_posinv_{a.tag}.npy", res, allow_pickle=True)
    print(f"\nsaved -> results/movmech_posinv_{a.tag}.npy")
