"""
T1-T5 sub-resolution recovery ladder on the moving-mechanism testbed
(spec v2 §"read between the pixels" + T1/T2/T4/T5; T6 SAE rung optional).

For each fine sub-source m (0..15), target = true Z_fine[m](t). Recovery corr at
levels of increasing model power:
  L0  coarse single frame  — ridge on the single-frame D_sub readout of m's
      (aliased, co-located-pair-mixing) envelope W_sub_coarse[m] @ obs[:,t].
      Resolution floor: the colliding pair shares one envelope, so L0 cannot
      separate them (~ the mixture baseline).
  L1  coarse K-window linear — ridge on a +/-HW-frame window of ALL parent- and
      sub-envelope-pooled coarse streams (temporal context; linear de-aliasing).
  L2  linear spatiotemporal surrogate — VAR-style: same as L1 but the window is
      the model input; kept as the linear ceiling (a *linear* dynamical readout).
  L3  GNN activations — ridge on W_sub_coarse[m]-pooled final node states (256).
  L4  SAE latents (optional) — encode L3 activations through the mixed SAE.

Operators (T1): run the whole ladder under D_sub (aliasing, recover) and D_avg
(low-pass, chance). Reported side by side.
T3 (hallucination): residual concentration — for the best recovered level, split
error power into the "trained-visible" part (correlation with the parent stream,
low-k) vs the "fine detail" part; real recovery -> residual flat, hallucination
-> residual piles on the fine component. Reported as residual_fine_frac.
T4 (twin null): sub-sources of blob 7 (indices 14,15) have identical phi+parent;
their INDIVIDUAL recovery must be ~chance at every level (their sum is fair).
T5 (dose-response): recovery vs the fast/slow phi gap is read across the
non-twin subs (Delta phi = 0.9-0.3 = 0.6) vs the twin (Delta phi = 0).

Usage:
  python3 sae/subres_ladder.py --ckpt checkpoints/movmech_place/best.pt \
      --variant plain --raw data/realisations_movmech_place \
      --raw-avg data/realisations_movmech_place --avg-key observations_avg \
      --tag place_plain [--sae sae_data_movmech_place/sae_mixed.pt] --nreal 20
Output: results/subres_ladder_<tag>.npy
"""

import sys, argparse, glob
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from gnn_forecaster import MeshGNN, K

N_SUB = 16
NY = NX = 50
L = NY * NX
HW = 8
RIDGE_LAM = 10.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ridge_cv(X, y, lam=RIDGE_LAM, folds=4):
    """corr(held-out prediction, y) via simple k-fold; returns (corr, residual)."""
    n = len(y); idx = np.arange(n); yhat = np.empty(n)
    for f in range(folds):
        te = idx[f::folds]; tr = np.setdiff1d(idx, te)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xtr = (X[tr] - mu) / sd; Xte = (X[te] - mu) / sd
        b = y[tr].mean()
        w = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ (y[tr] - b))
        yhat[te] = Xte @ w + b
    yc, hc = y - y.mean(), yhat - yhat.mean()
    corr = float(yc @ hc / (np.linalg.norm(yc) * np.linalg.norm(hc) + 1e-12))
    return corr, y - yhat


def window_feats(streams, HW):
    """streams: (T, P) pooled coarse streams -> (T-2HW, (2HW+1)*P) windowed."""
    T = streams.shape[0]
    return np.stack([streams[t-HW:t+HW+1].ravel() for t in range(HW, T-HW)])


def load_model(ckpt, variant):
    if variant == "plain":
        m = MeshGNN(ny=NY, nx=NX, k=K).to(DEVICE)
    else:
        from mesh_gnn_variants import VariantMeshGNN
        m = VariantMeshGNN(ny=NY, nx=NX, variant=variant, k=K).to(DEVICE)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE)["model_state"])
    m.eval()
    cap = {}
    m.decoder[0].register_forward_hook(lambda mod, i, o: cap.__setitem__("H", i[0]))
    return m, cap


def gnn_pool(model, cap, obs, Wsub, Wpar, ts, bs=64):
    """Pool final node states by each sub envelope -> (len(ts), 16, 256)."""
    frames = obs.T.reshape(-1, NY, NX)
    Ws = torch.from_numpy(Wsub).float().to(DEVICE)
    out = []
    with torch.no_grad():
        for i in range(0, len(ts), bs):
            chunk = ts[i:i+bs]
            win = torch.from_numpy(np.stack([frames[t:t+K] for t in chunk])).float().to(DEVICE)
            model(win); H = cap["H"].reshape(len(chunk), L, -1)
            out.append(torch.einsum("ml,blc->bmc", Ws, H).cpu().numpy())
    return np.concatenate(out, 0)


def run(ckpt, variant, raw_dir, obs_key, tag, nreal, sae_path=None):
    paths = sorted(glob.glob(f"{raw_dir}/realisation_*.npz"))[:nreal]
    model, cap = load_model(ckpt, variant)
    sub_meta = np.load(paths[0])["sub_meta"]                    # (16,5) phi,parent,lag,coeff,pat
    dphi = np.abs(sub_meta[:, 0] - np.array([0.3 if i % 2 == 0 else 0.9 for i in range(16)]))
    # per-level accumulators
    levels = ["L0", "L1", "L3"]
    if sae_path: levels.append("L4")
    acc = {lv: [[] for _ in range(N_SUB)] for lv in levels}
    resid_fine = [[] for _ in range(N_SUB)]

    sae = None
    if sae_path:
        from eval_sae_metrics import load_sae, encode as sae_encode
        sae, sm, ss = load_sae(sae_path)

    for p in paths:
        d = np.load(p)
        obs = d[obs_key].astype(np.float32)                    # (2500, T)
        Zf = d["Z_fine"].astype(np.float32)                    # (16, T)
        Zpar = d["latent_states"].astype(np.float32)           # (8, T)
        Wsub = d["W_sub_coarse"].astype(np.float32)            # (16, 2500)
        Wpar = d["W"].astype(np.float32)                       # (8, 2500)
        T = obs.shape[1]
        ts = list(range(0, T - K, 1))                          # stride 1
        tgt_idx = np.array([t + K for t in ts])

        env = (Wsub @ obs).T                                    # (T, 16) single-frame env
        par = (Wpar @ obs).T                                    # (T, 8)
        streams = np.concatenate([env, par], 1)                 # (T, 24)
        Wf = window_feats(streams, HW)                          # (T-2HW, 24*(2HW+1))
        win_ts = np.arange(HW, T - HW)

        acts = gnn_pool(model, cap, obs, Wsub, Wpar, ts)        # (len(ts),16,256)

        for m in range(N_SUB):
            y_all = Zf[m]
            # L0 single frame env (co-located pair mixed)
            c0, _ = ridge_cv(env[tgt_idx][:, [m]], y_all[tgt_idx])
            acc["L0"][m].append(c0)
            # L1 windowed linear (window centre t -> Z at t, nowcast)
            c1, res1 = ridge_cv(Wf, Zf[m][win_ts])
            acc["L1"][m].append(c1)
            # L3 GNN acts
            c3, res3 = ridge_cv(acts[:, m], y_all[tgt_idx])
            acc["L3"][m].append(c3)
            # T3 residual concentration: fraction of residual power explained by
            # the parent (trained-visible, low-k) stream = 1 - fine-detail frac
            par_m = Zpar[int(sub_meta[m, 1])][tgt_idx]
            pc = par_m - par_m.mean(); rc = res3 - res3.mean()
            expl = (pc @ rc) ** 2 / ((pc @ pc) * (rc @ rc) + 1e-12)
            resid_fine[m].append(1.0 - float(expl))            # high => residual is fine-detail
            if sae:
                code = sae_encode(sae, acts[:, m], sm, ss)
                c4, _ = ridge_cv(code, y_all[tgt_idx])
                acc["L4"][m].append(c4)

    print(f"\n=== T2 ladder [{tag}] obs={obs_key}  (corr to true Z_fine, {nreal} reals) ===")
    hdr = "sub  phi  par dphi | " + " ".join(f"{lv:>5}" for lv in levels) + " | rFine"
    print(hdr)
    summ = {}
    for m in range(N_SUB):
        row = {lv: float(np.mean(acc[lv][m])) for lv in levels}
        rf = float(np.mean(resid_fine[m]))
        summ[m] = dict(**row, resid_fine=rf, phi=float(sub_meta[m, 0]),
                       parent=int(sub_meta[m, 1]), dphi=float(dphi[m]))
        twin = " TWIN" if m in (14, 15) else ""
        print(f"X{m:>2} {sub_meta[m,0]:.2f} {int(sub_meta[m,1]):>3} {dphi[m]:.2f} | "
              + " ".join(f"{row[lv]:>5.2f}" for lv in levels) + f" | {rf:.2f}{twin}")
    # T5 dose-response: mean L3 for non-twin (dphi 0.6) vs twin (dphi 0)
    nt = [m for m in range(N_SUB) if m not in (14, 15)]
    print(f"\n  T5 dose: L3 non-twin (dphi~0.6) = {np.mean([summ[m]['L3'] for m in nt]):.3f}"
          f"  vs twin (dphi=0) X14/X15 = {summ[14]['L3']:.3f}/{summ[15]['L3']:.3f}")
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", default="plain")
    ap.add_argument("--raw", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--nreal", type=int, default=20)
    ap.add_argument("--sae", default=None)
    ap.add_argument("--operators", default="sub,avg", help="sub=D_sub obs, avg=D_avg obs (T1 fork)")
    a = ap.parse_args()
    keymap = {"sub": "observations", "avg": "observations_avg"}
    out = {}
    for op in a.operators.split(","):
        out[op] = run(a.ckpt, a.variant, a.raw, keymap[op], f"{a.tag}_{op}", a.nreal, a.sae)
    Path("results").mkdir(exist_ok=True)
    np.save(f"results/subres_ladder_{a.tag}.npy", out, allow_pickle=True)
    print(f"\nsaved -> results/subres_ladder_{a.tag}.npy")
