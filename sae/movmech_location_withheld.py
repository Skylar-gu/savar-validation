"""
P1-withheld — the location-WITHHELD left-vs-right test (spec v2 confound fix).

Motivation
----------
The oracle P1 (movmech_position_invariance.py) pools activations at the
ground-truth moving footprint  feat_k = W[k,:] @ H  before the readout. That
HANDS the readout the blob location; it only has to read CONTENT, which is the
same wherever the blob sits. So the "location invariance" it reports is partly
GIVEN by the W-pool, not learned. A translation-equivariant backbone passes it
trivially (P3 equivariance = 1.00).

The sharp test: WITHHOLD the location. Replace W-pooling with a LEARNED
attention pool that gets NO W — it must discover mode k's signature from content
(H) alone, and that signature must TRANSFER from one region to another:

    logits_k = H @ q_k                 (per-node score, q_k learnable, 256-dim)
    a_k      = softmax(logits_k)        (attention over the 2500 nodes)
    feat_k   = a_k @ H                  (256-dim pooled vector, location not given)
    pred_k   = feat_k · w_k + b_k

Train q_k,w_k,b_k on LEFT realisations of mode k (blob-centre column < mid),
evaluate on held-out RIGHT (out-of-region) and held-out LEFT (in-region);
repeat train-on-RIGHT; average. On the SAME window subset we also fit the oracle
W-pool ridge, so the contrast is apples-to-apples.

Read-out
--------
  oracle out-R^2 ≈ withheld out-R^2   -> a genuine location-transferable
      "what-code" exists: attention found mode k anywhere, invariance is LEARNED.
  withheld out-R^2 << oracle out-R^2  -> the only way to read mode k is to be
      TOLD its location; invariance was GIVEN by the W-pool, not learned.

Usage:
  python3 sae/movmech_location_withheld.py \
      --ckpt checkpoints/movmech_place/best.pt --variant plain \
      --raw data/realisations_movmech_place --tag place_plain \
      [--nwin 16 --steps 300 --seed 0]
Output: results/movmech_withheld_<tag>.npy  (+ prints a per-mode table)
"""

import sys, argparse, glob
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import torch
from pathlib import Path

NY = NX = 50
L = NY * NX
N_MODES = 8
DIM = 256
FINE_MID = 50.0
RIDGE_LAM = 10.0


# ── oracle W-pool ridge (numpy) — same readout family as movmech_position_invariance ──

def ridge_out_in(Xtr, ytr, Xin, yin, Xout, yout, lam=RIDGE_LAM):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xs = (Xtr - mu) / sd
    w = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ (ytr - ytr.mean()))
    b = ytr.mean()
    def r2(X, y):
        yh = ((X - mu) / sd) @ w + b
        ss = ((y - y.mean()) ** 2).sum()
        return float(1 - ((y - yh) ** 2).sum() / (ss + 1e-12))
    return r2(Xout, yout), r2(Xin, yin)


# ── learned attention pool (torch), location withheld ────────────────────────

def attn_out_in(Htr, ytr, Hin, yin, Hout, yout, dev, steps, seed, priors=None):
    """H*: (n_windows, L, 256) fp32 on GPU. Fit q,w,b on Htr; eval on Hin/Hout.
    Targets standardised by train stats. Returns (out_r2, in_r2).

    priors: optional (prior_tr, prior_in, prior_out), each (n, L) — a per-window
    soft bump at that realisation's TRUE blob centre. This HANDS localisation to
    the SAME attention/readout (logits += gamma * prior, gamma learnable). It is
    the capability CONTROL: if the identical probe recovers oracle skill once the
    location is provided, the withheld collapse is a self-localisation failure,
    not a weak readout."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.zeros(DIM, device=dev, requires_grad=True)
    w = (0.01 * torch.randn(DIM, generator=g)).to(dev).requires_grad_(True)
    b = torch.zeros(1, device=dev, requires_grad=True)
    params = [q, w, b]
    gamma = None
    if priors is not None:
        gamma = torch.ones(1, device=dev, requires_grad=True)  # prior sharpness
        params.append(gamma)
    ym, ysd = ytr.mean(), ytr.std() + 1e-8
    yt = ((ytr - ym) / ysd).to(dev)
    opt = torch.optim.Adam(params, lr=1e-2)

    def pooled(H, prior):
        logits = H @ q                                # (n, L)
        if prior is not None:
            logits = logits + gamma * prior           # per-window centre bump
        a = torch.softmax(logits, dim=1)
        return torch.einsum("nl,nlc->nc", a, H)       # (n, 256)

    p_tr = priors[0] if priors else None
    for _ in range(steps):
        opt.zero_grad()
        pred = pooled(Htr, p_tr) @ w + b
        loss = ((pred - yt) ** 2).mean()
        loss.backward()
        opt.step()

    def r2(H, y, prior):
        with torch.no_grad():
            yh = (pooled(H, prior) @ w + b).cpu().numpy() * float(ysd) + float(ym)
        y = y.cpu().numpy()
        ss = ((y - y.mean()) ** 2).sum()
        return float(1 - ((y - yh) ** 2).sum() / (ss + 1e-12))
    p_in = priors[1] if priors else None
    p_out = priors[2] if priors else None
    return r2(Hout, yout, p_out), r2(Hin, yin, p_in)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", default="plain")
    ap.add_argument("--raw", required=True, help="realisation dir (obs, latent_states, W, centres)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--nwin", type=int, default=16, help="windows cached per realisation")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--n-real", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--control", action="store_true",
                    help="positive control: also give the attention scorer (x,y) "
                         "coords, so it CAN localise by position (probe-capacity check)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # coarse-mesh (50x50) node coordinates, for the capability-control centre bump
    gy, gx = np.meshgrid(np.arange(NY), np.arange(NX), indexing="ij")
    GX = torch.from_numpy(gx.ravel().astype(np.float32)).to(dev)   # (L,)
    GY = torch.from_numpy(gy.ravel().astype(np.float32)).to(dev)

    # ── model + hook (final node H, exactly as the extract/oracle pipeline) ──
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
    from gnn_forecaster import MeshGNN, K
    if a.variant == "plain":
        model = MeshGNN(ny=NY, nx=NX, k=K).to(dev)
    else:
        from mesh_gnn_variants import VariantMeshGNN
        model = VariantMeshGNN(ny=NY, nx=NX, variant=a.variant, k=K).to(dev)
    ck = torch.load(a.ckpt, map_location=dev)
    model.load_state_dict(ck["model_state"]); model.eval()
    cap = {}
    model.decoder[0].register_forward_hook(lambda m, i, o: cap.__setitem__("H", i[0]))
    print(f"Loaded {a.variant}  val RMSE={ck.get('val_rmse', float('nan')):.4f}  dev={dev}  "
          f"nwin={a.nwin} steps={a.steps}")

    # ── cache H for nwin windows/realisation on GPU (fp32) ──
    paths = sorted(glob.glob(f"{a.raw}/realisation_*.npz"))[:a.n_real]
    R = len(paths)
    H_all = torch.empty((R, a.nwin, L, DIM), dtype=torch.float32, device=dev)
    Z_all = np.empty((R, N_MODES, a.nwin), dtype=np.float32)
    Wpool = np.empty((R, N_MODES, a.nwin, DIM), dtype=np.float32)  # oracle W-pool feats
    centres = np.empty((R, N_MODES, 2), dtype=np.float32)          # (row, col) fine grid
    with torch.no_grad():
        for r, p in enumerate(paths):
            d = np.load(p)
            obs = d["observations"].astype(np.float32)     # (2500, T)
            Z = d["latent_states"].astype(np.float32)      # (8, T)
            W = d["W"].astype(np.float32)                  # (8, 2500)
            centres[r] = d["centres"]
            T = obs.shape[1]
            starts = np.linspace(200, T - K - 1, a.nwin).astype(int)
            frames = obs.T.reshape(T, NY, NX)
            win = torch.from_numpy(np.stack([frames[t:t + K] for t in starts])).to(dev)
            model(win)
            H = cap["H"]                                    # (nwin, 2500, 256)
            H_all[r] = H
            Wt = torch.from_numpy(W).to(dev)
            Wpool[r] = torch.einsum("jl,nlc->jnc", Wt, H).cpu().numpy()   # (8, nwin, 256)
            Z_all[r] = Z[:, [t + K for t in starts]]
            if (r + 1) % 20 == 0:
                print(f"  cached [{r+1}/{R}]")

    # ── per-mode, both directions: oracle W-pool ridge vs learned attention ──
    print(f"\n=== {a.tag}  P1-WITHHELD (learned attention, no location) vs oracle (W-pool) ===")
    print(f"{'mech':>4} | {'oracle out':>10} {'withheld out':>12} {'wh-orc gap':>10} | "
          f"{'oracle in':>9} {'withheld in':>11} | nL/nR")
    rows = []
    SIG = 3.0  # coarse-px width of the capability-control centre bump
    def centre_bump(reals, k):
        """(n_r*nwin, L) soft bump at each realisation's TRUE mode-k centre."""
        cx = torch.from_numpy(centres[reals, k, 1] / 2.0).to(dev)   # fine->coarse col
        cy = torch.from_numpy(centres[reals, k, 0] / 2.0).to(dev)   # fine->coarse row
        d2 = (GX[None] - cx[:, None]) ** 2 + (GY[None] - cy[:, None]) ** 2  # (n_r, L)
        bump = -d2 / (2 * SIG ** 2)                                  # log-Gaussian logits
        return bump[:, None, :].expand(-1, a.nwin, -1).reshape(-1, L)

    for k in range(N_MODES):
        left = np.where(centres[:, k, 1] < FINE_MID)[0]
        right = np.where(centres[:, k, 1] >= FINE_MID)[0]
        if len(left) < 4 or len(right) < 4:
            rows.append(dict(k=k, orc_out=np.nan, wh_out=np.nan, orc_in=np.nan,
                             wh_in=np.nan, nL=len(left), nR=len(right)))
            continue
        orc_out, orc_in, wh_out, wh_in = [], [], [], []
        for A, B in ((left, right), (right, left)):
            Ap = rng.permutation(A)
            cut = max(1, int(0.8 * len(Ap)))
            tr, inr = Ap[:cut], Ap[cut:]
            # oracle W-pool ridge on the same windows
            def wp(reals):
                return (Wpool[reals, k].reshape(-1, DIM).astype(np.float64),
                        Z_all[reals, k].reshape(-1).astype(np.float64))
            Xtr, ytr = wp(tr); Xin, yin = wp(inr); Xout, yout = wp(B)
            o_out, o_in = ridge_out_in(Xtr, ytr, Xin, yin, Xout, yout)
            orc_out.append(o_out); orc_in.append(o_in)
            # learned attention (location withheld) on the same windows
            def hp(reals):
                H = H_all[reals].reshape(-1, L, DIM)
                y = torch.from_numpy(Z_all[reals, k].reshape(-1).astype(np.float32))
                return H, y
            Htr, yt = hp(tr); Hin, yi = hp(inr); Hout, yo = hp(B)
            priors = ((centre_bump(tr, k), centre_bump(inr, k), centre_bump(B, k))
                      if a.control else None)
            w_out, w_in = attn_out_in(Htr, yt, Hin, yi, Hout, yo, dev, a.steps,
                                      a.seed + k, priors=priors)
            wh_out.append(w_out); wh_in.append(w_in)
        row = dict(k=k, orc_out=float(np.mean(orc_out)), wh_out=float(np.mean(wh_out)),
                   orc_in=float(np.mean(orc_in)), wh_in=float(np.mean(wh_in)),
                   nL=len(left), nR=len(right))
        rows.append(row)
        print(f"X{k:>3} | {row['orc_out']:>10.3f} {row['wh_out']:>12.3f} "
              f"{row['wh_out']-row['orc_out']:>10.3f} | {row['orc_in']:>9.3f} "
              f"{row['wh_in']:>11.3f} | {row['nL']}/{row['nR']}")

    valid = [r for r in rows if not np.isnan(r["orc_out"])]
    m_orc = float(np.mean([r["orc_out"] for r in valid]))
    m_wh = float(np.mean([r["wh_out"] for r in valid]))
    # retention: how much of the oracle's out-region skill survives location-withholding
    ret = m_wh / m_orc if m_orc > 1e-3 else float("nan")
    print(f"\n  MEAN out-R^2:  oracle={m_orc:.3f}  withheld={m_wh:.3f}  "
          f"retention={ret:.2f}")
    verdict = ("LEARNED what-code (invariance transfers w/o location)" if ret > 0.7
               else "LOCATION-GIVEN (invariance collapses when withheld)" if ret < 0.3
               else "PARTIAL what-code")
    print(f"  VERDICT: {verdict}")

    Path("results").mkdir(exist_ok=True)
    out = dict(rows=rows, mean_oracle_out=m_orc, mean_withheld_out=m_wh,
               retention=ret, verdict=verdict, nwin=a.nwin, steps=a.steps)
    np.save(f"results/movmech_withheld_{a.tag}.npy", out, allow_pickle=True)
    print(f"\nsaved -> results/movmech_withheld_{a.tag}.npy")


if __name__ == "__main__":
    main()
