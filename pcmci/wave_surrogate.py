"""
Traveling-wave / spatiotemporal-Fourier surrogate — the T2 "wave rung"
(spec v2 §T2; Wave-RNN, Keller & Welling ICLR 2024, arXiv:2309.08045).

Moving structure is naturally wave-like: a blob advecting at speed v puts its
energy on the dispersion line omega = v*k (temporal frequency proportional to
spatial wavenumber). Two fine scales that COLLIDE spatially under D_sub
(same observed k) but advect at different speeds separate in the space-time
(k, omega) plane where a static per-frame model cannot tell them apart.

This surrogate builds a LINEAR readout of Z_fine from the space-time Fourier
representation of a coarse patch around each sub-source, and compares it to a
static (single-frame) linear readout — the gain is the "wave de-aliasing"
signal. Used mainly for MOVE=advect; on MOVE=place (no within-sequence drift)
it should give ~no gain over the static/windowed linear rung (a useful null
that attributes any advect gain to drift, not to the extra features).

For each sub-source m: extract a P x P coarse patch centred on the blob, take a
sliding space-time window (Tw frames), 3-D rFFT -> low-|k|,low-|omega| complex
coefficients as real features, ridge (k-fold) to Z_fine(t at window centre).
Metric: corr to true Z_fine; reported vs the static per-frame patch readout.

Usage:
  python3 pcmci/wave_surrogate.py --raw data/realisations_movmech_advect \
      --obs-key observations --tag advect --nreal 15
Output: results/wave_surrogate_<tag>.npy
"""

import sys, argparse, glob
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
from pathlib import Path

NY = NX = 50
N_SUB = 16
PATCH = 10          # coarse patch half-not; PxP window
TW = 12             # temporal window
KMAX = 4            # keep |k|<=KMAX spatial, |omega|<=WMAX temporal coeffs
WMAX = 4
RIDGE_LAM = 10.0


def ridge_cv(X, y, lam=RIDGE_LAM, folds=4):
    n = len(y); idx = np.arange(n); yhat = np.empty(n)
    for f in range(folds):
        te = idx[f::folds]; tr = np.setdiff1d(idx, te)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xtr = (X[tr]-mu)/sd; Xte = (X[te]-mu)/sd
        b = y[tr].mean()
        w = np.linalg.solve(Xtr.T@Xtr + lam*np.eye(Xtr.shape[1]), Xtr.T@(y[tr]-b))
        yhat[te] = Xte@w + b
    yc, hc = y-y.mean(), yhat-yhat.mean()
    return float(yc@hc/(np.linalg.norm(yc)*np.linalg.norm(hc)+1e-12))


def patch_coords(cy, cx, half=PATCH):
    r0, c0 = int(np.clip(cy-half, 0, NY-2*half)), int(np.clip(cx-half, 0, NX-2*half))
    return r0, c0


def run(raw_dir, obs_key, tag, nreal):
    paths = sorted(glob.glob(f"{raw_dir}/realisation_*.npz"))[:nreal]
    static_acc = [[] for _ in range(N_SUB)]
    wave_acc   = [[] for _ in range(N_SUB)]
    for p in paths:
        d = np.load(p)
        obs = d[obs_key].astype(np.float32)                 # (2500, T)
        Zf  = d["Z_fine"].astype(np.float32)                # (16, T)
        sub_pos = d["sub_pos"].astype(np.float32)           # (16, 2) FINE coords
        T = obs.shape[1]
        frames = obs.T.reshape(T, NY, NX)
        for m in range(N_SUB):
            cy, cx = sub_pos[m] / 2.0                        # fine->coarse
            r0, c0 = patch_coords(cy, cx)
            patch = frames[:, r0:r0+2*PATCH, c0:c0+2*PATCH]  # (T, P, P)
            centres = np.arange(TW//2, T - TW//2)
            # static: single-frame flattened patch
            Xs = patch[centres].reshape(len(centres), -1)
            static_acc[m].append(ridge_cv(Xs, Zf[m][centres]))
            # wave: space-time window 3D rFFT, keep low k/omega band
            W = np.stack([patch[c-TW//2:c+TW//2] for c in centres])  # (n, TW, P, P)
            F = np.fft.rfftn(W, axes=(1, 2, 3))
            F = F[:, :WMAX+1, :KMAX+1, :KMAX+1]
            Xw = np.concatenate([F.real.reshape(len(centres), -1),
                                 F.imag.reshape(len(centres), -1)], 1)
            wave_acc[m].append(ridge_cv(Xw, Zf[m][centres]))
    print(f"\n=== wave surrogate [{tag}] obs={obs_key} ({nreal} reals) ===")
    print(f"{'sub':>4} {'static':>7} {'wave':>7} {'gain':>7}")
    summ = {}
    for m in range(N_SUB):
        s, w = float(np.mean(static_acc[m])), float(np.mean(wave_acc[m]))
        summ[m] = dict(static=s, wave=w, gain=w-s)
        tw = " TWIN" if m in (14, 15) else ""
        print(f"X{m:>2} {s:>7.3f} {w:>7.3f} {w-s:>+7.3f}{tw}")
    mg = np.mean([summ[m]['gain'] for m in range(N_SUB) if m not in (14, 15)])
    print(f"  mean non-twin wave gain = {mg:+.3f}")
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--obs-key", default="observations")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--nreal", type=int, default=15)
    a = ap.parse_args()
    out = run(a.raw, a.obs_key, a.tag, a.nreal)
    Path("results").mkdir(exist_ok=True)
    np.save(f"results/wave_surrogate_{a.tag}.npy", out, allow_pickle=True)
    print(f"\nsaved -> results/wave_surrogate_{a.tag}.npy")
