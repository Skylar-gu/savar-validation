"""
DYNOTEARS worker — runs INSIDE the isolated causalnex venv (.venv_causalnex).

causalnex pins numpy<1.24 / pandas<2.0, incompatible with the main env (numpy 2.0).
So DYNOTEARS is never imported in the main process: the adapter (pcmci/methods/
dynotears.py) writes a stacked input array, shells out to this script with the venv
python, and reads back a stacked score array.

I/O
---
--in   : .npy of shape (R, T, N)  — R realisations, T timesteps, N variables
--out  : .npy of shape (R, N, N, tau_max+1) — signed DYNOTEARS weights, indexed
         [realisation, cause, eff, tau]; tau=0 is the contemporaneous slice.
         Edge cause(t-tau) -> eff(t) has weight from node "{cause}_lag{tau}" ->
         "{eff}_lag0" in the fitted StructureModel.
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from causalnex.structure.dynotears import from_pandas_dynamic


def fit_one(Z_TN, tau_max, lambda_w, lambda_a, w_threshold, max_iter):
    """Z_TN: (T, N). Returns signed score matrix (N, N, tau_max+1) [cause,eff,tau]."""
    T, N = Z_TN.shape
    df = pd.DataFrame(Z_TN, columns=list(range(N)))
    sm = from_pandas_dynamic(
        df, p=tau_max,
        lambda_w=lambda_w, lambda_a=lambda_a,
        w_threshold=w_threshold, max_iter=max_iter,
    )
    score = np.zeros((N, N, tau_max + 1), dtype=np.float64)
    for u, v, d in sm.edges(data=True):
        # nodes are "{var}_lag{tau}"; edges land on a lag0 effect node
        cause, ctau = u.split("_lag")
        eff,   etau = v.split("_lag")
        cause, ctau, eff, etau = int(cause), int(ctau), int(eff), int(etau)
        if etau != 0:
            continue                      # only edges into the present matter
        score[cause, eff, ctau] = d["weight"]
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--tau_max", type=int, default=2)
    ap.add_argument("--lambda_w", type=float, default=0.05)
    ap.add_argument("--lambda_a", type=float, default=0.05)
    ap.add_argument("--w_threshold", type=float, default=0.0)
    ap.add_argument("--max_iter", type=int, default=100)
    a = ap.parse_args()

    data = np.load(a.inp)                  # (R, T, N)
    R, T, N = data.shape
    out = np.zeros((R, N, N, a.tau_max + 1), dtype=np.float64)
    for r in range(R):
        out[r] = fit_one(data[r], a.tau_max, a.lambda_w, a.lambda_a,
                          a.w_threshold, a.max_iter)
        if (r + 1) % 10 == 0:
            print(f"  [dynotears worker] {r+1}/{R}", flush=True)
    np.save(a.out, out)
    print(f"  [dynotears worker] saved {out.shape} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
