"""
TSCI adapter (main env).

Tangent Space Causal Inference is *pairwise* and *not lag-resolved*: for two scalar
signals it returns a directional strength X->Y and Y->X. So TSCI yields only a
SUMMARY graph (cause -> eff, lag collapsed) — evaluated with AUROC/AUPRC, as on the
CausalDynamics leaderboard. There is no honest lag-resolved P/R/F1 for TSCI.

discover_all(data_RTN) returns a summary score array (R, N, N) where
score[r, cause, eff] = mean TSCI strength for cause -> eff in realisation r.
Diagonal is 0.
"""

import numpy as np

from .tsci.tsci_core import (
    tsci_nn, delay_embed, discrete_velocity, lag_select, false_nearest_neighbors,
)


def _embed_mode(sig, tau, Q):
    """Return (state, dstate) delay embeddings of a (T,) signal and its velocity."""
    sig = np.asarray(sig, dtype=float).reshape(-1, 1)
    state = delay_embed(sig, tau, Q)
    dstate = delay_embed(discrete_velocity(sig), tau, Q)
    return state, dstate


def _pair_scores(xi, yj, tau_x, Q_x, tau_y, Q_y, fraction_train):
    xs, dxs = _embed_mode(xi, tau_x, Q_x)
    ys, dys = _embed_mode(yj, tau_y, Q_y)
    L = min(xs.shape[0], ys.shape[0])
    xs, dxs, ys, dys = xs[-L:], dxs[-L:], ys[-L:], dys[-L:]
    s_x2y, s_y2x = tsci_nn(xs, ys, dxs, dys, fraction_train=fraction_train)
    return float(np.mean(s_x2y)), float(np.mean(s_y2x))


def discover_one(Z_TN, tau=1, Q=3, auto_embed=False, fraction_train=0.8):
    """Z_TN: (T, N). Returns summary score (N, N) [cause, eff], diagonal 0."""
    Z_TN = np.asarray(Z_TN, dtype=float)
    T, N = Z_TN.shape

    # Per-mode embedding hyperparameters
    if auto_embed:
        taus, Qs = [], []
        for j in range(N):
            tj = lag_select(Z_TN[:, j], theta=0.5)
            taus.append(tj)
            Qs.append(false_nearest_neighbors(Z_TN[:, j], tj, fnn_tol=0.01))
    else:
        taus = [tau] * N
        Qs = [Q] * N

    score = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            s_ij, s_ji = _pair_scores(
                Z_TN[:, i], Z_TN[:, j], taus[i], Qs[i], taus[j], Qs[j], fraction_train)
            score[i, j] = s_ij      # i -> j
            score[j, i] = s_ji      # j -> i
    return score


def discover_all(data_RTN, tau=1, Q=3, auto_embed=False, fraction_train=0.8,
                 verbose=True):
    """data_RTN: (R, T, N). Returns summary scores (R, N, N)."""
    data_RTN = np.asarray(data_RTN, dtype=float)
    R = data_RTN.shape[0]
    N = data_RTN.shape[2]
    out = np.zeros((R, N, N), dtype=np.float64)
    for r in range(R):
        out[r] = discover_one(data_RTN[r], tau=tau, Q=Q,
                              auto_embed=auto_embed, fraction_train=fraction_train)
        if verbose and (r + 1) % 10 == 0:
            print(f"  [tsci] {r+1}/{R}", flush=True)
    return out
