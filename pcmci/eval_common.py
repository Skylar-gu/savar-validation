"""
Shared causal-discovery evaluation helpers.

Used by run_baselines.py for PCMCI, DYNOTEARS, and TSCI so every method is scored
the same way against the same ground truth G.

Index conventions (identical to run_pcmci.py)
---------------------------------------------
Ground truth G:
  G[eff, cause, tau-1] = coefficient of X_cause(t-tau) in X_eff(t)

Method outputs in this repo are stored as (cause, eff, tau):
  p_matrix[cause, eff, tau]    — p-value for X_cause(t-tau) -> X_eff(t)   (PCMCI)
  score_matrix[cause, eff, tau]— signed strength for the same link        (all methods)

Two evaluation granularities:
  * lag-resolved : edge tuple (cause, eff, tau), tau in [1, tau_max]
                   — only meaningful for methods that resolve lag (PCMCI, DYNOTEARS)
  * summary graph: edge tuple (cause, eff), "is there any lagged link cause->eff"
                   — fair for every method incl. TSCI (which has no lag).
                   AUROC/AUPRC are computed here; this matches the CausalDynamics
                   leaderboard, which scores summary-graph accuracy.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


# ── lag-resolved ground truth / detection ─────────────────────────────────────

def gt_edges(G, cross_only=True):
    """Set of (cause, eff, tau) for nonzero entries in G; tau in [1, tau_max]."""
    N, _, tau_max = G.shape
    out = set()
    for eff in range(N):
        for cause in range(N):
            if cross_only and cause == eff:
                continue
            for tau in range(1, tau_max + 1):
                if G[eff, cause, tau - 1] != 0:
                    out.add((cause, eff, tau))
    return out


def detected_from_pmatrix(p_matrix, alpha, cross_only=True):
    """Set of (cause, eff, tau) where p_matrix[cause, eff, tau] < alpha (PCMCI)."""
    N = p_matrix.shape[0]
    tau_max = p_matrix.shape[2] - 1
    out = set()
    for cause in range(N):
        for eff in range(N):
            if cross_only and cause == eff:
                continue
            for tau in range(1, tau_max + 1):
                if p_matrix[cause, eff, tau] < alpha:
                    out.add((cause, eff, tau))
    return out


def detected_from_scores(score_matrix, thresh, cross_only=True):
    """Set of (cause, eff, tau) where |score_matrix[cause, eff, tau]| > thresh.

    For score-based methods (DYNOTEARS) that emit signed weights, not p-values.
    score_matrix shape (N, N, tau_max+1); tau=0 (contemporaneous) is ignored.
    """
    N = score_matrix.shape[0]
    tau_max = score_matrix.shape[2] - 1
    out = set()
    for cause in range(N):
        for eff in range(N):
            if cross_only and cause == eff:
                continue
            for tau in range(1, tau_max + 1):
                if abs(score_matrix[cause, eff, tau]) > thresh:
                    out.add((cause, eff, tau))
    return out


def prf(tp, fp, fn):
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec  = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return prec, rec, f1


# ── summary graph (lag collapsed) ─────────────────────────────────────────────

def summary_gt_matrix(G, cross_only=True):
    """Binary (N, N) summary adjacency: [cause, eff] = 1 if any tau has an edge."""
    N = G.shape[0]
    A = np.zeros((N, N), dtype=int)
    for eff in range(N):
        for cause in range(N):
            if cross_only and cause == eff:
                continue
            if np.any(G[eff, cause, :] != 0):
                A[cause, eff] = 1
    return A


def summary_score_matrix(score_matrix, cross_only=True):
    """Collapse lag: (N, N) where [cause, eff] = max_tau |score[cause, eff, tau]|.

    Accepts either a lag-resolved (N, N, tau_max+1) array or an already-summary
    (N, N) array (returned as |.| with the diagonal zeroed).
    """
    s = np.abs(np.asarray(score_matrix, dtype=float))
    if s.ndim == 3:
        s = s[:, :, 1:].max(axis=2)   # drop tau=0, max over lags
    if cross_only:
        np.fill_diagonal(s, 0.0)
    return s


def auroc_auprc(summary_score, G, cross_only=True):
    """AUROC/AUPRC of a (N, N) summary score against the summary ground truth.

    Off-diagonal entries only (when cross_only). Returns (auroc, auprc); either is
    nan if the label set is degenerate (all-0 or all-1) for that realisation.
    """
    A = summary_gt_matrix(G, cross_only=cross_only)
    N = A.shape[0]
    mask = ~np.eye(N, dtype=bool) if cross_only else np.ones((N, N), dtype=bool)
    y_true = A[mask].astype(int)
    y_score = np.asarray(summary_score)[mask]
    if y_true.min() == y_true.max():
        return np.nan, np.nan
    return roc_auc_score(y_true, y_score), average_precision_score(y_true, y_score)
