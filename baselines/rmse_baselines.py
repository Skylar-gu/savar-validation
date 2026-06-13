"""
Compute persistence and oracle-floor RMSE baselines.
Evaluated over all 100 realisations for both noise variants.

Persistence baseline : predict y_{t+1} = y_t
Oracle floor         : predict using true latent states Z and true Phi coefficients
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
from pathlib import Path


# ── core functions ────────────────────────────────────────────────────────────

def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def persistence_rmse(X):
    """X: (L, T). Forecast y_{t+1} = y_t."""
    return rmse(X[:, :-1], X[:, 1:])


def oracle_rmse(X, Z, G, W_plus):
    """
    X      : (L, T)          — observations (target)
    Z      : (N, T)          — true latent states
    G      : (N, N, tau_max) — Phi coefficients; G[j,i,tau-1] = coeff of X_i(t-tau) in X_j(t)
    W_plus : (L, N)          — pseudoinverse of W

    For each t in [tau_max, T-2], forecast y_{t+1}:
      x_pred = sum_{tau=1}^{tau_max} G[:,:,tau-1] @ Z[:, t - tau + 1]
      y_pred = W_plus @ x_pred
    """
    tau_max = G.shape[2]
    T       = X.shape[1]

    forecasts = []
    targets   = []

    for t in range(tau_max, T - 1):
        x_pred = np.zeros(Z.shape[0])
        for tau in range(1, tau_max + 1):
            x_pred += G[:, :, tau - 1] @ Z[:, t - tau + 1]
        forecasts.append(W_plus @ x_pred)
        targets.append(X[:, t + 1])

    forecasts = np.array(forecasts).T   # (L, T_eval)
    targets   = np.array(targets).T

    return rmse(forecasts, targets)


# ── evaluate one experiment ───────────────────────────────────────────────────

def evaluate(real_dir, ckpt_path, label):
    paths = sorted(Path(real_dir).glob("realisation_*.npz"))
    assert len(paths) > 0, f"No realisations found in {real_dir}"

    persist_scores = []
    oracle_scores  = []

    for path in paths:
        d       = np.load(path)
        X       = d["observations"].astype(np.float64)   # (L, T)
        Z       = d["latent_states"].astype(np.float64)  # (N, T)
        G       = d["ground_truth_graph"].astype(np.float64)
        W_plus  = d["W_plus"].astype(np.float64)

        persist_scores.append(persistence_rmse(X))
        oracle_scores.append(oracle_rmse(X, Z, G, W_plus))

    p_mean, p_std = np.mean(persist_scores), np.std(persist_scores)
    o_mean, o_std = np.mean(oracle_scores),  np.std(oracle_scores)

    # load CNN best val RMSE from checkpoint
    cnn_rmse = None
    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cnn_rmse = float(ckpt["val_rmse"])

    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  Realisations      : {len(paths)}")
    print(f"  Persistence RMSE  : {p_mean:.4f}  ±  {p_std:.4f}")
    print(f"  Oracle floor RMSE : {o_mean:.4f}  ±  {o_std:.4f}")

    if cnn_rmse is not None:
        gap_vs_persist = p_mean - cnn_rmse
        gap_vs_oracle  = cnn_rmse - o_mean
        frac_closed    = gap_vs_persist / (p_mean - o_mean) * 100 if (p_mean - o_mean) > 0 else float("nan")
        print(f"  CNN best.pt RMSE  : {cnn_rmse:.4f}")
        print()
        print(f"  CNN improvement over persistence : {gap_vs_persist:+.4f}")
        print(f"  CNN gap above oracle floor       : {gap_vs_oracle:+.4f}")
        print(f"  Fraction of gap closed           : {frac_closed:.1f}%")
    else:
        print(f"  CNN best.pt RMSE  : (checkpoint not found at {ckpt_path})")

    return {
        "persist_mean": p_mean, "persist_std": p_std,
        "oracle_mean":  o_mean, "oracle_std":  o_std,
        "cnn_rmse":     cnn_rmse,
    }


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("RMSE Baselines")
    print("==============")

    baseline = evaluate(
        real_dir  = "data/realisations",
        ckpt_path = "checkpoints/best.pt",
        label     = "D_y = 1.00 × I_L  (baseline)",
    )

    dy005 = evaluate(
        real_dir  = "data/realisations_dy005",
        ckpt_path = "checkpoints_dy005/best.pt",
        label     = "D_y = 0.05 × I_L  (low-noise experiment)",
    )

    print(f"\n{'─'*60}")
    print("  Summary comparison")
    print(f"{'─'*60}")
    print(f"  {'Metric':<32} {'D_y=I':>10}  {'D_y=0.05I':>10}")
    print(f"  {'-'*54}")
    print(f"  {'Persistence RMSE':<32} {baseline['persist_mean']:>10.4f}  {dy005['persist_mean']:>10.4f}")
    print(f"  {'Oracle floor RMSE':<32} {baseline['oracle_mean']:>10.4f}  {dy005['oracle_mean']:>10.4f}")
    if baseline['cnn_rmse'] and dy005['cnn_rmse']:
        print(f"  {'CNN best.pt RMSE':<32} {baseline['cnn_rmse']:>10.4f}  {dy005['cnn_rmse']:>10.4f}")
    print()
    print("  Oracle floor sanity check (should ≈ sqrt(D_y scale)):")
    print(f"    D_y=1.00 → expected ≈ 1.000, got {baseline['oracle_mean']:.4f}")
    print(f"    D_y=0.05 → expected ≈ {0.05**0.5:.4f}, got {dy005['oracle_mean']:.4f}")
