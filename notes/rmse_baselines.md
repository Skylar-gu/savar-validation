# Calculating Persistence and Oracle Floor RMSE

## Purpose

These two baselines bracket what your CNN's RMSE means:

```
Oracle floor        CNN best.pt       Persistence baseline
     |___________________|___________________|
     
  theoretical       your model          dumbest
  minimum                               possible forecast
```

- **Persistence baseline:** RMSE of predicting y_{t+1} = y_t. If your CNN doesn't beat this, it has learned nothing.
- **Oracle floor:** RMSE of predicting the structured component perfectly using the true Phi coefficients and the observed Z = W @ y_t as mode states. This is the irreducible minimum given this model's noise structure.

### Correct oracle floor formula

The residual after a perfect oracle prediction is:

```
w_{t+1} = W_plus @ eps_x_{t+1}  +  eps_y_{t+1}
```

Both noise sources contribute, so the oracle floor per cell is:

```
oracle floor = sqrt( (1/L) * ||W_plus||_F²  +  sigma_y² )
```

For this model, `(1/L) * ||W_plus||_F² = 0.1257` (fixed by W, independent of D_y). Consequently:

| D_y | sigma_y² | (1/L)·‖W⁺‖²_F | Oracle floor |
|-----|----------|----------------|--------------|
| I_L | 1.000 | 0.126 | √1.126 ≈ **1.061** |
| 0.05·I_L | 0.050 | 0.126 | √0.176 ≈ **0.419** |

**Important:** the naive approximation *oracle floor ≈ sigma_y* only holds when `(1/L)·‖W⁺‖_F² << sigma_y²`. For D_y = 0.05·I_L the mode-noise term dominates (0.126 > 0.05), so the floor is 0.419, not sqrt(0.05) ≈ 0.224.

---

## Setup

```python
import numpy as np
from pathlib import Path

# Load one realisation
data = np.load("path/to/realisation_000.npz")

X = data["observations"]       # shape (L, T) — grid observations y_t
Z = data["latent_states"]      # shape (N, T) — true mode states x_t (no noise)
G = data["ground_truth_graph"] # shape (N, N, tau_max) — Phi coefficients
W = data["W"]                  # shape (N, L) — mode weight matrix
W_plus = data["W_plus"]        # shape (L, N) — pseudoinverse of W

L, T = X.shape
N, _, tau_max = G.shape
```

---

## 1. Persistence Baseline RMSE

The persistence forecast predicts y_{t+1} = y_t — just repeat the current frame.

```python
def rmse(a, b):
    """Per-cell RMSE between arrays of shape (L, T)."""
    return np.sqrt(np.mean((a - b) ** 2))

def persistence_rmse(X):
    """
    X: (L, T) observation array.
    Forecast: y_{t+1} = y_t for t = 0, ..., T-2.
    Target:   y_{t+1}       for t = 0, ..., T-2.
    """
    forecast = X[:, :-1]   # y_0, y_1, ..., y_{T-2}
    target   = X[:, 1:]    # y_1, y_2, ..., y_{T-1}
    return rmse(forecast, target)

rmse_persist = persistence_rmse(X)
print(f"Persistence RMSE: {rmse_persist:.4f}")
```

**What to expect:** For D_y = I_L, approximately 1.48. For D_y = 0.05 · I_L, approximately 0.54. Higher noise means larger persistence RMSE because consecutive frames differ more.

---

## 2. Oracle Floor RMSE

The oracle forecast uses the **true latent states Z** (stored in the .npz) and the **true causal coefficients Phi** to predict the structured component of y_{t+1} perfectly. The residual error is purely the irreducible noise ε_{t+1}.

**Critical:** do not use W @ y_t to estimate mode states — this contaminates the estimate with projected observation noise, making the oracle worse than it should be. Always use Z directly.

```python
def oracle_rmse(X, Z, G, W_plus, tau_max):
    """
    X:       (L, T)          — grid observations (target only, not used for forecast)
    Z:       (N, T)          — TRUE latent mode states (from .npz, not estimated)
    G:       (N, N, tau_max) — ground truth Phi coefficients
    W_plus:  (L, N)          — pseudoinverse of W
    tau_max: int             — longest lag in causal graph

    For each t in [tau_max, T-1]:
      1. Apply true causal dynamics in mode space using Z
      2. Project predicted mode state back to grid via W_plus
      3. Compare to actual observation y_{t+1}

    Returns per-cell RMSE.
    """
    T = X.shape[1]
    forecasts = []
    targets = []

    for t in range(tau_max, T - 1):
        # Step 1: apply true VAR dynamics in mode space
        # x_pred = sum_tau Phi(tau) @ x_{t-tau}
        x_pred = np.zeros(Z.shape[0])  # (N,)
        for tau in range(1, tau_max + 1):
            # G[:, :, tau-1] is Phi(tau), shape (N, N)
            # G[j, i, tau-1] = coefficient of X_i(t-tau) in equation for X_j(t)
            x_pred += G[:, :, tau - 1] @ Z[:, t - tau + 1]
            # Note: Z[:, t-tau+1] is x_{t-tau} when forecasting t+1

        # Step 2: project predicted mode state to grid
        y_pred = W_plus @ x_pred   # (L,)

        forecasts.append(y_pred)
        targets.append(X[:, t + 1])

    forecasts = np.array(forecasts).T   # (L, T_eval)
    targets   = np.array(targets).T     # (L, T_eval)

    return rmse(forecasts, targets)

rmse_oracle = oracle_rmse(X, Z, G, W_plus, tau_max)
print(f"Oracle floor RMSE: {rmse_oracle:.4f}")
```

**What to expect:** See the oracle floor formula above. The correct expected values are:
- D_y = I_L → oracle RMSE ≈ 1.061
- D_y = 0.05 · I_L → oracle RMSE ≈ 0.419

These are not sigma_y. They include the projected mode-noise contribution `(1/L)·‖W⁺‖_F²`.

---

## 3. Lag Indexing — Common Source of Error

The reading convention in your ground truth graph is:

```
G[j, i, tau-1] = coefficient of X_i(t-tau) in the equation for X_j(t)
```

When forecasting y_{t+1} (one step ahead), you want:

```
x_pred[j] = sum over tau of: G[j, :, tau-1] @ Z[:, (t+1) - tau]
           = G[j, :, 0] @ Z[:, t]       # tau=1 contribution
           + G[j, :, 1] @ Z[:, t-1]     # tau=2 contribution
```

Verify with a sanity check: compute oracle RMSE on training data where Z is known exactly. It should equal sqrt(mean(diag(D_y))) to within sampling noise.

---

## 4. Putting It Together

```python
# Full calibration
rmse_persist = persistence_rmse(X)
rmse_oracle  = oracle_rmse(X, Z, G, W_plus, tau_max)

print(f"Persistence RMSE : {rmse_persist:.4f}")
print(f"Oracle floor RMSE: {rmse_oracle:.4f}")
print(f"CNN best.pt RMSE : 1.0723  (from checkpoints/best.pt)")
print()
print(f"CNN improvement over persistence : {rmse_persist - 1.0723:.4f}")
print(f"CNN gap above oracle floor       : {1.0723 - rmse_oracle:.4f}")
```

The CNN gap above the oracle floor should be small but positive. If it is:

| Gap | Interpretation |
|---|---|
| < 0 (CNN beats oracle) | Bug in oracle calculation — check lag indexing and use of Z not W@X |
| 0.0 – 0.1 | CNN has learned causal structure well; proceed to SAE |
| 0.1 – 0.3 | CNN is learning but not optimal; acceptable for Phase 1 |
| > 0.3 | CNN has not learned the causal structure; fix signal-to-noise before SAE |

---

## 5. Averaging Across Realisations

For reliable estimates, compute both baselines across all 100 realisations and report mean ± std:

```python
persist_scores = []
oracle_scores  = []

for path in sorted(Path("data/").glob("realisation_*.npz")):
    data    = np.load(path)
    X       = data["observations"]
    Z       = data["latent_states"]
    G       = data["ground_truth_graph"]
    W_plus  = data["W_plus"]
    tau_max = G.shape[2]

    persist_scores.append(persistence_rmse(X))
    oracle_scores.append(oracle_rmse(X, Z, G, W_plus, tau_max))

print(f"Persistence RMSE: {np.mean(persist_scores):.4f} ± {np.std(persist_scores):.4f}")
print(f"Oracle floor RMSE: {np.mean(oracle_scores):.4f} ± {np.std(oracle_scores):.4f}")
```

These become the reference numbers reported alongside CNN RMSE in your paper.
