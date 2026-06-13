# SAVAR Synthetic Dataset Requirements

## Purpose

This document specifies requirements for generating a synthetic SAVAR dataset for Phase 1 of the validation pipeline. The dataset must simultaneously satisfy three goals:

1. **Methodological validity** — assumptions of PCMCI and Mapped-PCMCI are cleanly satisfied so that failures in later pipeline phases are attributable to the methodology, not the data
2. **CNN trainability** — the spatiotemporal structure is rich enough that a convolutional forecaster must learn non-trivial internal representations to achieve good forecast skill
3. **Ground truth evaluability** — the latent causal structure is recoverable from discovered SAE features in Phase 7, and causal centrality varies enough across nodes to make Phase 8 a meaningful test

---

## 1. Spatial Domain

| Parameter | Value | Rationale |
|---|---|---|
| Grid shape | 50×50 | Gives ~312 grid points per mode for N=8; matches paper's benchmark scale |
| L (total grid points) | 2,500 | Above the ~500 point plateau where performance stabilises (paper Figure 7b) |
| Mode layout | Non-overlapping Gaussian blobs | Required for Markovian SCM assumption; avoids contemporaneous cross-mode confounding |
| Blob placement | Homogeneous, deterministic | Reproducibility; avoids degenerate random placements |

Non-overlapping modes are a hard requirement for Phase 1. Overlapping modes introduce contemporaneous correlations between mode time series through shared noise, violating PCMCI's causal sufficiency assumption and producing false positive edges. Introduce overlap only in later stress-test phases.

---

## 2. Latent Mode Structure

| Parameter | Value | Rationale |
|---|---|---|
| N (number of modes) | 8 | Enough for meaningful causal centrality variance; comfortable on 50×50 grid |
| W shape | (8, 2500) | One weight vector per mode over L grid points |
| W construction | Symmetric Gaussian blobs, random=False | Reproducible; clean spatial footprints for Varimax recovery |
| W normalisation | L1 per row | Consistent with paper; ensures mode scalars are interpretable weighted averages |

**Why N=8 specifically:** With N=3, causal centrality is nearly predetermined (one obvious root, one obvious sink). Phase 8's hypothesis test — whether causal centrality predicts forecast importance — requires genuine variance in centrality scores across nodes. N=8 with the graph structure below provides a hub node, intermediate nodes, and sink nodes with meaningfully different centrality profiles.

---

## 3. Noise Parameters

| Parameter | Value | Rationale |
|---|---|---|
| D_x | I_N (identity) | Independent latent innovations; satisfies PCMCI causal sufficiency exactly |
| D_y | I_L (identity) | Per-grid-point independent noise; positive definite Σ_y; matches paper baseline |
| λ (noise_strength) | 1.0 | Matches paper's synthetic dataset; strong enough mode signal for clean Varimax recovery |

**D_x = I_N** is required for Phase 1. Off-diagonal D_x introduces contemporaneous correlation between mode time series through the noise channel, creating hidden common causes at lag zero that PCMCI cannot account for and that produce spurious edges in the recovered graph.

**D_y = I_L** is preferred over D_y = 0 for two reasons. First, D_y = 0 produces a positive semi-definite (not positive definite) Σ_y, causing numerical issues during sampling for grid points outside all mode regions. Second, per-grid-point noise makes the mode recovery problem harder in a realistic way — Varimax must separate mode-structured variance from isotropic noise, which is closer to the condition your SAE will face on GraphCast activations.

**Implied covariance:**
```
Σ_y = λ · W⁺ D_x (W⁺)ᵀ + D_y
     = 1.0 · W⁺ I_N (W⁺)ᵀ + I_L
```
Verify positive definiteness: eigenvalue range should be strictly positive. With D_y = I_L this is guaranteed.

---

## 4. Causal Graph G = Φ(τ)

### Requirements

The graph must satisfy all of the following:

- **Stability:** spectral radius of VAR companion matrix strictly less than 1
- **Varied centrality:** hub nodes (high out-degree), intermediate nodes, and sink nodes (zero out-degree) must all be present
- **Mixed lags:** both lag-1 and lag-2 edges present; some nodes connected only at lag 2
- **Negative coefficients:** at least 2–3 negative cross-mode edges distributed across the graph
- **No pure chain:** at least one converging path (two modes driving the same target) and one diverging path (one mode driving two targets) — these create the confounding structure that separates PCMCI from correlation
- **No feedback loops at Phase 1:** bidirected contemporaneous edges require PCMCIplus; introduce in later phases only

### Coefficient Ranges

Following paper Section 4.1.1:

| Coefficient type | Range | Distribution |
|---|---|---|
| Autocorrelation (diagonal Φ) | 0.3–0.6, varying per mode | Truncated Gaussian, mean 0.3, outside (−0.2, 0.2) |
| Cross-mode (off-diagonal Φ) | 0.2–0.45 in magnitude | Truncated Gaussian, mean 0.3, P(negative) = 0.5 |
| Number of cross-dependencies | ~12 edges for N=8 | ~1.5× number of modes, following paper's N=5 default of 5 edges scaled up |

### Suggested Graph Structure for N=8

```
Autocorrelations (all modes, varying):
  X0: 0.45,  X1: 0.50,  X2: 0.35,  X3: 0.55
  X4: 0.40,  X5: 0.30,  X6: 0.50,  X7: 0.45

Cross-mode edges:
  X0(t-1) → X1(t)   +0.35   hub: X0 drives multiple
  X0(t-1) → X3(t)   +0.30
  X0(t-2) → X5(t)   -0.20   long-range negative
  X1(t-1) → X2(t)   +0.40   chain forward
  X1(t-2) → X4(t)   +0.25   lag-2 only edge
  X2(t-1) → X3(t)   -0.30   converging negative into X3
  X3(t-1) → X6(t)   +0.30
  X4(t-1) → X5(t)   +0.35
  X5(t-2) → X6(t)   +0.25   two paths into X6
  X6(t-1) → X7(t)   +0.20   sink
  X3(t-2) → X7(t)   -0.15   second path to sink, negative
  X2(t-2) → X0(t)   +0.22   weak feedback from downstream
```

This gives:
- **X0:** high out-degree, moderate in-degree — primary hub
- **X3:** converging inputs from two directions — high betweenness
- **X7:** pure sink, zero out-degree — lowest causal centrality
- **Mixed negative edges:** distributed across hubs and sinks
- **Lag-2 only edge X1→X4:** hard case for PCMCI, tests lag detection

### Stability Check (Required Before Use)

```python
import numpy as np

def check_stability(Phi_list):
    """
    Phi_list: list of NxN arrays [Phi(tau=1), Phi(tau=2), ...]
    Returns spectral radius of companion matrix.
    Must be < 1 for stability.
    """
    N = Phi_list[0].shape[0]
    p = len(Phi_list)
    
    top_row = np.hstack(Phi_list)
    bottom = np.hstack([np.eye(N * (p-1)), np.zeros((N * (p-1), N))])
    companion = np.vstack([top_row, bottom])
    
    spectral_radius = np.max(np.abs(np.linalg.eigvals(companion)))
    print(f"Spectral radius: {spectral_radius:.4f}")
    assert spectral_radius < 1.0, "VAR process is unstable — reduce coefficient magnitudes"
    return spectral_radius
```

If the spectral radius exceeds 1, reduce the largest cross-mode coefficients proportionally until stability is achieved. Do not reduce autocorrelations below 0.2 as this removes the autocorrelation challenge PCMCI is designed to handle.

---

## 5. Time Series

| Parameter | Value | Rationale |
|---|---|---|
| T (usable timesteps) | 500 | Matches paper's synthetic baseline |
| Burn-in | 200 | Ensures stationarity before usable samples begin |
| Total generated | 700 | T + burn-in; discard first 200 |
| Number of realisations | 100 | Required for confidence intervals on evaluation metrics, matching paper |
| τ_max | 2 | Matches longest lag in causal graph |

---

## 6. Data Split

| Split | Fraction | Timesteps (approx) |
|---|---|---|
| Train | 70% | 350 |
| Validation | 15% | 75 |
| Test | 15% | 75 |

Split along the time axis only. Do not shuffle — temporal order must be preserved for both CNN training and causal discovery. The causal discovery pipeline (Phase 6) should be run on the full 500 timesteps, not just the training split, since PCMCI needs as many samples as possible.

---

## 7. Storage Format

Store each realisation as:

```python
{
    "observations":        X,   # shape (L, T) = (2500, 500) — the grid field y_t
    "latent_states":       Z,   # shape (N, T) = (8, 500)    — the mode time series x_t
    "ground_truth_graph":  G,   # shape (N, N, tau_max)      — Phi coefficients
    "W":                   W,   # shape (N, L) = (8, 2500)   — mode weight matrix
    "W_plus":              Wp,  # shape (L, N) = (2500, 8)   — pseudoinverse
    "metadata":            {...} # N, L, T, lambda, seed, spectral_radius
}
```

**Retaining latent_states Z is critical.** Phase 7 evaluation requires computing corr(SAE_feature_i, Z_j) for every feature-mode pair. Without Z stored at generation time this evaluation is impossible.

Suggested S3 structure:
```
s3://savar-project/
  raw/
    realisation_000.npz
    realisation_001.npz
    ...
    realisation_099.npz
  processed/
    train/
    val/
    test/
  metadata/
    ground_truth_graph.npy
    W.npy
    stability_check.json
```

---

## 8. CNN Forecasting Requirements

The dataset must be structured to support the forecaster defined in Phase 2.

**Forecast target:**
```
y_{t-k:t} → y_{t+1}
```
where k is the temporal window (recommend k=3, giving the CNN access to enough history to detect lag-2 causal effects).

**Input tensor shape per sample:**
```
(k, H, W) = (3, 50, 50)
```
reshaped from the flat (L, T) observations array. The 50×50 spatial structure is essential — it gives the 3D convolutional forecaster a meaningful spatial inductive bias to exploit, which is what creates rich internal representations for the SAE to analyse.

**Why this matters for the pipeline:** If the CNN can achieve good forecast skill with a trivially simple computation (e.g. just copying the last frame), its internal activations will be low-rank and uninteresting for SAE analysis. The causal graph structure — especially the lag-2 edges and negative coefficients — ensures the forecaster must learn to integrate information across time and space in a non-trivial way, producing richer activations.

**Forecast quality target:** The CNN should achieve substantially better RMSE than a persistence baseline (y_{t+1} = y_t). If it doesn't, the causal structure is too weak or the noise too strong — increase λ or reduce coefficient magnitudes.

---

## 9. Verification Checklist

Before committing to any realisation for downstream pipeline phases, verify:

- [ ] Spectral radius of companion matrix < 1
- [ ] Σ_y is positive definite (all eigenvalues > 0)
- [ ] W rows are non-overlapping (confirm zero intersection between mode support sets)
- [ ] W rows are L1-normalised
- [ ] Mode time series Z are stationary (ADF test p < 0.05 for each mode)
- [ ] latent_states Z are stored alongside observations X
- [ ] CNN trained on this data achieves RMSE below persistence baseline
- [ ] Causal centrality varies across nodes (out-degree not uniform)
- [ ] At least one negative cross-mode coefficient present
- [ ] Both lag-1 and lag-2 edges present in G

---

## 10. Progression After Phase 1

Once the pipeline succeeds on this clean baseline, stress-test by modifying one parameter at a time:

| Phase | Modification | What it tests |
|---|---|---|
| 1b | Increase D_x off-diagonal to 0.3 | Contemporaneous confounding |
| 1c | Introduce mild mode overlap | Assumption violation robustness |
| 1d | Add nonstationary trend (Ornstein-Uhlenbeck) | Nonstationarity robustness |
| 1e | Reduce λ from 1.0 to 0.2 | Weak mode signal — harder recovery |
| 1f | Increase N to 15, scale grid to 70×70 | Higher dimensionality |

Each modification should degrade performance gracefully rather than catastrophically. If a single modification causes complete pipeline failure, that points to a specific assumption the method relies on too heavily — important to know before applying to GraphCast.
