# Phase 7 SAE Findings

## Result

7/8 modes pass alignment (|r| ≥ 0.35). 0/8 modes are monosemantic.

The CNN's res3 features encode **global system dynamics**, not mode-specific states.

---

## Setup

- **Extraction:** Mode-weighted pooling — `feat_j[t, c] = W[j, :] @ act[t, c, :]` — yields (100, 8, 497, 256)
- **SAE:** TopK, input=256, features=512, K=25
- **Training:** 8 per-mode SAEs, each on one mode's 49,700 samples (100 × 497)
- **Evaluation threshold:** 0.35 (derived from theoretical ceilings below)

---

## Theoretical ceilings

Maximum achievable Pearson |r| between any linear combination of the 256-dim
res3 activations and Z_j (from Ridge regression, out-of-fold):

| Mode | Ceiling | Notes |
|------|---------|-------|
| X0   | 0.49    | border |
| X1   | 0.57    | can exceed 0.5 |
| X2   | 0.48    | border |
| X3   | 0.59    | can exceed 0.5 |
| X4   | 0.45    | below 0.5 |
| X5   | 0.36    | fundamentally limited |
| X6   | 0.58    | can exceed 0.5 |
| X7   | 0.47    | border |

The PCA structure of mode-weighted activations explains the limits:
- PC0: 86% variance (shared global activity across all modes)
- PC1: 12% variance
- PC2+: <1% total

The Z_j signal lives almost entirely in PC0+PC1. Modes with stronger causal
connectivity (X1, X3, X6) have higher ceilings because global activity tracks
hub-node states more reliably.

---

## Per-mode SAE results

| Mode | Best feat | max\|r\| | Ceiling | Frac | Specificity | Status |
|------|-----------|---------|---------|------|-------------|--------|
| X0   | f65       | 0.377   | 0.489   | 0.77 | −0.085      | ALIGN / polysemantic |
| X1   | f306      | 0.446   | 0.575   | 0.78 | −0.017      | ALIGN / polysemantic |
| X2   | f307      | 0.367   | 0.479   | 0.77 | −0.091      | ALIGN / polysemantic |
| X3   | f135      | 0.465   | 0.589   | 0.79 | +0.006      | ALIGN / polysemantic |
| X4   | f478      | 0.353   | 0.450   | 0.79 | −0.118      | ALIGN / polysemantic |
| X5   | f163      | 0.284   | 0.358   | 0.79 | −0.186      | FAIL / polysemantic |
| X6   | f444      | 0.449   | 0.584   | 0.77 | −0.008      | ALIGN / polysemantic |
| X7   | f26       | 0.355   | 0.465   | 0.76 | −0.110      | ALIGN / polysemantic |

**Specificity** = r(feature, Z_j) − max_k r(feature, Z_k): positive means the
feature is more correlated with mode j than with any other mode.

---

## Why features are polysemantic

Every mode's best SAE feature correlates nearly equally with all 8 modes' Z
values. The root cause is that the dominant direction in res3 activations (86%
of variance, PC0) is a **global activity** signal that tracks the overall
excitation level of the system.

This global activity correlates with all modes' Z values because:
1. Any active mode increases the overall observation Y(t), raising global CNN activation
2. Hub modes (X1, X3, X6) have higher correlations (~0.52) because global activity is
   more directly driven by their incoming/outgoing connections
3. Leaf modes (X5, X7) have lower correlations (~0.33) because they're more isolated

Differential analysis (feature maximizing r_j − max_k r_k) confirms this: the
most mode-specific feature for any mode has differential < 0.03 and r < 0.16 —
negligible mode-specific signal.

---

## Interpretation

The CNN's res3 layer has learned to track **global system state** (how active is
the system overall) rather than **individual mode states**. This is still
effective for forecasting — val RMSE = 1.0723 vs oracle floor = 1.061, gap of
0.011 — because future observations depend on the overall dynamics.

This finding motivates Phase 8: modes with higher causal centrality (more
connections in the PCMCI graph) should have their Z values better captured by
the global activity feature. If so, causal centrality and forecast-feature
importance (gradient or ablation) should be correlated.

---

## Pass/fail summary

| Criterion | Result |
|-----------|--------|
| Alignment (|r| ≥ 0.35) | 7/8 PASS |
| Strong alignment (|r| ≥ 0.5) | 0/8 FAIL |
| Monosemanticity | 0/8 FAIL |
| Fraction of ceiling | 77–79% (consistent) |

Phase 7 reports: CNN has mode information but not mode-specific representations.
Global-activity encoding is the dominant strategy.
