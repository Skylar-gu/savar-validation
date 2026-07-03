# Agent session results — 2026-07-03

Branch `agent/session-2026-07-03`. Plan: `notes/agent_progress.md` +
`notes/agent_session_plan.md`. Numbers first, interpretation second.
Closing section (status table, verdicts, follow-ups) at the bottom.

---

## Block A — VPD on the hetdynamics_eqvar checkpoint

**Question.** Did VPD components collapse on finecadence because the GNN computes
one mechanism (data problem), or because of VPD config (method problem)?
hetdynamics_eqvar has a 22.8× per-mode timescale spread → separable mechanisms
should exist. Baselines (finecadence run `p-0625f7dd`, layer0.mlp0):
PR = 1.14/64, median pairwise gate-map |r| = 0.79, blob-usage CV = 0.04.
Success bar: PR meaningfully > 3/64 AND blob CV ≫ 0.04 with mode-preferential
components.

**Primary run** (`vpd_out/runs/p-6b9ba3ba`, M4 config, C=64, 5000 steps,
byte-identical loss/CI block to p-0625f7dd; only checkpoint + split changed).
Results: `results/vpd_eqvar_redundancy.npy`.

| module | PR (/64) | med pairwise r | med blob CV | #mode-pref (/64) | sp(φ, blob-entropy) |
|---|---|---|---|---|---|
| layers_0_mlp_0 | 1.08 | 0.81 | 0.038 | 0 | −0.08 |
| layers_0_mlp_2 | 1.18 | 0.54 | 0.066 | 1 | −0.32 |
| layers_1_mlp_0 | 1.15 | 0.03 | 0.074 | 6 | −0.30 |
| layers_1_mlp_2 | 1.27 | 0.24 | 0.052 | 7 | −0.66 |
| layers_2_mlp_0 | 1.31 | 0.24 | 0.058 | 4 | −0.48 |
| layers_2_mlp_2 | 1.54 | 0.35 | 0.117 | 12 | −0.31 |
| layers_3_mlp_0 | 1.27 | 0.54 | 0.098 | 6 | −0.05 |
| layers_3_mlp_2 | 1.29 | 0.36 | 0.229 | 14 | +0.34 |

**Outcome: success criterion NOT met.** All 8 modules sit at PR 1.08–1.54, far
below the >3/64 bar; layer-0 blob CV (0.038) is *identical* to the finecadence
baseline (0.04–0.05). The 22.8× timescale spread did not make VPD components
differentiate. There is only mild late-layer movement (layers_2_mlp_2 PR
1.29→1.54, mode-preferential count 8→12; layers_3_mlp_2 7→14 with the
φ-vs-entropy Spearman flipping sign to +0.34 — slow modes attracting slightly
more dedicated components at the output end).

**Secondary run** (config test: C=16, ImportanceMinimality ×30 — the loss-balance
amplifier flagged in results_gnn §5): launched
(`vpd/config_gnn_vpd_m4_eqvar_c16.yaml`); diagnostics below when complete.

*(section to be completed when the C=16 run finishes)*

---

## Block B — Impulse-response dynamical test of the frozen eqvar GNN

**Setup.** Frozen `checkpoints_hetdynamics_eqvar/best.pt`; impulse = mode-j
spatial pattern (W_plus column, unit peak) × 1σ pixel std added to the last
input frame; 240 windows × 8 impulses; 12-step autoregressive rollout; response
projected onto W rows → R[i,j,τ]; edges by permutation null (α=0.01, 1000
pixel-permutations). Script `pcmci/impulse_response_gnn.py`, results
`results/impulse_response.npy`.

**Graph recovery (pair level, cross edges only):**
TP=5 FP=3 FN=7 → **P=0.625, R=0.417, F1=0.500** (PCMCI+ on true Z: 0.823).

- Every detected true edge peaks at (or 1 step after) its designed lag:
  X0→X1 (τ̂=1, ℓ=1), X0→X3 (1,1), X1→X2 (2,1), X3→X6 (2,2), X4→X5 (3,2).
- Missed edges are dominated by slow-mode effects: (0→5), (1→4), (3→7),
  (5→6), (6→7) all have effects in X4–X7 (φ≥0.68) whose responses build
  slowly; plus (2→0), (2→3).

**Self-response timescales:** Spearman(τ_design, τ_efold) = **0.738** (n=8),
but e-folding saturates at ≈1.8–2.1 steps for the slow modes (X5–X7 designed
4.0–12.0) — the 12-step rollout window and model damping truncate slow decays.

**Interpretation.** The frozen GNN demonstrably internalized a *directed,
lag-correct* subset of Φ (all five detected edges lag-consistent; precision
0.625 ≫ chance ≈ 0.21), and it orders the modes' intrinsic timescales
correctly, but the internals-free channel recovers only half the graph at this
budget — well short of the PCMCI-on-Z ballpark the success bar asked for.
A 24-step / 480-window sensitivity run (targeting the slow-mode misses) is
reported below.

**Sensitivity run** (24 steps, 480 windows, `results/impulse_response_s24.npy`):
edge detection is *byte-identical* — same TP/FP/FN (F1=0.500), same detected /
missed / false-positive sets, same peak lags. The graph-recovery result is
robust to doubling both the window count and the rollout horizon; the misses
are not a budget artifact. The e-folding estimates, however, are NOT robust:
with a 24-step fit window every mode collapses to τ̂ ≈ 0.8–1.2 (Spearman flips
to −0.71). Beyond ~10 rollout steps the model's response decays at a uniform,
model-imposed damping rate regardless of the designed φ — so (1) the frozen GNN
does not carry the designed slow timescales through long autoregressive
rollouts, and (2) the 12-step Spearman 0.74 should be read as "ordering visible
in the early response only", not as quantitative timescale recovery.

---

*(further blocks appended as they complete)*
