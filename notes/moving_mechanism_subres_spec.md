# Moving-mechanism sub-resolution recovery — spec

**Supersedes the position-locked framing in
[[subres_spatial_dealiasing_plan]].** Reuses its de-aliasing machinery (operator
fork `D_sub` vs `D_avg`, the recoverability ladder, the true-realization vs
statistics control) but fixes the design flaw that makes the whole thing
collapse into a position readout.

## Why the position-locked version can't answer the question

Session finding (2026-07-03, all three followups): the GNN carries full fine-lag
content but as **one mode-agnostic operator — mode identity lives in data
*locations*, not channels or mechanisms** ([[project_savar_gridlock]]). This was
*guaranteed* by the data: SAVAR modes are nailed to fixed disjoint `W` footprints
across every realisation, so a positional code is optimal and the network has
zero pressure to build a location-invariant channel/mechanism code. SAE (reads
channels) and VPD (carves mechanisms) therefore probe for something the data
forbids. A sub-resolution test on position-locked modes just re-measures "position
is decodable" — the grid-lock trap again.

## Hypothesis

If the **same dynamical mechanism appears at *varying* locations**, the network
is *forced* to encode "which mechanism" separately from "where" to generalise.
Then, and only then:
- SAE uniqueness can exceed its ~0 ceiling (a mechanism = a channel, not a place);
- VPD can find mechanism-preferential components;
- sub-grid de-aliasing becomes a genuine test (recovery can't be a fixed-pixel
  readout).

Falsifiable null: even with moving mechanisms the model builds no abstraction
(e.g. it just tracks blobs frame-to-frame through the K=3 window and never needs
cross-location invariance) → tools still fail → the barrier is deeper than
position-locking. Either outcome is informative.

## Generative design — moving-mechanism SAVAR (`generate_movmech.py`)

Fork `generate_hetdynamics.py`; keep the linear-mode dynamics (per-mechanism
self-loop φ, cross-mechanism causal graph Φ, spatially-correlated noise,
saturating nonlinearity, skew-normal innovations). **Change: decouple mechanism
identity from location.**

- **Mechanism = (φ_k, causal parents), invariant across realisations.** The
  causal graph Φ (X0→X1, X0→X3, …) is the SAME every realisation.
- **Location varies.** Two movement modes, both implemented, compared:
  - `MOVE=place` — draw each mechanism's blob centre freshly per realisation
    from a distribution over the grid (mechanism k is a blob *somewhere*, moving
    across realisations). Forces cross-realisation position-invariance.
  - `MOVE=advect` — blobs translate *within* a sequence at mechanism-specific
    speeds; speed itself can be part of the mechanism signature. Also creates the
    scale↔frequency (dispersion) coupling that powers spatial de-aliasing.
- **Non-overlap bookkeeping:** re-run the Block-G Jaccard check per realisation;
  allow controlled overlap in a later rung but start disjoint so pooling stays
  clean.
- **Fine sub-mechanisms (sub-resolution angle):** within each parent blob place
  2 fine sub-sources at distinct fine-grid positions with distinct φ (slow/fast)
  and distinct parents. Observe through `D_sub` (subsample → aliasing, primary)
  or `D_avg` (block-average → low-pass, control). Save fine ground truth
  `Z_fine(t)` per sub-source AND the per-realisation blob centres/positions.

## Test battery (rerun "all relevant tests" on the moving-mechanism testbed)

Carry over T0–T6 from [[subres_spatial_dealiasing_plan]], add the
position-invariance controls (P-tests) that are the whole point of this spec.

**T0 — spatial de-aliasing identifiability floor (clean Fourier, no GNN).**
Unchanged go/no-go gate: can an oracle de-alias `D_sub` given distinct φ? If it
fails, the sub-res angle is dead regardless of movement.

**P1 — cross-location generalisation of the mechanism readout (THE test).**
Train a readout (mechanism amplitude ← activations) on realisations where
mechanism k lives in region A; test on realisations where it lives in region B.
- Location code → fails (R² collapses out-of-region).
- Mechanism code → passes (R² holds across held-out locations).
This directly measures whether the abstraction the hypothesis predicts actually
formed. Report per-mechanism held-out-region R², and the gap vs in-region.

**P2 — position-shuffle null.** Shuffle the position labels; a mechanism code
should still decode identity from channels, a location code should not.

**T1 — operator fork.** `D_sub` recovery vs `D_avg` (should be recover vs chance).

**T2 — recoverability ladder** (coarse frame → K-window linear → linear
spatiotemporal surrogate → GNN activations → SAE latents), metric = corr to true
`Z_fine`.

**T3 — true-realisation vs statistics control** (hallucination detector).

**T4 — identical-twin null** (dynamically identical sub-pair must be
unrecoverable at every level/operator).

**T5 — dose–response on Δφ** (identifiability curve; connect to the 22.8× spread,
[[project_savar_mode_specialization]]).

**T6 — SAE + VPD re-runs on the moving-mechanism checkpoint (the payoff).**
- SAE: Hungarian MCC + **uniqueness** vs mechanism GT (`sae/eval_sae_metrics.py`,
  matched to `Z_fine` / mechanism amplitudes). Success = uniqueness meaningfully
  > 0 (baseline ~0; ZCA-whitening ceiling +0.068 from followup 3).
- VPD: M4 config on the new checkpoint, **with the gate-decorrelation loss**
  (apply `vpd/param_decomp_gate_decorrelation.patch`; the param-decomp repo was
  left dirty — re-apply cleanly). Success = mode-preferential component count
  rises above the followup-2 baseline (decorrelation fixed *pattern* PR but not
  mode-preference on position-locked data — the whole question is whether moving
  mechanisms change that).

## Predictions (scorecard)

| result | reading |
|---|---|
| P1 holds across held-out regions | abstraction formed — the core hypothesis |
| SAE uniqueness ≫ 0, VPD mode-pref ↑ | tools now bite: mechanism = channel/circuit |
| `D_sub` recovers, `D_avg` chance (T1) | genuine spatial de-aliasing |
| T3 realisation-corr high | reading between the lines, not hallucinating |
| P1 fails / tools still flat | barrier is deeper than position-locking (null, still informative) |

## Metrics
Per-mechanism cross-location R² (P1); channel-decode under shuffle (P2);
corr(recovered, true `Z_fine`) per level (T2); R²(`D_sub`)−R²(`D_avg`) (T1);
realisation-corr vs spectrum-only (T3); identical-twin recovery ≈ chance (T4);
R² vs Δφ curve (T5); Hungarian MCC + uniqueness vs mechanism GT, VPD
mode-preferential counts (T6).

## Files (distinct; no collision with prior work)
- `data_gen/generate_movmech.py`, `data_gen/split_movmech.py`
- `pcmci/subres_identifiability.py` (T0/T1 oracle+linear)
- `sae/movmech_position_invariance.py` (P1/P2), `sae/subres_ladder.py` (T2/T6)
- reuse `sae/extract_activations*.py`, `sae/eval_sae_metrics.py` (add
  `--ground-truth` path), `pcmci/dmd_timescales.py` (linear surrogate),
  param-decomp VPD (+ gate-decorrelation patch)
- results → `results/movmech_*.npy`, `results/subres_*.npy`
- checkpoints → `checkpoints_movmech_place/`, `checkpoints_movmech_advect/`

## Sequencing
1. **T0** first (cheap, no training) — go/no-go for the de-aliasing angle.
2. **Generator + train** the moving-mechanism forecaster (`place` first).
3. **P1/P2** — does abstraction form? This is the load-bearing result; if P1
   fails the sub-res/SAE/VPD payoff is expected null but still run T6 to confirm.
4. **T1–T5** de-aliasing battery, then **T6** SAE/VPD re-runs.
Write each block into `notes/session_results.md` (numbers first), commit per
block, do not push.
