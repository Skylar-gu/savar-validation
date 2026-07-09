# Pre-registration — R4b: fair-pool scale rung (selector re-test at N=24)

**Written & committed BEFORE running. Declares pool, method, and success bar so the
result counts as pre-registered, not tuned.** Companion to the R4 pre-registered
result (PX Spearman +0.357, a real miss) and the post-hoc R4-fair *diagnostic*
(homogeneous 5-candidate pool → Spearman +0.667, Pearson +0.963).

## Question
R4's selector miss was traced to **candidate-pool resolution heterogeneity**: the
pool mixed 24-mode maps (Leiden) with 12-mode maps (varimax at C0=12, `coarse4`)
and other off-count anchors. PX compares each candidate's integration graph against
every *other* candidate's dynamics graph; when candidates carry different mode
counts those cross-comparisons are scored on mismatched supports and PX compresses
toward zero (all R4 candidates fell in PX∈[0,0.08]). This rung asks: **on the same
N=24 world, does a resolution-homogeneous candidate pool restore the selector to the
pre-registered bar?**

## Candidate pool (declared — all target the true count N=24)
Data-driven builders run at **C0=24** (coherence pruning may trim N̂ below 24; that
is honest discovery variance and is reported per candidate — distinct from the
*intentional* count-changing anchors we exclude below):
- `leiden`   — pixel-side, unknown-N (found N̂=24 in R4)
- `vmax_pix` — varimax on pixels, C0=24
- `km_pix`   — k-means on pixel time-courses, C0=24
- `vmax_act` — varimax on GNN node activations, C0=24
- `km_act`   — k-means on activations, C0=24
- `dmd_act`  — DMD-mode clustering on activations, C0=24

Count-preserving synthetic anchors (exactly 24 modes by construction):
- `oracle` — true W (upper anchor)
- `blur`   — Gaussian-smeared footprints (σ=6)
- `shift5` — footprints rolled +5 columns
- `diag8`  — footprints rolled (+8,+8) diagonally (lower anchor; ~0 overlap)

**= 10 candidates**, every one targeting 24 modes.

## Excluded a priori — the heterogeneity source under test
`coarse4` (→12 modes), `fine16` (→48), `merge01` (→23), `split7` (→25). These
builders intentionally change the mode count; they ARE the mechanism this rung
isolates, so including them would re-inject the confound. Exclusion is declared
here, before the run — not chosen after seeing scores.

## Method (unchanged from every prior rung)
- World: `data/realisations_scale24` (N=24, 80×80, disjoint 16×16 blobs), checkpoint
  `checkpoints/scale24/best.pt`, test split `data/splits_scale24/test`.
- E1 discovery on last E1_DISC=4 reals; E4 PX on the held-out test split.
- Scoring: behaviour-matched (Hungarian on |corr(pooled series, true Z)| ≥ 0.3;
  <4 matches → 0). Never footprint cosine. `ground_truth_graph[effect, cause, lag]`.
- CI test: ParCorr (linear) — valid here (bilinear cross_edges ⊆ linear G).
- SEED / COH_MIN / alpha left at script defaults; no per-candidate tuning.

## Success bar (pre-registered, program-wide — do not tune)
**Primary: Spearman(PX, behaviour-matched truth-F1 pair) ≥ 0.8.**
Secondary (reported, not gating): Pearson(PX, truth-F1); position of these 10 points
on the E4-final trust-dial line (accuracy ≈ 1.48·PX + 0.13). **Report whatever it
is — a miss is a finding, not a bug.**

## Prediction (stated before running)
The R4-fair diagnostic already gave Pearson +0.963 (magnitude relation intact) with
Spearman held down to +0.667 by only 5 candidates + one anchor rank-inversion.
Adding the genuinely-weak-but-homogeneous acts-side candidates (expected low truth-F1)
widens the truth-F1 range and should let a 10-candidate homogeneous pool clear
Spearman ≥ 0.8. If it does not, that bounds the fix: homogenising resolution is
necessary but not sufficient, and the residual miss is intrinsic to N=24 selection.

## Outputs
`results/litext_e1_discovery_scale24pre.npy`, `results/litext_e4_agreement_scale24pre.npy`.
Tag `_scale24pre` (does not clobber R4 `_scale24` or diagnostic `_scale24fair`).
