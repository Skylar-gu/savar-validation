# Next Steps — Diagnosis and Plan (2026-07-03)

Companion to [results_cnn.md](results_cnn.md) and [results_gnn.md](results_gnn.md).
Goal unchanged: validate an SAE + VPD + causal-discovery pipeline on SAVAR ground
truth, then apply it to GraphCast to investigate/reconstruct its learned causal
structure.

---

## 0. The diagnosis: data problem or architecture problem?

**Primarily a data/task problem, with one architecture interaction that is itself a
finding (and is shared with GraphCast).** The two symptoms have one root cause.

### (a) SAE directions don't track modes, yet modes are probeable

Two different things are "probeable", and they fail differently:

1. **Mode content** (the value of Z_j(t)) — DATA problem, already half-solved.
   The finecadence task barely rewards per-mode circuitry (modes dynamically
   homogeneous; optimal forecaster = one shared shrink; GNN R² over mean = 0.125),
   so the representation contains little per-mode signal (ridge ceilings 0.36–0.50,
   PC0 = 88–92%). The SAEs are *not* underperforming: they hit 66–88% of the linear
   ceiling in every dataset, dictionary size, and config. Causal confirmation:
   widening per-mode timescales (hetdynamics/eqvar) raises alignment monotonically
   with timescale (X7 mixed-SAE r = 0.63; 3/8 strong), and eqvar ≥ as-is isolates
   timescale over amplitude. **The tool was fine; the data had nothing to find.**

2. **Mode identity** (which mode is this?) — ARCHITECTURE×DATA interaction,
   structural. Identity is 71%-decodable but only as a static per-mode mean offset
   (demeaned probe → 16%); the dynamic content code is *shared across modes* because
   the weight-shared, equivariant MP MLPs put "my local value" in the same channels
   at every node — identity exists only in the pooling weights W[j,:]. A constant
   offset has near-zero within-mode variance, so a reconstruction-driven TopK SAE
   never spends a feature on it (identity survives distributed: 71% probe on SAE
   codes). **Identity-monosemantic features are structurally impossible here.**
   This is not a bug to fix by tweaking SAEs — no SAE variant can carve a feature
   along a direction the model never uses dynamically.

### (b) VPD: faithful reconstruction, near-identical components

Same root cause seen from weight space. Recon + faithfulness losses are satisfiable
by *any* redundant shredding of W; differentiation must come from
importance-minimality, which only bites if the network's computation contains
separable input-dependent mechanisms. The frozen finecadence GNN computes ~one
mechanism at every node, so the 64 rank-1 components are shards of it:
- gate-map participation ratio ≈ **1.1 / 64** at every layer (one spatial pattern,
  scale/sign variants);
- blob-usage CV across the 8 modes ≈ 0.04 (every component fires on every mode);
- all 64 components stay active everywhere.

Secondary (config) amplifiers, not root cause: C=64 ≫ plausible mechanism count;
rank-1 splitting; minimality term small in absolute loss terms (~0.15 vs recon
~0.02 vs faithfulness ~7600 after coefficients). The method demonstrably works on
this model — the depth gradient (hubs early → blobs late) and the hub-locked ⇒
causally-inert cross (corr ≈ −0.48) are real, replicated structure.

**The decisive discriminator experiment has not been run: VPD on the
hetdynamics_eqvar checkpoint** (where per-mode mechanisms should pay off). If
components differentiate by timescale → data problem confirmed; if they stay
identical → method/config problem. That's Direction 1.

### Implication for GraphCast (why this is good news)

The SAVAR testbed was *too symmetric*: identical mode dynamics + no static inputs +
equivariant processor = nothing mode-specific for any interp tool to find. GraphCast
differs on exactly the relevant axes: (i) real atmospheric dynamics are strongly
heterogeneous (tropics vs midlatitudes, land vs ocean timescales); (ii) GraphCast
**receives static geographic inputs** (lat/lon, orography, land-sea mask), which
re-inject "identity" as content the processor can encode in channels. The SAVAR
failure modes therefore predict what to expect in GraphCast (content-typed features;
distributed identity; epiphenomenal mesh structure) rather than invalidating the
pipeline. Each direction below either closes a realism gap or hardens the toolkit.

---

## Direction 1 — Close the loop: VPD on the heterogeneous-dynamics GNN
**The single highest-value, cheapest next experiment.** (~15 min on the L40S per
run; all code exists.)

- Run the M4 config against `checkpoints_hetdynamics_eqvar/best.pt` (and
  `checkpoints_hetdynamics/best.pt`); regenerate diagnostics + the faithful figure.
- **Prediction:** late-layer components differentiate along the timescale axis —
  gate maps stop being one pattern; `usage_by_mode` CV rises, especially separating
  slow blobs (X5–X7) from fast (X0–X2); participation ratio of gate maps ≫ 1.1.
- **Read-outs:** gate-map participation ratio per layer; per-component blob-usage
  CV; corr(component's mode-usage profile, mode timescales); ablation-drop spread.
- **If components differentiate** → data-problem verdict confirmed end-to-end; VPD
  is validated as sensitive to real mechanism structure; proceed to Direction 5
  (component time series → PCMCI+).
- **Backup if they don't differentiate** (method/config problem), in order:
  1. Raise `ImportanceMinimalityLoss.coeff` 10–100× (sweep 1e-3, 1e-2) and/or
     stronger frequency penalty (beta 0.5→1.0); watch adversarial recon for
     degradation — the tension is the point.
  2. Shrink C (64 → 8/16) to force competition between components.
  3. Decompose only late layers (2–3), where mode content lives.
  4. If still monolithic while SAE alignment says per-mode signal exists → write it
     up as a VPD limitation on weak-signal regressors (valuable negative result;
     compare against an SPD-only run to isolate the adversarial term).

## Direction 2 — Re-inject identity faithfully: static node inputs
**Tests whether the monosemanticity failure disappears when identity is available
as content — the GraphCast-faithful fix.**

- Add fixed (non-learned) static features to the GNN *input*: sin/cos of x and y
  coordinates + a hub-indicator flag, concatenated to the K frames (small change in
  `MeshGNN.forward`; keep `node_emb=0` — a learned table remains a confound,
  whereas static inputs are exactly GraphCast's lat/lon/orography channels).
- Retrain on hetdynamics_eqvar (~3 h); rerun SAE per-mode + mixed + identity probes
  + 3D viz; rerun VPD (after Direction 1).
- **Prediction:** demeaned identity probe jumps above 16% is not required — rather,
  the encoder can now bind content to location, so mixed-SAE **specificity** should
  finally go positive for slow modes (features like "slow blob at location (2,1)").
- **Backups:**
  1. If specificity stays ≈ 0: check whether the network even *uses* the static
     inputs (ablate them at inference; if forecast unchanged, the task still
     doesn't reward binding — strengthen Direction 3 first).
  2. Compare against the `GNN_EMB_DIM>0` learned-embedding control to bound how
     much identity a cheap lookup would buy (upper bound for the static-input run).
  3. If binding appears but SAEs still miss it, try SAEs with per-node (unpooled)
     samples tagged by position — the spatial-SAE recipe from the CNN gridlock work.

## Direction 3 — Make the data earn specialization: harder SAVAR rungs
Each rung adds ONE failure mode (ladder discipline). Ordered by expected payoff:

1. **Longer-horizon / coarser target.** Predict t+s (s=4–8) instead of t+1, or
   train multi-step rollout. At 1 step, fast modes are noise-dominated and slow
   modes carry all skill; longer horizons *widen* the per-mode skill spread and
   mirror GraphCast's rollout training. Cheap (dataset index change / loss change).
2. **Spatial transport.** Advecting/propagating structures (drifting blobs or a
   traveling-wave term) so spatial patterns matter dynamically — the GNN must learn
   motion, not per-cell damping; the single biggest realism gap after nonlinearity
   (audit Part II). This is also the rung that could create genuinely *spatial*
   mechanisms for VPD to separate.
3. **Overlapping (non-disjoint) modes** (`future_plans.md` sketch: closer centers,
   σ=6, target cosine overlap ~0.2 then >0.5). Breaks the clean grid→mode
   projection; recovery must use pinv(W) — stresses SAE dilution exactly as
   GraphCast's non-orthogonal physical modes will.
4. **More modes / bigger graph** (N 8→20+) to stress PCMCI+ parent search and
   feature grouping at realistic dimensionality.
- **Backups:** if a rung makes the forecaster fail to train (skill ≈ climatology),
  back the knob off 50% rather than abandoning; keep every rung's edge SET identical
  where possible so discovery scores stay comparable across rungs.

## Direction 4 — Metric reform: evaluate what the theory says is achievable
The pipeline's scoring still asks for something (identity-monosemantic features)
that §0 shows is structurally impossible under weight sharing. Fix the yardstick:

1. Report alignment as **fraction-of-ridge-ceiling** (already computed) as the
   headline, with raw |r| secondary.
2. Add a **distributed-code metric**: linear probe accuracy for Z (content) and
   identity from SAE codes vs raw activations — "does the SAE preserve the
   information" rather than "is it one feature".
3. Adopt **causal feature importance** (ablation ΔRMSE) as the primary feature
   ranking, per the Phase-5/VPD lesson that decodability ≠ used.
4. Build the **unsupervised grouping path** needed for GraphCast anyway:
   Ising/inverse-Ising couplings + Leiden on SAE feature co-firing (design note §4)
   → communities → community activation time series. Validate on hetdynamics_eqvar
   where ground truth exists: do communities align with modes/timescale groups?
- **Backup:** if Ising grouping is unstable on continuous TopK codes, fall back to
  conditional co-activation graphs or decoder-direction clustering with the
  redundancy caveats from the feature-splitting study.

## Direction 5 — Causal discovery on discovered objects (the pipeline payoff)
The end-to-end claim is "recover Φ from the trained model's internals". Now that
hetdynamics gives aligned features, rerun the discovery stage on them:

1. **PCMCI+ over mixed-SAE feature series** (hetdynamics_eqvar): pick top features
   per community/mode (via Direction 4 grouping, not hand-picking), compare
   recovered graph to fine ground truth, alongside the latent-Z ceiling (F1 0.82).
2. **VPD component time series** (per-component contribution magnitude per step) →
   PCMCI+ → compare to Φ (vpd note §8's bridge experiment). Needs Direction 1 first.
3. **Contemporaneous orientation:** the aliasing sweep leaves τ=0 edges unoriented
   (orient rate 0.08–0.30) while the data is non-Gaussian by construction — add a
   LiNGAM-style orientation pass (e.g. pairwise likelihood-ratio on residuals) on
   PCMCI+'s `o-o` edges; score orientation accuracy vs ground truth. Self-contained,
   publishable methods piece.
- **Backups:** if feature-series discovery degrades badly, diagnose with the known
  ladder (alignment quality → edge strength → lag aliasing) — the CNN chapter's
  feature-space F1 0.695 vs 0.825 shows what graceful degradation looks like; if
  orientation fails, verify innovations' skewness survives the pooling/SAE chain
  (each nonlinearity Gaussianizes).

## Direction 6 — Architecture ladder toward GraphCast
After Directions 1–3 establish *what* transfers:

1. **`GNN_MP_MODE=graphcast`** (directional edge MLPs — already implemented, never
   trained): retrain on hetdynamics_eqvar; VPD then decomposes `edge_mlp` — the
   natural home for *oriented* (causal-direction) mechanisms. Does edge-mechanism
   structure align with Φ's directed edges? Nothing in the current gcn mode can
   represent direction; this is the first architecture that could.
2. **Multivariate fields** (≥2 coupled channels) — cross-variable causality is
   where most real atmospheric structure lives.
3. **Scale up mesh realism** (icosahedral-style multi-level refinement instead of
   the stride-5 hub lattice) only if grid-lock questions demand it.
4. Then the real thing: MacMillan & Ouellette-style SAE on GraphCast layer-8 node
   embeddings on teacher-forced trajectories (deseasonalized), plus VPD on
   processor MLPs — with the validated metric suite from Direction 4 and the
   discovery stage from Direction 5.

---

## Suggested sequencing

| # | Step | Cost | Blocked by |
|---|---|---|---|
| 1 | VPD on hetdynamics_eqvar (+ as-is) | ~1 h total | — |
| 2 | Longer-horizon target variant (D3.1) | ~3 h train | — |
| 3 | Static-node-input GNN + SAE/probe rerun (D2) | ~4 h | — |
| 4 | Metric reform + Ising/Leiden grouping (D4) | 1–2 days code | — |
| 5 | PCMCI+ on mixed-SAE features, hetdynamics (D5.1) | hours | D4 helpful |
| 6 | LiNGAM orientation of τ=0 edges (D5.3) | 1–2 days | — |
| 7 | VPD component series → PCMCI+ (D5.2) | ~1 day | 1 |
| 8 | Spatial transport rung (D3.2) | 2–3 days | — |
| 9 | graphcast MP mode retrain + edge-MLP VPD (D6.1) | ~1 day | 1 |
| 10 | Overlapping modes rung (D3.3) | 1–2 days | — |

Steps 1–3 are independent and can run in parallel (GPU permitting). Steps 1+3
together fully adjudicate "data vs architecture vs method-config" for both the SAE
and VPD symptoms.

## Updated recommendations after literature review (2026-07-03)

Full review with per-source paragraphs: [literature_review.md](literature_review.md).
The literature confirms the overall program (synthetic ground-truth validation of
SAE claims is now standard practice on the LLM side too — SynthSAEBench — and
nobody has yet run causal discovery on weather-emulator internals), and it
*revises* the plan in six places:

1. **Metric suite upgrade (revises Direction 4, do first — it re-scores everything
   else).** Adopt SynthSAEBench's metrics on SAVAR: Hungarian-matched mean
   correlation coefficient between decoder columns and ground-truth directions,
   feature uniqueness, and per-latent F1 — replacing the single best-|r| scalar.
   Add reseed error bars (≥3 SAE seeds) per "Are SAE benchmarks reliable?".
   Recalibrate targets: even under ideal linear conditions no SAE architecture
   reaches supervised-probe recovery (best F1≈0.88 vs 0.97), so our 66–88% of
   ceiling is in-family; stop treating <1.0 frac-of-ceiling as failure.
2. **Diagnose the dilution regime explicitly (extends Direction 4).** Fit
   inverse-Ising couplings on binarized mixed-SAE codes (hetdynamics_eqvar),
   cluster with Leiden, and classify each mode's feature group as
   capture/tiling/dilution per the concept-manifolds taxonomy. Prediction: slow-
   mode content shows tiling (negative couplings among same-mode features);
   identity shows dilution. This turns our "distributed code" claim into the
   field's standard vocabulary.
3. **New cheap control arm: dynamical testing of the frozen GNN (new direction,
   insert alongside Direction 1).** Hakim–Masanam-style: perturb mode j's blob in
   the input window, read the frozen GNN's response at t+1..t+k against the
   ground-truth impulse response of Φ. Internals-free causal-structure probe —
   if the GNN's implicit graph already matches Φ, interp methods have something
   real to find; if not, no feature method can recover what the model never
   learned. ~1 day, no training.
4. **Ground the orientation work in existing theory (revises D5.3).** Gong et
   al. (ICML 2015) prove the fine-cadence VAR is identifiable from subsampled
   data given non-Gaussian noise — which our generator has by design. Benchmark a
   LiNGAM/NG-EM estimator on the existing subsample sweep and score orientation
   accuracy of the aliased τ=0 edges; do not hand-roll a heuristic.
5. **Validate the aggregation step before trusting feature-level graphs (new,
   gates D3.3 and the GraphCast rung).** Our W-pooling is a spatial-average
   aggregation map; arXiv:2505.10476 shows such maps can corrupt discovery and
   provides consistency scores + the Adag wrapper. Run their consistency check on
   (a) current disjoint W (expect pass), (b) the overlapping-modes rung (expect
   stress), (c) any learned grouping used on GraphCast.
6. **Add two baselines the field will ask for (extends Direction 5/6).**
   (a) DAG-VAE-style causal representation learning (graph-in-the-latent) on
   SAVAR, scored against Φ — the "build causality in" philosophy vs our "discover
   it post-hoc" one; (b) SINDy-SHRED as a white-box forecaster bound on what a
   black-box forecaster's internals could yield. Optional third: DMD/Koopman
   spectrum of GNN activation dynamics as a timescale cross-check.
7. **SAE architecture backups sharpened (revises D2/D4 backups).** If alignment
   plateaus below ceiling after static inputs: try Matryoshka/BatchTopK (best
   latent quality per SynthSAEBench) before dictionary-size games; try KAN-SAE
   (per-feature spline gates) specifically because our dynamics are saturating
   and thresholds are what linear gates miss. Never select SAEs by reconstruction
   MSE alone (recon and recovery dissociate).
8. **Steering as a validation channel (new, cheap).** Both weather-SAE papers
   validate features by dose–response steering; the CFD-GNN paper warns static
   injections fail on oscillatory latents. On SAVAR we can steer the best
   slow-mode feature and score dY against ground truth W⁺ΔZ — with the
   time-aware modulation caveat.

**Revised sequencing (deltas in bold):**

| # | Step | Cost | Source |
|---|---|---|---|
| 1 | VPD on hetdynamics_eqvar | ~1 h | unchanged |
| 2 | **Impulse-response dynamical test of frozen GNN vs Φ** | ~1 day | new (lit §C) |
| 3 | **Metric suite upgrade + reseed error bars** | ~1 day | revised (lit §A) |
| 4 | Static-node-input GNN + SAE/probe rerun | ~4 h | unchanged |
| 5 | **Ising/Leiden regime diagnosis on mixed-SAE codes** | ~1 day | sharpened |
| 6 | PCMCI+ on mixed-SAE features (hetdynamics) | hours | unchanged |
| 7 | **LiNGAM/NG-EM orientation on subsample sweep** | 1–2 days | grounded |
| 8 | **Aggregation-consistency check on W-pooling** | ~1 day | new (lit §B) |
| 9 | **Steering dose–response validation of top features** | ~1 day | new (lit §A/C) |
| 10 | Longer-horizon target variant | ~3 h train | unchanged |
| 11 | Spatial transport rung | 2–3 days | unchanged |
| 12 | graphcast MP mode + edge-MLP VPD | ~1 day | unchanged |
| 13 | **DAG-VAE + SINDy-SHRED baselines** | 2–4 days | new (lit §C/D) |
| 14 | Overlapping modes rung (gated by #8) | 1–2 days | unchanged |

## Global risk register

- **Weak-signal ceiling is intrinsic to 1-step SAVAR** (~96% of achievable corr
  already reached). Don't chase forecast skill; chase per-mode *structure*. If a
  rung needs more skill headroom, use the longer-horizon task, not bigger models.
- **Hub lattice overlaps mode blobs** — always disentangle hub-tied vs mode-tied
  usage explicitly (regression on hub indicator + mode membership), never read
  "localized" as "grid-locked".
- **Any per-position learned parameter is a confound** — GraphCast's identity comes
  from static *inputs*; keep it that way in SAVAR.
- **Ensemble-mean deseasonalization stops being exact under nonlinear dynamics**
  (E[f(x)] ≠ f(E[x])) — if cycles are re-added to nonlinear rungs, switch to
  per-realisation harmonic fits or phase conditioning.
- **Chronological splits + full-timeline SAE extraction** partially leak — keep the
  Phase-5 lesson: report held-out-test numbers for every causal/importance claim.
- **Negative results here are findings** (grid-lock epiphenomenal, mixed-SAE failure
  = homogeneity detector, VPD redundancy = monolithic mechanism detector): document
  them as *pipeline sensitivity*, which is exactly what a validation ladder is for.
