# Literature-extension experiments (litext) — from verification to a recovery technique

**Written 2026-07-06.** Companion to [[literature_review]], [[session_results]],
[[next_steps_plan]], [[moving_mechanism_subres_spec_v2]],
[[graphcast_interp_design]]. Supersedes the "verify the diagnosis" mode of the
last three sessions: the diagnosis is done. This note (i) states what the
accumulated results actually establish, (ii) identifies the technique gap they
leave open, (iii) specifies a candidate technique assembled from the reviewed
literature, and (iv) lays out the experiment ladder that develops it on SAVAR
*and* rebuilds SAVAR so that each validated claim is a claim about GraphCast.

---

## 1. Analysis — why the current experiments feel like re-verification

The last three sessions (Blocks A–H, Follow-ups 1–3, movmech v2 battery) were
adjudication experiments: data-vs-method-vs-architecture forks, each closing a
hypothesis. They closed. The stable facts, stated once:

| # | Established fact | Where |
|---|---|---|
| F1 | The frozen GNN's pooled activations carry the **full causal graph**: PCMCI+ on stride-1 W-pooled activations recovers Φ at F1 0.855 = the true-Z ceiling (0.853), edge-for-edge | FU1 |
| F2 | The GNN implements **one mode-agnostic amplitude-dynamics operator**: no mode-unique linear structure (uniqueness ceiling 0.024), no mode-factored weight mechanisms even after the VPD objective is fixed (pattern-PR 34/64, zero mode-preferential gain) | FU2, FU3 |
| F3 | Mode identity lives in **positions, not channels**: W-pooling is necessary and sufficient to read a mode; withheld-location attention collapses (retention −0.62); moving the mechanisms does not induce a what-code, in any architecture (plain/blurpool/refframe/slot) | P1-withheld, arch sweep |
| F4 | Aggregation is lossless when footprints are disjoint (pooled pixels = true-Z, agreement 1.000) | Block G |
| F5 | The internals-free impulse-response arm recovers a lag-correct **half** of the graph (F1 0.500, P 0.625); recall is capped by rollout damping, which is a property of the generative loop, not the representation (DMD-on-acts Spearman 0.976 = pixels) | Block B, FU1(iii) |
| F6 | SAE features tile one shared subspace (Ising: one community; uniqueness ≈ 0; steering leakage 0.81–0.94 = global amplitude knob); ZCA whitening is the only intervention that buys identity specificity (+0.068, at MCC cost) | Blocks C/D/F, FU3 |
| F7 | τ=0 orientation of aliased edges is at chance despite Gong-identifiability (skewness too weak at our T) | Block E |
| F8 | Activation extraction must be at **native cadence** (the stride-5 bug class inverted an entire verdict) | FU1 |

The re-verification feeling is accurate: each remaining item on
[[next_steps_plan]]'s list (more rungs, more metrics, more nulls) adds
confidence to F1–F8 but no new capability. The plan below pivots on the one
result that is *positive* and *unexhausted*:

**F1 is an existence proof.** The causal structure behind the emulator is
recoverable from its internals at the theoretical ceiling — *given* the
ground-truth aggregation map W. Every downstream failure (SAE uniqueness, VPD
mode-mechanisms, steering specificity) is a failure to recover things the model
provably does not contain (F2, F3). What the model does contain — the graph,
readable through the right spatial pooling — we have only ever read with an
oracle.

**The technique gap, precisely:** on GraphCast there is no W. To make the F1
existence proof into a method, three sub-problems must be solved *unsupervised*:

- **(G1) Mode discovery** — find the aggregation map (spatial footprints +
  count) from the model's internals alone.
- **(G2) Selection without ground truth** — decide *which* candidate
  aggregation/graph to trust when Φ is unknown (on SAVAR we score F1; on
  GraphCast we can't).
- **(G3) Validation without ground truth** — attach a calibrated confidence to
  the recovered graph (the deliverable a reviewer, or a downstream Nowack-style
  model-evaluation user, needs).

No prior work solves these for forecaster internals — [[literature_review]] §B
confirms causal discovery on weather-emulator internals validated against
ground truth remains our differentiated claim; MacMillan & Ouellette explicitly
stop before feature–feature/causal structure.

**And F3 is prescriptive, not just diagnostic.** "Where-code not what-code"
sounds like a defeat, but it *tells us how to do G1*: since the network binds
mode identity to spatial position (and GraphCast's features are likewise
geographically anchored — its "specific geographical coding" bucket, its static
lat/lon/orography inputs), mode discovery should be **spatial-footprint-based**
(clustering/rotating the activation field), not feature-identity-based (hoping
for a monosemantic "this is ENSO" channel). The three sessions of nulls are the
justification for the design of the technique.

---

## 2. The technique — the W-free pipeline (internals → modes → graph → calibrated confidence)

Assembled from the reviewed literature; each stage names its sources.

```
frozen emulator + input trajectories (teacher-forced, native cadence)   [F8; MacMillan&Ouellette]
        │
        ▼
STAGE 1  candidate aggregation maps Ŵ from internals          (G1)
         a. Varimax⁺-rotated PCA on per-node activations       [Runge/SAVAR lineage — the
            (the classic climate mode-discovery operator,       operator SAVAR was built to
            applied to activations instead of raw fields)       benchmark, never yet run on
                                                                 internals]
         b. space-time SAE features → inverse-Ising couplings
            → Leiden communities → community footprint maps    [Goodfire concept-manifolds
                                                                 arXiv:2604.28119; ST-SAE
                                                                 arXiv:2604.03919; the §4
                                                                 composition of
                                                                 graphcast_interp_design,
                                                                 never built]
         c. DMD spatial modes of the activation field          [Koopman family §D; Block H
                                                                 showed blob cos 0.64–0.90
                                                                 from pixels — free candidate]
         d. spatial clustering of per-node activation
            trajectories (k-means / spectral, cosine on
            time courses)                                       [cheap floor baseline]
        │
        ▼
STAGE 2  aggregation-consistency gate: score every Ŵ with the (G2)
         Adag consistency scores; refine/select the map that
         preserves the independence model                       [arXiv:2505.10476 — used as an
                                                                 UNSUPERVISED MODEL-SELECTION
                                                                 criterion, which is beyond how
                                                                 the paper uses it; E2 tests
                                                                 whether that extension is valid]
        │
        ▼
STAGE 3  pool activations through selected Ŵ (nowcast readout,(—)
         stride 1) → PCMCI+ (ParCorr; CMIknn on nonlinear
         rungs) → graph Ĝ_int                                  [Runge 2020 PCMCI+; F1/F8]
        │
        ▼
STAGE 4  independent second channel: impulse/forcing responses (G3)
         of the frozen emulator, read out at TEACHER-FORCED
         single steps through the same Ŵ → graph Ĝ_dyn         [Hakim–Masanam line §C;
                                                                 upgraded per F5 — see E4]
        │
        ▼
STAGE 5  graph agreement(Ĝ_int, Ĝ_dyn) + seed/layer stability (G3)
         → calibrated confidence: on SAVAR, learn the mapping
         agreement → truth-F1; on GraphCast, report agreement
         and the SAVAR-calibrated confidence                    [Nowack et al. 2020 graph-
                                                                 fingerprint template]
```

Two channels matter because they have disjoint failure modes: Stage 3 reads
correlational structure out of representations (fails with readout/pooling
error); Stage 4 physically intervenes on the model (fails with excitation/SNR
error). Agreement between them is evidence neither could fabricate alone —
that's what makes Stage 5 a legitimate no-ground-truth validator, *if* SAVAR
shows agreement tracks accuracy (E5 tests exactly this).

In-house lessons folded in: native-cadence extraction everywhere (F8);
ZCA-whitened activations as a variant wherever identity specificity matters
(F6); nowcast (δ=0) readout, never the forecast-target readout (FU1(ii));
expect and accept a global operator — the pipeline never requires per-mode
mechanisms or monosemantic features to exist (F2/F3-proof by design).

---

## 3. Experiments

Ordered so each experiment gates the next. E1–E2 need **no new training** (run
on the existing hetdynamics_eqvar checkpoint + stride-1 activations,
`sae_data_hetdynamics_eqvar/activations_stride1_sel.npy`). Every experiment
reports vs two anchors: the PCMCI-on-true-Z ceiling and the oracle-W activation
result (F1 0.855).

### E1 — Unsupervised mode-discovery bake-off (G1)

**Question.** How much of the oracle-W F1 0.855 survives when Ŵ must be
discovered from internals?

**Setup.** New `sae/discover_modes.py` producing candidate Ŵ's (Stage-1 a–d)
from stride-1 per-node activations (and, as a control, from raw pixels — does
the *emulator's representation* make mode discovery easier or harder than the
data itself?). For (b), add the space-time SAE (T6 machinery, ≥3 seeds per the
seed-instability literature arXiv:2606.12138) and the Ising/Leiden grouping
already prototyped in `sae/ising_regimes.py`. New
`pcmci/run_pcmci_discovered.py`: pool → PCMCI+ → score.

**Metrics.** (i) Footprint recovery: Hungarian-matched IoU / cosine of Ŵ rows
vs true W rows, + estimated mode count N̂ vs 8. (ii) Downstream: PCMCI+ F1 of
Ĝ_int vs Φ, per candidate. (iii) The gap decomposition: F1(oracle W) −
F1(Ŵ) = the price of discovery.

**Predictions.** On disjoint static blobs, (a) and (d) recover footprints
nearly exactly (blobs are variance-coherent spatial units; F3 says the
representation is position-organized, which *helps*), so F1 ≥ 0.8; (b) is the
interesting one — Block D found one shared Ising community across all modes at
the *pooled* level, but per-node/space-time codes re-introduce position, so
Leiden should now split spatially; (c) lands mid-pack (Block H cos 0.64–0.90).

**Success bar.** Best candidate within 0.05 F1 of oracle-W. If nothing gets
close, the W-free program needs Stage-1 work before anything else matters —
that's the finding.

**Cost.** ~2 days, no training (space-time SAE seeds are the only GPU hours).

### E2 — Aggregation consistency as unsupervised selection (G2) — **the load-bearing experiment**

**Question.** Do the Adag consistency scores (arXiv:2505.10476), computable
WITHOUT ground truth, rank the E1 candidates in the same order as their
ground-truth F1?

**Setup.** Extend `pcmci/aggregation_consistency.py` from its Block-G
oracle-check role: compute the three consistency scores for every E1 candidate
Ŵ (plus deliberately corrupted Ŵ's — merged modes, split modes, shifted
footprints, wrong N — to populate the quality axis). Correlate score vs truth-F1
across the candidate set.

**Prediction.** Spearman(score, F1) ≥ 0.8 with the corrupted maps clearly
separated. Merging two modes creates spurious edges (marginalization), splitting
one creates deterministic near-duplicates (faithfulness violations) — both are
what the consistency scores measure.

**Success bar / branch.** If the ranking holds → we possess an unsupervised
model-selection criterion, and the GraphCast rung inherits it. If it fails →
fall back to selecting by Stage-5 cross-channel agreement (E5 then carries both
G2 and G3), and report the Adag negative as a scope result on the paper's
method (their scores were designed for vector-valued *data*, not learned
representations).

**Cost.** ~1–2 days, no training.

### E3 — Upgrade the internals-free arm to close F5's recall gap (Stage 4)

**Question.** Block B's impulse response reached F1 0.500, recall capped by
slow-mode responses dying in autoregressive rollout. FU1 proved the
*representation* holds the timescales — the damping is the generative loop. Can
excitation + readout redesign close 0.500 → ~0.85?

**Setup.** Extend `pcmci/impulse_response_gnn.py` with three orthogonal
upgrades, ablated separately:
1. **Teacher-forced response readout** — read the perturbation's effect from
   pooled *activations* at single teacher-forced steps along the true
   trajectory (nowcast readout), never through long free rollout. Directly
   targets the damping mechanism.
2. **Sustained/resonant forcing** — replace the single impulse with a sustained
   or on-timescale-modulated forcing of blob j (the CFD-GNN steering paper's
   lesson: static injections fail on oscillatory/slow latents; interventions
   must respect the mode's temporal structure). Slow modes integrate forcing —
   SNR grows with excitation length.
3. **Dose ladder** (±1σ, ±3σ) with linearity check per edge, reusing the
   Block-F dose–response machinery as an edge-level screen.

**Prediction.** Upgrade 1 alone recovers most missed slow-effect edges ((0→5),
(1→4), (3→7), (5→6), (6→7) all have slow-mode effects); 1+2 approaches the
PCMCI-on-Z ballpark.

**Success bar.** F1 ≥ 0.75 (vs 0.853 ceiling). This channel is the most
GraphCast-portable piece of the whole program (it needs only forward passes and
is exactly the Hakim–Masanam protocol), so every point of F1 here transfers.

**Cost.** ~2 days, no training.

### E4 — Agreement→accuracy calibration (G3): the statement that transfers

**Question.** Does cross-channel graph agreement predict truth-accuracy well
enough to serve as GraphCast's confidence measure?

**Setup.** New `pcmci/graph_agreement.py`. Across every (dataset rung ×
candidate Ŵ × channel) combination produced by E1/E3/E5-rungs — plus degraded
variants (shorter T, higher noise, corrupted Ŵ) to spread the accuracy axis —
compute pairwise agreement between Ĝ_int and Ĝ_dyn (edge-set F1 between the
two channels' graphs, orientation agreement separately per F7), and regress
truth-F1 on agreement. Report the calibration curve with uncertainty.

**Prediction.** Monotone, tight enough to be decision-useful (e.g. agreement
> 0.7 ⇒ truth-F1 > 0.7 at 90% confidence). The two channels' errors are close
to independent by construction; shared failure modes to watch: both inherit
Ŵ (test by pairing channels across *different* Ŵ's), both lose slow edges at
short T.

**Deliverable.** This is the transfer artifact: on GraphCast we will never see
truth-F1, but we will see agreement — SAVAR supplies the mapping from the
observable to the quantity of interest. (Consumer template: Nowack et al.'s
causal-network model evaluation, where graph agreement is already the currency.)

**Cost.** ~2 days once E1/E3 exist (it's an analysis layer over their outputs).

### E5 — Rebuild SAVAR toward GraphCast: the transfer rungs

The second half of the user-goal: construct SAVAR so validated statements are
statements about GraphCast. Current SAVAR lacks five GraphCast properties
([[graphcast_interp_design]] §5.2, next_steps §D2/D3/D6). One rung each, one
failure mode each; the full W-free pipeline (E1→E2→E3→E4 stages) reruns on
every rung, scored against that rung's own PCMCI-on-Z ceiling.

| rung | change | GraphCast property matched | expected stress point | cost |
|---|---|---|---|---|
| R1 **overlap** | non-disjoint W (cosine overlap 0.2 → 0.5), `future_plans` recipe | non-orthogonal physical modes | Stage 1 footprints blur; Stage 2 gate goes from trivially-passed (F4) to live — the rung E2 was built for | ~2 d |
| R2 **static inputs** | concat sin/cos(x,y) + hub flag to GNN input (next_steps D2, unrun) | lat/lon/orography channels — identity as *content* | may create the first genuine what-code → re-run P1-withheld; if attention can now self-localize, Stage 1 gains a content-based candidate | ~4 h train + reruns |
| R3 **multivariate** | 2–3 coupled observed channels per node, cross-channel edges in Φ | coupled atmospheric fields | Ŵ must become (channel × space); cross-channel edges test PCMCI+ conditioning | ~3 d |
| R4 **scale** | N = 24 modes, larger grid, unknown-N discovery | realistic mode count; N unknown a priori | N̂ estimation (Stage 1) and PCMCI+ parent search; Ising/Leiden favored over varimax here | ~3 d |
| R5 **rollout training** | fine-tune the forecaster on 4–8-step rollout (GraphCast's curriculum) | autoregressive training | does trained-on-rollout change F5's damping and E3's channel? Also widens per-mode skill spread (next_steps D3.1) | ~1 d |
| R6 **atmosphere regime** (added 2026-07-06; band moderated same day) | push φ band → **0.90–0.99** per step (τ ≈ 9.5–100 steps, ratio compressed 22.8×→~10× — preserving the full ratio at φ_min=0.95 would put the slow mode at φ≈0.998, ~5 independent samples per realisation: ceiling collapse by under-sampling, not regime, confounding the test); **T 2400→~9600** so the slow mode keeps ~100 e-folds per realisation (ceiling shift then attributable to autocorrelation structure, not sample starvation); fastest mode R²≈0.81 crosses the mostly-signal window boundary; optional second dose 0.95–0.998 only if the first sails through (fork generate_hetdynamics: HD_PHI band; verify spectral radius < 1, re-tune eqvar innovation scaling); cut pixel observation noise (DY_SCALE); innovations stay exogenous/per-mode/skew-normal | GraphCast's one-step *statistical* regime: near-unit memory, one-step R² ≈ 0.9+ (vs current 0.125), high SNR from memory+structure — NOT from scaling ξ, which is a no-op (in a self-exciting VAR, signal is filtered past noise; amplitude cancels) | extreme autocorrelation: effective sample size per unit T collapses, lag pinning blurs (does the PCMCI-on-Z anchor itself sink?); near-determinism flirts with the faithfulness pathology that destroyed fine16 (E2 scores must stay discriminative); E3's integral stat should *gain* power (persistent responses); at R² ≈ 0.9 the forecaster is no longer gradient-starved — rerun identity probes: if "one shared operator" persists with abundant skill, that finding gets its strongest form; **SAE readout (window-composition test)**: the model sees only K=3 frames, and SAE dictionaries are variance-weighted — in the current regime the window is mostly innovation (φ=0.15 ⇒ ~98% noise), so features encode noise-dominated directions; per-mode alignment already rises monotonically with φ (X0 0.12 → X7 0.63) and MCC orders finecadence<het<eqvar — R6 puts the whole band in the mostly-signal regime GraphCast's 2-frame input lives in (near-deterministic given full state) → rerun the Block-C metric suite (Hungarian MCC, uniqueness, 3 seeds): prediction, alignment rises across ALL modes and the SAVAR→GraphCast SAE calibration becomes regime-matched | ~1 d (generator fork + ~3 h retrain + rerun E1/E2/E3 scripts + Block-C SAE suite, already parameterized by datadir/ckpt) |

Rung discipline unchanged: keep Φ's edge set fixed wherever possible so F1s
are comparable down the ladder (R6 keeps the edge SET; coefficients rescale
with the φ band, so it gets its own PCMCI-on-Z ceiling); when a rung breaks
the pipeline, the *stage* it breaks is the finding (that attribution is what
the ladder is for).

**The transfer table** (maintained as rungs complete) — each row is a sentence
we will be able to assert about GraphCast with a SAVAR-calibrated confidence:

| SAVAR-validated statement | GraphCast statement it licenses |
|---|---|
| footprint-based discovery ≈ oracle on R1–R4 | discover GraphCast "modes" spatially from layer-8 activations (varimax⁺/Leiden), don't hunt monosemantic identity channels |
| pipeline numbers hold on R6 (high-memory regime) | the calibration is valid at GraphCast's one-step SNR, not just at SAVAR's noisy-index SNR — closes the regime-mismatch objection |
| Adag scores rank Ŵ correctly (E2) | select the GraphCast aggregation unsupervised, with a known error rate |
| teacher-forced forcing responses reach F1 ≥ 0.75 (E3) | the Hakim–Masanam arm on GraphCast yields a graph of known expected quality |
| agreement→accuracy curve (E4) | a confidence statement for the recovered GraphCast graph, despite no ground truth |
| whitening ↑ identity specificity at tracker cost (F6) | preprocessing choice for GraphCast SAEs, decided by whether the question is "which feature" or "how strong" |
| native-cadence extraction (F8) | extract GraphCast activations at 6 h, τ in 6 h units, no silent striding |

### E6 — Orientation levers (smaller, self-contained; carries F7)

Block E left τ=0 orientation at chance despite theoretical identifiability
(Gong et al. 2015). One controlled sweep to find where identifiability *cashes
out*: innovation family (skew-normal → Student-t → stable), T (2.4k → 20k),
subsampling vs temporal **aggregation** (Gong's companion result: aggregation
is the friendlier operator — and closer to what 6 h reanalysis cadence actually
does to atmospheric dynamics). Score orientation accuracy of aliased pairs vs
(tail-weight, T, operator). If no climate-plausible setting works, publish the
boundary: sub-cadence edge *orientation* from 6 h GraphCast trajectories is
beyond current estimators — adjacency-only claims at τ=0 (a scoping statement
the field needs anyway). ~2 days.

### E7 — Philosophy baselines (post-hoc vs built-in vs white-box)

Deferred until E1–E4 produce a positive pipeline result, then run once on the
best rung: (a) DAG-VAE-style graph-in-the-latent (arXiv:2603.02879) trained on
the same data, scored against the same Φ — "build causality in" vs our
"discover post-hoc"; (b) SINDy-SHRED (arXiv:2501.13329) as the white-box
forecaster bound (what a maximally interpretable model of the same data
yields). These make the eventual writeup's comparison table; they discover no
technique themselves. ~3–4 days.

---

## 4. What is deliberately dropped (and why)

- **Chasing per-mode mechanisms in weights** (VPD mode-preferential components)
  — F2 is a model-level fact; the decorrelation patch stays as tooling, but no
  more runs whose success bar requires mode-factored weights.
- **Chasing monosemantic identity features / steering specificity** — F3/F6
  bound these from the model side (Jacobian homogeneity, no what-code). The
  pipeline is now designed to not need them.
- **Movmech T2/advect + remaining sub-res rungs** — the sub-resolution ladder
  answered its question (position-leakage, not dynamics-reading; GNN below the
  L1 linear baseline). The advect wave-surrogate rung only re-tests F3 in a
  harder setting. Park unless R3/R5 resurrect a concrete need.
- **KAN/Matryoshka SAE variants** — C2 tie; encoder expressivity is not the
  binding constraint (F2 is).

## 5. Sequencing

| # | experiment | needs training? | blocked by | days |
|---|---|---|---|---|
| 1 | E1 mode-discovery bake-off | no (ST-SAE seeds only) | — | ~2 |
| 2 | E2 Adag-as-selection | no | E1 candidates | ~1.5 |
| 3 | E3 dynamical-arm upgrade | no | — (parallel with E1) | ~2 |
| 4 | E4 agreement calibration v1 | no | E1+E3 | ~2 |
| 5 | R1 overlap rung + pipeline rerun | generator + ~3 h train | E2 | ~2 |
| 6 | R2 static-inputs rung | ~4 h train | — (parallel) | ~1 |
| 6b | R6 atmosphere-regime rung + E1/E2/E3 rerun | generator + ~3 h train | E2 | ~1 |
| 7 | E6 orientation sweep | no | — (fill GPU-idle time) | ~2 |
| 8 | R3/R4/R5 rungs | yes | E4 v1 | ~7 |
| 9 | E4 final calibration across all rungs | no | 8 | ~1 |
| 10 | E7 baselines | yes | positive E1–E4 | ~4 |

E1+E3 start immediately on existing checkpoints/activations. The decision point
is after E4-v1 (step 4): if the W-free pipeline holds F1 ≥ 0.75 with a working
unsupervised selector on the *current* rung, the program is "harden down the
rungs, then GraphCast"; if not, the bottleneck stage is identified by
construction and becomes the research object.

## 6. Risks

- **E2 is the keystone and the least-precedented step** (using Adag scores for
  model selection over learned representations). Mitigation: E5-rung agreement
  (E4) is an independent selector; the plan survives an E2 null.
- **Static-blob discovery may be *too* easy** (E1 near-ceiling on the current
  rung says little). The overlap + scale rungs (R1, R4) are where G1 earns its
  keep — don't declare victory before them.
- **Two-channel agreement can be confounded by shared Ŵ** — always include
  cross-Ŵ agreement cells in E4.
- **Rung comparability**: keep Φ's edge set frozen across R1/R2/R5; R3/R4
  necessarily change it — score those against their own ceilings only.
- **Compute**: everything before step 5 is CPU/analysis + minutes of GPU;
  training-heavy steps are batched late and parallelizable.
