# GraphCast Rung — Design Document (2026-07-16)

The finale the ladder was built for: run the validated answer-key-free recipe on
real GraphCast, where no ground truth exists, and read the trust dial. This doc
consolidates every empirical lesson from the SAVAR rungs (see
`literature_extension_results.md`), the original design discussion
(`graphcast_interp_design.md`), and the literature review
(`literature_review.md`) into a concrete execution plan.

**Deliverable:** the directed, lagged causal graph among GraphCast's internal
modes, produced with NO answer key, accompanied by a calibrated trust statement
(PX rank + predicted accuracy band), plus the SAE/SPD side-programs that the
bigger model newly justifies.

---

## 1. Model and infrastructure

- **Model:** DeepMind `graphcast_small` — 1°, 13 pressure levels, mesh M5,
  encode–process–decode, 16 message-passing layers, latent 512/node. JAX.
  Weights from the public GCS bucket. Fits the L40S (48 GB) with headroom.
  (Full 0.25° GraphCast is 16× the data and does not fit the local disk budget;
  small first, and the recipe is resolution-agnostic by construction.)
- **Scope caveat (state in the paper):** results characterize
  `graphcast_small`'s internals, not the 0.25° flagship's — siblings by
  architecture and training recipe, but different networks. The mode-level
  claims target synoptic/teleconnection scales (≫1°), where 1° is the right
  level of description; the 0.25° extension is a defined follow-up rung
  (streamed WB2 + larger instance), not a design change.
- **Single-model coherence rule:** every quantitative artifact in the rung —
  activations, SAE features, candidate Ŵs, both graphs, steering — comes from
  ONE network (`graphcast_small`) and ONE data resolution (1° ERA5). Features
  and graphs from different networks live in different latent spaces and are
  never compared quantitatively (see §5).
- **Compute:** the g6e.8xlarge (L40S + 32 vCPU). Teacher-forced extraction is
  ~22k forward passes (trivial); the dominant CPU cost is PCMCI+ over the
  window ensemble (the robustness rungs bounded this: an 8-mode, 24-realisation
  E1+E4 battery ≈ 2–4 h on 24 workers; budget ~2× for N≈16–24 and 10 windows).
- **Storage — S3 for everything large, local disk as cache only.** Local disk
  is 400 GB (321 free) and must also hold the repo, checkpoints, results.

### S3 bucket layout

```
s3://<bucket>/graphcast-rung/
  era5/                    # WB2 zarr subset, ~200–300 GB (or streamed, see §2)
  activations/
    raw/                   # float16 layer dumps, SAE-training subset ONLY (~50–100 GB)
    mode_series/           # projected candidate mode series — MB-scale, the workhorse
  candidates/              # candidate Ŵ footprints + provenance JSON
  sae/                     # trained SAE weights + feature dictionaries
  spd/                     # SPD subcomponents + gate maps
  results/                 # all litext_gc_* npy (mirrored to repo git for small files)
  figures/
```

Rule of thumb from this week: **project early, store small.** The full-rate
layer-8 dump (512 × ~10k mesh nodes × 21,900 steps) is ~450 GB/layer at fp32 —
never materialize it. Project activations onto candidate footprints on the fly
and store the N-dim mode series (megabytes). Raw activations are landed (fp16,
temporally strided) only for the SAE/SPD training subset.

---

## 2. Data plan: ERA5

**Amount: 15 years of 6-hourly, 1°, 13-level ERA5 (e.g. 2005–2019), ~21,900
steps.** Derivation from our own validated operating point plus PCMCI
requirements:

1. Every validated rung ran PCMCI+ at **T ≈ 2000 samples** per series (τ_max
   5–6, N ≤ 24, RobustParCorr) — the empirically calibrated regime.
2. Earth is ONE realisation. The consensus vote (edge kept if detected in ≥50%
   of realisations) transfers to **time windows as pseudo-realisations**
   (`E4_NWIN` machinery already exists). Meaningful vote ⇒ ~8–12 windows ×
   ~2000 samples ≈ **20k steps ≈ 14–15 years at 6-hourly**.
3. **Deseasonalization is a hard precondition** (seasonal rung: PX +0.132 raw →
   +0.769 deseasonalized) and a stable seasonal fit wants ≥10 annual cycles.
   NOTE: the ensemble-mean deseasonalization used on SAVAR is unavailable (one
   realisation) and would be inexact anyway (nonlinear dynamics — design note
   §II.4). Use per-window harmonic regression at the known periods (annual +
   diurnal + leading harmonics), plus **linear detrend** (the warming trend is
   a common-cause confounder PCMCI+ will otherwise eat).
4. Claim scope at this cadence: synoptic-to-intraseasonal edges (τ_max a few
   days). ENSO-timescale edges are NOT recoverable from 15 years at any
   cadence; a coarsened monthly parallel run over 1979–present can be an
   appendix, with weaker claims.

**Sizing:** ~83 fields/step (6 atm vars × 13 levels + surface/forcing) × 181×360
× 4 B ≈ 22 MB/step → **~470 GB fp32, ~200–300 GB as WB2 zarr**. Two options:
land the subset in `s3://…/era5/` once (preferred; reproducible), or stream WB2
GCS chunks during extraction and never store inputs at all. Hold out 2019 (or
the last year) entirely for liveness/dyn checks and any calibration-transfer
test.

**Regime:** teacher-forced trajectory (consecutive single steps on reanalysis)
— approximately stationary, PCMCI-compatible. NOT autoregressive rollout
(non-stationary; error accumulation is a confounder). R5 says the selector
survives rollout-*trained* models; the *extraction* regime still must be
teacher-forced.

---

## 3. Preconditions and conventions (every one bought with a rung)

| Precondition | Source rung | Enforcement on GraphCast |
|---|---|---|
| Deseasonalize (+ detrend) before any CI test | seasonal (+0.132→+0.769) | harmonic regression per mode series; verify residual spectrum flat at 1/yr, 1/day |
| RobustParCorr as CI test | NG rung (F1 0.923 vs 0.889; FP α-calibrated) | `*_CITEST=robustparcorr` (default) |
| Resolution-homogeneous candidate pool | R4 (+0.357 het → +0.667/+0.790 fair) | fix one N for the whole pool (see §4); pool-resolution spread is an unsupervised red flag |
| Dyn liveness check before trusting PX | R6 (dead dyn ⇒ PX inapplicable) | perturbation probe must show responsive, non-degenerate Ĝ_dyn; report inapplicability honestly |
| PX is a screen + calibrated line, not a fine ranker | NL rung (in-cluster rank = noise; top-pick cost 0.057) | report top-K set, not a single winner, when PX spread < ~0.02 |
| Absolute calibration is per-domain | trust-dial v2 (8/52 in old band; slope 1.48→2.58 across domains) | quote rank + ±0.16 band; never quote a universal line |
| Behavior-based matching for ALL scoring | project convention | match candidate modes across decompositions by series correlation, never footprint cosine |
| Windows-as-realisations consensus, ≥50% | e4_agreement.py | n_win ≈ 10 × T ≈ 2000 |

---

## 4. G1 — the core experiment: PX-ranked causal graph with a trust dial

### Phase 0 — plumbing (gate for everything)
JAX + CUDA on the L40S; load `graphcast_small`; one teacher-forced year end to
end; hook processor **layer-8 mesh-node embeddings** (MacMillan's site; also
grab layers 4/12 cheaply for a depth ablation). Smoke-test the on-the-fly
projection path. Exit criterion: mode series for one candidate, one year, lands
in `s3://…/activations/mode_series/`.

### Phase 1 — candidate pool
One fixed N (start N=24 — R4b showed the machinery works at 24 with a fair
pool; +0.790 within one rank-swap of the bar), 6–10 candidates, all built at
that N:

- `vmax_pix` — varimax on the input/output grid fields (the reliable performer
  on every rung: F1 0.668–0.971)
- `km_pix` — k-means on grid fields
- `vmax_act`, `km_act` — same on mesh activations (see §6: expected weak on our
  MeshGNN evidence, but GraphCast's richer representation is exactly the retry
  hypothesis)
- `leiden` — community detection, the unknown-N scale winner (R4: N̂=24, 23/24
  matched, while varimax collapsed). Run once unconstrained as the N̂ estimate
  that *sets* the pool N, then constrained to the pool N.
- `dmd_act` — DMD on activations
- `sae` — SAE-feature-cluster candidate (§5)
- corrupted anchors: `blur`, `shift5` ported; count-preserving only (R4b
  lesson: exclude count-changing anchors a priori)

No oracle exists — that is the point. The pool IS the measurement instrument.

### Phase 2 — two graphs per candidate
- **Ĝ_int:** PCMCI+ (RobustParCorr, τ_max ≈ 8–12 6-h steps, pc_alpha 0.05) on
  the deseasonalized Ŵ-pooled series, consensus across ~10 windows.
- **Ĝ_dyn:** the perturbation probe, re-engineered for JAX: inject
  mode-patterned perturbations into the input fields (and/or encoder output),
  integrate teacher-forced for k steps, regress response amplitudes on
  perturbed-mode amplitude — the e4 dyn-stage logic
  (`pcmci/e4_agreement.py` dyn/int stages) with the PyTorch forward swapped for
  a jitted GraphCast call. THIS IS THE MAIN NEW ENGINEERING (~1–2 weeks).
  Liveness check is built into this stage.

### Phase 3 — the readout
PX(A) = mean over B≠A of pair-F1(Ĝ_int(A), Ĝ_dyn(B)), behavior-matched. Rank
the pool; report the top-K near-tie set; read predicted accuracy off the v2
dial **as a band, with the per-domain caveat stated**. Deliverable figure =
`trust_dial_paper.pdf` style with the GraphCast pool overlaid at its PX values.

### Phase 4 — physics plausibility (paper payoff)
Sanity-check the winning candidate's graph against known synoptic/teleconnection
structure (e.g. tropical→extratropical lags, land–sea contrasts). This is a
plausibility read, NOT validation — the validation is the ladder. Position
against MacMillan & Ouellette: they produced features; the feature–feature
causal graph with a trust readout is their stated open problem.

---

## 5. G2 — SAE experiments (the representation program)

SAEs are currently NOT in the validated pipeline (the litext rungs use
varimax/kmeans/DMD/Leiden; `sae/` is legacy naming). GraphCast justifies
re-opening them — but note the coherence constraint: **MacMillan & Ouellette
trained their SAEs on GraphCast proper (0.25°). Their features live in that
network's latent space and CANNOT be reused against `graphcast_small`'s modes
or graphs — different weights, different mesh (M6 vs M5), different latent
geometry.** So we train our own:

1. **Port the MacMillan recipe to `graphcast_small`** (TopK SAE, layer-8
   embeddings, teacher-forced 1° ERA5 — their released method, our model).
   Qualitative check: do 1° analogs of their feature classes appear
   (atmospheric rivers, sea-ice extent, diurnal/seasonal cycles, grid-locked)?
   Fine-scale cyclone-core features may genuinely not exist at 1° — report
   what does. This is method transfer, not replication; the credibility anchor
   is that the recipe found rich features on a sibling network.
2. **SAE-cluster candidate for the G1 pool** (from OUR small-model SAE): group
   its features (retry #1 below) and build a Ŵ from cluster spatial
   signatures. The pool structure makes this bet FREE — if the SAE candidate
   is bad, PX screens it (that is what PX is for); if good, it wins and the
   SAE program gets its causal handle.
3. **Feature-level steering on `graphcast_small`** (retry of Block F): scale
   individual features of OUR SAE during teacher-forced steps; measure
   dose-response AND cross-mode leakage. MacMillan showed monotone
   hurricane-intensity steering on GraphCast proper — evidence the experiment
   class works on this architecture family, though their number is not our
   baseline; our leakage metric on our model is the quantitative contribution.

Optional appendix, clearly fenced: a straight replication of MacMillan on
GraphCast proper (0.25°) as a standalone credibility exercise — no
cross-model comparison to any G1 artifact. Only worth it if 0.25° activation
extraction turns out cheap (single-step, no probe); otherwise skip.

---

## 6. Retry ledger — experiments that failed on SAVAR/MeshGNN but may flip on GraphCast

The unifying reason for optimism, made precise by our own diagnostics: **every
representation-side failure traced to the small forecaster's representation
being low-rank/entangled, not to the method.** Block G localized it — pooling
PIXELS through candidate footprints preserved causal signal (F1 0.853 = true-Z
ceiling) while pooling our GNN's ACTIVATIONS destroyed it (F1 0.020);
design-note §II.2 explains why (linear-Gaussian worlds ⇒ near-linear optimal
map ⇒ PC0 ≈ 87% collapse). GraphCast is 512-wide × 16 layers, deeply nonlinear,
trained on real atmospheric data with genuinely modal structure — and MacMillan
& Ouellette's results are direct evidence its representation is feature-rich.

| # | Failed experiment | SAVAR result | Why it may flip on GraphCast | Cost |
|---|---|---|---|---|
| 1 | **Ising/Leiden grouping of SAE atoms** (Block D) | all 8 modes in ONE community; median identity-R² 0.03 | failure was upstream (SAE atoms were global knobs, §II.2 collapse); MacMillan's features are mode-like, so couplings have structure to find. Composition Ising→(communities)→PCMCI+ is designed and unbuilt (design note §4) | low (CPU; pipeline exists in `sae/ising_regimes.py`) |
| 2 | **SAE steering as mode handles** (Block F) | dose-response R² 0.99 but leakage 0.81–0.94 | already demonstrated on real GraphCast by prior art; our leakage metric is the *new* quantitative contribution | low-med |
| 3 | **Activation-side discovery** (`vmax_act`/`km_act`) | collapse at scale (R4: F1 0.032–0.105; R3 acts-arm degenerate: km_act 0.000, one hidden/node can't split co-located channels) | GraphCast has 512 dims/node (vs our 1/node bottleneck) — the exact capacity our diagnosis said was missing | free (already pool members) |
| 4 | **VAR-LiNGAM / hybrid orientation** (Block E) | chance (0.50–0.56) at our T and skewness | real atmosphere is strongly non-Gaussian (extremes), T is 10× larger, and the NG rung showed our CI stack stays calibrated under skew — the identifiability bonus (LiNGAM: linear+non-Gaussian ⇒ full DAG) finally has the moments to cash in | low (CPU, mode series only) |
| 5 | **Grid-locked feature detection** | impossible on our CNN (translation-equivariant, no per-position params) and undifferentiated on our homogeneous-mesh MeshGNN | GraphCast's icosahedral multi-mesh IS heterogeneous (some nodes structurally better connected) — the substrate exists for the first time. Test: features whose firing tracks mesh-node degree/position, invariant to input content; predicted causally inert (ablatable) | low |

Each retry gets the same discipline as the rungs: pre-registered success bar,
committed before the run; a miss is a finding.

---

## 7. G3 — SPD / parameter decomposition (the mechanism program)

From design note §3, updated: SPD reached transformers (GPT-2-small,
Christensen & Riggs Smith Nov 2025) but never a GNN. Two-step plan that uses
the ladder properly:

1. **De-risk on our MeshGNN first** (known Φ = ground-truth mechanisms —
   exactly what SPD's unsolved cluster-naming problem needs). Decompose the
   shared message-MLP of the trained overlap02 checkpoint; test whether
   subcomponent clusters align with the 8 known modes / their causal edges.
   This is a NEW SAVAR rung (cheap, local GPU) and publishable on its own —
   "SPD ported to GNNs, validated against known mechanisms."
2. **Then GraphCast:** decompose the shared processor MLPs; per-node gates give
   **spatial usage maps** per mechanism (the interpretable artifact); compare
   mechanism spatial structure against the G1 mode footprints — do
   weight-space mechanisms and activation-space modes agree? Agreement is
   itself a PX-style two-independent-views check, now at the mechanism level.

Honest risks (design note): per-node masking breaks weight sharing in the
masked pass; subcomponent→mechanism clustering is hand-done in all prior work
(step 1 exists to fix precisely this); compute is many forward passes (budget
GPU-days, checkpoint gates to `s3://…/spd/`).

---

## 8. Budget summary

| Item | Size / time | Where |
|---|---|---|
| ERA5 subset (15 y, 1°, 13 lev, 6 h) | ~200–300 GB zarr | S3 (or streamed from WB2) |
| Teacher-forced extraction (22k fwd passes + proj) | ~1–2 GPU-days incl. I/O | mode series → S3 (MBs) |
| Raw activation subset for SAE/SPD (fp16, strided) | 50–100 GB | S3 |
| PCMCI+ battery (10 candidates × 10 windows) | ~1–2 days on 32 vCPU | results → git + S3 |
| Perturbation probe (Ĝ_dyn) engineering | 1–2 weeks human/agent time | — |
| SAE training (TopK, layer 8) | ~1 GPU-day | S3 |
| SPD de-risk on MeshGNN | ~2–3 GPU-days | local + S3 |
| SPD on GraphCast | GPU-weeks (gated on de-risk) | S3 |

Local disk (400 GB) holds only: repo, the active zarr chunk cache, current
window's tensors. Everything else lands in S3 with lifecycle rules.

---

## 9. Pre-registration and success bars

Committed BEFORE any GraphCast number is looked at (the R4b discipline):

- **G1 primary:** the pool's PX ranking with the top-K near-tie set; liveness
  and pool-homogeneity checks pass. There is no truth-F1 — the pre-registered
  claim is the *procedure*: PX ≥ (parent-rung PX of corrupted anchors) for the
  winning candidate, corrupted anchors ranked below all data-driven candidates
  (the unsupervised sanity ordering), and dial-predicted accuracy quoted as
  rank + ±0.16 band.
- **Retry bars:** #1 ≥2 distinct communities with identity structure (vs 1 on
  SAVAR); #2 leakage < 0.5 for ≥1 feature (vs 0.81 floor); #3 acts-candidates
  within 0.1 PX of pixel-candidates (vs collapse); #4 orientation accuracy
  > 0.65 on high-skew mode pairs (vs 0.56); #5 ≥1 grid-locked feature found
  AND shown causally inert.
- Every run logged to `out/` with the orchestrator status-file pattern;
  results npy in `results/` with `litext_gc_` prefix; large artifacts to S3;
  misses reported as findings.

## 10. Sequencing

1. Phase 0 plumbing (gate) → 2. ERA5 landing + extraction → 3. candidate pool
+ Ĝ_int (CPU, overlappable) → 4. perturbation probe build → Ĝ_dyn → 5. PX +
dial readout (G1 DONE) → 6. SAE program (G2) + retry ledger, reusing the
landed activations → 7. SPD de-risk rung on MeshGNN (can start anytime,
independent) → 8. SPD on GraphCast (gated) → 9. paper assembly.

Steps 3, 6, 7 are mutually parallel once 2 lands — the two-lane orchestrator
pattern from the robustness pipeline applies directly.
