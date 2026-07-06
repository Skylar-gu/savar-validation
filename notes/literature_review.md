# Literature Review — SAVAR→GraphCast Interpretability Pipeline (2026-07-03)

Screened by abstract across areas (mech interp, causality, climate,
atmospheric science, dynamical systems, and — §E, added 2026-07-03 for the
moving-mechanism spec — object-centric learning, equivariance, and
super-resolution); the sources below passed screening and were read (full text
where accessible, abstract+key-sections otherwise). One paragraph per source.
Screening rejects listed at the end. The action items this review generates are
consolidated in [next_steps_plan.md](next_steps_plan.md) §"Updated recommendations"
and, for §E, in [[moving_mechanism_subres_spec_v2]].

---

## A. Mechanistic interpretability

**MacMillan & Ouellette (2025), "Towards mechanistic understanding in a
data-driven weather model" (arXiv:2512.24440).** The direct predecessor of our
target experiment: TopK SAEs on GraphCast's intermediate (processor) activations,
trained on teacher-forced ERA5 steps. They recover features for tropical
cyclones, atmospheric rivers, diurnal/seasonal cycles, large-scale precipitation,
sea-ice extent, and — notably for us — "specific geographical coding" (their
grid-locked bucket), and validate by intervention: sparsely scaling a tropical-
cyclone feature steers hurricane intensity while the model's response stays
physically consistent. What they do *not* do is any feature–feature interaction /
circuit analysis or any causal-graph recovery — exactly the gap our
SAE→PCMCI+ program targets — and they do not test whether the geographic-coding
features are causally used. Our SAVAR grid-lock result (position decodable but
epiphenomenal, in both SAE-ablation and VPD-gate forms) is a concrete,
falsifiable prediction about their geographic features; running their steering
protocol against SAVAR ground truth (dose–response of Z_j under feature scaling)
is a validation rung we should add.

**KAN-SAE (arXiv:2605.17493, May 2026), "Beyond Linear Superposition:
Discovering Climate Features in AI Weather Models with KAN-SAE."** Replaces the
SAE encoder's ReLU with a learnable per-feature cubic B-spline gate (1,024
features, 9 control points each) on layer-5 residual activations of a 20.5M-param
weather transformer. The argument: atmospheric features are regime-dependent —
present only above a threshold, saturating once a synoptic pattern matures — so a
universal linear-then-ReLU gate misallocates them. Results: 95% vs 55% feature
utilisation, 72% more alive features, lower redundancy, at ~2% reconstruction
cost; a European-heatwave feature with a sharp convex threshold profile that the
linear SAE misses entirely (nearest linear feature 51° away), plus a
steering dose–response of r² > 0.99. Relevance: our hetdynamics data has
saturating-tanh dynamics, and our slow-mode features plateau at 66–88% of the
*linear* ridge ceiling — a per-feature nonlinear gate is the natural backup
architecture if alignment saturates below ceiling, and their threshold argument
suggests our "content lives in low-variance channels off PC0" geometry may be
gate-shaped rather than direction-shaped.

**SynthSAEBench (arXiv:2602.14687, Feb 2026).** A synthetic-activation benchmark
with ground-truth features and controllable superposition (mean-max cosine),
low-rank feature correlation, hierarchy (children fire only with parents), and
Zipfian firing — the LLM-side twin of our SAVAR ladder, which strongly validates
the ladder's premise (the field now agrees SAE claims need ground-truth
testbeds). Two findings recalibrate our expectations: (1) *no* SAE architecture
achieves perfect feature recovery even when the linear representation hypothesis
holds exactly — best per-latent probing F1 ≈ 0.88 vs 0.97 for supervised probes,
so our SAEs sitting at 66–88% of the ridge ceiling is normal, not a bug; (2)
reconstruction quality and feature recovery *dissociate* (Matching-Pursuit SAEs
get the best reconstruction and the worst features by exploiting superposition
noise; Matryoshka SAEs the reverse). Their metric suite — Hungarian-matched mean
correlation coefficient between decoder columns and ground-truth directions,
feature uniqueness, per-latent F1 — is straightforwardly portable to SAVAR and
strictly better than our current best-|r| alignment scalar.

**"Are Sparse Autoencoder Benchmarks Reliable?" (arXiv:2605.18229, May 2026).**
Audits SAEBench-style metrics via reseed noise, synthetic ground truth, and
discriminability along training trajectories. Verdict: targeted-probe-perturbation
and spurious-correlation-removal metrics fail outright; most others are noisier
and less discriminative than assumed; k-sparse probing (sae-probes variant)
survives best but cannot separate variants of one architecture. For us: report
reseed error bars on every SAE alignment number (we currently train one seed per
config; the dictionary-size sweep suggests our headline numbers are stable, but
that should be shown, not assumed), and prefer probe-based metrics — which we
already moved toward with `probe_mode_identity.py`.

**"Do Sparse Autoencoders Capture Concept Manifolds?" (Goodfire,
arXiv:2604.28119, Apr 2026).** Extends the linear-representation hypothesis to an
additive mixture of low-dimensional concept *manifolds* and identifies three SAE
regimes — capture (one compact atom group spans the manifold), tiling/shattering
(atoms partition it, strong negative Ising couplings), dilution (redundant
overlapping atoms, the regime real LLM SAEs occupy). Methodologically, pairwise
inverse-Ising couplings on binarized codes + community detection cleanly recover
which features jointly represent a manifold, while decoder cosine fails. This
gives our Direction-4 grouping plan its concrete recipe and a diagnosis
vocabulary: our distributed mode-identity code (probe 71%, no single feature) is
textbook dilution, and slow-mode content in hetdynamics is plausibly a 1-D
manifold (Z_j amplitude) that our TopK SAEs tile — testable by fitting Ising
couplings on our mixed-SAE codes and checking for the negative-coupling tiling
signature among same-mode features.

**Stochastic Parameter Decomposition (Bushnaq, Braun & Sharkey,
arXiv:2506.20790) and APD (arXiv:2501.14926).** The lineage under our VPD runs:
APD framed interpretability as minimising mechanistic description length via
attribution-based decomposition but was cost- and hyperparameter-fragile; SPD
replaced attribution with stochastic ablation masks and causal-importance gates,
recovering ground-truth mechanisms in toy models without shrinkage. Two details
matter for our redundancy problem: SPD's validation is precisely
"ground-truth-mechanism recovery in toy models" — our SAVAR GNN is arguably the
first *non-toy, non-transformer* test — and the paper documents rank-1
*splitting* (one mechanism shredded across components) as the known failure mode
when minimality pressure is weak relative to the mechanism count, which is the
config-side hypothesis for our participation-ratio-1.1 gate maps.

**Goodfire, "Interpreting Language Model Parameters" (VPD, May 2026).** The
method we run: SPD plus PGD-adversarial ablation masks and a superlinear
frequency-minimality penalty, demonstrated on a 4-layer Pile transformer
including attention, with component-level circuit tracing and targeted model
edits. Read against our results, the salient points are that their
specialization demonstrations are all on models whose tasks *demand* many
distinguishable mechanisms (token-level language), and that they tune the
minimality/recon balance per target; neither condition held in our M4 run
(monolithic forecast task; faithfulness term dominating the loss 5 orders of
magnitude above minimality). Their circuit-tracing over subcomponent
interactions is the natural follow-on once components differentiate on
hetdynamics.

**Hu, Glatt & Liu, "SAEs as a Steering Basis for Phase Synchronization in
Graph-Based CFD Surrogates" (arXiv:2604.04946, Apr 2026).** The closest
architecture match to our setting: SAEs trained on frozen **MeshGraphNet**
embeddings (weight-shared message passing on a mesh, like our MeshGNN and
GraphCast). They find sparse SAE features beat dense embeddings and PCA as a
*control* basis, but static feature injection fails — steering oscillatory flows
requires dynamically phase-aware interventions (Hilbert/SVD-informed rotations)
that respect amplitude–phase structure. Two transfers: (1) independent evidence
that frozen-GNN SAE features are usable causal handles on mesh surrogates; (2) a
warning that our planned steering validation on SAVAR should modulate features
*along trajectories* (time-aware), not as static offsets — especially for the
slow modes whose content is effectively a phase/amplitude variable.

## B. Causality and causal discovery

**Runge (2020), "Discovering contemporaneous and lagged causal relations in
autocorrelated nonlinear time series" — PCMCI+ (arXiv:2003.03685, UAI).** The
algorithm our discovery stage now standardises on. Key properties confirmed
against the paper: separate lagged/contemporaneous condition-set phases with
momentary-conditional-independence tests, order-independence, benefits (rather
than suffers) from autocorrelation, and much higher contemporaneous orientation
recall than PC-style baselines while controlling false positives. It is
explicitly motivated by "time resolutions too coarse to resolve time delays" —
our aliasing experiment instantiates exactly this and confirms the claimed
advantage (CON-F1 0.63/0.49/0.38 vs 0 for plain PCMCI at stride 2/3/4). Its known
limit — τ=0 edges left as Markov-equivalent `o-o` under ParCorr — is what the
LiNGAM-family work below resolves.

**Gong et al. (ICML 2015), "Discovering Temporal Causal Relations from
Subsampled Data."** The theory behind our orientation gap: with Gaussian noise
the fine-cadence VAR is *not identifiable* from subsampled observations (many
fine models produce the same coarse covariance), but with non-Gaussian
innovations the fine transition matrix is identifiable under mild conditions,
with EM and variational estimators given. Companion work (UAI 2017) extends this
to temporal *aggregation*, where identifiability survives averaging as the
aggregation factor grows. Our finecadence generator is deliberately non-Gaussian
(skew-normal), so the aliased τ=0 edges PCMCI+ leaves unoriented are, by this
theory, orientable in principle — we should not invent an orientation heuristic
but benchmark a LiNGAM/NG-EM-style estimator on the subsample sweep, scoring
orientation accuracy against the known fine graph.

**"Causal discovery on vector-valued variables and consistency-guided
aggregation" (arXiv:2505.10476, 2025).** Directly targets the step our whole
pipeline takes for granted: collapsing grid-level fields to mode-level scalars
before discovery. They show naive aggregation (spatial averages — our W-pooling
is exactly this) can create or destroy edges relative to the vector-valued
ground truth, define three aggregation-consistency scores testing whether an
aggregation map preserves the independence model, and give a wrapper (Adag) that
optimises aggregation for discovery reliability. For disjoint SAVAR blobs our
W-pooling is provably benign, but the moment we move to the overlapping-modes
rung (plan D3.3) or to GraphCast (where "modes" must be discovered), aggregation
consistency becomes a live failure mode — their scores belong in our pipeline as
a pre-discovery diagnostic on any learned pooling/grouping.

**"Causality for Earth Science — A Review on Time-series and Spatiotemporal
Causality Methods" (arXiv:2404.05746).** Broad survey spanning Granger,
constraint-based (PC/PCMCI family), noise-based (LiNGAM), score-based, and
state-space/CCM methods, with Earth-science-specific challenges (latent
confounding, aggregation, nonstationarity, teleconnections) and an inventory of
synthetic/simulated/observational benchmarks and tools. Useful to us mainly as
(1) confirmation that our method roster (PCMCI+/DYNOTEARS/TSCI + planned LiNGAM)
covers the method families a reviewer would expect, and (2) a source of
positioning language: SAVAR appears in this literature as *the* standard
mode-level benchmark, and no prior work runs causal discovery on *forecaster
internals* validated against it — that remains our differentiated claim.

**Runge et al. (2019/2023 lineage: Nature Comms perspective; Nowack et al. 2020,
"Causal networks for climate model evaluation and constrained projections").**
The applied payoff pattern for causal graphs in climate: PCMCI-derived causal
networks computed from model output vs reanalysis serve as process-level
fingerprints — models whose causal networks match observations are weighted
higher, constraining projections. Relevance: this is the consumer-side template
for what our pipeline would deliver on GraphCast — a causal network over
discovered internal features, comparable across models/reanalysis — and it
justifies the investment in getting feature-level graphs trustworthy on SAVAR
first, since the evaluation currency is graph agreement.

## C. Climate and atmospheric science

**Tibau et al. (2022), "SAVAR: A spatiotemporal stochastic climate model for
benchmarking causal discovery" (Environmental Data Science).** Our generative
substrate; re-read for what we still under-use. SAVAR's stated purpose is
benchmarking *discovery* methods at grid level given mode-level ground truth;
its two "essential climate properties" are spatial aggregation and
local+long-range dependency. Our extensions (fine-cadence heterogeneous lags,
nonlinear saturating/bilinear dynamics, non-Gaussian innovations, heterogeneous
per-mode timescales) all preserve its mode-level VAR contract, which is what
keeps ground truth exact — worth stating in any writeup, since it is the
property none of the LLM-side synthetic benchmarks (e.g. SynthSAEBench) have: a
*dynamical* ground truth with a causal graph, not just a feature dictionary.

**Lam et al. (2023), GraphCast (Science).** The eventual target: 37M-param
encode–process–decode GNN on an icosahedral multi-mesh (16 MP layers, latent 512/
node), 6 h steps, autoregressive rollout training. The architectural facts that
shaped our testbed remain the load-bearing ones: *shared* processor MLPs over a
*heterogeneous* multi-mesh (grid-lock substrate), static geographic inputs
(lat/lon, orography, land-sea mask — the "identity as content" channel our SAVAR
GNN lacks by design), multivariate coupled fields, and rollout training. Each is
one rung of the remaining ladder, in that order of cost.

**"Disentangling regional impacts of joint teleconnections using causal
representation learning" (DAG-VAE, arXiv:2603.02879, Mar 2026).** The
representation-learning *alternative* to our post-hoc route: a VAE per variable
(tropical-Pacific SST, Indian-Ocean SST, East-African precipitation) whose
latent prior is a physics-informed structural causal model with LASSO sparsity,
trained on SEAS5 hindcasts and ERA5. Latents align with ENSO/IOD by
construction, beat PCA-regression (anomaly correlation 0.68 vs 0.51), and
counterfactual basin-swap experiments reproduce known SST-replacement responses.
Limitations they concede — instantaneous-only coupling, linear latent SCM,
dataset sensitivity — are exactly where our machinery is stronger (lags,
nonlinear tests, ground-truth scoring). Action: DAG-VAE-style "graph-in-the-
latent" is a baseline our SAVAR ladder can score fairly against the
SAE→PCMCI+ route — same data, same ground truth Φ, two philosophies.

**"Causal Climate Emulation with Bayesian Filtering" (arXiv:2506.09891).**
Second data point in the same trend: an emulator whose latent dynamics are
constrained by learned causal structure, with a Bayesian filter for stable
long-horizon autoregression, validated on a realistic synthetic dataset plus two
climate models. Confirms (1) synthetic-with-ground-truth validation is now the
accepted methodology for causal-ML climate work, and (2) causal structure is
being moved *into* emulators. For us it is a second candidate baseline for the
"discover structure from the trained emulator vs build it in" comparison, and
its stability-via-filtering trick is relevant if we add rollout training.

**Dynamical-testing line (Hakim & Masanam-style controlled experiments on
Pangu/GraphCast-class emulators; surfaced via 2025–26 attribution studies, e.g.
arXiv:2408.16433).** A parallel interpretability channel that uses *no*
internals: apply controlled initial-condition or boundary perturbations to the
frozen emulator and read the response against known dynamics (Matsuno–Gill
response, baroclinic development, geostrophic adjustment, hurricane genesis;
SST-forcing sensitivity differs across emulators — Pangu captures cases
FourCastNet-v2 misses). Relevance is high and cheap: on SAVAR we can perturb
mode j's blob in the input window and read the frozen GNN's response against the
ground-truth impulse response of Φ — a direct causal-structure probe of the
forecaster that bypasses SAEs/VPD entirely and gives the pipeline an
internals-free control arm.

## D. Dynamical systems modelling

**SINDy-SHRED (arXiv:2501.13329; PNAS 2026).** Joint sensing + shallow recurrent
decoding + sparse identification: a GRU over sparse sensors with a latent space
regularised to converge to a SINDy-class (or, restricted to linear, Koopman)
functional — yielding a symbolic, interpretable latent dynamical model with a
provably well-behaved loss landscape, validated on turbulence, sea-surface
temperature, and video. For us this is the strongest "white-box forecaster"
baseline: on SAVAR its latent SINDy model should recover something close to the
ground-truth VAR — bounding what *any* interpretability method could hope to
extract from a black-box forecaster of the same data, and giving a
reference point for our GNN's ceiling-limited representations.

**Koopman-operator interpretability family (KoopGen arXiv:2602.14011,
Deep-Koopman-KANDy arXiv:2605.06000, and kin).** Lift nonlinear dynamics to
linear evolution over learned observables; interpretability arrives via the
spectrum (timescales!) and symbolic dictionaries. Screened as *context, not
adoption*: the relevant idea is that eigenvalue spectra are the natural
coordinates for heterogeneous timescales — our hetdynamics result ("alignment
rises monotonically with mode timescale") is a Koopman-flavoured statement, and
a DMD/Koopman spectral decomposition of the frozen GNN's activation dynamics
would give a cheap, assumption-light cross-check of which timescales the network
actually represents.

**TSCI (Butler et al., NeurIPS 2024) — already vendored in-repo.** Tangent-space
CCM for deterministic dynamical systems; degenerates to chance on stochastic
SAVAR (verified in our Phase-6 baselines, reproducing their Rössler–Lorenz
result as a faithfulness check). Kept in the roster because the GraphCast rung
*is* the deterministic-emulator case TSCI was built for — the open question from
our baseline table ("is an intrinsically deterministic emulator TSCI's regime?")
is still live and cheap to answer there.

## E. Object-centric learning, equivariance, and super-resolution (moving-mechanism spec)

Added 2026-07-03 for [[moving_mechanism_subres_spec_v2]]. These reshape the
moving-mechanism plan from "move the patterns and the tools will work" into a
two-axis (data × architecture) design, and harden the sub-resolution angle.

**Mansouri et al., "Object-centric architectures enable efficient causal
representation learning" (arXiv:2310.19054).** The single most plan-changing source.
Standard causal representation learning assumes the observation is an injective
function of the latents; with multiple objects that breaks, and disentanglement
fails. Their fix pairs a Slot-Attention encoder (one representation per object) with
weak supervision from *sparse perturbations*, recovering each object's properties
far more data-efficiently than a monolithic encoder. The load-bearing consequence
for us: a monolithic encoder that collapses the scene into one vector — exactly our
W-pooled GNN — provably cannot preserve per-object causal independence when objects
move. So our v1 hypothesis ("moving mechanisms force a which-vs-where split") is
predicted to *fail* for the current architecture: the split needs an object-centric
prior, not just moving data. This converts the session's recurring
data/method/architecture adjudication into a designed 2×2 and supplies the
architecture arm (A1/A2) of the v2 spec, plus a "build causality in" baseline to
score against our post-hoc route.

**Invariant Slot Attention (Biza et al., ICML 2023, arXiv:2302.04973).** Standard
slot attention entangles an object's identity with its pose; ISA bakes in
equivariance to per-object translation/scale/rotation by transforming the position
encodings against each slot's own reference frame, yielding identity codes invariant
to where the object sits. This is the concrete recipe behind the A1 rung ("describe
things relative to the blob's own centre"): the minimal change that gives a
location-invariant "what" channel on the mesh GNN. (Companion: Dual-State Slot
Attention for video, arXiv:2606.12601, decouples appearance from identity across
frames — relevant if we go to the drifting `advect` regime; and "Does object binding
emerge in pretrained ViTs?" arXiv:2510.24709, the null-emergence question our plain
network instantiates.)

**Zhang, "Making Convolutional Networks Shift-Invariant Again" (arXiv:1904.11486).**
Downsampling (strided conv, max/avg-pool) ignores the sampling theorem, so small
input shifts scramble internal feature maps; a low-pass (blur) filter before
downsampling restores shift-equivariance nearly for free. Our stride-5 hub lattice +
W-pooling is exactly such a sampling-theorem-violating downsample, so a *moving* blob
aliases inside the network before it can be understood. Gives the cheapest
architecture rung (A0, blur-pool) and the P3 control that separates a
signal-processing bug (internal aliasing) from a representational one (missing
binding prior) — two different failures with very different fixes. (Follow-on:
translation-invariant polyphase sampling, arXiv:2404.07410.)

**"The False Promise of Zero-Shot Super-Resolution in Machine-Learned Operators"
(arXiv:2510.06646) and "Limits of Resolution Equivariance in Fourier Neural
Operators" (arXiv:2606.00677).** Trained operators systematically fail at inference
above (or below) their training resolution, via two mechanisms: resolution-
interpolation failure (energy spikes at unseen sampling frequencies) and
information-extrapolation failure (assigning energy to frequencies never observed) —
i.e. aliasing and hallucinated high-wavenumber content. Their diagnostic is the
*normalized residual spectrum*: real recovery leaves residual energy flat across
wavenumber, hallucination piles it at the un-trained high-k. Two transfers to the
sub-resolution angle: (1) the trained-GNN ladder levels can fail even where the T0
oracle succeeds, so keep the in-principle vs trained-model split sharp; (2) their
residual-spectrum test *is* our T3 hallucination detector, promoted to the primary
sub-res metric.

**Off-grid spectral super-resolution (Candès & Fernández-Granda; atomic-norm line
spectral estimation, e.g. arXiv:1612.01459).** Recovering closely spaced spectral
lines from coarse Fourier data is possible via convex optimisation *iff* a
separation condition on the support holds. Gives the T0 identifiability gate a
theoretical floor: a T0 failure is attributed to a genuine resolution limit
(sub-separation), not a weak method — and tells us how different the colliding
sub-sources' scales must be before recovery is even possible.

**Keller & Welling, "Traveling Waves Encode the Recent Past and Enhance Sequence
Learning" (Wave-RNN, ICLR 2024, arXiv:2309.08045).** A recurrent net whose hidden
state carries induced traveling waves stores a short-term memory of a sequence more
efficiently than wave-free RNNs. For the drifting (`advect`) regime this is a
positive template, not just a null: a *good* moving-mechanism code should look like a
traveling wave carrying its recent past, and the colliding fine scales separate by
their dispersion (phase speed vs wavenumber). Motivates the T2 traveling-wave /
spatiotemporal-Fourier surrogate rung and predicts what to probe for in the `advect`
GNN's activations.

**Spatio-temporal SAEs for video (arXiv:2604.03919) and SAE seed instability
(arXiv:2606.12138).** The first trains SAEs over space×time patches of video
representations — the right function class for a *moving* feature, which a per-frame
pooled SAE cannot represent as one atom; this upgrades T6's SAE. The second shows SAE
features are seed-dependent while the spanned subspace is reproducible — reinforcing
the ≥3-seed, report-the-stable-subspace discipline (already flagged via "Are SAE
benchmarks reliable?") specifically for the moving-mechanism SAE re-run.

**GraphCast/Pangu storm-tracking skill (operational track-error evaluations,
2024–2026).** Both emulators track moving cyclones with position errors competitive
with or better than ECMWF-HRES — i.e. the target models *already* solve the
moving-mechanism problem operationally. So the abstraction we are trying to force on
SAVAR is one GraphCast must possess; the payoff of the SAVAR exercise is pinning down
*where* such an abstraction lives (locations vs channels vs slots) before we go
looking for it in the real model.

---

## Screening rejects (abstract-level, with reason)

| Source | Reason |
|---|---|
| Weight-sparse transformers (arXiv:2511.13653) | transformer-circuit specific; no transfer to GNN regressors yet |
| Domain-filtered knowledge graphs from SAE features (arXiv:2604.23829) | LLM text pipeline; not activation-dynamics relevant |
| MR-GNF regional forecasting (arXiv:2603.13563) | forecasting-skill paper, no interp content |
| Space-time stencil causal discovery (JGR-ML 2025) | local-stencil discovery; our aggregation/mode question differs; revisit if we go gridpoint-level |
| "Causally-informed DL to improve climate models" (JGR 2023) | causal feature *selection* for parameterizations; different layer of the stack |
| SAE scaling with feature manifolds (arXiv:2509.02565) | theory of SAE scaling curves; folded into the concept-manifolds paragraph |
| Assimilative causal inference (arXiv:2505.14825) | DA-centric single-trajectory inference; not our regime |
| Heatwave attribution with AI forecasts (2025EF006453, arXiv:2408.16433) | application of emulators, not interpretation; kept only as pointer to the dynamical-testing line |
| KoopGen / KANDy details | context only, see §D |
| Concept-description survey (arXiv:2510.01048) | LLM auto-interp; not applicable to continuous fields yet |
