# Interpreting GraphCast on a SAVAR Ladder — Design Notes

Consolidated from discussion. Two parts: (I) the design we reasoned through, organized
by theme rather than chronologically; (II) a comparison against the current
implementation (`generate_diurnal.py` + CNN→PCMCI→SAE), with the points where the
implemented results correct the design called out explicitly.

---

## 1. Goal and overall strategy

The object is to understand *what structure GraphCast actually learned* — not just that
it forecasts well, but what computation/representation underlies the skill. Two
complementary probes:

- **Activations** — train SAEs on internal hidden states; ask what features exist and
  how they relate (the representation).
- **Weights** — decompose the parameters into mechanisms (SPD / parameter decomposition);
  ask what computational sub-operations exist (the mechanism).

Neither can be validated on real GraphCast, because there is no ground-truth list of
"the" features/mechanisms or "the" causal graph among them. Hence a **synthetic ladder
with known ground truth (SAVAR)**: build a system where the modes, the causal graph, and
the forcing are all known by construction, validate the pipeline there, then climb toward
GraphCast losing ground truth only at the top rung. The single discipline that makes the
ladder worth its cost: **add one failure mode per rung so a downstream loss can be
attributed** to a specific stage rather than to the system as a whole.

---

## 2. Probing the activations (SAE on GraphCast)

### 2.1 What the activations are

"Activations" are *internal hidden states*, not the raw input and not the final forecast.
GraphCast is encode–process–decode: an encoder lifts the lat-lon grid onto an icosahedral
multi-mesh, a 16-layer message-passing GNN processes the mesh (latent width **512** per
node — note: *not* 768; 768 is FastNet), a decoder maps back to a residual grid update.
Two orthogonal choices define "which activation":

- **Site:** encoder grid/mesh embeddings (≈ input re-encoding), processor mesh-node
  embeddings per layer (where the learned dynamics live; the usual choice), or decoder
  embeddings.
- **Data-generating distribution:** single-step teacher-forced on reanalysis; a
  teacher-forced *trajectory* (consecutive single steps on ground truth → a time series,
  still on-distribution and ≈ stationary); or true autoregressive rollout (off-distribution,
  **non-stationary** — error accumulates, fields blur with lead time).

For causal discovery over feature time series the right regime is the **teacher-forced
trajectory, deseasonalized** — it gives a time axis while keeping the approximate
stationarity PCMCI+ assumes. Rollout injects non-stationarity that is a confounder for
conditional-independence discovery, unless the closed-loop behaviour is itself the object
of study.

### 2.2 Prior art (MacMillan & Ouellette, Dec 2025)

TopK SAE on **layer-8 node embeddings**, single-step teacher-forced on ERA5. Recovered
interpretable features (tropical cyclones, atmospheric rivers, diurnal/seasonal cycles,
sea-ice extent — which is not even an I/O variable — and spurious "grid-locked" features),
and showed monotone steering of hurricane intensity by scaling one feature, with steered
output respecting hydrostatic/mass/gradient-wind balance. Their stated open problem is
**feature–feature interaction / circuits** — i.e. the causal-discovery-over-features
program is the differentiated contribution.

---

## 3. Probing the weights (SPD / parameter decomposition)

Stochastic Parameter Decomposition (Bushnaq, Braun, Sharkey) writes each weight matrix as
a sum of rank-one subcomponents `U[:,c] V[c,:]ᵀ` under faithfulness (they sum to W),
minimality (few causally important per input), and simplicity (rank-one, single-layer).
"Causally important" = ablatable-without-effect, measured by learning per-subcomponent
gates `g_c(x) ∈ [0,1]` trained via **stochastic masking** (scale each subcomponent by a
random amount and force the masked network to reproduce the output).

Status update that changes the risk: as of mid-2025 it was toy-only, but Christensen &
Riggs Smith (Nov 2025) ported SPD to transformers and located interpretable subcomponents
in GPT-2-small; Goodfire's repo now runs it on small LMs. So weight-space decomposition
has reached real (small) models — but only **transformers**, never a GNN.

For GraphCast: decompose the *shared* message-passing MLPs into mechanisms; because the
MLP is applied at every mesh node, the per-input gate evaluated per node becomes a
**spatial usage map** of each mechanism. Obstacles, in order of severity: (1) transformer→
GNN port (expensive forward passes; gate redesigned for spatial not sequential data);
(2) **clustering subcomponents into named mechanisms is unsolved** and the authors do it
by hand using toy-model ground truth; (3) per-node masking breaks weight-sharing in the
masked pass; (4) interpretation is one step removed from plottable fields. SAVAR de-risks
all of these by supplying ground-truth mechanisms.

---

## 4. From fragments to mechanisms: the grouping problem

Both SPD and SAEs emit *fragments* (subcomponents / atomized features) that must be grouped
into mechanisms/concepts. The Goodfire "concept manifolds" paper supplies a released
pipeline: cluster SAE features by statistical dependence in firing patterns, then read each
cluster's geometry. Of five similarity measures (decoder cosine, co-activation, correlation,
mutual information, **inverse-Ising couplings**), the Ising couplings and conditional
co-activation give the cleanest block-diagonal separation; decoder cosine and Pearson fail.
Community detection (Leiden) on the chosen graph yields the groups.

**Ising vs PCMCI+.** The inverse-Ising coupling `J_ij = 0 ⇔ s_i ⟂ s_j | s_rest` is exactly
the *contemporaneous, undirected, all-conditioning* special case of what PCMCI+ estimates
with lags and orientation. They are not competitors for one job:

- For **i.i.d. concept-probe grouping** (no time axis), Ising is the right tool; a PCMCI+
  substitution is a type-mismatch and fights near-deterministic mutual-exclusion via
  faithfulness violations.
- For **directed temporal dynamics on a real time axis** (the GraphCast trajectory case),
  PCMCI+ is the right tool and Ising is the limited contemporaneous shadow.

Proposed composition (not substitution): Ising/Leiden for cheap, robust *static grouping*
of thousands of atoms into communities → then PCMCI+ on the *continuous activation time
series of those communities* for the directed, lagged structure. This shrinks the variable
set before any CI test, sidestepping PCMCI+'s high-dimensional parent-selection cost.

---

## 5. The SAVAR validation ladder

### 5.1 Generative model and the transition/emission split

SAVAR (Tibau et al. 2022) is a state-space model:

- **Transition (dynamics):** a VAR at the mode level,
  `x_t = Σ_τ Φ_τ x_{t−τ} + ξ_t`, where **Φ is the ground-truth causal graph**.
- **Emission (observation):** the instantaneous, memoryless map from latent modes to the
  observed grid, `d_t = g(x_t)`. Linear emission `d_t = W⁺ x_t` is the standard
  spatial-aggregation operator; each mode contributes a fixed spatial pattern scaled by its
  amplitude.

The split is load-bearing because the two pieces feed different stages: **Φ is the discovery
target** (scored by PCMCI+), while the **emission is what the SAE sees** (where representation
geometry lives). Crucially, the emission carries no lags, so changing it cannot corrupt Φ.

### 5.2 Difficulty knobs (what makes the benchmark match the real problem)

What transfers from SAVAR to GraphCast is *matched difficulty*, not cosmetic resemblance to
ENSO/NAO. The knobs, roughly in order of how much they hurt:

| Knob | Effect | Stresses |
|---|---|---|
| Mode spatial **overlap** | breaks clean grid→mode projection | SAE dilution, recovery |
| Number of modes **N** | high-dim conditioning | PCMCI+ parent search, Ising fit |
| Link density / coefficient strength | weaker/denser graph | discovery F1 |
| Noise (mode vs grid) | masks low-rank signal | SAE, recovery |
| **Non-Gaussian** innovations | changes *identifiability* (see below) | CI test calibration |
| **Nonlinearity** (dynamics and/or emission) | curved geometry / nonlinear dependence | SAE manifold, ParCorr→CMIknn |
| **Periodic forcing** | cyclostationarity | PCMCI+ stationarity, deseasonalization |

**Non-Gaussianity is not cosmetic.** In the linear-Gaussian regime contemporaneous causal
direction is fundamentally unidentifiable (X→Y and Y→X give the same covariance). Non-Gaussian
noise breaks that symmetry — the LiNGAM result: linear + non-Gaussian + acyclic ⇒ the full DAG
is identifiable, because higher moments carry the directional information second-order
statistics cannot. It simultaneously *helps* identifiability and *hurts* ParCorr's calibration,
which is exactly the tension the rung exists to measure. The swap is one line — the VAR
recursion and stationarity condition depend only on Φ, not the noise distribution.

### 5.3 Feature types map to three independently controllable layers

| Feature type | Lives in | Controlled by |
|---|---|---|
| **Grid-locked** | the **computational mesh** (architecture) | mesh *heterogeneity* — multi-scale/irregular connectivity |
| **Diurnal / seasonal / annual** | the **data** (exogenous forcing) | injected periodic drive |
| **Content** | the **dynamics** | modes + causal graph Φ |

Grid-locked features are *not* a property of the underlying physics nor of data collection —
they are artifacts of the discretization/mesh the model *computes on* (in GraphCast, the
icosahedral multi-mesh refinement makes some nodes structurally better-connected). The
consequence: a homogeneous architecture produces no grid-locked features, so validating that
distinction *requires* deliberate mesh heterogeneity.

### 5.4 Rungs and attribution

- **Calibration number (not a milestone):** PCMCI+ on the clean mode series, to record the
  achievable F1 at *these* settings as the ceiling the full pipeline is measured against.
  PCMCI+ itself is already validated elsewhere — this is bookkeeping computed in passing, not
  a sequenced rung.
- **SAE grouping rung:** does an SAE recover the modes from the grid / forecaster activations?
  Requires nonlinearity (otherwise nothing to tile). The honest hypothesis under test: a
  forecaster trained on a modal system need not represent those modes internally — failure is
  a finding, not a bug.
- **Architecture rung:** a forecaster with mesh heterogeneity, for grid-locked features.
- **Stress rungs:** add seasonality, then non-Gaussianity, then nonlinear dynamics, one at a
  time, for attribution.

---

## 6. Build status from this discussion

- `savar.py` — modes + grid + VAR with known Φ, plus stationarity / mode-recovery / low-rank
  sanity checks (5-mode demo: recovery corr ≈ 0.999, ρ = 0.30).
- `savar_nonlinear.py` — *emission* nonlinearity (morphing bump → curved manifold, participation
  ratio 1.00 linear vs 1.65 morphing) and periodic-forcing spectrum demos.

---

# II. Differences vs the current implementation

The implemented pipeline (`generate_diurnal.py`, CNN→PCMCI→SAE, results 2026-06-13) differs
from the design discussion on several axes. Where the difference is an *implemented result
correcting the design*, it is marked **[corrects]**.

| Axis | Design discussion | Current implementation | Assessment |
|---|---|---|---|
| Forecaster | GNN (to match GraphCast message-passing; needed for grid-locked) | **CNN** (k=3), val RMSE 0.596 | divergence — see (1) |
| Nonlinearity location | proposed **emission**-first (keep Φ clean) | diagnosed need for **dynamics** nonlinearity; `generate_nonlinear.py` adds saturating AR + bilinear advection | **[corrects]** — see (2) |
| Where the SAE manifold was shown | demonstrated curvature in **raw grid** space | the SAE is on **forecaster activations**; collapse is a property of the *forecaster representation* | **[corrects]** — see (2) |
| Periodic forcing | sinusoid added to a mode's drive | injected in **latent space before the VAR recurrence**, propagating through Φ; + afternoon heteroskedasticity; semidiurnal dropped (Nyquist aliasing at 6 h) | implementation is richer — see (3) |
| Deseasonalization | generic "subtract the cycle" | **ensemble mean over realisations** = *exact* forced cycle (because dynamics linear + forcing shared) | implementation is sharper, but linearity-bound — see (4) |
| Non-Gaussianity | flagged as important (identifiability) | data is explicitly **linear-Gaussian**; not yet added | gap — see (5) |
| SAE grouping method | Ising couplings + Leiden community detection (unsupervised) | **supervised** per-mode best-feature \|r\| + specificity | scope difference — see (6) |
| N modes / grid | N≈5 → 20–50; 30×30 | **N=8** on a 3×3 blob grid; width 50 = one solar sweep | minor, well-motivated |
| Discovery method label | PCMCI+ (contemporaneous + lagged) | "PCMCI" | confirm whether + variant is used |

### (1) CNN, not GNN — and the cost for grid-locked features
A plain CNN is translation-equivariant with shared kernels, so it has **no per-position
parameters** to latch onto — it cannot produce grid-locked features in the GraphCast sense
(those come from mesh *heterogeneity*, §5.3). The current pipeline therefore cannot validate
the grid-locked-vs-content distinction that the thesis quantifies as causally inert; that
specific desideratum needs a GNN on a heterogeneous/multi-scale mesh. The CNN is a perfectly
good, cheaper choice for the *content* and *cycle* experiments — it just forecloses one of the
three feature-type layers.

### (2) Emission vs dynamics nonlinearity — the implementation is right and the design was mis-targeted
This is the substantive correction. I proposed putting nonlinearity in the **emission**
(morph the spatial pattern with amplitude) and demonstrated curvature in the **raw grid**.
But the SAE is trained on **forecaster activations**, not the raw grid, and the implemented
result shows the binding constraint is the *forecaster representation rank*: with
linear-Gaussian dynamics the optimal one-step map is linear (`d_{t+1}=W⁺ΦW d_t`), so the CNN's
representation is near-low-rank (PC0 ≈ 87%) and the SAE collapses onto the dominant shared
direction — and this persists after deseasonalization, proving the collapse is intrinsic to
linear-Gaussian *generation*, not the cycle. **Dynamics** nonlinearity (saturating AR +
bilinear advective coupling) is therefore both the correct fix and the more faithful analog of
real atmospheric nonlinearity (advection is dynamical), and it was verified to add real
nonlinear predictive signal (+0.066 R² vs +0.010 on the linear data).

Where the emission idea is *not* wrong: emission nonlinearity also makes the
observation→observation forecasting map nonlinear (invert g, linear step, re-apply g), so it
would also break the collapse — and it is the cleaner knob if you ever train an SAE *directly
on the grid field* rather than on forecaster activations, and the safer one for keeping Φ exact.
But for the SAE-on-forecaster setup that is actually implemented, it mis-identifies the
bottleneck; the dynamics-first choice targets the right space.

### (3) Cycle injection before the recurrence — a genuine improvement
Injecting the diurnal/annual drive into the latent forcing *before* the VAR step makes the
cycle a **shared confounder that propagates through Φ**, which is what produces the realistic
failure mode the results show: the top raw-PCMCI false positives are same-cycle-phase mode
pairs (X0→X4, X4↔X6, …), and raw F1 collapses to 0.293 (FP 58.1). My standalone `periodic_forcing`
added a cycle to a mode independently and would not reproduce that propagated confounding. The
heteroskedasticity (innovation variance peaking past local noon) and the Nyquist-aliasing
argument for dropping the semidiurnal tide are both correct refinements not discussed.

### (4) Ensemble-mean deseasonalization is exact — but only while linear
Subtracting the ensemble mean over realisations recovers the forced cycle *exactly* because,
for a linear system with shared deterministic forcing and independent zero-mean noise,
`E[Z_t]` over realisations converges to the deterministic forced trajectory. This is elegant
and beats estimated harmonic deseasonalization. **Caveat to carry into the nonlinear rung:**
once dynamics are nonlinear, `E[f(x)] ≠ f(E[x])`, so the ensemble mean is **no longer the exact
forced cycle** and this deseasonalization stops being exact. The nonlinear follow-up will need
a different deseasonalization (per-realisation harmonic fit, or conditioning on phase), or the
PCMCI F1 = 0.825 "exact restoration" result won't carry over.

### (5) Non-Gaussianity is still unexplored
The current data is linear-Gaussian by construction. Per §5.2 this leaves contemporaneous
links in a Markov-equivalence class (unidentifiable direction) and never stresses ParCorr's
Gaussian assumption. If real GraphCast features are skewed/heavy-tailed (atmospheric extremes
are), the benchmark is currently easier on identifiability and on CI-test calibration than the
target. Adding skewed/heavy-tailed innovations is a one-line, high-value change orthogonal to
the nonlinear-dynamics work.

### (6) Supervised feature-mode alignment vs unsupervised Ising grouping
The implemented SAE metric (does a feature track each *known* mode, with specificity) is the
*supervised* evaluation — appropriate for SAVAR, where the modes are known, and arguably the
cleaner metric there. The Ising/Leiden unsupervised grouping (and the Ising-vs-PCMCI+ analysis)
is for the **GraphCast** setting, where there is no ground-truth mode list and grouping must be
discovered. So this is a scope difference, not a flaw — but the Ising→PCMCI+ composition (§4)
is not yet anywhere in the implemented pipeline and remains to be built for the top rung.

### Points of agreement the implementation confirms
- Deseasonalization is central to *both* stages — and the same ensemble-mean operation fixes
  the PCMCI confounding (F1 0.293→0.825) and removes the cycle from the dominant SAE direction
  (R²(PC0~cycle) 0.32→0.00). The asymmetry is the real result: PCMCI recovery is *restored*,
  while SAE interpretability is *unmasked* as intrinsically limited under linear-Gaussian
  generation rather than improved.
- The linearity concern raised in discussion is confirmed empirically, and the fix (nonlinearity)
  is agreed — the only correction is *which* nonlinearity (dynamics, not emission).
- Deseasonalized PCMCI+ F1 = 0.825 matches the privileged-latent baseline, consistent with the
  earlier result that deseasonalizing is what recovers discovery skill.
