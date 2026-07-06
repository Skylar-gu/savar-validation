# Project summary — plain-language orientation

*A one-read orientation for a collaborator who is not in this subfield. Written
after the 2026-07-03 sessions, updated after 2026-07-06; reflects the
reorganized repo layout.*

---

## What this project is

We build **synthetic "toy climate" data** where we know the ground truth, train
neural-network **forecasters** on it, and then use **interpretability tools** to
ask whether the network actually learned the underlying physical mechanisms — or
just a shortcut that mimics them. The long-term goal is to validate this whole
toolchain here, where we can check every answer, and then point it at a real
large weather model (**GraphCast**) to reverse-engineer the causal structure it
learned.

The synthetic data comes from **SAVAR**: a field on a 2-D grid (e.g. 50x50) that
is a sum of a handful of spatial **"modes"** — localized blobs, each an evolving
weather-pattern-like signal — plus noise. Because we define the modes and how
they drive each other over time, we know the exact answer the interpretability
tools are supposed to recover: which modes exist, where they are, how fast they
evolve, and which causes which.

We generated several **variants** of this data to stress different questions
(each now lives under `data/`, checkpoints under `checkpoints/<variant>/`, SAE
activations under `sae_data/<variant>/`):
- **base / dy005 / diurnal** — early datasets; `diurnal` adds day/night + annual
  cycles to look more like GraphCast's inputs.
- **finecadence** — sampled finely in time, then sub-sampled, to study temporal
  *aliasing* (fast dynamics disguised as slow ones).
- **hetdynamics / hetdynamics_eqvar** — modes given deliberately *different*
  timescales (up to ~23x spread) so, in principle, the network has a reason to
  represent them separately. `eqvar` equalizes their variance so speed is the
  only cue.
- **movmech (moving mechanism)** — the newest: each mode's blob is placed at a
  *random location every run*, while its dynamics stay fixed. This forces the
  question "did the net learn *what* a mode is, independent of *where* it sits?"

## What we built

**Forecasters** (predict the next frame from recent frames):
- **CNN** (`train/cnn/`, `SpatioTemporalCNN`, ~3.2M params) — first generation.
- **MeshGNN** (`train/gnn/`, ~0.9M params) — a graph/message-passing network on
  a multi-scale mesh over the grid. This is the current workhorse: architecturally
  closer to GraphCast, and translation-equivariant by construction. Variants
  (blurpool, ref-frame, slot-attention) live in `train/gnn/mesh_gnn_variants.py`.

**Interpretability tools** (all applied to a *frozen* trained forecaster):
- **SAE (Sparse Autoencoder)** — learns a dictionary of sparse "features" from the
  network's internal activations, hoping each feature corresponds to one mode.
- **VPD (adVersarial Parameter Decomposition)** — tries to split the network's
  *weights* into separable mechanistic components. Uses the vendored
  `param-decomp/` library; configs and runners in `vpd/`.
- **PCMCI / PCMCI+** — causal-discovery algorithms that recover the "which mode
  drives which, at what time lag" graph from time series. PCMCI+ additionally
  catches *contemporaneous* (lag-0) edges that plain PCMCI misses under temporal
  aliasing (`pcmci/`).
- **DMD / Koopman** — a model-free spectral method used as an independent check on
  the modes' timescale ordering.
- **Supporting probes**: linear readout probes, feature-steering dose-response,
  inverse-Ising community structure, impulse-response tests.

## What the experiments found

The work ran in phases: build the data and forecasters, get the SAE working
per-mode, then a large battery of tests trying to isolate *mechanism*. The
recurring result across every angle:

> **The network reads WHERE a mode is, not WHAT it is.** Mode identity lives in
> position; there is no location-independent "what-code."

Concrete pillars supporting this (2026-07-03 sessions):
- **Every representation-level *measurement* works.** Linear probes recover mode
  states; SAE features rank-match modes; DMD nails the timescale ordering
  (correlation 0.98 with the designed spectrum); steering a feature produces
  smooth, predictable effects.
- **Every attempt to *isolate a single mode's mechanism* fails**, each for its own
  reason: VPD collapses everything into one blob instead of separate components
  (a limitation of the objective, *not* the data — it persists even on the 23x
  heterogeneous-timescale checkpoint); steering moves *all* modes at once (a global
  volume knob, not a per-mode handle); causal discovery run on the network's
  activations collapses even though it is lossless on the true mode signals.
- **The moving-mechanism ("movmech v2") battery** was designed to force a
  what-code and settled the question:
  - An **architecture sweep** (blurpool / ref-frame / slot-attention) predicted to
    "rescue" a plain net that fails a left-vs-right test — but there was nothing to
    rescue: the GNN is already translation-equivariant, so the predicted failure
    never happens (prediction cleanly *inverted*).
  - The decisive **"location-withheld"** test replaces the ground-truth pooling
    with a probe that must *find* each mode by content. It collapses to zero skill
    — yet the *same* probe, handed the true blob center, recovers full skill. So
    the network genuinely has no content signature for "this is mode k"; position
    is both necessary and sufficient.
  - The **sub-resolution ladder** (can the net "read between the pixels"?) and the
    **space-time SAE + anti-redundancy VPD** payoff tests were all nulls in the
    same direction: the fine signal is recoverable *in principle* (a clean Fourier
    oracle confirms it), but the GNN doesn't exploit it, and every apparent
    "recovery" through the network is position leakage, not dynamics.

The triangulated cause: the GNN concentrates all mode information in **one shared,
spatially-homogeneous subspace** that supports linear *readout* but not per-mode
*surgery*. Whether a mode is fast or slow shows up inside that single shared
representation, but no tool can pry the modes apart because the network never
separated them in the first place.

## The July 6 turn: from diagnosing failures to building a recovery recipe

By July 6 the diagnosis felt finished — new experiments kept re-confirming the
same facts. But hidden inside them was one strongly *positive* result that had
never been exploited: if you average the network's internal activity over each
mode's true location and run causal discovery on those averaged signals, you
recover the **full cause-and-effect graph** — exactly as well as if you had the
true mode signals themselves. Everything needed to reconstruct the causal
structure *is* inside the network. The only "cheat" was that we told the tool
where each mode lives. On GraphCast, nobody will tell us that.

So the problem reduces to three honest questions: **(1)** can we *find* the
mode locations ourselves, from the network alone? **(2)** with no answer key,
how do we decide which candidate "map of modes" to trust? **(3)** with no
answer key, how much should we trust the final graph? This reframing also
flips the project's biggest negative finding into an instruction: since the
network stores identity in *places*, look for modes as **places** (regions
whose activity moves together), not as abstract labels.

The new plan is `notes/literature_extension_experiments.md`; the first three
experiments ran on 2026-07-06 (`notes/literature_extension_results.md`):

- **Finding the modes without the answer key — works, cheaply.** A classic
  climate-science technique (rotated principal components — the same math used
  to find El Niño in real data), pointed at the network's internal activity,
  finds all 8 blobs. Graph-recovery score with the true locations: 0.853; with
  locations discovered from the raw data: 0.853 (identical — and the method
  even gets the *number* of modes right on its own); discovered purely from
  the network's internals: 0.819. **The price of not being told where the
  modes are is about 4%.** Two side-lessons: cheaper clustering methods do
  much worse (method choice matters), and slightly *blurred* locations
  actually beat the exact ones (blurring averages away pixel noise) — you
  don't need to draw the blobs' outlines perfectly, you need the averaged
  signals to keep the right dependence structure.

- **Poking the model and watching the ripples — much better now, and it
  exposed a real flaw in the network.** The old "poke one mode, watch what
  responds" test recovered only half the graph, because responses were read
  through the model's own free-running forecasts, which fade unrealistically.
  Reading the ripples along the *true* data instead removes the fading (the
  speed-ordering of the modes is now recovered almost perfectly), and a
  statistic that *accumulates* slow responses finds **9 of the 12 true links**
  with almost no false alarms — and every false alarm but one is actually a
  real two-step chain (A→B→C showing up as A→C). A final untangling step
  (deconvolution) that separates chains from direct links produces **zero**
  false alarms. Best part: the 3 links still missed turn out to be links the
  network **never learned** — on one of them the true system responds strongly
  and the network not at all, even though the network's internal encodings
  *contain* the correlation. The test can now point at specific physical
  couplings an emulator failed to internalize — exactly the kind of statement
  we ultimately want to make about GraphCast.

- **Choosing without an answer key — first attempt failed usefully; fix is
  running.** Literature scores that check whether a candidate mode-map
  "preserves independence structure" turned out to be gameable: a map that
  destroys *all* signal is perfectly "consistent" (nothing depends on anything,
  at any level) and scored near the top while being worthless. The fixed
  version — which also demands the map keep some signal and not create
  near-duplicate modes — is running now. Either outcome is fine: if it ranks
  candidates correctly we have our answer-key-free selector; if not, the
  backup selector is agreement between the two independent channels above.

## Where it's headed

1. **Finish the answer-key-free machinery**: the fixed selection scores (E2),
   then E4 — calibrate how well *agreement* between the two independent
   channels (read-the-activations vs poke-the-model) predicts actual accuracy,
   because agreement is the only score computable on GraphCast. The July-6
   result that the two channels disagree exactly where the network failed to
   learn a coupling suggests disagreement is not just noise — it's diagnostic.
2. **Rebuild SAVAR toward GraphCast, one property per rung (R1–R6)**:
   overlapping blobs, geography-as-input, several coupled variables, many
   modes, rollout training, and — added July 6 — an **"atmosphere-regime"
   variant (R6)**: today one step of SAVAR is mostly unpredictable noise,
   whereas one step of real weather is nearly deterministic; R6 makes all the
   modes much slower-moving (and the observations cleaner) so the network —
   and the SAEs, which only ever see 3 frames — finally operate in the
   signal-rich setting the real target lives in.
3. Each rung re-runs the same recipe and adds one row to a **transfer table**:
   a sentence we can assert about GraphCast with a known, SAVAR-calibrated
   confidence.
4. End goal unchanged: point the validated recipe at **GraphCast**.

---

### Where things live (post-reorg)

| area | location |
|---|---|
| Data generators / splitters | `data_gen/` |
| Generated data (gitignored) | `data/` |
| CNN forecaster code | `train/cnn/` |
| GNN forecaster code | `train/gnn/` |
| Trained checkpoints (gitignored) | `checkpoints/<variant>/` |
| SAE + probing scripts | `sae/` |
| SAE activations (gitignored) | `sae_data/<variant>/` |
| Causal discovery / DMD | `pcmci/` |
| VPD (weight decomposition) | `vpd/` (+ vendored `param-decomp/`) |
| Baselines / diagnostics / viz | `baselines/`, `visualization/` |
| Results arrays, figures | `results/`, `figures/` |
| Notes (design, results, plans) | `notes/` |
| Ad-hoc / historical scripts | `scratch/` |

For a deeper technical account see `notes/session_results.md` (the full 2026-07-03
run log), `notes/literature_extension_results.md` (the 2026-07-06 run log),
`notes/literature_extension_experiments.md` (the current plan),
`notes/results_cnn.md` / `notes/results_gnn.md` (consolidated forecaster
results), and `notes/repo_summary_and_audit.md`.
