# Project summary — plain-language orientation

*A one-read orientation for a collaborator who is not in this subfield. Written
after the 2026-07-03 sessions; reflects the reorganized repo layout.*

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

## Where it's headed

The diagnosis phase is considered done; the focus shifts from "verify the
problem" to "engineer around it" (see `notes/literature_extension_experiments.md`
and `notes/next_steps_plan.md`). The live threads:
1. **Break the shared subspace before interpreting it** — project out the single
   global amplitude direction (or use decorrelated-dictionary / group-sparse SAE
   objectives) so features have a chance to become mode-selective; re-score with
   the stricter SAE metric suite (uniqueness and steering-leakage are the numbers
   to move).
2. **VPD objective surgery** — add explicit anti-redundancy so weight components
   actually split, since the gates already register the timescale lever.
3. **A moving/advecting-mechanism dataset** (blobs that drift within a sequence)
   to probe temporal, not just spatial, mechanism reading.
4. The end goal is unchanged: once the pipeline can extract genuine mechanism on
   SAVAR ground truth, apply it to **GraphCast**.

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
run log), `notes/results_cnn.md` / `notes/results_gnn.md` (consolidated forecaster
results), and `notes/repo_summary_and_audit.md`.
