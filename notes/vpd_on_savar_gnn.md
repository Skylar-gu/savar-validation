# adVersarial Parameter Decomposition (VPD) on the SAVAR GNN Forecaster

How to run **VPD** on the SAVAR message-passing GNN (`train/gnn_forecaster.py`)
using the `param-decomp` library (cloned at `./param-decomp/`).

Method: VPD (Goodfire, "Interpreting Language Model Parameters",
`param-decomp/papers/`) is the adversarial parameter-decomposition method. The
**same `param-decomp` repo implements both SPD and VPD** — VPD builds on SPD
(Stochastic Parameter Decomposition) with two additions we treat as **core, not
optional**:
1. **Adversarial ablation masks** — gradient-ascent (PGD) worst-case masks
   (`PersistentPGDReconLoss`) replace stochastic-only sampling, enforcing
   mechanistic faithfulness under the *worst* ablation, not the average one.
2. **Frequency-minimality** — a superlinear penalty + p-norm annealing on how often
   a subcomponent is causally important (the `beta` + `p_anneal_*` fields of
   `ImportanceMinimalityLoss`), encouraging specialized subcomponents.

The short version: this is mostly *configuration*, not new code. `param-decomp`
provides the rank-1 decomposition, the causal-importance (CI) gates, stochastic +
adversarial (PGD) mask sampling, frequency-minimality, and the training loop. Our
job is a thin adapter (load the frozen GNN + a dataloader), the VPD config, and
SAVAR-specific diagnostics.

Environment (built 2026-06-28): a dedicated venv at `param-decomp/.venv` — Python
3.13.14, torch 2.8.0+cu128, CUDA on the L40S. Run everything VPD-side with
`param-decomp/.venv/bin/python` (the base 3.9 env cannot import the library).

---

## 0. The spine: grid-lock is a *shared mechanism with a position-locked gate*

This is the single idea the whole note hangs on, and it is exactly how the SPD
transformer paper handles position.

VPD (like SPD, which it builds on) decomposes each weight matrix into **one shared
set** of rank-1 components, `W ≈ Σ_c U[:,c] V[c,:]`. There is no per-position copy of
the components — the mechanism is global. What varies by position is the **causal-importance gate**
`g_c(x) ∈ [0,1]`: evaluated per input element, it says how necessary component `c`
is *there* (operationally: how much it can be masked without changing the output).

Because the GNN's input carries a **node** axis, the gate is `g_c(node)`. The
message-passing MLP is applied at every mesh node with **shared weights**, but the
gate can fire it at some nodes and not others. So:

> **position-specificity emerges through the CI function, not through per-position
> parameters.** The SPD paper's finding — "subcomponents are assigned largely to
> the relevant positions" — is this: shared weights, position-locked gate. The
> paper does **not** decompose embeddings; position-locking lives in the gate.

For SAVAR this gives a clean operational definition. Average `g_c(node)` over
inputs and reshape to `(50, 50)` — a **spatial usage map** per component. Then:

- **grid-locked mechanism** ⇒ its usage map concentrates on fixed **mesh-structural**
  locations (hubs), *regardless of the content/mode present there* — `g_c` is a
  function of the node's structural identity.
- **content mechanism** ⇒ its usage map tracks the active **mode/signal**, wherever
  it appears — `g_c` follows content, not fixed position.

Grid-lock detection is therefore: *does a shared component's gate depend on mesh
position or on content?* Nothing is stored per-position; we read the gate.

Why this is the right substrate. The SAVAR generator hand-specifies **no**
grid-locked structure: the only fixed spatial structure is the mode layout `W`
(that is *content*, tied to the causal graph Φ), and the observation noise is
`D_y = I_L` (spatially homogeneous). The GNN's only source of mesh-structural
asymmetry is the **heterogeneous multi-scale mesh** (`build_mesh`: hub nodes of
degree 32 vs ordinary cells of degree 8), processed by **shared** MP MLPs. So any
component whose gate locks to hubs is attributable to the mesh — which is precisely
the grid-lock claim (design note §5.3). The shared MP MLPs are the decomposition
target; the per-node gate is the readout.

**Analysis caveat.** The hub lattice (stride-5) and the mode blocks are different
but spatially overlapping layouts. A gate concentrated near a mode centre that
happens to sit on a hub is ambiguous. The diagnostics in §7 must therefore
**disentangle hub-tied (mesh) from mode-tied (content)** usage explicitly, not read
"localized" as "grid-locked".

---

## 1. Target model

`MeshGNN` (`train/gnn_forecaster.py`):

```text
Input  x: (B, K, ny, nx)        K=3 past frames
Nodes  L = ny*nx = 2500         50x50 SAVAR grid
Output y: (B, 1, ny, nx)        next frame
```

```text
feats = x reshaped to (B, L, K)
H0    = encoder(feats)
for each message-passing layer i:
    agg = A_hat @ H                 # fixed sparse adjacency (mesh heterogeneity)
    upd = layers[i].mlp([H, agg])   # SHARED MLP, applied per node
    H   = LayerNorm(H + upd)
forecast = decoder(H)
```

Decomposable weight modules (matched by `fnmatch`):

```text
encoder.0            Linear(K -> hidden)
encoder.2            Linear(hidden -> hidden)
layers.{0..3}.mlp.0  Linear(2*hidden -> hidden)   # SHARED message-passing MLP  <- primary target
layers.{0..3}.mlp.2  Linear(hidden -> hidden)     # SHARED message-passing MLP  <- primary target
decoder.0            Linear(hidden -> hidden//2)
decoder.2            Linear(hidden//2 -> 1)
```

The **shared MP MLPs** (`layers.*.mlp.*`) are the object of interest: they are the
single mechanism applied at every node, so their per-node gate is where grid-lock
(if any) must appear. Recommended first GNN-relevant run decomposes one MP layer:

```text
layers.0.mlp.0
layers.0.mlp.2
```

then expand to all four.

### 1.1 Checkpoint health — verify before investing

Use a **converged** `checkpoints_finecadence/best.pt` (the 40-epoch cosine run, with
`history.npy` present), not a mid-training one. The fine-cadence one-step forecast
is a **weak-signal** target: val RMSE ≈ 0.43 against a data std (≈ predict-the-mean
RMSE) ≈ 0.46 — it beats persistence but beats climatology by only ~7%. Before
decomposing:

- Report forecast **correlation**, not just RMSE, to confirm there is real structure
  to decompose.
- Temper magnitude claims: "component `c` matters" is relative to a target only
  marginally better than the mean. Lean on the **structure** of usage maps, not
  absolute effect sizes.

If grid-lock turns out faint, that is itself a finding (mesh heterogeneity alone
may be a weak driver); the response is to widen the MP MLPs, never to add
per-position parameters.

---

## 2. How this maps onto `param-decomp`

`param-decomp` decomposes each target `W ≈ Σ_c V[:,c] U[c,:] (+ optional Δ)`
(`components.py`) and a CI function emits, per component, a gate `g[..., c] ∈ [0,1]`.
The leading `...` are the layer's input batch dims. For the LM target that is
`(batch, token)`. **For the GNN it is `(batch, node)`** — the node axis falls out
for free, because each decomposed `nn.Linear` inside the MP MLP sees an input of
shape `(B, L, d_in)`. So:

```text
g[b, node, c] ∈ [0, 1]
```

with no custom mask plumbing. A masked forward attenuates component `c`'s
contribution at node `node` for example `b`; the component stays shared — the mask
says only *where and when* the shared mechanism is used. Average `g[:, :, c]` over
examples and reshape to `(50, 50)`: the **spatial usage map** of §0.

### 2.1 CI-function choice — layerwise `vector_mlp`, not the transformer

The pile/LM configs use `ci_config.fn_type: global_shared_transformer`, which runs
RoPE self-attention over the sequence axis. For the GNN that axis is 2500 nodes in
row-major order; RoPE would impose a meaningless 1-D positional bias over a 2-D
grid, and attention over 2500 positions is expensive. Use a **layerwise** CI fn
(`ci_fns.py:17` — `LayerwiseCiFnType = {"mlp", "vector_mlp", "shared_mlp"}`):

```yaml
ci_config:
  mode: layerwise
  fn_type: vector_mlp
  hidden_dims: [64]
```

`vector_mlp` consumes the layer's own input `[..., d_in]` and emits `[..., C]`. For
`layers.i.mlp.0` that input is `concat([H, agg])` — i.e. **the gate sees both the
node's own hidden state and its neighbour-aggregated state.** That is exactly the
signal that distinguishes a hub (large, structured `agg`) from an ordinary cell, so
a position-locked gate can form here *without* any explicit positional encoding —
the GNN analogue of the paper's position-aware importance function. (`fn_type: mlp`
is the per-component scalar variant; fine as a cheap smoke-test, `vector_mlp` is the
default for spatial gating.)

---

## 3. What node-level masking means (and does not)

For a decomposed linear the per-node masked forward is conceptually:

```text
out[b, node] = Σ_c m[b, node, c] · U_c · <V_c, x[b, node]> + Δ·x[b, node]
```

with `m[b, node, c] ∈ [g[b, node, c], 1]`. Shapes for `C=64, B=8, L=2500` →
`m.shape = (8, 2500, 64)`.

Per-node masking **breaks exact weight-sharing during the masked pass** — that is
intended: the underlying component is shared; its mask is node-specific because we
are asking *where* the shared mechanism is used. The primary artifact is the reduced
map `mean_g[c] → (50,50)`. Do not over-read individual `g[b, node, c]` values, and
do not interpret node-specific masks as node-specific parameters.

This is **not** whole-node ablation. Deleting nodes answers "which locations
matter?"; VPD answers "which learned parameter mechanisms matter, and where are they
used?". Whole-node ablation is a separate baseline.

Note: components are shared across a layer 
---

## 4. Losses — VPD config (mirror the pile VPD config)

Loss metrics are a `loss_metrics` list in the `pd:` block; each entry's `type` is a
`Literal` discriminator selecting a `Metric` subclass (`param_decomp/metrics/`).
Reconstruction is MSE to the **frozen GNN's own output** (`recon_loss_mse`,
`param_decomp_lab.batch_and_loss_fns`); for VPD masked-recon must hold under the
*adversarial* masks, not just stochastic ones.

Mirror `param_decomp_lab/experiments/lm/pile_llama_simple_mlp-12L.yaml` (the VPD
paper's config), adapting `position` → `node`:

```yaml
pd:
  loss_metrics:
    - type: ImportanceMinimalityLoss     # minimality + FREQUENCY-minimality (the VPD knob)
      coeff: 1e-4
      pnorm: 2.0
      beta: 0.5                          # >0 ⇒ superlinear frequency penalty
      p_anneal_start_frac: 0.0
      p_anneal_final_p: 0.4
      p_anneal_end_frac: 1.0
    - type: StochasticReconSubsetLoss    # stochastic subset masks
      coeff: 0.5
      routing: {type: uniform_k_subset}
    - type: PersistentPGDReconLoss       # ADVERSARIAL masks — the core of VPD
      coeff: 0.5
      optimizer: {type: adam, beta1: 0.5, beta2: 0.99, eps: 1e-8,
                  lr_schedule: {start_val: 0.01, warmup_pct: 0.025, fn_type: constant, final_val_frac: 1.0}}
      scope: {type: per_batch_per_position}   # position == NODE for the GNN
      use_sigmoid_parameterization: false
      n_warmup_steps: 2
    - type: FaithfulnessLoss             # Δ-faithfulness  Σ‖W - ΣUV‖²
      coeff: 1e7
```

`scope: per_batch_per_position` makes the adversary choose worst-case masks **per
node** — exactly the granularity the spatial usage map needs. All `type` strings are
verified in `param_decomp/metrics/`. For bring-up you may omit
`PersistentPGDReconLoss` for the first few hundred steps (pure stochastic) to confirm
recon converges, then switch it on — turning it on is what makes the run **VPD**
rather than SPD. (`FaithfulnessLoss` as a loss term is the pile/VPD style; the
`resid_mlp` config instead uses a `faithfulness_warmup_*` phase — either works.)

Forecast-quality metrics vs ground truth (`RMSE/forecast_corr(y_masked, y_true)`)
are logged, not optimized; recon is always to `y_ref = frozen_gnn(x)`. For
`eval.metrics` keep regression-agnostic ones (`CI_L0`, `ComponentActivationDensity`,
`CIMeanPerComponent`, `CIHistograms`, `StochasticHiddenActsReconLoss`).

---

## 5. Stochastic and adversarial masks (provided by the library)

The learned `g` lower-bounds how active a component must stay; sampled masks satisfy
`m ∈ [g, 1]`:

```text
stochastic:   u ~ U(0,1);  m = g + (1-g)·u
adversarial:  init m stochastic, gradient-ascent m to MAXIMIZE recon loss,
              clamp to [g,1] each step; train the decomposition + CI fn to
              reconstruct under that worst case  (PersistentPGDReconLoss)
```

Both are library-provided; select via `loss_metrics`, don't write PGD. Node masks
are large, so for pilots: `batch_size ≤ 8`, `C ≤ 32–64`, PGD `n_warmup_steps`
small (1–3), `autocast` bf16.

---

## 6. Experiment structure

```text
vpd/
  gnn_target.py     # load MeshGNN, load checkpoint, .eval(), requires_grad_(False), rebuild _A
  gnn_data.py       # wrap MultiRealisationDataset -> DataLoader yielding (x, y)
  run_gnn_vpd.py    # build PDConfig + Trainer from config_gnn.yaml; Trainer.run(...)
  config_gnn.yaml   # decomposition_targets, ci_config(layerwise vector_mlp), loss_metrics
  diagnostics.py    # mean_g -> (C,50,50) maps; hub/mode projections; ablations
```

`run_gnn_vpd.py` mirrors `param_decomp_lab/experiments/resid_mlp/run.py`:

1. Build target: load `MeshGNN`, load `checkpoints_finecadence/best.pt`
   (`ckpt["model_state"]`), `model.eval()`, **`requires_grad_(False)`** (the
   `ComponentModel` constructor asserts this), rebuild the cached adjacency `_A`
   on-device. Do **not** hand-roll `DecomposedLinear`/gates/masks — `ComponentModel`
   wraps the frozen module and replaces selected submodules via forward hooks, so
   the GNN's own `forward` (sparse `Â H`, LayerNorm, reshapes) is left untouched.
2. Build train/val `DataLoader`s from `data/splits_finecadence` via
   `MultiRealisationDataset` (returns `(x, y)`; use `run_batch_first_element`).
3. `Trainer(target_model=..., run_batch=run_batch_first_element,
   reconstruction_loss=recon_loss_mse, pd_config=cfg.pd, runtime_config=cfg.runtime)`
   then `trainer.run(train_loader, sink, cadence, eval_loop)`. The `Trainer` builds
   the `ComponentModel` (component replacement + CI fn) internally from `cfg.pd`.
4. Save checkpoints, losses, masks, and the §7 diagnostics.

Decomposition targets (config) — shared MP MLPs only:

```yaml
decomposition_targets:
  - module_pattern: layers.0.mlp.0
    C: 64
  - module_pattern: layers.0.mlp.2
    C: 64
sigmoid_type: leaky_hard
use_delta_component: true
```

---

## 7. Diagnostics to save

Per decomposed matrix / component:

```text
mean_g_map.npy        (C, 50, 50)   mean CI gate per node, reshaped  <- the artifact
component_norms.npy   (C,)
delta_norm.npy
reconstruction_rmse.npy
usage_by_mode.npy     (C, n_modes)  content alignment
usage_by_hub.npy      (C, 2)        mesh/structural alignment (hub vs non-hub)
ablation_drop.npy     (C,)          recon/forecast change when component c is masked
```

Content alignment (mode maps `W[j]`, shape `(n_modes, L)`):

```text
usage_by_mode[c, j] = Σ_node mean_g[c, node] · W[j, node]
```

Mesh alignment (hub ids from `build_mesh`):

```text
hub_usage[c]    = mean mean_g[c, hub_nodes]
nonhub_usage[c] = mean mean_g[c, nonhub_nodes]
```

**The grid-lock test (the disentanglement from §0).** For each component regress /
partial-correlate `mean_g[c, node]` against a hub indicator (structural) and against
mode membership (content):

- **grid-locked** ⇒ gate explained by hub-ness, *not* by any mode.
- **content** ⇒ gate explained by a mode, *not* by hub-ness.
- **mixed/ambiguous** ⇒ both (expected where a hub sits on a mode block; report it).

**Causal-native grid-lock (the payoff over SAEs, §8).** Because `g` is
ablation-defined, cross `hub_usage[c]` with `ablation_drop[c]`:

- high hub_usage **and** high ablation_drop ⇒ a *used* grid-locked mechanism.
- high hub_usage **and** ~zero ablation_drop ⇒ a **grid-locked but epiphenomenal**
  mechanism — decodable position that the computation does not rely on. This
  reproduces the earlier SAE finding ("position decodable but causally unused") in a
  **single** object, with no separate ablation experiment.

---

## 8. Why this is an alternative to causal discovery on SAEs

Both programs target the same end — what mechanisms/features exist and how they
relate — by different routes:

| | SAE + causal discovery (PCMCI+) | VPD (this note) |
|---|---|---|
| object | **features** = directions in *activation* space | **components** = mechanisms in *weight* space |
| causality | inferred **post-hoc** via CI tests on feature **time series** | **built in** — importance is *ablation-defined* (`g_c` = how much you can mask) |
| time axis | required → inherits the **subsampling/aliasing** problem (fast couplings → τ=0) | none → the spatial usage map is the readout |
| grid-lock readout | a feature firing at fixed positions → needs a *separate* ablation to test relevance | the gate map *is* the causal-importance map → locked + relevant in one object |
| circuits | feature–feature causal graph over activation dynamics | component co-usage / attribution graph over mechanisms |

The sharpest contrast is grid-lock: SAEs found a grid-locked feature but it took a
second ablation to discover it was epiphenomenal; VPD's `g` is itself a
causal-importance gate, so "locked **and** inert" falls out of one map (§7). VPD
also sidesteps the temporal-aliasing pitfalls entirely — there is no time axis to
subsample. The trade-off: VPD probes **mechanism** (what computation exists), the
SAE route probes **representation** (what is linearly encoded), so they remain
complementary for the "what concepts are represented / how do they co-evolve"
questions. For **grid-lock** and **circuits** specifically, VPD is the
causality-native tool.

A cross-check that bridges both: build component **time series** (per-component
contribution magnitude averaged over nodes/modes) and run the existing PCMCI feature
pipeline on them, comparing the recovered graph to the ground-truth Φ — VPD
mechanisms feeding the causal-discovery program rather than competing with it.

---

## 9. Milestones

**M1 — smoke test.** Decompose `decoder.0` only, `ci_config.fn_type: mlp` (scalar),
stochastic recon + faithfulness warmup. Verify the wrapped model reconstructs the
frozen forecast (recon RMSE → ~0). Validates the adapter.

**M2 — first GNN-relevant run.** Decompose `layers.0.mlp.0`, `layers.0.mlp.2`;
`vector_mlp` CI; node-level stochastic masks. Save the `(C,50,50)` usage maps and
run the §7 hub/mode disentanglement + ablation cross.

**M3 — turn on the adversarial term (this is what makes it VPD).** Add
`PersistentPGDReconLoss` + frequency-minimality (`beta>0`, `p_anneal_*`); confirm
worst-case recon stays low; tune for sparse, localized usage.

**M4 — all processor layers.** Decompose `layers.0..3` MP MLPs. Compare maps across
depth: early local/mesh mechanisms vs later mode/global ones.

**M5 — causal evaluation.** Project usage onto SAVAR modes; ablate high-usage
components for forecast importance; run PCMCI over component time series vs Φ; test
whether components aligned to high out-degree modes are more forecast-critical
(Phase-8 centrality tie-in).

---

## 10. Risks

**Weak target signal.** ~7%-over-climatology skill ⇒ small component contributions;
interpret usage **structure**, not magnitudes (§1.1).

**Faint grid-lock.** Mesh heterogeneity alone may induce only weak position-locking.
That is a finding; widen the MP MLPs if needed — do not introduce per-position
parameters (they would manufacture the answer and have no GraphCast analogue).

**Memory.** `B×L×C` masks are large (B=32, L=2500, C=64 → 5.12M values/layer).
Mitigate: small batch, one layer first, bf16, smaller C for pilots.

**Identifiability.** Rank-1 components may split one mechanism across atoms.
Mitigate: importance minimality, co-usage clustering, and the hub/mode + ablation
validation in §7.

---

## 11. Bottom line

VPD on the SAVAR GNN is feasible and mostly configuration on top of `param-decomp`:
a frozen `MeshGNN`, a `(x, y)` dataloader, a layerwise `vector_mlp` CI fn, MSE
reconstruction to the frozen forecaster, and the **shared message-passing MLPs** as
decomposition targets. The per-node CI gate becomes a plottable, per-component
mechanism-use map over the 50×50 mesh — and grid-lock is read off it as a *shared
mechanism with a mesh-locked gate*. That is the bridge to GraphCast:

```text
SAVAR node = 50x50 grid cell        ->  GraphCast node = multi-mesh node
SAVAR shared MP-MLP component map    ->  GraphCast processor-MLP component map
grid-lock = hub-locked CI gate       ->  grid-lock = refinement-locked CI gate
```

Start small: one converged GNN checkpoint, one message-passing layer, stochastic
node-level masks, and component maps evaluated against the known SAVAR modes and hub
structure.
</content>
