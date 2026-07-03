# VPD (adVersarial Parameter Decomposition) — Repository Guide & Method Overview

**Document combining:**
- The VPD research paper: [Interpreting Language Model Parameters](https://www.goodfire.ai/research/interpreting-lm-parameters) (Goodfire AI, May 2026)
- The reference implementation: [goodfire-ai/param-decomp](https://github.com/goodfire-ai/param-decomp)

---

## 1. What is VPD?

**adVersarial Parameter Decomposition (VPD)** is a method for decomposing a neural network's parameters into interpretable subcomponents that each implement a small part of the model's learned algorithm. Unlike activation-based methods (e.g., transcoders, SAEs), VPD operates directly on the **parameters** of the network, producing mechanistically faithful descriptions that map directly to the weights and nonlinearities doing the computation.

### Key properties VPD optimizes for:

| Property | Description | How enforced |
|----------|-------------|--------------|
| **Parameter-faithful** | Subcomponents sum to the original model's parameter vector | Δ-components + L2 loss |
| **Minimal** | As few subcomponents as possible are causally important on any input | Importance minimality loss |
| **Mechanistically faithful** | Any subset containing the important subcomponents suffices to reconstruct output | Adversarial + stochastic reconstruction loss |
| **Simple** | Each subcomponent involves minimal computational machinery | Rank-1 constraint + frequency minimality loss |

### VPD vs. SPD (Stochastic Parameter Decomposition)

VPD builds on SPD but introduces two critical differences:
1. **Adversarial ablations**: Instead of only sampling ablations stochastically, VPD uses *gradient-ascent-optimized* adversarial ablation masks to find the worst-case ablations that still preserve output. This ensures robustness.
2. **Frequency minimality loss**: A superlinear penalty on how often subcomponents are causally important, encouraging specialization.

---

## 2. The Core Mathematical Method

### 2.1 Parameter Subcomponents

Each weight matrix $W^l$ is decomposed into rank-1 subcomponents (outer products):

$$W^l \approx \sum_{c} \vec{U}^l_c (\vec{V}^l_c)^\top = U^l (V^l)^\top$$

Where $c$ indexes subcomponents. There may be **more subcomponents than rows/columns**, allowing superposition.

### 2.2 Δ-Components (Parameter Faithfulness)

$$\Delta^l := W^l - \sum_{c} \vec{U}^l_c (\vec{V}^l_c)^\top$$

The residual $\Delta^l$ is penalized with an L2 loss to keep the decomposition close to the original parameters.

### 2.3 Causal Importance Function

A neural network predicts, for each subcomponent $c$ of matrix $l$ at batch $b$ and position $t$, a causal importance value:

$$g^l_{b,t,c} \in [0,1]$$

- $g = 0$: subcomponent can be fully/partially ablated without affecting output
- $g = 1$: subcomponent is necessary

### 2.4 Ablation Masks & Reconstruction Loss

Ablation masks $m^l_{b,t,c} \in [g^l_{b,t,c}, 1]$ define masked weight matrices $W'^{l}_{b,t}$. The reconstruction loss demands that the KL-divergence between the original output and masked output be small:

$$\mathcal{L}_{\text{masked-recon}} = \frac{1}{B} \sum_{b=1}^{B} D_{KL}(f(x_b | W), f(x_b | W'_b(m)))$$

**Two sampling strategies:**
1. **Stochastic**: $m^{\text{stoch}}$ drawn from uniform distributions → $\mathcal{L}_{\text{stochastic-recon}}$
2. **Adversarial**: $m^{\text{adv}}$ optimized via gradient ascent to *maximize* reconstruction loss → $\mathcal{L}_{\text{adversarial-recon}}$

### 2.5 Full Loss Function

$$\mathcal{L}_{\text{VPD}} = \beta_1 \mathcal{L}_{\text{adversarial-recon}} + \beta_2 \mathcal{L}_{\text{stochastic-recon}} + \beta_3 \mathcal{L}_{\text{importance-minimality}} + \beta_4 \mathcal{L}_{\text{frequency-minimality}} + \beta_5 \mathcal{L}_{\text{Delta-L2}}$$

---

## 3. Repository Structure

```
goodfire-ai/param-decomp/
├── pyproject.toml                          # Core package: param-decomp
├── param_decomp/                           # Core library (import as param_decomp)
│   ├── configs.py                          # PDConfig, RuntimeConfig, Cadence, metric configs
│   ├── optimize.py                         # Main optimize() entry point
│   ├── metrics/
│   │   ├── base.py                         # Metric base classes (LossMetricConfig, etc.)
│   │   └── dispatch.py                     # LOSS_METRIC_CLASSES dispatch
│   └── ...
├── param_decomp_lab/                       # Lab package (import as param_decomp_lab)
│   ├── pyproject.toml                      # Separate distribution: param-decomp-lab
│   ├── experiments/                        # In-repo experiment scripts
│   │   ├── tms/                            # Toy model: TMS (tms_5-2_config.yaml)
│   │   ├── resid_mlp/                      # Toy model: Residual MLP (resid_mlp1_config.yaml)
│   │   └── lm/                             # Language model: 4L Pile (pile_llama_simple_mlp-4L.yaml)
│   ├── batch_and_loss_fns.py               # recon_loss_mse, run_batch_first_element
│   ├── run_sink.py                         # RunSink (output directory management)
│   └── eval_metrics.py                     # EVAL_METRIC_CLASSES dispatch
├── Makefile                                # Dev commands
└── tests/                                  # Test suite
```

### Two Python Distributions

| Distribution | Import Path | Contents |
|--------------|-------------|----------|
| `param-decomp` | `param_decomp` | Core library: configs, optimize loop, metrics base |
| `param-decomp-lab` | `param_decomp_lab` | Experiments, CLI, postprocessing, app tooling |

---

## 4. Installation

```bash
# Full dev install (core + lab + dev deps + pre-commit hooks)
make install-dev

# Core package only
make install

# Core + lab, without dev dependencies
make install-lab
```

The repo uses `uv` workspace for local development, so absolute imports work for both packages after install.

---

## 5. Running VPD

### 5.1 Quick Start: In-Repo Experiments

The `pd-*` CLI commands are installed by `param-decomp-lab`. Each experiment is a self-contained script that reads a YAML config and calls `optimize()`:

```bash
# Toy model: TMS
pd-tms       param_decomp_lab/experiments/tms/tms_5-2_config.yaml

# Toy model: Residual MLP
pd-resid-mlp param_decomp_lab/experiments/resid_mlp/resid_mlp1_config.yaml

# Language model: 4-layer Pile model
pd-lm        param_decomp_lab/experiments/lm/pile_llama_simple_mlp-4L.yaml
```

### 5.2 Writing a Custom Experiment

For a brand-new experiment, write a `run.py` that:

1. **Builds the target model** (the model you want to decompose)
2. **Creates train/eval dataloaders**
3. **Defines eval metrics** as a list of pre-instantiated `Metric` objects
4. **Configures `PDConfig`** and `RuntimeConfig`
5. **Sets a `Cadence`** (how often to log, save checkpoints)
6. **Creates a `RunSink`** (where output goes)
7. **Calls `optimize()`**

**Minimal template:**

```python
from param_decomp.configs import Cadence, PDConfig, RuntimeConfig
from param_decomp.optimize import EvalLoop, optimize
from param_decomp_lab.batch_and_loss_fns import recon_loss_mse, run_batch_first_element
from param_decomp_lab.run_sink import RunSink

optimize(
    target_model=my_target_module,
    train_loader=train_loader,
    run_batch=run_batch_first_element,      # or your own batch runner
    reconstruction_loss=recon_loss_mse,     # or your own loss function
    pd_config=PDConfig(...),                 # hyperparameters for VPD
    runtime_config=RuntimeConfig(device=device),
    cadence=Cadence(
        train_log_every=100,
        save_every=5000,
    ),
    sink=RunSink.local(out_dir),            # local output directory
    eval_loop=EvalLoop(
        loader=eval_loader,
        metrics=[...],                       # list of Metric objects
        n_steps=10,
        every=1000,
        slow_every=5000,
    ),
)
```

**Reference `run.py` files:**
- `param_decomp_lab/experiments/tms/run.py`
- `param_decomp_lab/experiments/resid_mlp/run.py`
- `param_decomp_lab/experiments/lm/run.py`

### 5.3 Configuring Loss Metrics

Training losses are configured in `pd.loss_metrics` as a list of `{type: "<ClassName>", ...}` entries. The `type` literal dispatches to a `Metric` subclass via `param_decomp.metrics.dispatch.LOSS_METRIC_CLASSES`.

**To add a new loss metric:**
1. Define the class in `param_decomp/metrics/`
2. Append the config to `AnyLossMetricConfig` in `configs.py`
3. Append the class to `LOSS_METRIC_CLASSES`

Loss metrics must set `coeff` (the loss coefficient $\beta$).

### 5.4 Configuring Eval Metrics

Eval metrics are caller-supplied. Instantiate `Metric` objects in your `run.py` and pass them via `EvalLoop(metrics=...)`.

The in-repo experiments validate the YAML `eval.metrics` list via `AnyEvalMetricConfig` (a discriminated union on `EvalConfig`), then dispatch each entry through `EVAL_METRIC_CLASSES` (both in `param_decomp_lab.eval_metrics`):

```python
eval_metrics = [EVAL_METRIC_CLASSES[m.type](m) for m in cfg.eval.metrics]
```

---

## 6. Key Hyperparameters & Training Recipe

From the paper (Appendix A.6), the most important metrics for tuning are:

| Metric | What it measures | Target |
|--------|------------------|--------|
| $\mathcal{L}_{\text{adversarial-recon}}$ | Worst-case reconstruction under adversarial ablation | Low |
| $L_0$ per datapoint | Average number of active subcomponents per input | Low (sparsity) |

**Key `PDConfig` fields to tune:**
- Number of subcomponents per matrix
- Loss coefficients $\beta_1$ (adversarial), $\beta_2$ (stochastic), $\beta_3$ (importance minimality), $\beta_4$ (frequency minimality), $\beta_5$ (Delta L2)
- Causal importance function architecture
- Adversarial sampling parameters (number of steps, learning rate)

---

## 7. Development Commands

```bash
make check     # ruff format/lint + basedpyright
make type      # basedpyright only
make format    # ruff lint + format
make test      # tests not marked slow
make test-all  # all tests
```

---

## 8. Architecture Decomposed by VPD

The paper decomposes a **4-layer, 67M parameter decoder-only transformer** trained on an uncopyrighted subset of The Pile. The architecture includes:

- **Embedding layer**: $W_E$
- **4 attention layers**: each with $W_Q, W_K, W_V, W_O$ (query, key, value, output projections)
- **MLP layers**: $W_{up}$, GELU activation, $W_{down}$
- **LayerNorm** between sublayers
- **Output head**: $W_U$

VPD decomposes **all weight matrices** (attention and MLP) into rank-1 subcomponents, enabling:
- **Cross-head attention subcomponents**: computations distributed across multiple attention heads
- **Attribution graphs**: tracing information flow between parameter subcomponents
- **Manual model editing**: predictable, interpretable parameter edits (e.g., rewriting emoticon prediction behavior)

---

## 9. Summary

| Aspect | Details |
|--------|---------|
| **Method** | VPD — adversarial parameter decomposition into rank-1 subcomponents |
| **Key innovation** | Adversarial ablation masks + frequency minimality for mechanistic faithfulness |
| **Target** | Any neural network; demonstrated on 4L 67M param LM on The Pile |
| **Repo** | `goodfire-ai/param-decomp` |
| **Core package** | `param-decomp` (import `param_decomp`) |
| **Lab package** | `param-decomp-lab` (import `param_decomp_lab`) |
| **Entry point** | `optimize()` in `param_decomp.optimize` |
| **Experiments** | `pd-tms`, `pd-resid-mlp`, `pd-lm` CLI commands |
| **Custom run** | Write `run.py` → build model → configure `PDConfig`/`RuntimeConfig` → call `optimize()` |

---

*This document synthesizes the VPD research paper and the reference implementation repository to provide a practical guide for running VPD experiments.*
