# litext session results — 2026-07-06

Executes [[literature_extension_experiments]] steps 1–4: E1 (mode-discovery bake-off), E2
(Adag-selection test), E3 (dynamical-arm upgrade). Numbers first. Anchors
throughout: PCMCI+ on true Z F1 = 0.853 (graph, exact-lag protocol) / 0.823
(pair-level Block-B protocol); oracle-W pooled activations F1 = 0.855 (FU1).

---

## E3 — internals-free dynamical arm upgraded (Block B F1 0.500 → …)

`pcmci/impulse_response_v2.py`. Three arms + statistic variants, all under
Block B's detection protocol (W-projected response, pixel-permutation null
α=0.01, 1000 perms, pair-level scoring vs the 12 gt cross edges). New:
**ancestor-level scoring** vs the transitive closure (29 ordered pairs) —
dynamical probing measures *total* effects, so a detected 2-hop path is not a
model error.

**Run 1** (240 windows, 12 steps, σ=1: `results/litext_e3_dynarm.npy`):

| arm | direct P/R/F1 | ancestor P/R/F1 | lag-ok | τ-Spearman |
|---|---|---|---|---|
| A baseline (Block B replica) | 0.625/0.417/**0.500** | 1.00/0.28/0.43 | 5/5 | 0.738 |
| B teacher-forced | 0.600/0.500/**0.545** | 1.00/0.34/0.51 | 6/6 | **0.952** |
| C sustained forcing | 0.625/0.417/0.500 | 1.00/0.28/0.43 | 0/5 | n/a |
| C sustained ×3σ | 0.556/0.417/0.476 | 1.00/0.31/0.47 | 0/5 | n/a |

- Arm A reproduces Block B byte-for-byte (TP=5 FP=3 FN=7, same sets) —
  protocol validated.
- **Arm B fixes the timescale channel**: e-folding times become monotone in φ
  (Spearman 0.738 → 0.952 at 12 steps; no saturation), causally confirming
  FU1's diagnosis — Block B's flat τ̂ was rollout damping, not representation.
- Sustained forcing (C) is a null for recall and destroys lag info: not the
  lever.
- Dose linearity (A): response-shape corr(1×,3×) = 0.999, amplitude ratio
  3.01 — the model's response is linear in the impulse over ±3σ.

**Run 2** (240 windows, **24 steps**, + integral statistic Σ_τ|R| —
slow-mode responses are broad and low, exactly what a max-statistic misses;
`results/litext_e3_dynarm_s24.npy`):

| arm/stat | direct P/R/F1 | ancestor P/R/F1 |
|---|---|---|
| A max | 0.625/0.417/0.500 | 1.00/0.28/0.43 |
| A integral | 0.389/0.583/0.467 | 0.83/0.52/0.64 |
| B max | 0.600/0.500/0.545 | 1.00/0.34/0.51 |
| **B integral** | 0.529/**0.750**/**0.621** | **0.941/0.552/0.696** |
| C ×3σ | 0.700/0.583/0.636 | 1.00/0.34/0.51 |

- **B+integral: recall 0.417 → 0.750** (9/12 direct edges, 8/9
  lag-consistent). Missed: (2→0), (2→3), (5→6) — the X2-sourced edges and the
  slow–slow lag-6 edge.
- Every direct-level FP but one is a true 2–3-hop path (ancestor precision
  0.941; the single non-path detection is (3,4)). **The arm essentially never
  hallucinates influence** — the remaining direct-FP problem is
  direct-vs-indirect separation, not detection error.
- At 24 steps the free-rollout arms degrade (A τ-Spearman flips negative —
  damping artifact, consistent with Block B's 24-step sensitivity run), while
  teacher-forced stays clean (0.810). Teacher-forcing is what makes longer
  horizons usable.

**Run 3** (arm B only, 720 windows, 24 steps, + **deconvolution scoring**:
the measured R[i,j,τ] is the model's total-effect Green's function; the
Volterra recursion B[τ] = T[τ] − Σ_{s<τ} B[s]·T[τ−s] extracts direct kernels,
with the permutation null pushed through the same recursion):
*(running — results below when complete)*

---

## E1 — unsupervised mode-discovery bake-off

`sae/discover_modes.py`. Discovery fit on 4 held-out realisations (96–99);
PCMCI eval on realisations 0–23 (Block-G protocol, exact-lag scoring,
Hungarian-strict variable mapping: edges touching unmatched discovered
variables = FP, gt edges at unmatched true modes = FN).

**Footprint recovery** (Hungarian-matched cosine to true W rows; N̂ selected
by coherence floor 0.25 from C0=12 initial components):

| candidate | N̂ | matched | mean cos | note |
|---|---|---|---|---|
| vmax_act | 12 | 8/8 | 0.701 | varimax-PCA on per-node activation scalar field |
| **vmax_pix** | **8** | 8/8 | **0.999** | varimax-PCA on raw pixels — near-exact, N̂ exactly 8 (coherence eigengap 0.99→0.20) |
| km_act | 10 | 4/8 | 0.610 | k-means on activation time courses |
| km_pix | 9 | 8/8 | 0.776 | |
| dmd_act | 8 | 4/8 | 0.405 | k-means on \|DMD mode\| loadings |
| merge01/coarse4/split7/fine16/shift5/diag8/blur | — | — | 0.96/0.71/0.96/0.71/0.00/0.00/0.50 | corrupted variants for E2's quality axis |

*(PCMCI battery running — graph F1 table below when complete)*

---

## E2 — Adag consistency scores as unsupervised selector

*(pending E1 battery)*
