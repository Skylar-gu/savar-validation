# Next-cluster handoff — resuming the R1 (overlap) rung and beyond

*Written 2026-07-07 ~23:00 UTC, after deliberately stopping all runs on the
current box (4 vCPU / L40S). The bottleneck step is pure-CPU statistics; the
next cluster should be chosen for CPU cores, not GPU. Everything below is
resumable — all long jobs cache per-unit partial results.*

---

## 1. Why we stopped, and what to rent

The R1 measurement phase was mid-run when we stopped. The long pole — the E1
PCMCI candidate battery — is **CPU-bound** (tigramite ParCorr conditional-
independence tests, embarrassingly parallel over 24 realisations × 13
candidates) and was pacing at ~1.4 h *per candidate* on 4 cores (~8–12 h
total). The GPU (L40S) sits idle during it.

**Recommended box: 16–32 vCPUs + any modern CUDA GPU** (L40S-class is already
more than enough; the GPU stages are minutes-to-~1 h). The next cluster is
**16 vCPUs**: with 14 workers the remaining battery is ~**2.5–3.5 h** (vs
~7–11 h here). The same trade holds for every future rung (R3/R4/R5 each
rerun a battery), so this pays off repeatedly.

Two code knobs to bump on the bigger box (both currently hard-coded to 4):
- `sae/discover_modes.py` — `ProcessPoolExecutor(max_workers=4, ...)` in
  `score_candidate` (~line 361): set to 14 on a 16-vCPU box (~`cores-2`).
- `pcmci/e4_agreement.py` — same pattern in its int/dyn stages (grep
  `max_workers`).
Keep the existing convention: `OMP_NUM_THREADS=1` **only inside the worker
initializer**, never globally (global setting cripples single-process BLAS —
learned the hard way).

## 2. Getting the artifacts onto the new cluster

**UPDATE 2026-07-08: the small irreplaceable artifacts now travel IN GIT** —
`checkpoints/overlap02/{best.pt,history.npy}` (3.4 MB) and
`sae_data/overlap02/{litext_node_scalar_acts.npy,litext_channel_pc1.npy}`
(91.4 MB) are force-added on this branch (one-off exception to the gitignore;
the acts file sits under GitHub's 100 MB limit). A plain clone + checkout of
`agent/session-2026-07-03` delivers them. **Do not retrain the checkpoint** —
retraining changes the model under test (val RMSE 1.2955, corr 0.495).

The big data dirs are regenerated on the new box (CPU-only, deterministic):

```bash
# 1. realisations (~2.1 G; numpy pinned — see below)
python3 data_gen/generate_overlap.py            # -> data/realisations_overlap02

# 2. verify array contents byte-for-byte against this box's manifest
python3 data_gen/manifests/verify_content_hashes.py \
        data_gen/manifests/overlap02_realisations.sha256   # expect "100 ok"

# 3. splits (70/15/15 along time; data_split.py is now env-parameterized)
SPLIT_REAL_DIR=data/realisations_overlap02 \
SPLIT_OUT_DIR=data/splits_overlap02 python3 data_gen/data_split.py
```

Determinism note: the manifest was built with **numpy 2.0.2 / scipy 1.13.1**
(py3.9). `default_rng` bit-streams are stable across platforms for a given
numpy major version — if the verify step reports mismatches, match the numpy
version rather than trusting eyeballed stats. The manifest hashes array
CONTENTS, not npz file bytes (zip metadata isn't reproducible).

Parent-rung artifacts (`realisations_hetdynamics_eqvar` + splits +
`checkpoints/hetdynamics_eqvar` + its acts cache, ~5.5 G) are only needed for
E4-final cross-rung calibration — same recipe applies (parent generator is
`data_gen/generate_hetdynamics.py`; checkpoint would need rsync or a
force-add if that box is ever GPU-less too).

Environment: system python3 with **tigramite + scikit-learn + pydmd + scipy**
(CPU work); torch+CUDA only for training/acts/E4-dyn.

### 2b. If the new box has no working GPU (2026-07-08 reality)

The first 32-vCPU box came up with a broken driver path (custom 6.18 kernel,
no kernel-devel). That blocks ONLY: acts regeneration (cache ships in git —
moot) and the **E4 dyn stage**. Division of labor that loses nothing:

- **new box (CPU):** E1 battery (max_workers≈30; ~24 useful — one per
  realisation) → commit+push `results/litext_e1_discovery_overlap02.npy` +
  the completed `litext_e4_int_partial_overlap02.npy`; then E2 screen.
  E1's step 0 finds the shipped acts cache and never touches the GPU; its
  step 4 (fully-internal readout, 2 candidates) falls back to CPU torch
  automatically — slower but fine.
- **GPU box (the old L40S, or any CUDA box with this branch + the git-shipped
  checkpoint):** pull, then E4 with `E4_STAGE=dyn` and finally `E4_STAGE=cal`
  (int stage reads the pushed partial). The PX headline comes out of cal.

## 3. Exactly what was running and where it stopped

- **E1 battery on overlap02** — stopped 1 of 13 candidates in.
  Partial caches (tracked in git, auto-resumed on relaunch):
  - `results/litext_e1_discovery_partial_overlap02.npy` — `vmax_act` done:
    **F1 = 0.185** (parent value: 0.819 — see finding below).
  - `results/litext_e4_int_partial_overlap02.npy` — per-real edge sets for
    `vmax_act` (E1 saves these as it goes; they make E4's int stage free).
  - Candidate building + footprint table already logged in
    `out/e1_full_overlap02.log` (regenerated deterministically on relaunch;
    the acts cache makes it fast).
- **Opus agent + monitors** — stopped; nothing else was live. E4/E2 had not
  started (both consume E1's output).

**First substantive R1 finding (from the one finished candidate):** discovery
from network *internals* degrades hard under footprint overlap — `vmax_act`
footprint cosine 0.541 (5/8 matched) and graph F1 0.185 vs 0.819 on the
parent. Pixel-side candidates looked healthy at build time (vmax_pix cos
0.987). Good for the selector test: wide quality spread is what PX needs to
demonstrate ranking power.

## 4. Resume commands (in order)

```bash
# 1. E1 battery (resumes from partial cache; several hours on 4 cores, ~1h on 32)
E1_DATA_DIR=data/realisations_overlap02 E1_SAE_DIR=sae_data/overlap02 \
E1_CKPT=checkpoints/overlap02/best.pt E1_TAG=_overlap02 \
nohup python3 sae/discover_modes.py > out/e1_full_overlap02.log 2>&1 &
# finishes by writing results/litext_e1_discovery_overlap02.npy
# + completing results/litext_e4_int_partial_overlap02.npy

# 2. E4 — int stage free from the partial; dyn stage is GPU (~1–2h on L40S)
E4_DATA_DIR=data/realisations_overlap02 E4_SPLIT=data/splits_overlap02/test \
E4_CKPT=checkpoints/overlap02/best.pt \
E4_CANDS=results/litext_e1_discovery_overlap02.npy E4_TAG=_overlap02 \
nohup python3 pcmci/e4_agreement.py > out/e4_overlap02.log 2>&1 &
# -> results/litext_e4_agreement_overlap02.npy

# 3. E2 consistency screen (CPU, can overlap with E4's dyn stage)
E2_DATA_DIR=data/realisations_overlap02 E2_TAG=_overlap02 \
nohup python3 pcmci/aggregation_selection_v2.py > out/e2_overlap02.log 2>&1 &
```

Useful targeted-rerun env knobs (already implemented): `E1_ONLY` /
`E1_BUILDERS` / `E1_SKIP_ACTS` / `E1_STAGE`; `E4_ONLY` / `E4_STAGE`
(`int|dyn|cal`); `E2_NREAL`.

## 5. The headline readout (pre-registered — do not tune)

R1 is the **out-of-sample test of the pool-crossed selector**:

> PX(A) = mean over B≠A of pair-level F1( Ĝ_int(A), Ĝ_dyn(B) ), variables
> mapped to mode space by behavior matching (Hungarian on |corr(pooled
> series, true Z)|, keep iff mean |corr| ≥ 0.3; candidates with <4 matches
> score 0). Success bar: **Spearman(PX, truth-F1_behavior-matched) ≥ 0.8**
> across the candidate set.

The rule is implemented verbatim in `pcmci/e4_agreement.py`'s cal stage and
was validated byte-for-byte against the parent's saved battery (+0.950
Spearman / +0.990 Pearson). **Report the number whatever it is** — a miss is
a finding, not a bug to fix.

Also read out (all in the E4 cal output / parent write-up format):
- **Dyn-channel liveness** first: if dyn graphs are empty across candidates,
  the selector is inapplicable (R6 lesson; unsupervised-detectable). DY_SCALE
  here is 0.05, same as the parent where dyn was live — expected live.
- Same-Ŵ agreement for contrast (parent: +0.38, fails for signed reasons).
- Confound checks: coarse4 self-agreement, shift5 read-fine/write-dead
  asymmetry, disagreement classifier (int-not-dyn edges vs the oracle-map dyn
  channel — on the parent these were exactly the 3 model-unimplemented
  couplings).
- E2 screen at overlap 0.2: expect screen-not-ranker to replicate.

## 6. Write-up checklist (after the numbers land)

1. Complete the "R1 — overlap rung" section of
   `notes/literature_extension_results.md` (battery table, liveness, PX vs
   bar as plain pass/fail, confounds, E2 row). Numbers-first style.
2. Add the R1 row to the transfer table in
   `notes/literature_extension_experiments.md` (§7 sequencing note).
3. `notes/summary_jul6.md` "July 7 sessions" point 4 says R1 is mid-run —
   append the outcome in the file's plain-language tone.
4. Update memory: `~/.claude/projects/-home-ec2-user-savar-project/memory/`
   `project_savar_litext.md` (currently ends at "NEXT: R1 overlap rung").
5. Commit per block, **never push**; commit trailer:
   `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

Mandatory conventions (hard-won, see plan §7): behavior-based matching for
ALL scoring (never footprint cosine); `ground_truth_graph[effect, cause,
lag]`; `fine_edges` rows are (cause, effect, lag, coeff); realisations obs
are (2500, T) but splits obs are (T, 50, 50).

## 7. Experiment ledger — done vs left

### Done

| experiment | what it settled | when |
|---|---|---|
| E1 (parent) | mode discovery works W-free; price ≈0.03 F1 (varimax) | Jul 6 |
| E2 v1+v2 (parent) | consistency = trustworthy screen, NOT a ranker (failed 0.8 bar usefully) | Jul 6–7 |
| E3 (parent) | poke-channel at ceiling: 9/9 implemented edges, zero-FP deconv core; 3 misses = model's own gaps | Jul 6 |
| E4-v1 (parent) | pool-crossed agreement selector: Spearman +0.95 (in-sample) | Jul 7 |
| R2 — geography rung | address-code substitutes for physics; skill metrics blind to it | Jul 7 |
| R6 — atmosphere rung | built in 3 tries (exposed parent generator saturation bug); SAE window-composition confirmed; shared-operator survives; dyn channel dead → liveness precondition | Jul 7 |
| R1 — generator + gates | ceiling F1 0.867, dynamics identical to parent, overlap real (cos 0.20) | Jul 7 |
| R1 — forecaster training | val RMSE 1.2955 / corr 0.495, converged | Jul 7 |
| R1 — acts extraction + 13 candidates | internals footprints degraded vs parent (expected stress) | Jul 7 |
| R1 — battery candidate 1/13 | vmax_act F1 0.185 (vs 0.819 parent) — internals discovery hurt by overlap | Jul 7 |

### Left (R1, resume in §4 order)

| step | estimate (4 cores / 16 cores, 14 workers) |
|---|---|
| E1 battery, remaining 12/13 candidates | ~7–11 h / **~2.5–3.5 h** |
| E4 int channel | ~free (edge sets saved by E1) |
| E4 dyn channel (GPU) + liveness | ~1–2 h (GPU-bound, same either way) |
| **PX Spearman — headline out-of-sample number** | minutes once E4 lands |
| E2 consistency screen | ~1–2 h / ~30 min (overlaps with E4 dyn) |
| write-up + commits + memory | ~15 min |

### Left (after R1)

| step | estimate |
|---|---|
| E4-final: pool agreement→accuracy calibration across rungs | ~1 h, mostly analysis |
| R4 — many modes, unknown count (Leiden route) | ~1 day (gen + train + battery) |
| R3 — several coupled variables | ~1 day |
| R5 — rollout-trained forecaster | ~1 day (longer training) |
| E7 — external baselines (DAG-VAE, SINDy-SHRED) | optional, ~half day |
| GraphCast rung | multi-day, the finale |

The single load-bearing pending result is the **PX number**: if it clears 0.8
out-of-sample, the recipe has an answer-key-free selector and trust dial, and
everything after is scaling rungs toward GraphCast.
