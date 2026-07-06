# Moving-mechanism sub-resolution recovery — spec v2 (plain-language rewrite)

**Replaces [[moving_mechanism_subres_spec_v1]] (v1)**, which replaced
[[subresolution_dealiasing_plan]]. Same core idea as v1 — make the weather
patterns *move around* so the network is forced to learn *what* each pattern is,
not just *where* it sits — but a literature sweep (2026-07-03) changed three things.
Sources: [[literature_review]] §E.

---

## The problem, in plain terms

We built a fake climate world where we know the true cause-and-effect. We train a
network to forecast it, then try to reverse-engineer those causes from inside the
network — practice for eventually doing this to a real weather model (GraphCast).

The wall we keep hitting: the network works, but it **smears all 8 patterns
together**. Our tools (SAEs read the network's "channels"; VPD carves its
"mechanisms") go looking for a clean "here's pattern 3" signal and find nothing.

Why it smears: in our simulation, **each pattern is nailed to a fixed spot on the
map**. Pattern 3 is always in the same place. So the network takes the shortcut —
it never learns "this is pattern 3," it just learns "this is the stuff at location
X." Identity and location become the same thing, so there's no reason to build a
separate "which pattern" code. Our tools hunt for something the setup made pointless
to build. (This is the whole session's finding: [[project_savar_gridlock]] — mode
identity lives in *locations*, not channels or mechanisms.)

## The idea, and the one big correction

**The idea:** make the patterns *move* to different spots. Then location no longer
tells you which pattern, so the network is forced to actually learn what each
pattern is — and our tools should finally have something to find.

**The correction from the literature:** moving the patterns is **not enough on its
own.** People who studied this exact "moving objects" problem in vision/video found
that a plain network still cheats — it just follows the blob frame-to-frame without
ever forming a real "what is this thing" concept. To actually force the separation
you have to **change the network's design** to give it the right machinery
(object-centric / slot architectures; arXiv:2310.19054, 2302.04973). Our pooled GNN
is exactly the plain "smush everything into one vector" design those papers say
fails.

So v1's plan ("move the patterns → tools work") becomes a **two-step** plan:

1. Move the patterns, keep the current network → **we expect it to still fail.**
   That's now a *predicted* result, not a mystery.
2. Then upgrade the network's design, cheapest change first, until it works — or
   until nothing works, which is itself a strong, useful finding.

Think of it as a 2×2:

| | current network (smush-to-one-vector) | network with the right building blocks |
|---|---|---|
| pattern fixed in place (done already) | smears — the session baseline | (skip) |
| **pattern moves around** | **step 1: expected to still smear** | **step 2: the real test** |

## The key test that runs through everything

**"Left side vs right side."** Train the network on realisations where pattern k
lives on the left of the map; then test whether it can still read pattern k when it
shows up on the right.

- If it *can* → it learned *what* the pattern is (the abstraction formed). ✅
- If it *can't* → it only learned *where* (the shortcut). ❌

Run this same test after each network change. The test tells you exactly which
change fixed it. (Formally this is P1 below.)

## Two ways to make patterns move (`generate_movmech.py`)

Fork `data_gen/generate_hetdynamics.py`. Keep everything that keeps the ground truth
exact (each pattern's timescale φ, the cause-effect graph Φ, the noise, the
nonlinearity). Only change **where** each pattern sits.

- **`MOVE=place`** — each realisation, drop each pattern's blob at a fresh random
  spot (kept from overlapping). This is the clean "left vs right" test. **Do this
  one first** — it isolates the "what vs where" question.
- **`MOVE=advect`** — blobs *drift* across the map during a single sequence, each at
  its own speed. Harder (now the network also has to track motion), but it creates a
  useful physics bonus: fast-moving and slow-moving structure separate in a way that
  helps the "read between the pixels" part below. **Do this second.**

Also (for the sub-pixel angle): inside each blob, hide 2 finer sub-sources with
different timescales, then observe the world through a coarse grid two ways —
`D_sub` (skip pixels → the "scrambled but recoverable" case) or `D_avg` (blur pixels
together → the "genuinely destroyed" case, our control that should fail). Save the
true fine-detail signal so we can score recovery honestly.

## The network upgrades to try (cheapest first)

Only escalate to the next one if the "left vs right" test still fails.

- **A0 — fix the network's own blurring (cheap, almost free).** Our network
  shrinks the map in a way that ignores a basic signal-processing rule (the sampling
  theorem), so a *moving* blob gets scrambled *inside the network itself* before it
  ever gets a chance to be understood (Zhang, arXiv:1904.11486 — "make CNNs
  shift-invariant again"). The fix is a standard smoothing step before shrinking.
  This might be the whole problem, and it's nearly free to try.
- **A1 — let the network describe things relative to the blob's own center**
  instead of absolute map position (a light version of Invariant Slot Attention,
  arXiv:2302.04973). This is the smallest change that gives a truly
  location-independent "what" code.
- **A2 — give the network proper "object slots"** (a small slot-attention head that
  competes to assign patterns to slots; arXiv:2310.19054). Heaviest; only if A0/A1
  don't do it. This doubles as the "build the causality in from the start" baseline
  the review wanted to compare against our "discover it afterward" approach.

**New control test — P3 ("is it aliasing or is it binding?"):** shift an input blob
by a few pixels and check whether the network's internal representation shifts with
it cleanly. If A0 (the smoothing fix) passes the left-vs-right test but the plain
network fails, then the problem was just the internal blurring (a cheap
signal-processing bug), not a deep "can't form concepts" problem. Different problem,
cheaper fix — worth knowing which one we have.

## The "read between the pixels" part, and its honesty check

Part of this tries to recover detail *finer than the grid the network sees*.
Literature warning (arXiv:2510.06646, "the false promise of zero-shot
super-resolution"): neural nets are notoriously bad at this and tend to
**hallucinate** plausible-looking detail instead of recovering the real thing.

So we do two things:

- **Run a cheap "is this even possible?" check first (T0).** Before building
  anything, a clean math oracle checks whether the fine detail is recoverable *in
  principle* given how different the two sub-sources' timescales are. There's a known
  theoretical limit for this (off-grid spectral estimation / separation condition),
  so if it fails we know it's a real limit, not a weak method. If T0 fails, drop the
  sub-pixel angle (but the moving-pattern tests above still run — they don't need it).
- **Use the standard hallucination check (T3).** For anything we "recover," compare
  it to the *true* signal AND look at where the leftover error concentrates. Real
  recovery = error spread evenly. Hallucination = the recovered signal *looks*
  statistically plausible but the error piles up exactly at the fine detail the
  network never saw during training. That pile-up is the tell. This directly settles
  the "is it really reading between the lines, or just making stuff up?" question.

## The rest of the test battery (carried from v1, lightly upgraded)

- **T1 — the two-operator comparison.** Recovery should work under `D_sub`
  (scrambled-but-there) and fail under `D_avg` (blurred-away). If they look the same,
  we're not really de-aliasing.
- **T2 — the ladder.** Try to recover the fine signal at each level: coarse single
  frame → coarse short window → a simple linear/wave model → the trained network →
  the SAE. See how far up the ladder the signal survives. **New rung:** for the
  drifting (`advect`) case, add a "traveling-wave" model — moving structure is
  naturally wave-like (Wave-RNN, ICLR 2024), and a wave model can separate the
  colliding scales where a static one can't.
- **T4 — the identical-twins null.** Two sub-sources with *identical* dynamics must
  be impossible to tell apart at every level. If we "recover" them, we're leaking
  position info, not reading dynamics.
- **T5 — the dial.** Slowly increase how different the two sub-sources' timescales
  are, from identical to very different, and watch recovery climb from zero to good.
  This draws the "how different is different enough?" curve.
- **T6 — the payoff: re-run the SAE and VPD on the moving-pattern network.**
  - **SAE, now over space *and* time** (arXiv:2604.03919) — a moving pattern can't
    be captured by a single-frame feature, so the SAE has to look at little
    space-time patches. Score whether features now cleanly pick out one pattern
    (uniqueness above ~0 = success). **Run ≥3 random seeds** and report the stable
    part, since SAE features are seed-jittery (arXiv:2606.12138).
  - **VPD** with the anti-redundancy fix from last session
    (`vpd/param_decomp_gate_decorrelation.patch`). Success = it finally finds
    components that prefer specific patterns.

## What each outcome would mean

| what happens | what it means |
|---|---|
| plain network fails "left vs right", A1/A2 passes | the fix was **network design**, not data — the clean, expected win |
| plain network fails, A0 (smoothing) passes | it was just internal blurring — a cheap signal-processing fix, not a deep problem |
| every network fails "left vs right" | the wall is deeper than we think — a strong, honest null that says post-hoc interp may *need* object-centric models, not just probes |
| any network: SAE features get unique, VPD finds per-pattern parts | **the tools finally bite** |
| `D_sub` recovers, `D_avg` fails, error spread evenly | real reading-between-the-pixels |
| recovered signal looks plausible but error piles at fine detail | hallucination, caught |
| wave model de-aliases the drifting case but the static one doesn't | the drift/dispersion is the usable signal |

## Files (new; no collision with existing work)
- `data_gen/generate_movmech.py`, `data_gen/split_movmech.py`
- `train/mesh_gnn_variants.py` — A0 smoothing, A1 relative-position pooling, A2 slots
  (flags on the existing MeshGNN; plain network stays the default)
- `pcmci/subres_identifiability.py` (T0/T1 oracle + the theoretical limit)
- `sae/movmech_position_invariance.py` (the left-vs-right test P1, plus P2/P3)
- `sae/subres_ladder.py` (T2/T6 ladder + SAE scoring)
- `pcmci/wave_surrogate.py` (the traveling-wave rung in T2)
- reuse `sae/extract_activations*.py` (**extract at stride 1** — last session's bug),
  `sae/eval_sae_metrics.py` (add a `--ground-truth` path), a space-time SAE flag on
  `sae/train_sae_mixed.py`, `pcmci/dmd_timescales.py`, param-decomp VPD (+ the patch)
- results → `results/movmech_*.npy`, `results/subres_*.npy`
- checkpoints → `checkpoints_movmech_place{,_blurpool,_refframe,_slot}/`,
  `checkpoints_movmech_advect*/`

## Order to run things
1. **T0** cheap "is it even possible" check (no training). In parallel, build the
   generator and **train the plain network on `place`**.
2. **Left-vs-right test (P1) + P2/P3 on the plain network** — expect it to fail;
   P3 tells us whether that's just internal blurring or a real concept-forming gap.
3. **Try the network upgrades on `place`, cheapest first: A0 → A1 → A2**, re-running
   the left-vs-right test each time until it passes (or all fail = the strong null).
   This is the main result.
4. **T1–T5** de-aliasing battery on whichever network passed (and on `advect` for
   the wave rung).
5. **T6** the SAE + VPD payoff on the winning network.

Write each block into `notes/session_results.md` (numbers first), commit per block,
don't push. A clean "data wasn't enough, network design was the fix" is one of the
most useful things we could learn — it's a direct warning about what to expect when
we point these tools at the real weather model.
