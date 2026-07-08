"""
E4-final: cross-rung agreement->accuracy calibration (the "trust dial").

Pools per-candidate (PX, truth-F1) pairs across all pre-registered rungs and asks
two questions:
  1. WITHIN each rung, does the pool-crossed selector PX rank accuracy? (already
     known per-rung; recomputed here for the pooled view)
  2. ACROSS rungs, is the PX->accuracy mapping STABLE (one line fits all -> the
     trust dial transfers to a new model), or rung-specific (only ranking
     transfers, not absolute calibration)?

Predictor  = PX (pool-crossed agreement; observable with NO answer key)
Target      = f1_pair (behavior-matched truth-F1; the accuracy we want to predict)
Excludes the _scale24fair diagnostic (post-hoc pool, not pre-registered).
"""
import numpy as np, os
from scipy import stats
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNGS = [
    ("parent",      ""),
    ("R1_overlap",  "_overlap02"),
    ("R5_rollout",  "_overlap02_rollout"),
    ("R4_scale",    "_scale24"),
    ("R3_multivar", "_multivar"),
]

recs = []   # (rung, cand, PX, agree, f1_pair)
for name, tag in RUNGS:
    f = os.path.join(ROOT, "results", f"litext_e4_agreement{tag}.npy")
    if not os.path.exists(f):
        print(f"[skip] {name}: {f} missing"); continue
    d = np.load(f, allow_pickle=True).item()
    px, rows = d["px"], d["rows"]
    for cand in px:
        if cand in rows and np.isfinite(rows[cand].get("f1_pair", np.nan)):
            recs.append((name, cand, float(px[cand]),
                         float(rows[cand].get("agree", np.nan)),
                         float(rows[cand]["f1_pair"])))

byr = collections.defaultdict(list)
for r in recs: byr[r[0]].append(r)

print(f"\n=== per-rung Spearman/Pearson(PX, accuracy) ===")
perrung = {}
for name, _ in RUNGS:
    rr = byr.get(name, [])
    if len(rr) < 3: continue
    px = np.array([x[2] for x in rr]); f1 = np.array([x[4] for x in rr])
    sp = stats.spearmanr(px, f1).statistic; pe = stats.pearsonr(px, f1)[0]
    sl, ic = np.polyfit(px, f1, 1)
    perrung[name] = dict(n=len(rr), spearman=sp, pearson=pe, slope=sl, icept=ic,
                         px_range=(px.min(), px.max()))
    print(f"  {name:<13} n={len(rr):2d}  Spearman={sp:+.3f}  Pearson={pe:+.3f}"
          f"  fit: f1={sl:+.2f}*PX{ic:+.2f}  PX∈[{px.min():.3f},{px.max():.3f}]")

PX = np.array([x[2] for x in recs]); AG = np.array([x[3] for x in recs]); F1 = np.array([x[4] for x in recs])
print(f"\n=== POOLED across {len(byr)} rungs (n={len(PX)} candidates) ===")
sp = stats.spearmanr(PX, F1).statistic; pe = stats.pearsonr(PX, F1)[0]
sl, ic, r, p, se = stats.linregress(PX, F1)
resid = F1 - (sl * PX + ic)
print(f"  Spearman(PX, accuracy) = {sp:+.3f}")
print(f"  Pearson (PX, accuracy) = {pe:+.3f}")
print(f"  TRUST DIAL: accuracy ≈ {sl:.2f}·PX {ic:+.2f}   (R²={r**2:.3f}, residual std={resid.std():.3f})")
# same-W agreement contrast, pooled
mask = np.isfinite(AG)
if mask.sum() > 3:
    print(f"  [contrast] Spearman(same-Ŵ agree, accuracy) = {stats.spearmanr(AG[mask], F1[mask]).statistic:+.3f}")

# cross-rung stability: do per-rung slopes/ranges agree, or is calibration rung-specific?
slopes = [v["slope"] for v in perrung.values()]
print(f"\n=== cross-rung stability ===")
print(f"  per-rung slopes: {[round(s,2) for s in slopes]}  (spread {min(slopes):.2f}..{max(slopes):.2f})")
print(f"  per-rung PX ranges differ: " +
      "; ".join(f"{n}:[{v['px_range'][0]:.2f},{v['px_range'][1]:.2f}]" for n,v in perrung.items()))
mean_within = np.mean([v["spearman"] for v in perrung.values()])
print(f"  mean WITHIN-rung Spearman = {mean_within:+.3f}   vs POOLED Spearman = {sp:+.3f}")
print(f"  -> if within >> pooled, ranking transfers but absolute PX scale is rung-specific")

np.save(os.path.join(ROOT, "results", "litext_e4_final_calibration.npy"),
        dict(recs=recs, perrung=perrung,
             pooled=dict(spearman=sp, pearson=pe, slope=sl, icept=ic, r2=r**2, resid_std=resid.std())),
        allow_pickle=True)
print("\nsaved -> results/litext_e4_final_calibration.npy")
