"""
Trust dial v2: original 5-rung calibration + robustness worlds as an
OUT-OF-SAMPLE test, then a refit including them.

The original dial (accuracy ~ 1.48*PX + 0.13, resid +/-0.13) was fit 2026-07-08
on the five pre-registered rungs. The 2026-07-15/16 robustness worlds were never
seen by that fit, so they test whether the line TRANSFERS: what fraction of new
per-candidate (PX, truth-F1) points fall inside the original +/-1 residual band?

overlapseas_raw is excluded from calibration on the pre-registered precondition
that seasonal inputs must be deseasonalized (its rung result: PX uninformative
on raw series, Spearman +0.13); it is still reported in the printout.

Renders results/plots/trust_dial.png (supersedes the 2026-07-08 figure) and
saves results/litext_e4_final_calibration_v2.npy.
"""
import collections
import os

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ORIG_RUNGS = [  # the 2026-07-08 pre-registered fit
    ("parent",      ""),
    ("R1_overlap",  "_overlap02"),
    ("R5_rollout",  "_overlap02_rollout"),
    ("R4_scale",    "_scale24"),
    ("R3_multivar", "_multivar"),
]
NEW_RUNGS = [   # 2026-07-15/16 robustness worlds (out-of-sample for the old fit)
    ("NL_nlgauss",     "_nlgauss"),
    ("NG_linskew",     "_linskew"),
    ("seas_deseas",    "_overlapseas_deseas"),
    ("combined_fine",  "_finecadence"),
]
EXCLUDED = [("seas_raw", "_overlapseas_raw")]  # precondition violated; report only


def load_points(rungs):
    recs = []  # (rung, cand, PX, f1_pair)
    for name, tag in rungs:
        f = os.path.join(ROOT, "results", f"litext_e4_agreement{tag}.npy")
        if not os.path.exists(f):
            print(f"[skip] {name}: {f} missing")
            continue
        d = np.load(f, allow_pickle=True).item()
        px, rows = d["px"], d["rows"]
        for cand in px:
            if cand in rows and np.isfinite(rows[cand].get("f1_pair", np.nan)):
                recs.append((name, cand, float(px[cand]), float(rows[cand]["f1_pair"])))
    return recs


old = load_points(ORIG_RUNGS)
new = load_points(NEW_RUNGS)
raw = load_points(EXCLUDED)

PXo = np.array([r[2] for r in old]); F1o = np.array([r[3] for r in old])
PXn = np.array([r[2] for r in new]); F1n = np.array([r[3] for r in new])

# --- original dial, recomputed exactly as e4_final_calibration.py ------------
sl, ic, r, p, se = stats.linregress(PXo, F1o)
resid_o = F1o - (sl * PXo + ic)
band = resid_o.std()
print(f"original dial (n={len(old)}): accuracy ~ {sl:.2f}*PX {ic:+.2f}  "
      f"R2={r**2:.3f}  resid std={band:.3f}")

# --- out-of-sample test -------------------------------------------------------
pred = sl * PXn + ic
res_n = F1n - pred
inside = np.abs(res_n) <= band
sp_n = stats.spearmanr(PXn, F1n).statistic
pe_n = stats.pearsonr(PXn, F1n)[0]
print(f"\nOUT-OF-SAMPLE robustness worlds (n={len(new)}):")
print(f"  within +/-{band:.2f} band of OLD line: {inside.sum()}/{len(new)} "
      f"({100*inside.mean():.0f}%)")
print(f"  new-point residual vs old line: mean {res_n.mean():+.3f}, std {res_n.std():.3f}")
print(f"  Spearman(PX, F1) on new points  = {sp_n:+.3f}   Pearson = {pe_n:+.3f}")
byr = collections.defaultdict(list)
for rrec, rr in zip(new, res_n): byr[rrec[0]].append(rr)
for name, rs in byr.items():
    print(f"    {name:<14} n={len(rs):2d}  mean resid {np.mean(rs):+.3f}")
if raw:
    PXr = np.array([r[2] for r in raw]); F1r = np.array([r[3] for r in raw])
    rr = F1r - (sl * PXr + ic)
    print(f"  [excluded] seas_raw n={len(raw)}: {np.mean(np.abs(rr) <= band)*100:.0f}% "
          f"in band, Spearman {stats.spearmanr(PXr, F1r).statistic:+.3f} "
          f"(precondition: deseasonalize first)")

# --- refit v2 = original + robustness worlds ----------------------------------
PX2 = np.concatenate([PXo, PXn]); F12 = np.concatenate([F1o, F1n])
sl2, ic2, r2, p2, se2 = stats.linregress(PX2, F12)
resid2 = F12 - (sl2 * PX2 + ic2)
band2 = resid2.std()
sp2 = stats.spearmanr(PX2, F12).statistic
print(f"\nTRUST DIAL v2 (n={len(PX2)}, 9 worlds): accuracy ~ {sl2:.2f}*PX {ic2:+.2f}  "
      f"Spearman {sp2:+.3f}  R2={r2**2:.3f}  resid std={band2:.3f}")

np.save(os.path.join(ROOT, "results", "litext_e4_final_calibration_v2.npy"),
        dict(old_recs=old, new_recs=new, raw_recs=raw,
             old_fit=dict(slope=sl, icept=ic, r2=r**2, resid_std=band),
             oos=dict(n=len(new), frac_in_band=float(inside.mean()),
                      resid_mean=float(res_n.mean()), resid_std=float(res_n.std()),
                      spearman=sp_n, pearson=pe_n),
             v2_fit=dict(slope=sl2, icept=ic2, r2=r2**2, resid_std=band2,
                         spearman=sp2)),
        allow_pickle=True)
print("saved -> results/litext_e4_final_calibration_v2.npy")

# --- figure -------------------------------------------------------------------
NEW_COLORS = {          # fixed categorical order
    "NL_nlgauss":    ("#4269D0", "non-linear world"),
    "NG_linskew":    ("#EFB118", "non-Gaussian world"),
    "seas_deseas":   ("#3CA951", "seasonal (deseasonalized)"),
    "combined_fine": ("#FF725C", "all-combined world"),
}
fig, ax = plt.subplots(figsize=(10, 7.5))
xx = np.linspace(0, max(PX2.max(), 0.55), 100)
ax.fill_between(xx, sl2 * xx + ic2 - band2, sl2 * xx + ic2 + band2,
                color="#9498A0", alpha=0.18,
                label=f"±{band2:.2f} F1 residual (v2)")
ax.plot(xx, sl * xx + ic, color="#9498A0", lw=1.6, ls="--",
        label=f"original dial (5 worlds):  {sl:.2f}·PX {ic:+.2f}")
ax.plot(xx, sl2 * xx + ic2, color="#222222", lw=2,
        label=f"trust dial v2 (9 worlds):  {sl2:.2f}·PX {ic2:+.2f}")
ax.scatter(PXo, F1o, s=38, color="#9498A0", alpha=0.55, edgecolors="white",
           linewidths=1.2, label="original rungs (in-fit, Jul 8)", zorder=3)
for name, (col, lab) in NEW_COLORS.items():
    m = [i for i, rrec in enumerate(new) if rrec[0] == name]
    ax.scatter(PXn[m], F1n[m], s=52, color=col, edgecolors="white",
               linewidths=1.4, label=lab + " (out-of-sample)", zorder=4)
ax.set_xlabel("PX  —  pool-crossed agreement  (observable, NO answer key)")
ax.set_ylabel("truth-F1  —  actual graph-recovery accuracy")
ax.set_title("The trust dial: observed agreement → predicted accuracy",
             fontsize=14, fontweight="bold")
ax.text(0.02, 0.98,
        f"out-of-sample: {inside.sum()}/{len(new)} robustness points "
        f"inside the original ±{band:.2f} band\n"
        f"v2 pooled  n={len(PX2)}  Spearman {sp2:+.2f}  R² {r2**2:.2f}",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="#F5F5F5", ec="#CCCCCC"))
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, linewidth=0.4, alpha=0.35)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_xlim(-0.01, max(PX2.max(), 0.55) + 0.01); ax.set_ylim(-0.02, 1.0)
fig.tight_layout()
out = os.path.join(ROOT, "results", "plots", "trust_dial.png")
fig.savefig(out, dpi=110, bbox_inches="tight")
print("figure ->", out)
