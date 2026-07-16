"""
NeurIPS-format trust-dial figure: the four 2026-07-15/16 robustness worlds only
(out-of-sample generation of the selector), fit on these points.

Output: results/plots/trust_dial_paper.pdf (vector, for the manuscript) and a
.png preview. No in-figure title — the LaTeX caption carries it. Sized for
\\textwidth 5.5in single-column NeurIPS at 9pt-ish annotation sizes.
"""
import os

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,          # embed TrueType (camera-ready requirement)
})

WORLDS = [  # (tag, label, color, marker) — fixed order, CVD-safe + shape-coded
    ("_nlgauss",            "non-linear",              "#4269D0", "o"),
    ("_linskew",            "non-Gaussian",            "#EFB118", "s"),
    ("_overlapseas_deseas", "seasonal (deseasonalized)", "#3CA951", "^"),
    ("_finecadence",        "all mechanisms combined", "#FF725C", "D"),
]

pts = []  # (world_idx, PX, f1)
for wi, (tag, *_rest) in enumerate(WORLDS):
    d = np.load(os.path.join(ROOT, "results", f"litext_e4_agreement{tag}.npy"),
                allow_pickle=True).item()
    for cand in d["px"]:
        if cand in d["rows"] and np.isfinite(d["rows"][cand].get("f1_pair", np.nan)):
            pts.append((wi, float(d["px"][cand]), float(d["rows"][cand]["f1_pair"])))
pts = np.array(pts)
PX, F1 = pts[:, 1], pts[:, 2]

sl, ic, r, _, _ = stats.linregress(PX, F1)
band = (F1 - (sl * PX + ic)).std()
sp = stats.spearmanr(PX, F1).statistic
pe = stats.pearsonr(PX, F1)[0]
print(f"robustness-only fit (n={len(PX)}): F1 ~ {sl:.2f}*PX {ic:+.2f}  "
      f"Spearman {sp:+.3f}  Pearson {pe:+.3f}  R2 {r**2:.3f}  resid {band:.3f}")

fig, ax = plt.subplots(figsize=(5.5, 3.4))
xx = np.linspace(0, PX.max() * 1.05, 100)
ax.fill_between(xx, sl * xx + ic - band, sl * xx + ic + band,
                color="#9498A0", alpha=0.16, linewidth=0,
                label=rf"$\pm{band:.2f}$ F1 residual")
ic_term = rf"- {abs(ic):.2f}" if ic < 0 else rf"+ {ic:.2f}"
ax.plot(xx, sl * xx + ic, color="#222222", lw=1.3,
        label=rf"fit: $\mathrm{{F1}} \approx {sl:.2f}\,\mathrm{{PX}} {ic_term}$")
for wi, (tag, label, color, marker) in enumerate(WORLDS):
    m = pts[:, 0] == wi
    ax.scatter(PX[m], F1[m], s=26, color=color, marker=marker,
               edgecolors="white", linewidths=0.7, zorder=4, label=label)
ax.text(0.03, 0.955,
        rf"$n={len(PX)}$   Spearman $\rho=+{sp:.2f}$   Pearson $r=+{pe:.2f}$",
        transform=ax.transAxes, va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#CCCCCC", lw=0.6))
ax.set_xlabel(r"PX — pool-crossed agreement (observable, no ground truth)")
ax.set_ylabel(r"truth-F1 — graph-recovery accuracy")
ax.legend(loc="lower right", frameon=True, framealpha=0.9,
          edgecolor="#CCCCCC", handletextpad=0.4, borderpad=0.5)
ax.grid(True, linewidth=0.3, alpha=0.3)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_xlim(-0.008, PX.max() * 1.05)
ax.set_ylim(-0.03, 1.02)
fig.tight_layout(pad=0.3)
for ext in ("pdf", "png"):
    out = os.path.join(ROOT, "results", "plots", f"trust_dial_paper.{ext}")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("->", out)
