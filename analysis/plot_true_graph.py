"""Ground-truth SAVAR causal graph (parent) — publication-style figure.
Red = positive coeff, blue = negative; darker+thicker = larger |coeff|;
solid = lag-1, dashed = lag-2; node ring = autocorrelation strength."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.cm as cm

# ---- graph data -----------------------------------------------------------
pos = {
    "X0": (0.0, 0.0), "X1": (1.3, 1.55), "X2": (1.3, -1.55), "X3": (2.85, 0.0),
    "X4": (2.95, 2.75), "X5": (4.25, 1.75), "X6": (4.85, -0.85), "X7": (6.25, 0.35),
}
auto = {"X0":0.45,"X1":0.50,"X2":0.35,"X3":0.55,"X4":0.40,"X5":0.30,"X6":0.50,"X7":0.45}
roles = {"X0":"hub", "X3":"collider", "X7":"sink"}
# (src, dst, coeff, lag, rad)  rad = arc curvature (routing)
edges = [
    ("X2","X0", 0.22, 2,  0.15),
    ("X0","X1", 0.35, 1,  0.12),
    ("X1","X2", 0.40, 1,  0.18),
    ("X0","X3", 0.30, 1,  0.00),
    ("X2","X3",-0.30, 1, -0.15),
    ("X1","X4", 0.25, 2,  0.10),
    ("X4","X5", 0.35, 1,  0.12),
    ("X0","X5",-0.20, 2,  0.34),
    ("X3","X6", 0.30, 1, -0.12),
    ("X5","X6", 0.25, 2,  0.12),
    ("X6","X7", 0.20, 1,  0.12),
    ("X3","X7",-0.15, 2,  0.30),
]

MAGMIN, MAGMAX = 0.12, 0.56          # |coeff| range for shading/width
def frac(c): return np.clip((abs(c)-MAGMIN)/(MAGMAX-MAGMIN), 0, 1)
def color(c):
    cmap = cm.Reds if c >= 0 else cm.Blues
    return cmap(0.32 + 0.68*frac(c))
def width(c): return 1.6 + 6.0*frac(c)

NR = 0.30                             # node radius
plt.rcParams.update({"font.size": 11})
fig, ax = plt.subplots(figsize=(11, 7.2), dpi=150)

# ---- nodes: fill + autocorrelation ring ----------------------------------
node_patch = {}
for n,(x,y) in pos.items():
    ring = color(auto[n])            # all positive -> red, darkness = memory
    c = Circle((x,y), NR, facecolor="#f7f7f7", edgecolor=ring,
               linewidth=2.0+5.0*frac(auto[n]), zorder=5)
    ax.add_patch(c); node_patch[n] = c
    ax.text(x, y, n, ha="center", va="center", fontsize=13, fontweight="bold",
            zorder=6, color="#222222")
    ax_dx, ax_dy = (0.42, -NR+0.02) if n == "X5" else (0.0, -NR-0.17)
    ax.text(x+ax_dx, y+ax_dy, f"{auto[n]:.2f}", ha="center", va="top", fontsize=7.5,
            color="#b03030", zorder=6)                     # auto coeff (memory)
    if n in roles:
        ax.text(x, y+NR+0.13, roles[n], ha="center", va="bottom", fontsize=8.5,
                style="italic", color="#555555", zorder=6)

# manual label positions for crowded edges (avoids collisions)
LBL = {("X1","X2"): (0.80, 0.60), ("X0","X3"): (1.5, 0.22)}

# ---- directed edges -------------------------------------------------------
for s,d,c,lag,rad in edges:
    a = FancyArrowPatch(pos[s], pos[d], connectionstyle=f"arc3,rad={rad}",
                        arrowstyle="-|>", mutation_scale=17,
                        linewidth=width(c), color=color(c),
                        linestyle="-" if lag==1 else (0,(5,3)),
                        patchA=node_patch[s], patchB=node_patch[d],
                        shrinkA=1, shrinkB=1, zorder=3, joinstyle="round")
    ax.add_patch(a)
    # coefficient label at arc midpoint, nudged perpendicular
    (x0,y0),(x1,y1) = pos[s], pos[d]
    mx,my = (x0+x1)/2, (y0+y1)/2
    dx,dy = x1-x0, y1-y0; nrm = np.hypot(dx,dy)
    px,py = -dy/nrm, dx/nrm
    off = 0.16 + 0.9*rad
    lx, ly = LBL.get((s,d), (mx+px*off, my+py*off))
    ax.text(lx, ly, f"{c:+.2f}", ha="center", va="center",
            fontsize=8.2, color=color(c), fontweight="bold", zorder=7,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))

# ---- legend ---------------------------------------------------------------
leg = [
    Line2D([0],[0], color=cm.Reds(0.85), lw=4, label="positive (excitatory)"),
    Line2D([0],[0], color=cm.Blues(0.85), lw=4, label="negative (inhibitory)"),
    Line2D([0],[0], color="#666666", lw=2.4, ls="-",       label="lag-1 edge"),
    Line2D([0],[0], color="#666666", lw=2.4, ls=(0,(5,3)), label="lag-2 edge"),
    Line2D([0],[0], marker="o", color="none", markerfacecolor="#f7f7f7",
           markeredgecolor=cm.Reds(0.85), markeredgewidth=3, markersize=15,
           label="node ring = autocorrelation"),
]
ax.legend(handles=leg, loc="upper left", fontsize=9.5, framealpha=0.95,
          borderpad=0.8, labelspacing=0.7, bbox_to_anchor=(0.005, 0.995))

# ---- magnitude gradient key (darker = larger |coeff|) --------------------
cax = fig.add_axes([0.66, 0.055, 0.26, 0.026])
grad = np.linspace(MAGMIN, MAGMAX, 256)
strip = np.vstack([cm.Reds(0.32+0.68*frac(g)) for g in grad])[None,:,:]
cax.imshow(strip, aspect="auto", extent=[MAGMIN, MAGMAX, 0, 1])
cax.set_yticks([]); cax.set_xticks([0.15,0.25,0.35,0.45,0.55])
cax.tick_params(labelsize=7.5, length=2, pad=1)
cax.set_title("darker + thicker  =  larger |coefficient|", fontsize=8.3, pad=3)

ax.set_title("SAVAR ground-truth causal graph (parent, N=8)",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlim(-0.7, 7.0); ax.set_ylim(-2.4, 3.4)
ax.set_aspect("equal"); ax.axis("off")
plt.tight_layout()
os.makedirs("results/plots", exist_ok=True)
out = "results/plots/true_graph.png"
plt.savefig(out, bbox_inches="tight", facecolor="white")
print("saved", out)
