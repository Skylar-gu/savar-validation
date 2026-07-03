"""
Faithful regeneration of figures/param_decomposition.png from the raw VPD gate
diagnostics (M4 run p-0625f7dd, layer layers.0.mlp.0).

Why this rewrite
----------------
The original figure split components into crisp "position-locked" vs "physics-locked
/ tracks pattern j" classes with per-panel colour scales and a hand-placed green
circle. Checking the raw gates showed that is not faithful:
  * every component fires on ALL 8 mode blobs almost equally (blob-gate CV ~0.05),
    so "tracks pattern j" singles out one of eight statistically-tied blobs;
  * per-panel normalisation made low-gate backgrounds look bright;
  * components fire on hubs AND blobs simultaneously (no clean dichotomy).

This version therefore:
  * uses ONE colormap + ONE fixed scale (gate g in [0,1]) for every map panel, with
    a single shared colorbar on the right (the map legend);
  * overlays hub nodes and all 8 mode centres identically on every panel;
  * summarises the 64 components by hub-contrast vs blob-contrast (gate on that
    structure minus background gate), coloured by ablation importance — the honest
    readout: structure on both axes, ~zero importance (epiphenomenal).
"""

import sys, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent.parent
DIAG = ROOT / "vpd_out/runs/p-0625f7dd/diagnostics"
LAYER = "layers_0_mlp_0"
OUT = ROOT / "figures/param_decomposition_faithful.png"
NY = NX = 50
HUB_STRIDE = 5
GATE_CMAP = "magma"     # single shared colormap for all gate maps

# ── load raw diagnostics ──────────────────────────────────────────────────────
mg  = np.load(DIAG / f"mean_g_map__{LAYER}.npy")        # (64, 50, 50) mean CI gate
abl = np.load(DIAG / f"ablation_drop__{LAYER}.npy")     # (64,) recon drop when masked
C = mg.shape[0]

# ── region masks (identical for every component) ──────────────────────────────
W = np.load(sorted(glob.glob(str(ROOT / "data/realisations_finecadence/realisation_*.npz")))[0])["W"]
centers = [tuple(divmod(int(W[j].argmax()), NX)) for j in range(8)]   # 8 mode centres

hub_mask = np.zeros((NY, NX), bool); hub_mask[::HUB_STRIDE, ::HUB_STRIDE] = True
blob_mask = np.zeros((NY, NX), bool)
for (r, c) in centers:
    blob_mask[max(0, r - 2):r + 3, max(0, c - 2):c + 3] = True
bg_mask = ~hub_mask & ~blob_mask

hub_r, hub_c = np.where(hub_mask)

# ── per-component summary metrics (contrasts remove the saturated-comp confound) ─
hub_gate  = mg[:, hub_mask].mean(1)
blob_gate = mg[:, blob_mask].mean(1)
bg_gate   = mg[:, bg_mask].mean(1)
hub_contrast  = hub_gate - bg_gate      # how much the gate concentrates on hubs
blob_contrast = blob_gate - bg_gate     # ... on mode blobs, relative to background

# example panels: reuse the original 8 components so the two figures are comparable
examples = [62, 17, 8, 0, 47, 44, 5, 41]

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 8.5))
gs = GridSpec(2, 6, figure=fig, width_ratios=[1.35, 1, 1, 1, 1, 0.08],
              hspace=0.32, wspace=0.12, left=0.05, right=0.94, top=0.88, bottom=0.09)

fig.suptitle("VPD gate maps for one message-passing layer (layers.0.mlp.0, 64 components) — "
             "faithful rendering from raw gates", fontsize=14, fontweight="bold")

# left: honest scatter summary
axs = fig.add_subplot(gs[:, 0])
sc = axs.scatter(hub_contrast, blob_contrast, c=abl * 1e3, s=55, cmap="viridis",
                 edgecolor="k", linewidth=0.4)
axs.scatter(hub_contrast[examples], blob_contrast[examples], s=170, facecolors="none",
            edgecolors="red", linewidth=1.8, label="examples (right)")
for c in examples:
    axs.annotate(str(c), (hub_contrast[c], blob_contrast[c]), fontsize=7,
                 xytext=(4, 4), textcoords="offset points")
lim = max(hub_contrast.max(), blob_contrast.max()) * 1.08
lo  = min(0.0, hub_contrast.min(), blob_contrast.min()) - 0.03
axs.plot([lo, lim], [lo, lim], "--", color="grey", lw=1, alpha=0.6)
axs.axhline(0, color="grey", lw=0.6, alpha=0.4); axs.axvline(0, color="grey", lw=0.6, alpha=0.4)
axs.set_xlim(lo, lim); axs.set_ylim(lo, lim)
axs.set_xlabel("hub-contrast   (gate on hubs − background)")
axs.set_ylabel("mode-blob-contrast   (gate on mode blobs − background)")
axs.set_title("Each dot = one component\n"
              "on/above diagonal = at least as mode- as hub-locked", fontsize=10)
axs.legend(loc="lower right", fontsize=8)
cb = fig.colorbar(sc, ax=axs, orientation="horizontal", pad=0.14, fraction=0.05)
cb.set_label("ablation importance  (recon drop ×10⁻³)", fontsize=9)

# right: 8 example maps, shared colormap + shared scale [0,1]
im = None
for i, comp in enumerate(examples):
    ax = fig.add_subplot(gs[i // 4, 1 + i % 4])
    im = ax.imshow(mg[comp], cmap=GATE_CMAP, vmin=0.0, vmax=1.0)
    # identical overlays on every panel
    ax.scatter(hub_c, hub_r, s=3, c="cyan", alpha=0.55, linewidths=0)
    for (r, cc) in centers:
        ax.add_patch(plt.Circle((cc, r), 3.2, fill=False, edgecolor="lime", lw=1.1))
    ax.set_title(f"comp {comp}\nhubΔ={hub_contrast[comp]:.2f}  "
                 f"blobΔ={blob_contrast[comp]:.2f}  abl={abl[comp]*1e3:.1f}e-3",
                 fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])

# shared colorbar = the map legend, on the right
cax = fig.add_subplot(gs[:, 5])
mcb = fig.colorbar(im, cax=cax)
mcb.set_label("mean CI gate  g   (0 = unused / maskable,  1 = kept active)", fontsize=10)

# marker legend for the maps
fig.text(0.60, 0.025,
         "cyan dots = mesh hubs (stride 5)      lime circles = the 8 SAVAR mode centres"
         "      (drawn identically on every panel)", fontsize=9, ha="center")

fig.savefig(OUT, dpi=110)
print(f"saved {OUT}")

# console summary of the honest reading
print("\nComponent summary (all 64):")
print(f"  hub-contrast   mean {hub_contrast.mean():.3f}  (max {hub_contrast.max():.3f})")
print(f"  blob-contrast  mean {blob_contrast.mean():.3f}  (max {blob_contrast.max():.3f})")
print(f"  ablation drop  mean {abl.mean()*1e3:.2f}e-3  (max {abl.max()*1e3:.2f}e-3)")
n_diag = int((blob_contrast >= hub_contrast).sum())
print(f"  components at least as mode- as hub-locked: {n_diag}/{C}")
