"""
Visualize the SAVAR model: spatial weight matrix W and data field.
Saves figures to ./figures/
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "savar"))

# Re-run the instantiation (imports model, W, links_coeffs, etc.)
exec(open(os.path.join(_ROOT, "data_gen", "instantiate_model.py")).read())

os.makedirs("figures", exist_ok=True)

# ── colour helpers ──────────────────────────────────────────────────────────
CMAP_W    = "YlOrRd"
CMAP_DATA = "RdBu_r"
MODE_COLS = ["#e6194b", "#3cb44b", "#4363d8"]   # red / green / blue per mode

# ============================================================
# FIG 1 — Spatial weight matrix W
# ============================================================
fig, axes = plt.subplots(1, N + 1, figsize=(4 * (N + 1), 3.6),
                         constrained_layout=True)
fig.suptitle("Spatial weight matrix W  (nonoverlapping Gaussian blobs)",
             fontsize=13, weight="bold")

vmax = W.max()
for i in range(N):
    im = axes[i].imshow(W[i], origin="upper", cmap=CMAP_W,
                        vmin=0, vmax=vmax, aspect="auto")
    axes[i].set_title(f"Mode {i}", fontsize=11)
    axes[i].set_xlabel("x (column)")
    axes[i].set_ylabel("y (row)")
    strip_rows = ny // N
    axes[i].axhline(strip_rows,     color="white", lw=0.8, ls="--")
    axes[i].axhline(strip_rows * 2, color="white", lw=0.8, ls="--")
    fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

# Superimposed sum (shows non-overlap clearly)
im2 = axes[N].imshow(W.sum(axis=0), origin="upper", cmap=CMAP_W,
                     vmin=0, vmax=vmax, aspect="auto")
axes[N].set_title("W.sum(axis=0)\n(should = max, no overlap)", fontsize=10)
axes[N].set_xlabel("x (column)")
axes[N].axhline(strip_rows,     color="white", lw=0.8, ls="--")
axes[N].axhline(strip_rows * 2, color="white", lw=0.8, ls="--")
fig.colorbar(im2, ax=axes[N], fraction=0.046, pad=0.04)

fig.savefig("figures/weight_matrix_W.png", dpi=150)
plt.close(fig)
print("Saved → figures/weight_matrix_W.png")

# ============================================================
# FIG 2 — Full data field heatmap  (L × T)
# ============================================================
Y = model.data_field   # shape (900, 500)

fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
vlim = np.percentile(np.abs(Y), 98)
im = ax.imshow(Y, origin="upper", aspect="auto", cmap=CMAP_DATA,
               vmin=-vlim, vmax=vlim,
               extent=[0, T, L, 0])
ax.set_title(f"Data field  y(ℓ, t)  —  shape (L={L}, T={T})", fontsize=12, weight="bold")
ax.set_xlabel("Time  t")
ax.set_ylabel("Grid point  ℓ  (0 = top-left)")
fig.colorbar(im, ax=ax, label="y value", fraction=0.02, pad=0.02)

# Mark mode strip boundaries
strip_L = (ny // N) * nx          # grid points per mode strip
for boundary in [strip_L, 2 * strip_L]:
    ax.axhline(boundary, color="black", lw=1.0, ls="--", alpha=0.6)
ax.text(T + 2, strip_L * 0.5, "Mode 0", va="center", fontsize=8, color="black", clip_on=False)
ax.text(T + 2, strip_L * 1.5, "Mode 1", va="center", fontsize=8, color="black", clip_on=False)
ax.text(T + 2, strip_L * 2.5, "Mode 2", va="center", fontsize=8, color="black", clip_on=False)

fig.savefig("figures/data_field_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved → figures/data_field_heatmap.png")

# ============================================================
# FIG 3 — Latent mode time series  X_i(t) = W_i · y(t)
#          plus spatial std per time step
# ============================================================
W_flat = W.reshape(N, -1)          # (3, 900)
X = W_flat @ Y                     # (3, 500) — mode projections
t = np.arange(T)

fig = plt.figure(figsize=(13, 8), constrained_layout=True)
fig.suptitle("Latent mode signals  X_i(t) = W_i · y(t)", fontsize=13, weight="bold")
gs = gridspec.GridSpec(N + 1, 1, figure=fig, hspace=0.05)

axes = [fig.add_subplot(gs[i]) for i in range(N + 1)]

for i in range(N):
    axes[i].plot(t, X[i], color=MODE_COLS[i], lw=0.9, label=f"X{i}(t)")
    axes[i].axhline(0, color="grey", lw=0.5, ls=":")
    axes[i].set_ylabel(f"X{i}", fontsize=10)
    axes[i].legend(loc="upper right", fontsize=8, framealpha=0.5)
    axes[i].set_xticklabels([])

# Bottom panel: spatial std over time (captures noise envelope)
spatial_std = Y.std(axis=0)
axes[N].fill_between(t, 0, spatial_std, color="slategrey", alpha=0.5, label="spatial std(t)")
axes[N].set_ylabel("spatial std", fontsize=10)
axes[N].set_xlabel("Time  t", fontsize=10)
axes[N].legend(loc="upper right", fontsize=8, framealpha=0.5)

fig.savefig("figures/mode_timeseries.png", dpi=150)
plt.close(fig)
print("Saved → figures/mode_timeseries.png")

# ============================================================
# FIG 4 — Combined overview (3 panels, publication-style)
# ============================================================
fig = plt.figure(figsize=(14, 10), constrained_layout=True)
fig.suptitle("SAVAR model overview", fontsize=14, weight="bold")

gs_top    = gridspec.GridSpec(1, N + 1, figure=fig,
                              left=0.05, right=0.97, top=0.93, bottom=0.62, wspace=0.35)
gs_mid    = gridspec.GridSpec(1, 1, figure=fig,
                              left=0.05, right=0.97, top=0.58, bottom=0.38, hspace=0)
gs_bottom = gridspec.GridSpec(N, 1, figure=fig,
                              left=0.05, right=0.97, top=0.34, bottom=0.05, hspace=0.05)

# — top row: individual modes + sum —
for i in range(N):
    ax = fig.add_subplot(gs_top[0, i])
    ax.imshow(W[i], origin="upper", cmap=CMAP_W, vmin=0, vmax=vmax, aspect="auto")
    ax.set_title(f"W  mode {i}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for bnd in [strip_rows, strip_rows * 2]:
        ax.axhline(bnd, color="white", lw=0.7, ls="--")

ax_sum = fig.add_subplot(gs_top[0, N])
im_s = ax_sum.imshow(W.sum(axis=0), origin="upper", cmap=CMAP_W,
                     vmin=0, vmax=vmax, aspect="auto")
ax_sum.set_title("W sum\n(non-overlap check)", fontsize=9)
ax_sum.set_xticks([]); ax_sum.set_yticks([])
for bnd in [strip_rows, strip_rows * 2]:
    ax_sum.axhline(bnd, color="white", lw=0.7, ls="--")
fig.colorbar(im_s, ax=ax_sum, fraction=0.07, pad=0.04, label="weight")

# — middle: full data field —
ax_heat = fig.add_subplot(gs_mid[0, 0])
im_h = ax_heat.imshow(Y, origin="upper", aspect="auto", cmap=CMAP_DATA,
                      vmin=-vlim, vmax=vlim, extent=[0, T, L, 0])
ax_heat.set_ylabel("Grid point ℓ", fontsize=9)
ax_heat.set_xlabel("")
ax_heat.set_title("Data field  y(ℓ, t)", fontsize=10)
ax_heat.set_xticklabels([])
for bnd in [strip_L, 2 * strip_L]:
    ax_heat.axhline(bnd, color="black", lw=0.8, ls="--", alpha=0.5)
fig.colorbar(im_h, ax=ax_heat, fraction=0.015, pad=0.01, label="y")

# — bottom: latent time series —
for i in range(N):
    ax = fig.add_subplot(gs_bottom[i, 0])
    ax.plot(t, X[i], color=MODE_COLS[i], lw=0.8)
    ax.axhline(0, color="grey", lw=0.4, ls=":")
    ax.set_ylabel(f"X{i}", fontsize=9)
    if i < N - 1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("Time  t", fontsize=9)

fig.savefig("figures/overview.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved → figures/overview.png")

print("\nAll figures saved to ./figures/")
