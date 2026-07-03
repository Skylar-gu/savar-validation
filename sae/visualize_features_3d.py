"""
Phase 7.5-GNN — 3D visualization of dominant per-mode SAE feature directions.

What we are looking at
----------------------
Each per-mode SAE (sae_data_gnn/sae_mode_{j}.pt) has a decoder whose 512 columns
are UNIT vectors in the 256-dim (normalised) mode-weighted activation space — i.e.
512 candidate "feature directions".  We want to *see* those directions, but 256-D
is not viewable, so we project each direction onto a 3-D basis.

The 3-D basis (why these three axes)
------------------------------------
Plain top-3 PCA is one option, but a semantic basis is far more interpretable for
this pipeline, so it is the default (`--basis semantic`):

  axis 1  = PC0        the dominant "global activity" direction (~72% of variance;
                       shared across all modes — the thing a single mixed SAE latches onto)
  axis 2  = Z-readout  the optimal linear direction (ridge) that decodes Z_j from the
                       activations — a feature aligned with the mode points along this axis
  axis 3  = residual   top principal component orthogonal to axes 1 & 2

Everything is done in the SAME normalised space the SAE decoder lives in
(x_norm = (x - act_mean) / act_std), so decoder columns, PCA and the Z-readout are
all directly comparable.  `--basis pca` falls back to top-3 PCs of that cloud.

Encoding of each feature in the plot
------------------------------------
  position   decoder column projected into the 3-D basis (arrow from origin; length
             = fraction of the unit direction captured by the 3 axes, <= 1)
  colour     signed Pearson r of the feature's activation with Z_j (diverging:
             blue = anti-correlated, red = correlated with the mode)
  size       feature "dominance" = mean SAE activation over the data (how much it fires)
  gold star  the single best-Z-aligned feature for that mode

A faint unit sphere is drawn for reference (all true directions have length 1 in 256-D;
what you see is their shadow in the 3-D subspace).

Outputs (figures/)
------------------
  sae_features_3d_gnn.png        2x4 grid, one 3-D scatter per mode X0..X7
  sae_features_3d_gnn_hero.gif   rotating view of one hero mode (--hero, default X3)

Usage
-----
  python3 sae/visualize_features_3d.py                 # all modes, semantic basis
  python3 sae/visualize_features_3d.py --basis pca
  python3 sae/visualize_features_3d.py --hero 1 --no-gif
"""

import sys, argparse
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm

_ap = argparse.ArgumentParser()
_ap.add_argument("--data", default="sae_data_gnn", help="per-mode SAE data dir")
_ap.add_argument("--out",  default="figures",      help="output dir for figures")
_ap.add_argument("--basis", choices=["semantic", "pca"], default="semantic",
                 help="3-D basis: 'semantic' (PC0, Z-readout, residual) or 'pca' (top-3 PCs)")
_ap.add_argument("--hero", type=int, default=3, help="mode index for the rotating GIF")
_ap.add_argument("--topN", type=int, default=12, help="draw arrows for the N most dominant feats")
_ap.add_argument("--no-gif", action="store_true", help="skip the rotating GIF")
_ap.add_argument("--subsample", type=int, default=40000, help="samples for PCA / ridge / dominance")
_ap.add_argument("--tag", default="gnn", help="filename tag (e.g. 'hetdynamics') to avoid clobbering")
_ap.add_argument("--mixed", action="store_true",
                 help="visualize the single mixed-mode SAE (sae_mixed.pt / alignment_mixed.npy) "
                      "instead of the per-mode SAEs")
_args = _ap.parse_args()

DATA = Path(_args.data)
OUT  = Path(_args.out); OUT.mkdir(exist_ok=True)
N_MODES, INPUT_DIM, N_FEATURES, K_TOPK = 8, 256, 512, 25
rng = np.random.default_rng(0)


class TopKSAE(nn.Module):
    def __init__(s, input_dim=INPUT_DIM, n_features=N_FEATURES, k=K_TOPK):
        super().__init__()
        s.k = k
        s.encoder = nn.Linear(input_dim, n_features)
        s.decoder = nn.Linear(n_features, input_dim)

    def encode(s, x):
        pre = s.encoder(x)
        v, i = torch.topk(pre, s.k, dim=-1)
        a = torch.zeros_like(pre)
        a.scatter_(-1, i, F.relu(v))
        return a


def ridge_direction(Xn, z, lam=10.0):
    """Optimal linear readout direction for z from normalised activations Xn (unit-norm)."""
    A = Xn.T @ Xn + lam * np.eye(Xn.shape[1])
    w = np.linalg.solve(A, Xn.T @ z)
    return w / (np.linalg.norm(w) + 1e-12)


def build_basis(Xn, z, kind):
    """Return orthonormal (3, 256) basis + human labels for the three axes."""
    Xc = Xn - Xn.mean(0)
    sub = rng.choice(len(Xc), min(_args.subsample, len(Xc)), replace=False)
    _, _, Vt = np.linalg.svd(Xc[sub], full_matrices=False)
    if kind == "pca":
        return Vt[:3].copy(), ["PC0", "PC1", "PC2"]

    pc0 = Vt[0]
    zdir = ridge_direction(Xn[sub], z[sub])
    # Gram-Schmidt: keep PC0, make Z-readout orthogonal to it, then residual PC.
    zdir = zdir - (zdir @ pc0) * pc0
    zdir /= np.linalg.norm(zdir) + 1e-12
    for v in Vt:                                  # first PC not in span{pc0, zdir}
        r = v - (v @ pc0) * pc0 - (v @ zdir) * zdir
        if np.linalg.norm(r) > 1e-6:
            resid = r / np.linalg.norm(r)
            break
    return np.stack([pc0, zdir, resid]), ["PC0 (global activity)", "Z-readout", "residual"]


def load_mode(j):
    name = "sae_mixed.pt" if _args.mixed else f"sae_mode_{j}.pt"
    ck = torch.load(DATA / name, map_location="cpu", weights_only=False)
    sae = TopKSAE(); sae.load_state_dict(ck["model_state"]); sae.eval()
    return sae, ck["act_mean"], ck["act_std"]


def mode_data(acts, Z, j, sae, mean_j, std_j):
    """Return per-feature 3D coords, signed r with Z_j, dominance, best-feat idx, basis, labels."""
    Aj = acts[:, j, :, :].reshape(-1, INPUT_DIM).astype(np.float32)
    Zj = Z[:, j, :].reshape(-1).astype(np.float64)
    Xn = (Aj - mean_j) / std_j

    basis, labels = build_basis(Xn, Zj, _args.basis)

    with torch.no_grad():
        fa = []
        for i in range(0, len(Xn), 8192):
            fa.append(sae.encode(torch.from_numpy(Xn[i:i + 8192])).numpy())
        fa = np.concatenate(fa)                                   # (N, 512)

    dom = fa.mean(0)                                              # dominance per feature
    # signed correlation of each feature activation with Z_j
    fc = fa - fa.mean(0); zc = Zj - Zj.mean()
    r = (fc.T @ zc) / ((np.linalg.norm(fc, axis=0) + 1e-12) * (np.linalg.norm(zc) + 1e-12))

    dec = sae.decoder.weight.detach().numpy()                    # (256, 512) unit cols
    coords = (basis @ dec).T                                     # (512, 3)
    best = int(np.abs(r).argmax())
    return coords, r, dom, best, labels


def draw(ax, coords, r, dom, best, labels, title, topN):
    live = dom > dom.max() * 0.02                                # ignore ~dead features
    idx = np.where(live)[0]
    c, rr, dd = coords[idx], r[idx], dom[idx]

    # faint reference unit sphere
    u, v = np.mgrid[0:2*np.pi:24j, 0:np.pi:12j]
    ax.plot_wireframe(np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v),
                      color="0.85", linewidth=0.3, alpha=0.5)

    norm = TwoSlopeNorm(vmin=-max(0.05, np.abs(r).max()), vcenter=0,
                        vmax=max(0.05, np.abs(r).max()))
    sizes = 20 + 400 * (dd / (dd.max() + 1e-12))
    sc = ax.scatter(c[:, 0], c[:, 1], c[:, 2], c=rr, cmap="coolwarm", norm=norm,
                    s=sizes, alpha=0.85, edgecolors="0.3", linewidths=0.3, depthshade=True)

    # arrows from origin for the top-N dominant features
    top = idx[np.argsort(-dom[idx])[:topN]]
    for f in top:
        p = coords[f]
        ax.plot([0, p[0]], [0, p[1]], [0, p[2]], color=cm.coolwarm(norm(r[f])),
                linewidth=1.4, alpha=0.7)

    # highlight best-Z-aligned feature
    p = coords[best]
    ax.scatter(*p, s=260, marker="*", color="gold", edgecolors="k", linewidths=0.8, zorder=6)
    ax.plot([0, p[0]], [0, p[1]], [0, p[2]], color="gold", linewidth=2.2, zorder=5)

    ax.set_xlabel(labels[0], fontsize=8); ax.set_ylabel(labels[1], fontsize=8)
    ax.set_zlabel(labels[2], fontsize=8)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=6)
    return sc


def main():
    acts = np.load(DATA / "activations_full.npy")
    Z    = np.load(DATA / "Z_full.npy")
    align_name = "alignment_mixed.npy" if _args.mixed else "alignment_per_mode.npy"
    align = np.load(DATA / align_name, allow_pickle=True).item()
    print(f"acts {acts.shape}  basis={_args.basis}")

    # ── grid figure: all 8 modes ──────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 10))
    per_mode = {}
    for j in range(N_MODES):
        sae, m, s = load_mode(j)
        coords, r, dom, best, labels = mode_data(acts, Z, j, sae, m, s)
        per_mode[j] = (coords, r, dom, best, labels)
        ax = fig.add_subplot(2, 4, j + 1, projection="3d")
        rj = align[j]["max_r"]
        sc = draw(ax, coords, r, dom, best, labels,
                  f"X{j}   best f{best}  |r|={rj:.2f}", _args.topN)
        ax.view_init(elev=22, azim=45)
        print(f"  X{j}: best f{best}  |r|={np.abs(r).max():.3f}  live={int((dom>dom.max()*0.02).sum())}")

    cbar = fig.colorbar(sc, ax=fig.axes, shrink=0.5, pad=0.02, location="right")
    cbar.set_label("signed corr(feature, Z_j)")
    sae_kind = "mixed-mode SAE" if _args.mixed else "per-mode SAE"
    fig.suptitle(f"Dominant GNN {sae_kind} feature directions per mode  —  basis: {_args.basis}  "
                 f"(size = dominance, ★ = best Z-aligned)", fontsize=13)
    grid_path = OUT / f"sae_features_3d_{_args.tag}.png"
    fig.savefig(grid_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved  {grid_path}")

    # ── rotating GIF for the hero mode ────────────────────────────────────────
    if not _args.no_gif:
        try:
            from matplotlib.animation import FuncAnimation, PillowWriter
            j = _args.hero
            coords, r, dom, best, labels = per_mode[j]
            figh = plt.figure(figsize=(8, 7))
            axh = figh.add_subplot(111, projection="3d")
            sc = draw(axh, coords, r, dom, best, labels,
                      f"GNN SAE features — mode X{j}  (best f{best})", _args.topN)
            figh.colorbar(sc, ax=axh, shrink=0.6, label="signed corr(feature, Z_j)")

            def upd(a):
                axh.view_init(elev=20, azim=a); return ()
            anim = FuncAnimation(figh, upd, frames=range(0, 360, 4), interval=80)
            gif_path = OUT / f"sae_features_3d_{_args.tag}_hero.gif"
            anim.save(gif_path, writer=PillowWriter(fps=15))
            plt.close(figh)
            print(f"Saved  {gif_path}")
        except Exception as e:
            print(f"  (GIF skipped: {e})")


if __name__ == "__main__":
    main()
