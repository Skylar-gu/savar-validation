"""
Task 1 — Re-fit content_R2 WITHOUT the mode one-hot (cached, no retrain).

The original content_R2 (Step 2b, train_spatial_sae.py) regressed each SAE
feature on [1, content, content^2, mode_one_hot(9)]. The 9-level mode one-hot is
a *position proxy* on this dataset (modes sit at fixed locations), so it leaks
WHERE-information into the supposedly content-only score, and it gives the
content design matrix 12 columns vs the per-pixel pos_R2's 2500 groups — an
asymmetry that always lets position win the gl-vs-content tie.

This script drops the one-hot entirely and scores content_R2 on [1, content,
content^2] only. It is a pure OLS re-fit on the *cached* per-pixel activations
(sae_data_diurnal/spatial_acts.npz) re-encoded through the *cached* trained SAE
(sae_data_diurnal/spatial_sae.pt) — no CNN or SAE retraining.

Reports max / median content_R2 and the gl-vs-content taxonomy under BOTH design
matrices so the confound's effect is visible directly.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(line_buffering=True)

OUT = Path("sae_data_diurnal")
INPUT_DIM, N_FEATURES, K_TOPK = 256, 512, 25
NY = NX = 50
BORDER = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TopKSAE(nn.Module):
    def __init__(self, d, n, k):
        super().__init__()
        self.k = k
        self.encoder = nn.Linear(d, n)
        self.decoder = nn.Linear(n, d)

    def encode(self, x):
        pre = self.encoder(x)
        v, i = torch.topk(pre, self.k, dim=-1)
        a = torch.zeros_like(pre); a.scatter_(-1, i, F.relu(v))
        return a


# ── cached data + cached SAE ───────────────────────────────────────────────────
d = np.load(OUT / "spatial_acts.npz")
X, ys, xs, content, mode_id = d["X"], d["ys"], d["xs"], d["content"], d["mode_id"]
N = len(X)
mean, std = X.mean(0), X.std(0) + 1e-8
Xn = torch.from_numpy(((X - mean) / std).astype(np.float32)).to(DEVICE)

sae = TopKSAE(INPUT_DIM, N_FEATURES, K_TOPK).to(DEVICE)
sae.load_state_dict(torch.load(OUT / "spatial_sae.pt", map_location=DEVICE))
sae.eval()
print(f"Loaded cached SAE + acts:  N={N:,}  feats={N_FEATURES}  K={K_TOPK}")

feats = np.empty((N, N_FEATURES), dtype=np.float32)
with torch.no_grad():
    for i in range(0, N, 8192):
        feats[i:i+8192] = sae.encode(Xn[i:i+8192]).cpu().numpy()

# ── position_R2 (unchanged) ────────────────────────────────────────────────────
pix = ys * NX + xs
npix = NY * NX
cnt = np.bincount(pix, minlength=npix).astype(np.float64)
grand = feats.mean(0)
sum_pp = np.zeros((npix, N_FEATURES)); np.add.at(sum_pp, pix, feats)
mu = sum_pp / np.maximum(cnt[:, None], 1)
ss_total = ((feats - grand) ** 2).sum(0) + 1e-12
position_R2 = (cnt[:, None] * (mu - grand) ** 2).sum(0) / ss_total

# border concentration (for the gl gate)
border = np.zeros((NY, NX), bool)
border[:BORDER] = border[-BORDER:] = True; border[:, :BORDER] = border[:, -BORDER:] = True
muc = mu - mu.mean(0)
emap = (muc ** 2).reshape(npix, N_FEATURES)
border_ratio = (emap[border.ravel()].sum(0) / (emap.sum(0) + 1e-12)) / border.mean()

act_rate = (feats > 0).mean(0)
alive = act_rate > 1e-4


def content_r2(with_onehot):
    cols = [np.ones((N, 1), np.float32), content[:, None].astype(np.float32),
            (content ** 2)[:, None].astype(np.float32)]
    if with_onehot:
        oh = np.zeros((N, 9), np.float32); oh[np.arange(N), mode_id + 1] = 1
        cols.append(oh)
    D = np.concatenate(cols, 1)
    coef, *_ = np.linalg.lstsq(D, feats, rcond=None)
    return 1 - ((feats - D @ coef) ** 2).sum(0) / ss_total


cR2_old = content_r2(True)    # [1, c, c^2, mode_onehot(9)]   (original, confounded)
cR2_new = content_r2(False)   # [1, c, c^2]                   (Task 1: position-proxy removed)


def taxonomy(cR2, tag):
    gl = alive & (position_R2 > 0.30) & (position_R2 > 1.5 * cR2) & (border_ratio > 1.5)
    ct = alive & (cR2 > 0.30) & (cR2 > position_R2)
    print(f"\n── {tag} ──")
    print(f"  GRID-LOCKED : {gl.sum():>3}      CONTENT : {ct.sum():>3}      "
          f"mixed/weak : {alive.sum()-gl.sum()-ct.sum():>3}   (alive {alive.sum()})")
    print(f"  content_R2  median {np.median(cR2[alive]):.3f}   max {cR2[alive].max():.3f}")
    print(f"  position_R2 median {np.median(position_R2[alive]):.3f}   max {position_R2[alive].max():.3f}")
    return gl, ct


print("\n" + "=" * 64)
print("CONTENT_R2 RE-FIT  (cached activations, OLS re-fit only)")
print("=" * 64)
gl_o, ct_o = taxonomy(cR2_old, "WITH mode one-hot  [1, c, c^2, oh(9)]  (original)")
gl_n, ct_n = taxonomy(cR2_new, "WITHOUT one-hot    [1, c, c^2]        (Task 1 fix)")

dmax = cR2_new[alive].max() - cR2_old[alive].max()
print(f"\nΔ max content_R2  (no-oh − oh): {dmax:+.4f}")
print(f"Per-feature mean |Δcontent_R2|: {np.abs(cR2_new - cR2_old)[alive].mean():.4f}")
print(f"Features moved gl→content by the fix: "
      f"{int((gl_o & ct_n).sum())}   content→gl: {int((ct_o & gl_n).sum())}")

np.savez(OUT / "content_r2_refit.npz",
         position_R2=position_R2, content_R2_with_oh=cR2_old,
         content_R2_no_oh=cR2_new, border_ratio=border_ratio,
         act_rate=act_rate, alive=alive,
         gl_with_oh=gl_o, ct_with_oh=ct_o, gl_no_oh=gl_n, ct_no_oh=ct_n)
print(f"\nSaved → {OUT}/content_r2_refit.npz")
