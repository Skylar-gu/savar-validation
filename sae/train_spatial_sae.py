"""
Step 2b — Train a spatial SAE on per-pixel res3 activations and probe whether
any feature is GRID-LOCKED (fires as a function of pixel position, independent
of content).

For each learned feature f we score:
  position_R2  = eta^2 of feature activation grouped by pixel (y,x)
                 = fraction of the feature's variance explained by WHERE it is.
                 High ⇒ the feature reports location, not weather.
  content_R2   = R^2 of feature activation regressed on local content
                 (mode one-hot + field value + field^2). High ⇒ reports content.
  border_ratio = energy of the per-pixel mean map in the border ring / chance.

A feature with position_R2 >> content_R2 (and border-concentrated) is a
grid-locked / architecture feature — the SAVAR analogue of GraphCast's mesh
features. A feature with content_R2 dominant is a genuine physical feature.
"""

import sys, argparse
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

_ap = argparse.ArgumentParser()
_ap.add_argument("--dy005", action="store_true")
_ap.add_argument("--diurnal", action="store_true")
_ap.add_argument("--epochs", type=int, default=40)
_a = _ap.parse_args()
OUT = Path("sae_data_diurnal" if _a.diurnal else "sae_data_dy005" if _a.dy005 else "sae_data")

INPUT_DIM, N_FEATURES, K_TOPK = 256, 512, 25
LR, BATCH = 1e-3, 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NY = NX = 50
BORDER = 3


class TopKSAE(nn.Module):
    def __init__(self, d, n, k):
        super().__init__()
        self.k = k
        self.encoder = nn.Linear(d, n); self.decoder = nn.Linear(n, d)
        nn.init.normal_(self.decoder.weight, std=1.0 / d**0.5)
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)
            self.encoder.weight.data = self.decoder.weight.data.T.clone()
        nn.init.zeros_(self.encoder.bias); nn.init.zeros_(self.decoder.bias)

    def encode(self, x):
        pre = self.encoder(x)
        v, i = torch.topk(pre, self.k, dim=-1)
        a = torch.zeros_like(pre); a.scatter_(-1, i, F.relu(v))
        return a

    def forward(self, x):
        a = self.encode(x); return a, self.decoder(a)

    @torch.no_grad()
    def norm_dec(self):
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)


# ── data ──────────────────────────────────────────────────────────────────────
d = np.load(OUT / "spatial_acts.npz")
X, ys, xs, content, mode_id = d["X"], d["ys"], d["xs"], d["content"], d["mode_id"]
N = len(X)
mean, std = X.mean(0), X.std(0) + 1e-8
Xn = torch.from_numpy(((X - mean) / std).astype(np.float32))
print(f"Spatial SAE: N={N:,}  dim={INPUT_DIM}  features={N_FEATURES}  K={K_TOPK}")

# ── train ─────────────────────────────────────────────────────────────────────
sae = TopKSAE(INPUT_DIM, N_FEATURES, K_TOPK).to(DEVICE)
opt = torch.optim.Adam(sae.parameters(), lr=LR)
last_fired = torch.zeros(N_FEATURES, dtype=torch.long, device=DEVICE)
gstep = 0
Xn_gpu = Xn.to(DEVICE)
spe = N // BATCH
for ep in range(1, _a.epochs + 1):
    sae.train(); perm = torch.randperm(N, device=DEVICE); tot = 0.0
    for i in range(spe):
        b = Xn_gpu[perm[i*BATCH:(i+1)*BATCH]]
        a, rec = sae(b); loss = F.mse_loss(rec, b)
        opt.zero_grad(); loss.backward(); opt.step(); sae.norm_dec()
        last_fired[(a.detach() > 0).any(0)] = gstep; gstep += 1; tot += loss.item()
        if gstep % 1000 == 0:    # resample dead features into high-residual directions
            dead = (gstep - last_fired) > 800
            if dead.any():
                s = Xn_gpu[torch.randperm(N, device=DEVICE)[:2048]]
                with torch.no_grad():
                    _, r = sae(s); res = F.normalize((s - r)[F.mse_loss(r, s, reduction='none').mean(1).argsort(descending=True)][:int(dead.sum())], dim=1)
                    di = dead.nonzero(as_tuple=True)[0][:len(res)]
                    sae.encoder.weight.data[di] = res; sae.decoder.weight.data[:, di] = res.T
    if ep % 5 == 0 or ep == 1:
        dead_now = ((gstep - last_fired) > 800).sum().item()
        print(f"  ep{ep:>3}  trainMSE={tot/spe:.4f}  dead={dead_now}")

# ── encode all samples ────────────────────────────────────────────────────────
sae.eval()
feats = np.empty((N, N_FEATURES), dtype=np.float32)
with torch.no_grad():
    for i in range(0, N, 8192):
        feats[i:i+8192] = sae.encode(Xn_gpu[i:i+8192]).cpu().numpy()

# ── scoring ───────────────────────────────────────────────────────────────────
pix = ys * NX + xs
npix = NY * NX
cnt = np.bincount(pix, minlength=npix).astype(np.float64)          # (2500,)
grand = feats.mean(0)                                              # (512,)
# per-pixel mean map per feature
sum_pp = np.zeros((npix, N_FEATURES))
np.add.at(sum_pp, pix, feats)
mu = sum_pp / np.maximum(cnt[:, None], 1)                          # (2500,512)
ss_between = (cnt[:, None] * (mu - grand) ** 2).sum(0)
ss_total = ((feats - grand) ** 2).sum(0) + 1e-12
position_R2 = ss_between / ss_total                                # eta^2 (512,)

# content R2: regress on [1, content, content^2] ONLY.
# The mode one-hot was dropped (was: ..., mode one-hot(9)): on this dataset modes
# sit at fixed locations, so the one-hot is a position proxy that leaks WHERE-info
# into the content score and gives content 12 design columns vs pos_R2's 2500
# groups. Verified (refit_content_r2.py) to leave max content_R2 unchanged
# (0.375→0.374): the content signal is genuinely low, not confound-suppressed.
D = np.concatenate([np.ones((N, 1), np.float32), content[:, None],
                    (content**2)[:, None]], 1)                     # (N, 3)
coef, *_ = np.linalg.lstsq(D, feats, rcond=None)
pred = D @ coef
content_R2 = 1 - ((feats - pred) ** 2).sum(0) / ss_total

# border concentration of per-pixel mean map
border = np.zeros((NY, NX), bool)
border[:BORDER] = border[-BORDER:] = True; border[:, :BORDER] = border[:, -BORDER:] = True
muc = mu - mu.mean(0)
e = (muc ** 2)
emap = e.reshape(npix, N_FEATURES)
border_ratio = (emap[border.ravel()].sum(0) / (emap.sum(0) + 1e-12)) / border.mean()

act_rate = (feats > 0).mean(0)
alive = act_rate > 1e-4

gridlocked = alive & (position_R2 > 0.30) & (position_R2 > 1.5 * content_R2) & (border_ratio > 1.5)
contentf = alive & (content_R2 > 0.30) & (content_R2 > position_R2)

print(f"\n── Spatial SAE feature taxonomy ({alive.sum()} alive / {N_FEATURES}) ──")
print(f"  GRID-LOCKED (position-driven) : {gridlocked.sum():>3}")
print(f"  CONTENT     (physics-driven)  : {contentf.sum():>3}")
print(f"  mixed / weak                  : {alive.sum()-gridlocked.sum()-contentf.sum():>3}")
print(f"\n  position_R2  median {np.median(position_R2[alive]):.3f}  max {position_R2[alive].max():.3f}")
print(f"  content_R2   median {np.median(content_R2[alive]):.3f}  max {content_R2[alive].max():.3f}")

top_gl = np.argsort(np.where(gridlocked, position_R2, -1))[::-1][:6]
print("\n  Top grid-locked features (f: posR2 / contR2 / border×):")
for f in top_gl:
    if gridlocked[f]:
        print(f"    f{f:<3}  pos={position_R2[f]:.3f}  cont={content_R2[f]:.3f}  border={border_ratio[f]:.1f}x  rate={act_rate[f]:.3f}")
top_ct = np.argsort(np.where(contentf, content_R2, -1))[::-1][:6]
print("  Top content features (f: contR2 / posR2):")
for f in top_ct:
    if contentf[f]:
        print(f"    f{f:<3}  cont={content_R2[f]:.3f}  pos={position_R2[f]:.3f}")

np.savez(OUT / "spatial_sae_scores.npz", position_R2=position_R2, content_R2=content_R2,
         border_ratio=border_ratio, act_rate=act_rate, mu=mu.astype(np.float32),
         gridlocked=gridlocked, contentf=contentf)
torch.save(sae.state_dict(), OUT / "spatial_sae.pt")

# figure: per-pixel mean maps of top grid-locked vs top content features
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sel_gl = [f for f in top_gl if gridlocked[f]][:4]
    sel_ct = [f for f in top_ct if contentf[f]][:4]
    ncol = max(len(sel_gl), len(sel_ct), 1)
    fig, ax = plt.subplots(2, ncol, figsize=(3*ncol, 6))
    ax = np.atleast_2d(ax)
    for k, f in enumerate(sel_gl):
        ax[0, k].imshow(mu[:, f].reshape(NY, NX)); ax[0, k].set_title(f"GRID-LOCK f{f}\npos={position_R2[f]:.2f}")
    for k, f in enumerate(sel_ct):
        ax[1, k].imshow(mu[:, f].reshape(NY, NX)); ax[1, k].set_title(f"CONTENT f{f}\ncont={content_R2[f]:.2f}")
    for a in ax.ravel(): a.axis("off")
    fig.tight_layout(); fig.savefig("figures/gridlock_step2_features.png", dpi=120)
    print("\nSaved figure → figures/gridlock_step2_features.png")
except Exception as e:
    print(f"(figure skipped: {e})")
print(f"Saved scores → {OUT}/spatial_sae_scores.npz")
