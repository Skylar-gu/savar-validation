"""
Task 3 — Is the dominant-direction "feature splitting" a real code or an
under-regularization artifact?

The default spatial SAE (dict=512, TopK K=25) puts its top-10 features at nearly
identical stats (var≈1.9%, pos≈0.44, cont≈0.16) — the classic redundant-clone
signature of an over-complete, under-sparse dictionary. If those are clones, then
raising sparsity (lower K — the TopK analogue of a larger L1) or shrinking the
dictionary should merge them: fewer alive features, lower participation ratio,
higher per-feature variance, and lower decoder cosine-redundancy among the top
features, with the position/content picture essentially preserved.

We retrain the SAE on the SAME cached per-pixel activations across a
dict-size × sparsity grid (cheap: the SAE is tiny; the CNN is NOT retrained) and
report, per config:
  alive, participation ratio (effective #feats), top-1/5/10 variance share,
  redundancy = mean |cosine| among top-10 decoder columns (1.0 = identical),
  and the grid-locked / content taxonomy + max pos_R2 / cont_R2.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(line_buffering=True)

OUT = Path("sae_data_diurnal")
INPUT_DIM = 256
NY = NX = 50
BORDER = 3
EPOCHS = 30
BATCH = 512
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DICTS = [128, 256, 512]
KS = [8, 16, 25]


class TopKSAE(nn.Module):
    def __init__(self, d, n, k):
        super().__init__()
        self.k = k
        self.encoder = nn.Linear(d, n); self.decoder = nn.Linear(n, d)
        nn.init.normal_(self.decoder.weight, std=1.0 / d ** 0.5)
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


# ── cached data ───────────────────────────────────────────────────────────────
d = np.load(OUT / "spatial_acts.npz")
X, ys, xs, content, mode_id = d["X"], d["ys"], d["xs"], d["content"], d["mode_id"]
N = len(X)
mean, std = X.mean(0), X.std(0) + 1e-8
Xn_gpu = torch.from_numpy(((X - mean) / std).astype(np.float32)).to(DEVICE)
pix = ys * NX + xs; npix = NY * NX
cnt = np.bincount(pix, minlength=npix).astype(np.float64)
border = np.zeros((NY, NX), bool)
border[:BORDER] = border[-BORDER:] = True; border[:, :BORDER] = border[:, -BORDER:] = True
Dc = np.concatenate([np.ones((N, 1), np.float32), content[:, None].astype(np.float32),
                     (content ** 2)[:, None].astype(np.float32)], 1)   # [1,c,c^2] (Task-1 fix)
print(f"Cached acts: N={N:,}  dim={INPUT_DIM}")


def train_and_score(n_feat, k):
    torch.manual_seed(0)
    sae = TopKSAE(INPUT_DIM, n_feat, k).to(DEVICE)
    opt = torch.optim.Adam(sae.parameters(), lr=LR)
    last_fired = torch.zeros(n_feat, dtype=torch.long, device=DEVICE); gstep = 0
    spe = N // BATCH
    for ep in range(EPOCHS):
        perm = torch.randperm(N, device=DEVICE)
        for i in range(spe):
            b = Xn_gpu[perm[i*BATCH:(i+1)*BATCH]]
            a, rec = sae(b); loss = F.mse_loss(rec, b)
            opt.zero_grad(); loss.backward(); opt.step(); sae.norm_dec()
            last_fired[(a.detach() > 0).any(0)] = gstep; gstep += 1
            if gstep % 1000 == 0:                       # resample dead features
                dead = (gstep - last_fired) > 800
                if dead.any():
                    s = Xn_gpu[torch.randperm(N, device=DEVICE)[:2048]]
                    with torch.no_grad():
                        _, r = sae(s)
                        worst = F.mse_loss(r, s, reduction='none').mean(1).argsort(descending=True)
                        res = F.normalize((s - r)[worst][:int(dead.sum())], dim=1)
                        di = dead.nonzero(as_tuple=True)[0][:len(res)]
                        sae.encoder.weight.data[di] = res; sae.decoder.weight.data[:, di] = res.T
    sae.eval()
    feats = np.empty((N, n_feat), np.float32)
    with torch.no_grad():
        for i in range(0, N, 8192):
            feats[i:i+8192] = sae.encode(Xn_gpu[i:i+8192]).cpu().numpy()

    grand = feats.mean(0)
    sum_pp = np.zeros((npix, n_feat)); np.add.at(sum_pp, pix, feats)
    mu = sum_pp / np.maximum(cnt[:, None], 1)
    st = ((feats - grand) ** 2).sum(0) + 1e-12
    pos_R2 = (cnt[:, None] * (mu - grand) ** 2).sum(0) / st
    coef, *_ = np.linalg.lstsq(Dc, feats, rcond=None)
    cont_R2 = 1 - ((feats - Dc @ coef) ** 2).sum(0) / st
    muc = mu - mu.mean(0); emap = (muc ** 2)
    border_ratio = (emap[border.ravel()].sum(0) / (emap.sum(0) + 1e-12)) / border.mean()
    act_rate = (feats > 0).mean(0); alive = act_rate > 1e-4

    var = np.maximum(st - 1e-12, 0.0); tv = var.sum(); order = np.argsort(var)[::-1]
    pr = float(tv ** 2 / (var ** 2).sum())                    # participation ratio
    gl = alive & (pos_R2 > 0.30) & (pos_R2 > 1.5 * cont_R2) & (border_ratio > 1.5)
    ct = alive & (cont_R2 > 0.30) & (cont_R2 > pos_R2)

    # redundancy of the top-10-by-variance decoder columns (1.0 = identical clones)
    top = order[:10]
    W = sae.decoder.weight.detach()[:, torch.from_numpy(top.copy()).to(DEVICE)]
    C = (W.T @ W).abs().cpu().numpy()
    iu = np.triu_indices(len(top), 1)
    redund_mean = float(C[iu].mean()); redund_max = float(C[iu].max())

    return dict(dict=n_feat, k=k, alive=int(alive.sum()), part_ratio=pr,
                top1=float(var[order[0]]/tv), top5=float(var[order[:5]].sum()/tv),
                top10=float(var[order[:10]].sum()/tv),
                redund_mean=redund_mean, redund_max=redund_max,
                n_gl=int(gl.sum()), n_ct=int(ct.sum()),
                max_pos=float(pos_R2[alive].max()), max_cont=float(cont_R2[alive].max()))


rows = []
print(f"\n{'dict':>5} {'K':>3} {'alive':>6} {'partR':>7} {'top1%':>6} {'top5%':>6} "
      f"{'top10%':>7} {'redMean':>8} {'redMax':>7} {'#gl':>4} {'#ct':>4} {'maxPos':>7} {'maxCont':>8}")
for nf in DICTS:
    for k in KS:
        r = train_and_score(nf, k)
        rows.append(r)
        print(f"{r['dict']:>5} {r['k']:>3} {r['alive']:>6} {r['part_ratio']:>7.1f} "
              f"{r['top1']*100:>6.2f} {r['top5']*100:>6.2f} {r['top10']*100:>7.2f} "
              f"{r['redund_mean']:>8.3f} {r['redund_max']:>7.3f} {r['n_gl']:>4} {r['n_ct']:>4} "
              f"{r['max_pos']:>7.3f} {r['max_cont']:>8.3f}")

np.save("results/feature_splitting_sweep.npy", rows)
print("\nSaved → results/feature_splitting_sweep.npy")
print("\nRead: if redundancy is high and falls (clones merge) as K↓ / dict↓ while "
      "participation ratio drops and per-feature top-k% rises, the '~93 effective "
      "features' count is partly an under-regularization artifact. If pos/content "
      "(#gl, #ct, maxPos, maxCont) stay put across the grid, the position-vs-content "
      "picture is robust to SAE hyperparameters.")
