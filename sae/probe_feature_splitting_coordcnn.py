"""
Task 3 (companion) — feature-splitting on the CoordCNN run that the
"~93 effective features / top-10 at var≈1.9% each" numbers actually came from
(physics-vs-position trade-off sweep, strength=1, seed=0; cf. probe_variance_diag).

Those numbers are NOT from the diurnal SAE (which concentrates much harder); they
are from the CoordCNN. So to test the user's exact claim — that the ~93 count is
an under-regularization artifact that collapses at higher sparsity / smaller
dictionary — we retrain *that* CoordCNN (CNN train is unavoidable: it is not
checkpointed), extract its per-pixel res3 acts once, then sweep the SAE
dict-size × sparsity grid on the SAME activations.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import physics_vs_position_tradeoff_sweep as S

STRENGTH, SEED = 1.0, 0
NY = NX = 50; BORDER = 3
INPUT_DIM = 256
EPOCHS, BATCH, LR = 30, 512, 1e-3
DEVICE = S.DEVICE
DICTS = [128, 256, 512]
KS = [8, 16, 25]


def extract_acts(model, seed=0):
    """Replicates spatial_probe's per-pixel res3 extraction (returns the arrays)."""
    S.set_seed(seed)
    cap = {}
    h = model.res3.register_forward_hook(lambda m, i, o: cap.__setitem__("a", o))
    import glob
    files = sorted(glob.glob(str(S.SPLIT / "val" / "realisation_*.npz")))[:S.N_REAL_VAL]
    W0 = np.load(files[0])["W"].astype(np.float32).reshape(8, NY, NX)
    mode_map = np.where(np.abs(W0).sum(0) > 0, np.abs(W0).argmax(0), -1)
    rng = np.random.default_rng(seed)
    X, ys_, xs_, cont, mid = [], [], [], [], []
    K = S.cnn_forecaster_K if hasattr(S, "cnn_forecaster_K") else __import__("cnn_forecaster").K
    with torch.no_grad():
        for f in files:
            obs = np.load(f)["observations"].astype(np.float32)
            fr = torch.from_numpy(obs).unsqueeze(1)
            tl = list(range(0, len(fr) - K, 20))
            wins = torch.stack([fr[t:t+K].permute(1, 0, 2, 3) for t in tl])
            tgt = obs[[t+K for t in tl]]
            for i in range(0, len(wins), 128):
                model(wins[i:i+128].to(DEVICE))
                a = cap["a"].permute(0, 2, 3, 1).cpu().numpy()
                tb = tgt[i:i+128]
                for b in range(a.shape[0]):
                    yy = rng.integers(0, NY, 100); xx = rng.integers(0, NX, 100)
                    X.append(a[b, yy, xx]); ys_.append(yy); xs_.append(xx)
                    cont.append(tb[b, yy, xx]); mid.append(mode_map[yy, xx])
    h.remove()
    return (np.concatenate(X).astype(np.float32), np.concatenate(ys_),
            np.concatenate(xs_), np.concatenate(cont).astype(np.float32),
            np.concatenate(mid))


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
        a = torch.zeros_like(pre); a.scatter_(-1, i, F.relu(v)); return a

    def forward(self, x):
        a = self.encode(x); return a, self.decoder(a)

    @torch.no_grad()
    def norm_dec(self):
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)


print(f"Training CoordCNN (strength={STRENGTH}, seed={SEED}) ...")
Xtr, Ytr = S.load_windows("train", S.N_REAL_TRAIN)
Xva, Yva = S.load_windows("val", S.N_REAL_VAL)
S.set_seed(SEED)
model, rmse = S.train_cnn(STRENGTH, Xtr, Ytr, Xva, Yva)
print(f"  valRMSE={rmse:.4f}")
X, ys, xs, content, mode_id = extract_acts(model, SEED)
N = len(X)
mean, std = X.mean(0), X.std(0) + 1e-8
Xn_gpu = torch.from_numpy(((X - mean) / std).astype(np.float32)).to(DEVICE)
pix = ys * NX + xs; npix = NY * NX
cnt = np.bincount(pix, minlength=npix).astype(np.float64)
border = np.zeros((NY, NX), bool)
border[:BORDER] = border[-BORDER:] = True; border[:, :BORDER] = border[:, -BORDER:] = True
Dc = np.concatenate([np.ones((N, 1), np.float32), content[:, None].astype(np.float32),
                     (content ** 2)[:, None].astype(np.float32)], 1)
print(f"CoordCNN acts: N={N:,}  dim={INPUT_DIM}")


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
            if gstep % 1000 == 0:
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
    pr = float(tv ** 2 / (var ** 2).sum())
    gl = alive & (pos_R2 > 0.30) & (pos_R2 > 1.5 * cont_R2) & (border_ratio > 1.5)
    ct = alive & (cont_R2 > 0.30) & (cont_R2 > pos_R2)
    top = order[:10]
    W = sae.decoder.weight.detach()[:, torch.from_numpy(top.copy()).to(DEVICE)]
    C = (W.T @ W).abs().cpu().numpy(); iu = np.triu_indices(len(top), 1)
    return dict(dict=n_feat, k=k, alive=int(alive.sum()), part_ratio=pr,
                top1=float(var[order[0]]/tv), top5=float(var[order[:5]].sum()/tv),
                top10=float(var[order[:10]].sum()/tv),
                redund_mean=float(C[iu].mean()), redund_max=float(C[iu].max()),
                n_gl=int(gl.sum()), n_ct=int(ct.sum()),
                max_pos=float(pos_R2[alive].max()), max_cont=float(cont_R2[alive].max()))


rows = []
print(f"\n{'dict':>5} {'K':>3} {'alive':>6} {'partR':>7} {'top1%':>6} {'top5%':>6} "
      f"{'top10%':>7} {'redMean':>8} {'redMax':>7} {'#gl':>4} {'#ct':>4} {'maxPos':>7} {'maxCont':>8}")
for nf in DICTS:
    for k in KS:
        r = train_and_score(nf, k); rows.append(r)
        print(f"{r['dict']:>5} {r['k']:>3} {r['alive']:>6} {r['part_ratio']:>7.1f} "
              f"{r['top1']*100:>6.2f} {r['top5']*100:>6.2f} {r['top10']*100:>7.2f} "
              f"{r['redund_mean']:>8.3f} {r['redund_max']:>7.3f} {r['n_gl']:>4} {r['n_ct']:>4} "
              f"{r['max_pos']:>7.3f} {r['max_cont']:>8.3f}")

np.save("results/feature_splitting_coordcnn.npy",
        dict(strength=STRENGTH, seed=SEED, val_rmse=rmse, rows=rows))
print("\nSaved → results/feature_splitting_coordcnn.npy")
print("Default (512,25) partR here should reproduce the ~93 from probe_variance_diag.")
