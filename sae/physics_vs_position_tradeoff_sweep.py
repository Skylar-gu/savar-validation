"""
Step 2c — Physics-vs-position trade-off sweep.

Inject a CONTROLLABLE position confound: append normalized x/y coordinate
channels (scaled by `strength`) to the CNN input. strength=0 hand-feeds no
position (control); larger strength feeds a stronger fixed position signal the
network can route into its activations. For each strength we retrain a CNN, then
re-run the spatial-SAE probe (Step 2b) and measure:

  val RMSE            — does the position channel help/hurt forecasting?
  #grid-locked feats  — how many SAE features become pure position detectors
  median pos_R2       — how position-dominated those features are
  #content feats / max content_R2 — how much physics survives

NOTE on the content metrics: cont_R2 is now scored on [1, cont, cont^2] only
(the mode one-hot, a position proxy, was removed — see refit_content_r2.py;
dropping it left max content_R2 unchanged at ~0.37). Even so the content bucket
(cont_R2 > pos_R2) stays near zero: pos_R2 is scored at per-pixel (2500-group)
resolution while content gets 3 columns, and the underlying content signal is
genuinely low. Treat #content / max content_R2 as a near-noise floor; the
trustworthy signals remain #grid-locked and median pos_R2.

Each strength is repeated over SEEDS (fresh CNN + fresh SAE per seed) and
reported as mean ± sd, so the monotone trends can be told apart from run-to-run
noise.

Kept small on purpose (subset of realisations, fewer epochs). Writes
results/physics_vs_position_tradeoff_sweep.npy and
figures/physics_vs_position_tradeoff_sweep.png.
"""

import sys, glob, time
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from cnn_forecaster import ResBlock2D, K, BASE_CH

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NY = NX = 50
N_REAL_TRAIN = 30
N_REAL_VAL = 8
EPOCHS = 15
BATCH = 64
STRENGTHS = [0.0, 1.0, 3.0]
SEEDS = [0, 1, 2]                 # repeat each strength to get mean ± spread
SPLIT = Path("data/splits_diurnal")


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# normalized coordinate grid in [-1,1]
_yy, _xx = np.meshgrid(np.linspace(-1, 1, NY), np.linspace(-1, 1, NX), indexing="ij")
COORD = torch.from_numpy(np.stack([_yy, _xx]).astype(np.float32)).to(DEVICE)   # (2,50,50)


class CoordCNN(nn.Module):
    """SpatioTemporalCNN with 2 extra (scaled) coordinate input channels."""
    def __init__(self, strength, base_ch=BASE_CH, k=K):
        super().__init__()
        self.strength = strength
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, base_ch // 2, (k, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_ch // 2), nn.GELU())
        self.res1 = ResBlock2D(base_ch // 2, base_ch)
        self.res2 = ResBlock2D(base_ch, base_ch)
        self.res3 = ResBlock2D(base_ch, base_ch)
        self.head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 4, 1, bias=False), nn.GELU(),
            nn.Conv2d(base_ch // 4, 1, 1))

    def forward(self, x):                                   # x (B,1,K,H,W)
        B, _, k, H, Wd = x.shape
        c = (self.strength * COORD).view(1, 2, 1, H, Wd).expand(B, 2, k, H, Wd)
        h = self.conv3d(torch.cat([x, c], 1)).squeeze(2)
        h = self.res3(self.res2(self.res1(h)))
        return self.head(h)


def load_windows(split, n_real):
    files = sorted(glob.glob(str(SPLIT / split / "realisation_*.npz")))[:n_real]
    xs, ys = [], []
    for f in files:
        obs = np.load(f)["observations"].astype(np.float32)   # (T,50,50)
        fr = torch.from_numpy(obs).unsqueeze(1)               # (T,1,50,50)
        for i in range(len(fr) - K):
            xs.append(fr[i:i+K].permute(1, 0, 2, 3)); ys.append(fr[i+K])
    return torch.stack(xs), torch.stack(ys)


class TopKSAE(nn.Module):
    def __init__(self, d=256, n=512, k=25):
        super().__init__(); self.k = k
        self.encoder = nn.Linear(d, n); self.decoder = nn.Linear(n, d)
        with torch.no_grad():
            nn.init.normal_(self.decoder.weight, std=1/d**0.5)
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, 0)
            self.encoder.weight.data = self.decoder.weight.data.T.clone()
        nn.init.zeros_(self.encoder.bias); nn.init.zeros_(self.decoder.bias)
    def encode(self, x):
        pre = self.encoder(x); v, i = torch.topk(pre, self.k, -1)
        a = torch.zeros_like(pre); a.scatter_(-1, i, F.relu(v)); return a
    def forward(self, x): a = self.encode(x); return a, self.decoder(a)
    @torch.no_grad()
    def norm_dec(self): self.decoder.weight.data = F.normalize(self.decoder.weight.data, 0)


def spatial_probe(model, seed=0, return_full=False):
    """Extract per-pixel res3 acts on val realisations, train SAE, score."""
    set_seed(seed)
    cap = {}
    h = model.res3.register_forward_hook(lambda m, i, o: cap.__setitem__("a", o))
    files = sorted(glob.glob(str(SPLIT / "val" / "realisation_*.npz")))[:N_REAL_VAL]
    W0 = np.load(files[0])["W"].astype(np.float32).reshape(8, NY, NX)
    mode_map = np.where(np.abs(W0).sum(0) > 0, np.abs(W0).argmax(0), -1)
    rng = np.random.default_rng(seed)
    X, ys_, xs_, cont, mid = [], [], [], [], []
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
    X = np.concatenate(X).astype(np.float32)
    ys_ = np.concatenate(ys_); xs_ = np.concatenate(xs_)
    cont = np.concatenate(cont).astype(np.float32); mid = np.concatenate(mid)
    N = len(X); mean, std = X.mean(0), X.std(0) + 1e-8
    Xn = torch.from_numpy(((X - mean) / std).astype(np.float32)).to(DEVICE)
    sae = TopKSAE().to(DEVICE); opt = torch.optim.Adam(sae.parameters(), 1e-3)
    for ep in range(25):
        perm = torch.randperm(N, device=DEVICE)
        for i in range(N // 512):
            b = Xn[perm[i*512:(i+1)*512]]
            _, r = sae(b); loss = F.mse_loss(r, b)
            opt.zero_grad(); loss.backward(); opt.step(); sae.norm_dec()
    feats = np.empty((N, 512), np.float32)
    with torch.no_grad():
        for i in range(0, N, 8192):
            feats[i:i+8192] = sae.encode(Xn[i:i+8192]).cpu().numpy()
    pix = ys_ * NX + xs_; cnt = np.bincount(pix, minlength=NY*NX).astype(np.float64)
    grand = feats.mean(0); sum_pp = np.zeros((NY*NX, 512)); np.add.at(sum_pp, pix, feats)
    mu = sum_pp / np.maximum(cnt[:, None], 1)
    sb = (cnt[:, None] * (mu - grand)**2).sum(0); st = ((feats - grand)**2).sum(0) + 1e-12
    pos_R2 = sb / st
    # content R2 on [1, cont, cont^2] only — mode one-hot dropped (it was a
    # position proxy; verified not to suppress content, see refit_content_r2.py).
    D = np.concatenate([np.ones((N, 1), np.float32), cont[:, None], (cont**2)[:, None]], 1)
    coef, *_ = np.linalg.lstsq(D, feats, rcond=None)
    cont_R2 = 1 - ((feats - D @ coef)**2).sum(0) / st
    alive = (feats > 0).mean(0) > 1e-4
    bord = np.zeros((NY, NX), bool); bord[:3] = bord[-3:] = True; bord[:, :3] = bord[:, -3:] = True
    muc = mu - mu.mean(0); em = muc**2
    br = (em[bord.ravel()].sum(0) / (em.sum(0) + 1e-12)) / bord.mean()
    gl = alive & (pos_R2 > 0.30) & (pos_R2 > 1.5*cont_R2) & (br > 1.5)
    ct = alive & (cont_R2 > 0.30) & (cont_R2 > pos_R2)
    # variance concentration: per-feature total SS (st) ∝ variance
    var = np.maximum(st - 1e-12, 0.0); order = np.argsort(var)[::-1]; tv = var.sum()
    pr = float(var.sum()**2 / (var**2).sum())   # participation ratio = effective #features
    out = dict(n_gridlocked=int(gl.sum()), median_pos=float(np.median(pos_R2[gl])) if gl.any() else 0.0,
               n_content=int(ct.sum()), max_cont=float(cont_R2[alive].max()),
               top1_var=float(var[order[0]] / tv), top5_var=float(var[order[:5]].sum() / tv),
               part_ratio=pr, top5_gl=int(gl[order[:5]].sum()))
    if return_full:
        out.update(st=st, var=var, order=order, pos_R2=pos_R2, cont_R2=cont_R2,
                   alive=alive, gl=gl, ct=ct)
    return out


def train_cnn(strength, Xtr, Ytr, Xva, Yva):
    model = CoordCNN(strength).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    n = len(Xtr)
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(n)
        for i in range(n // BATCH):
            idx = perm[i*BATCH:(i+1)*BATCH]
            xb, yb = Xtr[idx].to(DEVICE), Ytr[idx].to(DEVICE)
            pred = model(xb); loss = F.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        se = tot = 0.0
        for i in range(0, len(Xva), 256):
            pred = model(Xva[i:i+256].to(DEVICE))
            se += ((pred - Yva[i:i+256].to(DEVICE))**2).sum().item(); tot += pred.numel()
    return model, (se / tot) ** 0.5


def main():
    print("Loading windows ...")
    Xtr, Ytr = load_windows("train", N_REAL_TRAIN)
    Xva, Yva = load_windows("val", N_REAL_VAL)
    print(f"  train {Xtr.shape}  val {Xva.shape}")

    METRICS = ["val_rmse", "n_gridlocked", "median_pos", "n_content", "max_cont"]
    rows = []
    for s in STRENGTHS:
        per_seed = []
        for seed in SEEDS:
            t0 = time.time()
            set_seed(seed)
            model, rmse = train_cnn(s, Xtr, Ytr, Xva, Yva)
            probe = spatial_probe(model, seed)
            per_seed.append(dict(seed=seed, val_rmse=rmse, **probe))
            print(f"  strength={s} seed={seed}:  valRMSE={rmse:.4f}  "
                  f"grid-locked={probe['n_gridlocked']} (med pos_R2={probe['median_pos']:.2f})  "
                  f"content={probe['n_content']} (max contR2={probe['max_cont']:.2f})   [{time.time()-t0:.0f}s]")
        agg = dict(strength=s, per_seed=per_seed)
        for m in METRICS:
            vals = np.array([r[m] for r in per_seed], float)
            agg[m + "_mean"] = float(vals.mean())
            agg[m + "_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(agg)

    np.save("results/physics_vs_position_tradeoff_sweep.npy", rows)
    print(f"\n══ PHYSICS-VS-POSITION TRADE-OFF SWEEP  (mean ± sd over {len(SEEDS)} seeds) ══")
    print(f"  {'strength':>8} {'valRMSE':>16} {'#gridlock':>14} {'medPosR2':>14} {'#content':>14} {'maxContR2':>14}")
    for r in rows:
        def ms(m): return f"{r[m+'_mean']:.3f}±{r[m+'_std']:.3f}"
        print(f"  {r['strength']:>8.1f} {ms('val_rmse'):>16} {ms('n_gridlocked'):>14} "
              f"{ms('median_pos'):>14} {ms('n_content'):>14} {ms('max_cont'):>14}")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        s = [r["strength"] for r in rows]
        def mean(m): return [r[m + "_mean"] for r in rows]
        def std(m):  return [r[m + "_std"] for r in rows]
        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        ax[0].errorbar(s, mean("val_rmse"), yerr=std("val_rmse"), fmt="o-", capsize=4)
        ax[0].set_title("val RMSE"); ax[0].set_xlabel("coord strength")
        ax[1].errorbar(s, mean("n_gridlocked"), yerr=std("n_gridlocked"), fmt="o-", capsize=4, label="#grid-locked")
        ax[1].errorbar(s, mean("n_content"), yerr=std("n_content"), fmt="s-", capsize=4, label="#content")
        ax[1].legend(); ax[1].set_title("feature counts"); ax[1].set_xlabel("coord strength")
        ax[2].errorbar(s, mean("max_cont"), yerr=std("max_cont"), fmt="o-", capsize=4)
        ax[2].set_title("max content_R2 (physics survival)"); ax[2].set_xlabel("coord strength")
        fig.tight_layout(); fig.savefig("figures/physics_vs_position_tradeoff_sweep.png", dpi=120)
        print("Saved figure → figures/physics_vs_position_tradeoff_sweep.png")
    except Exception as e:
        print(f"(figure skipped: {e})")
    print("Saved → results/physics_vs_position_tradeoff_sweep.npy")


if __name__ == "__main__":
    main()
