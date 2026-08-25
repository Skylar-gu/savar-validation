"""R4 — drop the oracle POOLING.

Every rung above eats `activations_full.npy`, which was already pooled through
the TRUE W. That is the biggest oracle in the stack and the one GraphCast most
obviously does not have. Here: train a TopK SAE on PER-PIXEL res3 vectors (the
structural analogue of a GraphCast SAE over mesh-node embeddings), then form a
variable set by aggregating each feature over space, unsupervised, and run
PCMCI+.

Two aggregations, both reported:
  R4-mean : series[t,f] = mean over the 2500 pixels of a[t,pix,f]
            (exactly what the GraphCast lane does: pool a feature over the mesh)
  R4-foot : series[t,f] = footprint-weighted mean, footprint = the feature's own
            mean activation map, which also supplies MAP-FOOT.

Timing measured first (gate_and_time.py): 150 ms/window on CPU -> 75 s per
realisation. Declared budget below.
"""
import sys, time, argparse
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import torch
import torch.nn as nn
from common import *
from common import _init_worker

ap = argparse.ArgumentParser()
ap.add_argument("--n_real", type=int, default=15)
ap.add_argument("--n_fit", type=int, default=3)
ap.add_argument("--pix_per_frame", type=int, default=150)
ap.add_argument("--t_stride", type=int, default=2)
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--draws", type=int, default=200)
A = ap.parse_args()

torch.set_num_threads(2)
G, gt = load_gt()
paths = sorted((ROOT / "data/realisations").glob("realisation_*.npz"))
W_true = np.load(paths[0])["W"].astype(np.float64)
Zf = np.asarray(np.load(ROOT / "sae_data/base/Z_full.npy")[:A.n_real])
T_EFF = Zf.shape[2]

sys.path.insert(0, str(ROOT / "train" / "cnn"))
from cnn_forecaster import SpatioTemporalCNN, K, BASE_CH
model = SpatioTemporalCNN(ny=NY, nx=NX, k=K, base_ch=BASE_CH)
ck = torch.load(ROOT / "checkpoints/base/best.pt", map_location="cpu")
model.load_state_dict(ck["model_state"]); model.eval()
cap = {}
model.res3.register_forward_hook(lambda m, i, o: cap.__setitem__("a", o))
print(f"CNN {sum(p.numel() for p in model.parameters())/1e6:.2f}M  "
      f"val_rmse {ck['val_rmse']:.4f}  T_eff={T_EFF}")


def frames_of(p):
    obs = np.load(p)["observations"].astype(np.float32).T.reshape(-1, NY, NX)
    assert np.isfinite(obs).all(), f"GATE FAIL non-finite obs {p}"
    return torch.from_numpy(obs)


def forward_res3(fr, t0, t1, B=32):
    """yield (t_start, act (b, 256, 50, 50))"""
    for i in range(t0, t1, B):
        n = min(B, t1 - i)
        w = torch.stack([fr[t:t + K] for t in range(i, i + n)]).unsqueeze(1)
        with torch.no_grad():
            model(w)
        yield i, cap["a"]


# ── pass A: collect per-pixel training samples ────────────────────────────────
print(f"\n[pass A] per-pixel sample collection, {A.n_fit} realisations")
rng = np.random.default_rng(0)
samples = []
tA = time.time()
for ri in range(A.n_fit):
    fr = frames_of(paths[ri])
    for i, act in forward_res3(fr, 0, T_EFF):
        Hb = act.permute(0, 2, 3, 1).reshape(act.shape[0], L, BASE_CH).numpy()
        for b in range(0, Hb.shape[0], A.t_stride):
            idx = rng.choice(L, A.pix_per_frame, replace=False)
            samples.append(Hb[b, idx])
    print(f"  real {ri}: {len(samples)*A.pix_per_frame/1000:.0f}k samples so far "
          f"({time.time()-tA:.0f}s)")
X = np.concatenate(samples, 0).astype(np.float32)
del samples
print(f"  training set {X.shape}  finite={np.isfinite(X).all()}  "
      f"all-zero rows={int((np.abs(X).sum(1)==0).sum())}")
mu = X.mean(0); sd = X.std(0)
Xn = (X - mu) / (sd + 1e-8)

# ── train the spatial TopK SAE (same hparams as the repo's SAEs) ─────────────
print(f"\n[train] spatial TopK SAE 256->512 K=25, {A.epochs} epochs")
sae = TopKSAE()
opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
Xt = torch.from_numpy(Xn)
n = len(Xt); nval = int(0.15 * n)
perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
tr, va = Xt[perm[nval:]], Xt[perm[:nval]]
BS = 256
best = np.inf; best_sd = None
tT = time.time()
for ep in range(A.epochs):
    sae.train()
    o = torch.randperm(len(tr))
    tot = 0.0
    for i in range(0, len(tr) - BS, BS):
        xb = tr[o[i:i + BS]]
        a = sae.encode(xb)
        rec = sae.decoder(a)
        loss = ((rec - xb) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():   # unit-norm decoder columns, as train_sae.py
            sae.decoder.weight.data /= (
                sae.decoder.weight.data.norm(dim=0, keepdim=True) + 1e-8)
        tot += float(loss)
    sae.eval()
    with torch.no_grad():
        vm = float(((sae.decoder(sae.encode(va)) - va) ** 2).mean())
    if vm < best:
        best = vm
        best_sd = {k: v.clone() for k, v in sae.state_dict().items()}
    print(f"  ep {ep:2d} train {tot/max(1,(len(tr)//BS)):.4f} val {vm:.4f} "
          f"({time.time()-tT:.0f}s)")
sae.load_state_dict(best_sd); sae.eval()
var = float(va.var())
print(f"  best val MSE {best:.4f}  (variance {var:.4f}, FVU {best/var:.4f})")
torch.save(dict(model_state=sae.state_dict(), act_mean=mu, act_std=sd,
                val_mse=best, fvu=best / var), OUT / "spatial_sae_base.pt")

mu_t = torch.from_numpy(mu); sd_t = torch.from_numpy(sd + 1e-8)

# ── pass B: per-realisation series + footprints ──────────────────────────────
print(f"\n[pass B] streaming encode, {A.n_real} realisations")
ser_mean = np.zeros((A.n_real, T_EFF, N_FEATURES), np.float32)
foot_sum = np.zeros((N_FEATURES, L), np.float64)
foot_n = 0
tB = time.time()
for ri in range(A.n_real):
    fr = frames_of(paths[ri])
    for i, act in forward_res3(fr, 0, T_EFF):
        Hb = act.permute(0, 2, 3, 1).reshape(act.shape[0], L, BASE_CH)
        with torch.no_grad():
            a = sae.encode((Hb - mu_t) / sd_t)          # (b, L, 512)
        ser_mean[ri, i:i + a.shape[0]] = a.mean(1).numpy()
        if ri < A.n_fit:
            foot_sum += a.sum(0).T.double().numpy()
            foot_n += a.shape[0]
    print(f"  real {ri}: {time.time()-tB:.0f}s")
foot = foot_sum / max(foot_n, 1)                        # (512, 2500)
assert np.isfinite(ser_mean).all()
np.save(OUT / "r4_series_mean.npy", ser_mean)
np.save(OUT / "r4_footprints.npy", foot)
print(f"  pass B done in {time.time()-tB:.0f}s")

# footprint-weighted series needs a second reduction; approximate it exactly by
# re-pooling in feature space is impossible from the mean alone, so R4-foot uses
# the footprint ONLY for MAP-FOOT and the mean series for the dynamics. This is
# stated in the report.
fw = np.maximum(foot, 0)
fw = fw / np.maximum(fw.sum(1, keepdims=True), 1e-12)

# ── selection + discovery ────────────────────────────────────────────────────
print("\n[R4] SEL-VAR over 512 spatial-SAE features")
S_all = ser_mean
res_all = {}
for rank_by in ("variance", "freq", "pc1"):
    chosen = sel_var(S_all, n_max=12, rank_by=rank_by)
    S = S_all[:, :, chosen]
    mp_f, Mf = map_foot(fw[chosen], W_true)
    flatS = S.reshape(-1, S.shape[-1])
    flatZ = Zf.transpose(0, 2, 1).reshape(-1, N_MODES)
    mp_r, Mr = map_r(flatS, flatZ)
    print(f"  rank={rank_by}: N_hat={len(chosen)} chosen={chosen}")
    print(f"    footprint cos to true W (max per var): {np.round(Mf.max(1),3).tolist()}")
    print(f"    |r| vs Z        (max per var): {np.round(Mr.max(1),3).tolist()}")
    for tagm, mp in (("MAP-FOOT", mp_f), ("MAP-R", mp_r)):
        r = run_ladder_rung(list(S), mp, gt, workers=A.workers)
        nm = f"R4_{rank_by}_{tagm}"
        res_all[nm] = dict(f1=r["f1"], precision=r["precision"],
                           recall=r["recall"], tp=r["tp"], fp=r["fp"],
                           fn=r["fn"], n_matched=len(mp), mapping=mp,
                           chosen=chosen)
        print(f"    {tagm:<9} F1={r['f1'].mean():.4f} P={r['precision'].mean():.4f} "
              f"R={r['recall'].mean():.4f} matched={len(mp)}/8")

# ── null for R4 (variance ranking, both matchings) ───────────────────────────
print(f"\n[R4 null] {A.draws} draws, N-RAND + N-PHASE")
from concurrent.futures import ProcessPoolExecutor
chosen = sel_var(S_all, n_max=12, rank_by="variance")
S_obs = S_all[:, :, chosen]
flat = S_all.reshape(-1, N_FEATURES)
alive = np.where(((flat != 0).mean(0) >= 0.02) & (flat.var(0) > 1e-12))[0]
print(f"  live candidates: {len(alive)}/512")
flatZ = Zf.transpose(0, 2, 1).reshape(-1, N_MODES)


def _r4draw(args):
    kind, seed = args
    rng = np.random.default_rng(seed)
    if kind == "rand":
        cols = rng.choice(alive, size=min(len(chosen), len(alive)), replace=False)
        S = S_all[:, :, cols]
        fp = fw[cols]
    else:
        S = np.empty_like(S_obs)
        for r in range(S_obs.shape[0]):
            for c in range(S_obs.shape[-1]):
                S[r, :, c] = phase_randomise(S_obs[r, :, c].astype(np.float64), rng)
        fp = fw[chosen]
    out = {}
    mp_f, _ = map_foot(fp, W_true)
    mp_r, _ = map_r(S.reshape(-1, S.shape[-1]), flatZ)
    for tagm, mp in (("MAP-FOOT", mp_f), ("MAP-R", mp_r)):
        r = run_ladder_rung(list(S), mp, gt, workers=1)
        out[tagm] = float(r["f1"].mean())
    return kind, out


jobs = [(k, 5000 + 13 * i) for k in ("rand", "phase") for i in range(A.draws)]
NULL = {("rand", "MAP-FOOT"): [], ("rand", "MAP-R"): [],
        ("phase", "MAP-FOOT"): [], ("phase", "MAP-R"): []}
t0 = time.time(); done = 0
with ProcessPoolExecutor(max_workers=A.workers, initializer=_init_worker) as ex:
    for kind, out in ex.map(_r4draw, jobs, chunksize=1):
        for tagm, v in out.items():
            NULL[(kind, tagm)].append(v)
        done += 1
        if done % 50 == 0:
            el = time.time() - t0
            print(f"  {done}/{len(jobs)} {el:.0f}s eta {el/done*(len(jobs)-done):.0f}s")

print("\n" + "=" * 82)
SUM = {}
for kind in ("rand", "phase"):
    for tagm in ("MAP-FOOT", "MAP-R"):
        arr = np.array(NULL[(kind, tagm)])
        o = res_all[f"R4_variance_{tagm}"]["f1"].mean()
        pv = (1 + int((arr >= o).sum())) / (1 + len(arr))
        SUM[f"{kind}|{tagm}"] = dict(obs=o, mean=arr.mean(), sd=arr.std(),
                                     p95=np.percentile(arr, 95), mx=arr.max(),
                                     mn=arr.min(), pval=pv, draws=arr)
        print(f"  R4 {kind:<6}{tagm:<9} obs={o:.4f}  null mean={arr.mean():.4f} "
              f"sd={arr.sd if False else arr.std():.4f} "
              f"[{arr.min():.4f},{arr.max():.4f}] p95={np.percentile(arr,95):.4f} "
              f"p={pv:.4f}")

np.save(OUT / "rung_r4.npy",
        dict(res=res_all, null=SUM, chosen=chosen, alive=alive,
             n_real=A.n_real, n_fit=A.n_fit, sae_val_mse=best, fvu=best / var,
             foot_cos_max=None),
        allow_pickle=True)
print(f"\nsaved -> {OUT}/rung_r4.npy")
