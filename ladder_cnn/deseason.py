"""The deseasonalisation contrast, re-derived ON THE LADDER (PREREG §6).

Known from the repo: on oracle features raw F1 0.351 -> deseason 0.570; on true
Z 0.293 -> 0.825. Question: does deseasonalisation rescue the UNSUPERVISED
rungs, or only the oracle ones?

`sae_data/diurnal*` ship per-mode SAEs but no mixed SAE, so one is trained here
on CPU with the repo's hyperparameters (256->512, TopK K=25) and cached in the
scratchpad. ~/savar-project stays read-only.
"""
import sys, time, argparse
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import torch
from scipy.stats import skew
from common import *
from common import _init_worker

ap = argparse.ArgumentParser()
ap.add_argument("--n_real", type=int, default=25)
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--workers", type=int, default=4)
A = ap.parse_args()
torch.set_num_threads(2)

G, gt = load_gt()   # verified identical for realisations_diurnal
Gd = np.load(ROOT / "data/realisations_diurnal/realisation_000.npz")["ground_truth_graph"]
assert np.array_equal(Gd.astype(np.float64), G), "GATE FAIL: diurnal graph differs"
MAP_ID = {j: j for j in range(N_MODES)}
Rn = A.n_real
OUTALL = {}


def train_mixed(acts, tag):
    p = OUT / f"mixed_sae_{tag}.pt"
    if p.exists():
        print(f"  [mixed SAE cached] {p}")
        return load_sae(p)
    X = acts.reshape(-1, INPUT_DIM)
    sub = X[np.random.default_rng(0).choice(len(X), min(len(X), 400000),
                                            replace=False)]
    mu, sd = sub.mean(0), sub.std(0)
    Xt = torch.from_numpy(((sub - mu) / (sd + 1e-8)).astype(np.float32))
    sae = TopKSAE()
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    nval = int(0.15 * len(Xt))
    perm = torch.randperm(len(Xt), generator=torch.Generator().manual_seed(0))
    tr, va = Xt[perm[nval:]], Xt[perm[:nval]]
    best, bsd = np.inf, None
    t0 = time.time()
    for ep in range(A.epochs):
        o = torch.randperm(len(tr))
        for i in range(0, len(tr) - 256, 256):
            xb = tr[o[i:i + 256]]
            loss = ((sae.decoder(sae.encode(xb)) - xb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                sae.decoder.weight.data /= (
                    sae.decoder.weight.data.norm(dim=0, keepdim=True) + 1e-8)
        with torch.no_grad():
            vm = float(((sae.decoder(sae.encode(va)) - va) ** 2).mean())
        if vm < best:
            best, bsd = vm, {k: v.clone() for k, v in sae.state_dict().items()}
        print(f"    mixed[{tag}] ep{ep} val {vm:.4f} ({time.time()-t0:.0f}s)")
    sae.load_state_dict(bsd); sae.eval()
    torch.save(dict(model_state=sae.state_dict(), act_mean=mu, act_std=sd,
                    val_mse=best), p)
    return sae, mu, sd


def signflip(S):
    return S * np.array([float(np.sign(skew(S[..., j].ravel()))) or 1.0
                         for j in range(S.shape[-1])])


for tag, sdir in (("diurnal", "sae_data/diurnal"),
                  ("deseason", "sae_data/diurnal_deseason")):
    print(f"\n{'='*74}\n### {tag}  ({sdir})")
    SD = ROOT / sdir
    acts = np.asarray(np.load(SD / "activations_full.npy", mmap_mode="r")[:Rn])
    Zf = np.asarray(np.load(SD / "Z_full.npy")[:Rn])
    align = np.load(SD / "alignment_per_mode.npy", allow_pickle=True).item()
    assert np.isfinite(acts).all() and np.isfinite(Zf).all(), "GATE FAIL"
    Rn_, _, T_eff, _ = acts.shape
    print(f"  acts {acts.shape}")
    res = {}

    # true-Z anchor
    r = run_ladder_rung([Zf[i].T.astype(np.float64) for i in range(Rn_)],
                        MAP_ID, gt, workers=A.workers)
    res["trueZ"] = r["f1"].mean(), r["precision"].mean(), r["recall"].mean()
    print(f"  trueZ   F1={r['f1'].mean():.4f} P={r['precision'].mean():.4f} "
          f"R={r['recall'].mean():.4f}")

    code_pm = np.empty((Rn_, N_MODES, T_eff, N_FEATURES), np.float32)
    for j in range(N_MODES):
        s, mu, sdv = load_sae(SD / f"sae_mode_{j}.pt")
        code_pm[:, j] = encode_block(s, acts[:, j].reshape(-1, INPUT_DIM), mu, sdv
                                     ).reshape(Rn_, T_eff, N_FEATURES)
    bf = [int(align[j]["best_feat"]) for j in range(N_MODES)]
    sg = [float(np.sign(align[j]["C_j"][bf[j]])) or 1.0 for j in range(N_MODES)]
    S0 = np.stack([sg[j] * code_pm[:, j, :, bf[j]] for j in range(N_MODES)], -1)
    r = run_ladder_rung(list(S0), MAP_ID, gt, workers=A.workers)
    res["R0"] = r["f1"].mean(), r["precision"].mean(), r["recall"].mean()
    print(f"  R0      F1={r['f1'].mean():.4f} P={r['precision'].mean():.4f} "
          f"R={r['recall'].mean():.4f}  (oracle feature, |r|="
          f"{np.round([align[j]['max_r'] for j in range(8)],2).tolist()})")

    for rank_by in ("variance", "freq", "pc1"):
        picks = [sel_var(code_pm[:, j], n_max=1, rank_by=rank_by)[0]
                 for j in range(N_MODES)]
        S = signflip(np.stack([code_pm[:, j, :, picks[j]]
                               for j in range(N_MODES)], -1))
        r = run_ladder_rung(list(S), MAP_ID, gt, workers=A.workers)
        nm = "R2" if rank_by == "variance" else f"R2_{rank_by}"
        res[nm] = r["f1"].mean(), r["precision"].mean(), r["recall"].mean()
        print(f"  {nm:<8}F1={r['f1'].mean():.4f} P={r['precision'].mean():.4f} "
              f"R={r['recall'].mean():.4f}")

    sae, mu, sdv = train_mixed(acts, tag)
    code_mx = encode_block(sae, acts.reshape(-1, INPUT_DIM), mu, sdv
                           ).reshape(Rn_, N_MODES, T_eff, N_FEATURES)
    picks_a = [sel_var(code_mx[:, j], n_max=1)[0] for j in range(N_MODES)]
    S = signflip(np.stack([code_mx[:, j, :, picks_a[j]]
                           for j in range(N_MODES)], -1))
    r = run_ladder_rung(list(S), MAP_ID, gt, workers=A.workers)
    res["R3a"] = r["f1"].mean(), r["precision"].mean(), r["recall"].mean()
    print(f"  R3a     F1={r['f1'].mean():.4f} P={r['precision'].mean():.4f} "
          f"R={r['recall'].mean():.4f}  picks={picks_a} "
          f"distinct={len(set(picks_a))}")

    pooled = code_mx.mean(1)
    del code_mx, code_pm
    chosen = sel_var(pooled, n_max=12)
    S = pooled[:, :, chosen]
    mp, M = map_r(S.reshape(-1, S.shape[-1]),
                  Zf.transpose(0, 2, 1).reshape(-1, N_MODES))
    r = run_ladder_rung(list(S), mp, gt, workers=A.workers)
    res["R3b"] = r["f1"].mean(), r["precision"].mean(), r["recall"].mean()
    print(f"  R3b     F1={r['f1'].mean():.4f} P={r['precision'].mean():.4f} "
          f"R={r['recall'].mean():.4f}  N_hat={len(chosen)} matched={len(mp)}/8 "
          f"maxr={np.round(M.max(1),2).tolist()}")
    OUTALL[tag] = res
    del acts, pooled

print(f"\n{'='*74}\n{'rung':<10}{'diurnal(raw)':>15}{'deseason':>12}{'delta':>10}")
for k in ["trueZ", "R0", "R2", "R2_freq", "R2_pc1", "R3a", "R3b"]:
    a = OUTALL["diurnal"][k][0]; b = OUTALL["deseason"][k][0]
    print(f"{k:<10}{a:>15.4f}{b:>12.4f}{b-a:>+10.4f}")
np.save(OUT / "deseason.npy", OUTALL, allow_pickle=True)
print(f"\nsaved -> {OUT}/deseason.npy")
