"""Rungs R0-R3 of the oracle-ablation ladder. Protocol frozen in PREREG.md.

R0  replication of pcmci/run_pcmci_features.py            (expect F1 0.695)
R1  - oracle sign        (sign = sign(skew), no Z)
R2  - oracle feature sel (argmax variance per mode, no Z)
R3a - oracle per-mode dictionary (mixed SAE, one feat per stream, MAP-ID)
R3b - oracle N + mode partition (mixed SAE, stream-pooled, SEL-VAR, MAP-R)
      <- THE GRAPHCAST-MATCHED RUNG
"""
import sys, time, argparse, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
from scipy.stats import skew
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--sae_dir", default="sae_data/base")
ap.add_argument("--tag", default="base")
ap.add_argument("--n_real", type=int, default=100)
ap.add_argument("--workers", type=int, default=4)
A = ap.parse_args()

SAE_DIR = ROOT / A.sae_dir
Rn = A.n_real
G, gt = load_gt()
MAP_ID = {j: j for j in range(N_MODES)}

print(f"# ladder R0-R3   sae_dir={A.sae_dir}  n_real={Rn}")
acts = np.asarray(np.load(SAE_DIR / "activations_full.npy", mmap_mode="r")[:Rn])
Zf = np.asarray(np.load(SAE_DIR / "Z_full.npy")[:Rn])          # (R, 8, T)
align = np.load(SAE_DIR / "alignment_per_mode.npy", allow_pickle=True).item()
Rn, _, T_eff, _ = acts.shape
print(f"  acts {acts.shape}  T_eff={T_eff}")
assert np.isfinite(acts).all() and np.isfinite(Zf).all(), "GATE FAIL: non-finite"

RES = {}
SPARSE = {}


def sparsity_report(name, S):
    """S (R, T, C). Zero fraction per variable + pairwise co-firing vs nominal T."""
    Rr, T, C = S.shape
    nz = (S != 0).mean(axis=(0, 1))
    co = np.zeros((C, C))
    for r in range(Rr):
        B = (S[r] != 0)
        co += B.T.astype(np.float64) @ B
    co /= Rr
    off = co[~np.eye(C, dtype=bool)]
    d = dict(zero_frac_mean=float(1 - nz.mean()), zero_frac_min=float(1 - nz.max()),
             zero_frac_max=float(1 - nz.min()), nz_per_var=nz.tolist(),
             cofire_mean=float(off.mean()), cofire_min=float(off.min()),
             cofire_max=float(off.max()), T_nominal=T,
             n_eff_ratio=float(off.mean() / T))
    SPARSE[name] = d
    print(f"    sparsity[{name}]: zero-frac {1-nz.mean():.3f} "
          f"(range {1-nz.max():.3f}-{1-nz.min():.3f})   "
          f"co-fire {off.mean():.0f}/{T} = n_eff/T {off.mean()/T:.3f} "
          f"(min {off.min():.0f}, max {off.max():.0f})")
    return d


def score(name, series, mapping, note=""):
    t0 = time.time()
    r = run_ladder_rung(list(series), mapping, gt, workers=A.workers, keep_vm=True)
    r["note"] = note
    r["n_matched"] = len(mapping)
    RES[name] = r
    print(f"  {name:<6} F1={r['f1'].mean():.4f}+-{r['f1'].std():.3f}  "
          f"P={r['precision'].mean():.4f}  R={r['recall'].mean():.4f}  "
          f"TP/FP/FN={r['tp'].mean():.1f}/{r['fp'].mean():.1f}/{r['fn'].mean():.1f}  "
          f"matched={len(mapping)}/8  [{time.time()-t0:.0f}s]  {note}")
    return r


# ── encode per-mode SAEs ──────────────────────────────────────────────────────
print("\n[encode] per-mode SAEs")
code_pm = np.empty((Rn, N_MODES, T_eff, N_FEATURES), np.float32)
best_feat, sign_oracle = [], []
for j in range(N_MODES):
    sae, mu, sd = load_sae(SAE_DIR / f"sae_mode_{j}.pt")
    code_pm[:, j] = encode_block(sae, acts[:, j].reshape(-1, INPUT_DIM), mu, sd
                                 ).reshape(Rn, T_eff, N_FEATURES)
    bf = int(align[j]["best_feat"])
    best_feat.append(bf)
    sign_oracle.append(float(np.sign(align[j]["C_j"][bf])) or 1.0)
print(f"  oracle best_feat = {best_feat}")
print(f"  oracle sign      = {sign_oracle}")

# ── R0 ────────────────────────────────────────────────────────────────────────
print("\n[R0] replication: per-mode SAE + oracle feature (max|r| vs Z) + oracle sign")
S0 = np.stack([sign_oracle[j] * code_pm[:, j, :, best_feat[j]]
               for j in range(N_MODES)], axis=-1)          # (R, T, 8)
sparsity_report("R0", S0)
score("R0", S0, MAP_ID, "oracle: pooling+N+per-mode SAE+feature+sign")

# ── R1: drop oracle sign ──────────────────────────────────────────────────────
print("\n[R1] drop oracle SIGN (sign = sign(skew of series), no Z)")
sign_unsup = []
for j in range(N_MODES):
    s = code_pm[:, j, :, best_feat[j]].ravel()
    sg = float(np.sign(skew(s))) or 1.0
    sign_unsup.append(sg)
print(f"  unsupervised sign = {sign_unsup}   (oracle was {sign_oracle})")
S1 = np.stack([sign_unsup[j] * code_pm[:, j, :, best_feat[j]]
               for j in range(N_MODES)], axis=-1)
score("R1", S1, MAP_ID, "oracle sign removed")

# ── R2: drop oracle feature selection ─────────────────────────────────────────
print("\n[R2] drop oracle FEATURE SELECTION (argmax variance per mode, no Z)")
R2_FEAT = {}
for rank_by in ("variance", "freq", "pc1"):
    picks = []
    for j in range(N_MODES):
        c = sel_var(code_pm[:, j], n_max=1, rank_by=rank_by)
        picks.append(c[0])
    R2_FEAT[rank_by] = picks
    S = np.stack([code_pm[:, j, :, picks[j]] for j in range(N_MODES)], axis=-1)
    S = S * np.array([float(np.sign(skew(S[..., j].ravel()))) or 1.0
                      for j in range(N_MODES)])
    nm = "R2" if rank_by == "variance" else f"R2_{rank_by}"
    if rank_by == "variance":
        sparsity_report("R2", S)
    # how good were the unsupervised picks, measured post hoc against Z?
    rs = []
    for j in range(N_MODES):
        a = S[..., j].ravel(); b = Zf[:, j].ravel()
        rs.append(abs(np.corrcoef(a, b)[0, 1]) if a.std() > 0 else 0.0)
    score(nm, S, MAP_ID,
          f"rank={rank_by} picks={picks} |r|vsZ={np.round(rs,2).tolist()}")

# ── mixed SAE ─────────────────────────────────────────────────────────────────
print("\n[encode] MIXED SAE (sae_best.pt) — one dictionary, no mode labels")
sae, mu, sd = load_sae(SAE_DIR.parent / "base" / "sae_best.pt"
                       if not (SAE_DIR / "sae_best.pt").exists()
                       else SAE_DIR / "sae_best.pt")
MIXED_SRC = str(SAE_DIR / "sae_best.pt") if (SAE_DIR / "sae_best.pt").exists() \
    else str(SAE_DIR.parent / "base" / "sae_best.pt")
print(f"  mixed SAE = {MIXED_SRC}")
code_mx = encode_block(sae, acts.reshape(-1, INPUT_DIM), mu, sd
                       ).reshape(Rn, N_MODES, T_eff, N_FEATURES)
alive_mx = ((code_mx != 0).reshape(-1, N_FEATURES).mean(0) >= 0.02).sum()
print(f"  live features (nz>=2%): {alive_mx}/512")

# ── R3a: mixed SAE, per-stream, MAP-ID ───────────────────────────────────────
print("\n[R3a] drop oracle PER-MODE DICTIONARY (mixed SAE, argmax-var per stream)")
picks_a = [sel_var(code_mx[:, j], n_max=1, rank_by="variance")[0]
           for j in range(N_MODES)]
print(f"  per-stream picks = {picks_a}  (distinct: {len(set(picks_a))})")
S3a = np.stack([code_mx[:, j, :, picks_a[j]] for j in range(N_MODES)], axis=-1)
S3a = S3a * np.array([float(np.sign(skew(S3a[..., j].ravel()))) or 1.0
                      for j in range(N_MODES)])
sparsity_report("R3a", S3a)
rs_a = [abs(np.corrcoef(S3a[..., j].ravel(), Zf[:, j].ravel())[0, 1])
        for j in range(N_MODES)]
score("R3a", S3a, MAP_ID,
      f"picks={picks_a} distinct={len(set(picks_a))} |r|vsZ={np.round(rs_a,2).tolist()}")

# ── R3b: THE GRAPHCAST-MATCHED RUNG ──────────────────────────────────────────
print("\n[R3b] drop oracle N + MODE PARTITION  <<< GRAPHCAST-MATCHED >>>")
print("      series_f[r,t] = mean over the 8 streams of a[r,j,t,f]  (512 candidates)")
pooled = code_mx.mean(axis=1)                              # (R, T, 512)
del code_mx
R3B = {}
for rank_by in ("variance", "freq", "pc1"):
    chosen = sel_var(pooled, n_max=12, rank_by=rank_by)
    S = pooled[:, :, chosen]                               # (R, T, N_hat)
    flatS = S.reshape(-1, S.shape[-1])
    flatZ = Zf.transpose(0, 2, 1).reshape(-1, N_MODES)
    mp, M = map_r(flatS, flatZ)
    nm = "R3b" if rank_by == "variance" else f"R3b_{rank_by}"
    if rank_by == "variance":
        sparsity_report("R3b", S)
        R3B["chosen"] = chosen
        R3B["map_matrix"] = M
        R3B["mapping"] = mp
        R3B["pooled_selected"] = S
    print(f"  rank={rank_by}: N_hat={len(chosen)} chosen={chosen}")
    print(f"    MAP-R best |r| per discovered var: "
          f"{np.round(M.max(1), 3).tolist()}")
    print(f"    matched {len(mp)}/8 true modes: {mp}")
    score(nm, S, mp, f"rank={rank_by} N_hat={len(chosen)} matched={len(mp)}/8")

np.save(OUT / f"ladder_r0_r3_{A.tag}.npy",
        dict(res={k: {kk: vv for kk, vv in v.items() if kk != "val_matrices"}
                  for k, v in RES.items()},
             sparsity=SPARSE, best_feat=best_feat, sign_oracle=sign_oracle,
             sign_unsup=sign_unsup, r2_feat=R2_FEAT, picks_r3a=picks_a,
             r3b_chosen=R3B.get("chosen"), r3b_map=R3B.get("mapping"),
             r3b_map_matrix=R3B.get("map_matrix"),
             mixed_src=MIXED_SRC, n_real=Rn, gt=sorted(gt),
             trueZ_anchor=0.8249),
        allow_pickle=True)
# the selected R3b series are needed by the null + FP-structure scripts
np.save(OUT / f"r3b_series_{A.tag}.npy",
        dict(series=R3B["pooled_selected"], chosen=R3B["chosen"],
             mapping=R3B["mapping"], map_matrix=R3B["map_matrix"]),
        allow_pickle=True)
print(f"\nsaved -> {OUT}/ladder_r0_r3_{A.tag}.npy")

print("\n" + "=" * 74)
print(f"{'rung':<12}{'F1':>8}{'P':>8}{'R':>8}{'matched':>9}")
for k, v in RES.items():
    print(f"{k:<12}{v['f1'].mean():>8.4f}{v['precision'].mean():>8.4f}"
          f"{v['recall'].mean():>8.4f}{v['n_matched']:>7}/8")
print(f"{'trueZ':<12}{0.8249:>8.4f}{0.7152:>8.4f}{0.9842:>8.4f}{8:>7}/8")
