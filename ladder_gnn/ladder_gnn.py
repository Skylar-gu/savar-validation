"""GNN port of rungs R0-R3 of the oracle-ablation ladder (protocol: ../savar_sae_pcmci/PREREG.md).

Forecaster: MeshGNN (checkpoints/hetdynamics_eqvar/best.pt), last MP layer H, W-pooled at
STRIDE 1. Dataset: hetdynamics_eqvar (T=2400, 12 true cross edges at lags 1-6, tau_max=6).
SAEs: the existing per-mode sae_mode_j.pt and mixed sae_mixed.pt of sae_data/hetdynamics_eqvar.

trueZ  the ladder's own ceiling: PCMCI on the true latent series, same protocol
R0     replication: per-mode SAE + oracle feature (max|r| vs Z) + oracle sign
R1     - oracle sign            R2  - oracle feature selection
R3a    - oracle per-mode dictionary (mixed SAE, one feat per stream, MAP-ID)
R3b    - oracle N + mode partition (mixed SAE, stream-pooled, SEL-VAR, MAP-R)  <- GRAPHCAST-MATCHED
"""
import sys, time, argparse
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
from scipy.stats import skew
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="gnn_eqvar")
ap.add_argument("--n_real", type=int, default=100)
ap.add_argument("--workers", type=int, default=24)
ap.add_argument("--quick", action="store_true", help="skip R1 and the freq/pc1 sensitivities")
A = ap.parse_args()
Rn = A.n_real
G, gt = load_gt()
MAP_ID = {j: j for j in range(N_MODES)}
print(f"# GNN ladder trueZ,R0-R3   n_real={Rn}  tau_max={TAU_MAX}  |gt|={len(gt)}")

acts = np.load(ACTS_CACHE, mmap_mode="r")[:Rn]
Zf = np.load(Z_CACHE)[:Rn]                                     # (R, 8, T)
align = np.load(SAE_DIR / "alignment_per_mode.npy", allow_pickle=True).item()
Rn, _, T_eff, _ = acts.shape
print(f"  acts {acts.shape}  T_eff={T_eff}")
assert np.isfinite(Zf).all(), "GATE FAIL: non-finite Z"
RES, SPARSE = {}, {}


def sparsity_report(name, S):
    Rr, T, C = S.shape
    nz = (S != 0).mean(axis=(0, 1))
    co = np.zeros((C, C))
    for r in range(Rr):
        B = (S[r] != 0); co += B.T.astype(np.float64) @ B
    co /= Rr
    off = co[~np.eye(C, dtype=bool)]
    SPARSE[name] = dict(zero_frac_mean=float(1 - nz.mean()), nz_per_var=nz.tolist(),
                        cofire_mean=float(off.mean()), T_nominal=T,
                        n_eff_ratio=float(off.mean() / T))
    print(f"    sparsity[{name}]: zero-frac {1-nz.mean():.3f} "
          f"(range {1-nz.max():.3f}-{1-nz.min():.3f})   co-fire {off.mean():.0f}/{T} "
          f"= n_eff/T {off.mean()/T:.3f}")


def score(name, series, mapping, note=""):
    t0 = time.time()
    r = run_ladder_rung(list(series), mapping, gt, workers=A.workers, keep_vm=False)
    r["note"] = note; r["n_matched"] = len(mapping); RES[name] = r
    print(f"  {name:<6} F1={r['f1'].mean():.4f}+-{r['f1'].std():.3f}  "
          f"P={r['precision'].mean():.4f}  R={r['recall'].mean():.4f}  "
          f"TP/FP/FN={r['tp'].mean():.1f}/{r['fp'].mean():.1f}/{r['fn'].mean():.1f}  "
          f"matched={len(mapping)}/8  [{time.time()-t0:.0f}s]  {note}")
    return r


# ── ceiling: PCMCI on the true latents, identical protocol ───────────────────
print("\n[trueZ] ceiling: PCMCI on true Z, same protocol")
paths = sorted(DATA_DIR.glob("realisation_*.npz"))[:Rn]
Ztrue = [np.load(p)["latent_states"].T.astype(np.float64) for p in paths]   # (T=2400, 8)
score("trueZ", Ztrue, MAP_ID, "true latent series (T=2400)")

# ── encode per-mode SAEs ─────────────────────────────────────────────────────
print("\n[encode] per-mode SAEs")
code_pm = np.empty((Rn, N_MODES, T_eff, N_FEATURES), np.float32)
best_feat, sign_oracle = [], []
for j in range(N_MODES):
    sae, mu, sd = load_sae(SAE_DIR / f"sae_mode_{j}.pt")
    code_pm[:, j] = encode_block(sae, np.asarray(acts[:, j]).reshape(-1, INPUT_DIM), mu, sd
                                 ).reshape(Rn, T_eff, N_FEATURES)
    bf = int(align[j]["best_feat"]); best_feat.append(bf)
    sign_oracle.append(float(np.sign(align[j]["C_j"][bf])) or 1.0)
print(f"  oracle best_feat = {best_feat}\n  oracle sign      = {sign_oracle}")

print("\n[R0] replication: per-mode SAE + oracle feature + oracle sign")
S0 = np.stack([sign_oracle[j] * code_pm[:, j, :, best_feat[j]] for j in range(N_MODES)], -1)
sparsity_report("R0", S0)
rs0 = [abs(np.corrcoef(S0[..., j].ravel(), Zf[:, j].ravel())[0, 1]) for j in range(N_MODES)]
score("R0", S0, MAP_ID, f"oracle: pooling+N+per-mode SAE+feature+sign |r|vsZ={np.round(rs0,2).tolist()}")

if not A.quick: print("\n[R1] drop oracle SIGN")
sign_unsup = [float(np.sign(skew(code_pm[:, j, :, best_feat[j]].ravel()))) or 1.0 for j in range(N_MODES)]
if not A.quick:
    S1 = np.stack([sign_unsup[j] * code_pm[:, j, :, best_feat[j]] for j in range(N_MODES)], -1)
    score("R1", S1, MAP_ID, f"unsup sign={sign_unsup}")

print("\n[R2] drop oracle FEATURE SELECTION")
R2_FEAT = {}
for rank_by in (("variance",) if A.quick else ("variance", "freq", "pc1")):
    picks = [sel_var(code_pm[:, j], n_max=1, rank_by=rank_by)[0] for j in range(N_MODES)]
    R2_FEAT[rank_by] = picks
    S = np.stack([code_pm[:, j, :, picks[j]] for j in range(N_MODES)], -1)
    S = S * np.array([float(np.sign(skew(S[..., j].ravel()))) or 1.0 for j in range(N_MODES)])
    nm = "R2" if rank_by == "variance" else f"R2_{rank_by}"
    if rank_by == "variance": sparsity_report("R2", S)
    rs = [abs(np.corrcoef(S[..., j].ravel(), Zf[:, j].ravel())[0, 1]) if S[..., j].std() > 0 else 0.0
          for j in range(N_MODES)]
    score(nm, S, MAP_ID, f"rank={rank_by} picks={picks} |r|vsZ={np.round(rs,2).tolist()}")
del code_pm

print("\n[encode] MIXED SAE (sae_mixed.pt) — one dictionary, no mode labels")
MIXED_SRC = SAE_DIR / "sae_mixed.pt"
sae, mu, sd = load_sae(MIXED_SRC)
code_mx = encode_block(sae, np.asarray(acts).reshape(-1, INPUT_DIM), mu, sd
                       ).reshape(Rn, N_MODES, T_eff, N_FEATURES)
print(f"  live features (nz>=2%): {((code_mx != 0).reshape(-1, N_FEATURES).mean(0) >= 0.02).sum()}/512")

print("\n[R3a] drop oracle PER-MODE DICTIONARY (mixed SAE, argmax-var per stream)")
picks_a = [sel_var(code_mx[:, j], n_max=1, rank_by="variance")[0] for j in range(N_MODES)]
S3a = np.stack([code_mx[:, j, :, picks_a[j]] for j in range(N_MODES)], -1)
S3a = S3a * np.array([float(np.sign(skew(S3a[..., j].ravel()))) or 1.0 for j in range(N_MODES)])
sparsity_report("R3a", S3a)
rs_a = [abs(np.corrcoef(S3a[..., j].ravel(), Zf[:, j].ravel())[0, 1]) for j in range(N_MODES)]
score("R3a", S3a, MAP_ID, f"picks={picks_a} distinct={len(set(picks_a))} |r|vsZ={np.round(rs_a,2).tolist()}")

print("\n[R3b] drop oracle N + MODE PARTITION  <<< GRAPHCAST-MATCHED >>>")
pooled = code_mx.mean(axis=1)                                   # (R, T, 512)
del code_mx
R3B = {}
flatZ = Zf.transpose(0, 2, 1).reshape(-1, N_MODES)
for rank_by in (("variance",) if A.quick else ("variance", "freq", "pc1")):
    chosen = sel_var(pooled, n_max=12, rank_by=rank_by)
    S = pooled[:, :, chosen]
    mp, M = map_r(S.reshape(-1, S.shape[-1]), flatZ)
    nm = "R3b" if rank_by == "variance" else f"R3b_{rank_by}"
    if rank_by == "variance":
        sparsity_report("R3b", S)
        R3B.update(chosen=chosen, map_matrix=M, mapping=mp, pooled_selected=S)
    print(f"  rank={rank_by}: N_hat={len(chosen)} chosen={chosen}")
    print(f"    MAP-R best |r| per discovered var: {np.round(M.max(1), 3).tolist()}")
    print(f"    matched {len(mp)}/8 true modes: {mp}")
    score(nm, S, mp, f"rank={rank_by} N_hat={len(chosen)} matched={len(mp)}/8")

np.save(OUT / f"ladder_{A.tag}.npy",
        dict(res={k: {kk: vv for kk, vv in v.items() if kk not in ("val_matrices",)} for k, v in RES.items()},
             sparsity=SPARSE, best_feat=best_feat, sign_oracle=sign_oracle, sign_unsup=sign_unsup,
             r2_feat=R2_FEAT, picks_r3a=picks_a, r3b_chosen=R3B["chosen"], r3b_map=R3B["mapping"],
             r3b_map_matrix=R3B["map_matrix"], mixed_src=str(MIXED_SRC), n_real=Rn, gt=sorted(gt),
             tau_max=TAU_MAX, forecaster="MeshGNN hetdynamics_eqvar, last MP layer, stride 1"),
        allow_pickle=True)
np.save(OUT / f"r3b_series_{A.tag}.npy",
        dict(series=R3B["pooled_selected"], chosen=R3B["chosen"], mapping=R3B["mapping"],
             map_matrix=R3B["map_matrix"]), allow_pickle=True)
print("\n" + "=" * 74)
print(f"{'rung':<12}{'F1':>8}{'P':>8}{'R':>8}{'matched':>9}")
for k, v in RES.items():
    print(f"{k:<12}{v['f1'].mean():>8.4f}{v['precision'].mean():>8.4f}{v['recall'].mean():>8.4f}{v['n_matched']:>7}/8")
