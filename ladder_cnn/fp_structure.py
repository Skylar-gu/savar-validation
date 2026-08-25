"""What predicts an edge at R3b? (PREREG §6, third item.)

The GraphCast lane is running the analogous "what predicts an edge" analysis on
the real model. This is the SAVAR ground-truth version: at the GraphCast-matched
rung, are the false positives random, or concentrated on feature pairs that
share the global-activity direction (PC0)?
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
from scipy.stats import spearmanr, pointbiserialr
from sklearn.decomposition import PCA
from common import *

G, gt = load_gt()
lad = np.load(OUT / "ladder_r0_r3_base.npy", allow_pickle=True).item()
r3b = np.load(OUT / "r3b_series_base.npy", allow_pickle=True).item()
S = r3b["series"]                      # (100, 497, 12)
chosen = r3b["chosen"]
mapping = r3b["mapping"]
dets = lad["res"]["R3b"]["detected"]   # list of sets, per realisation
Rn, T, C = S.shape
print(f"R3b: {Rn} realisations, T={T}, N_hat={C}, chosen={chosen}")
print(f"     mapping (discovered -> true mode): {mapping}")
unmatched = [c for c in range(C) if c not in mapping]
print(f"     unmatched discovered variables: {unmatched}")

# ── the global-activity direction on the FULL 512-feature pooled matrix ──────
acts = np.asarray(np.load(ROOT / "sae_data/base/activations_full.npy",
                          mmap_mode="r")[:Rn])
sae, mu, sd = load_sae(ROOT / "sae_data/base/sae_best.pt")
code = encode_block(sae, acts.reshape(-1, INPUT_DIM), mu, sd
                    ).reshape(Rn, N_MODES, T, N_FEATURES)
pooled = code.mean(1).reshape(-1, N_FEATURES)
del code, acts
p = PCA(n_components=10, random_state=0).fit(pooled - pooled.mean(0))
evr = p.explained_variance_ratio_
print(f"\nMixed-SAE pooled feature matrix: PC0 explains {evr[0]*100:.1f}% "
      f"of variance; PC0-4 {evr[:5].sum()*100:.1f}%")
print(f"  evr[:10] = {np.round(evr,4).tolist()}")
pc0 = p.components_[0]                              # (512,)
load = np.abs(pc0[chosen])                          # PC0 loading of chosen vars
print(f"  |PC0 loading| of the 12 chosen: {np.round(load,4).tolist()}")

# projection of each chosen series onto PC0 (share of its own variance)
proj = np.abs(np.corrcoef(np.vstack([pooled[:, chosen].T,
                                     (pooled - pooled.mean(0)) @ pc0]))[-1, :C])
print(f"  |r(series, PC0 score)| per chosen var: {np.round(proj,3).tolist()}")

# ── decoder-direction cosine between chosen features ────────────────────────
Wdec = sae.decoder.weight.detach().numpy()          # (256, 512)
Wn = Wdec / (np.linalg.norm(Wdec, axis=0, keepdims=True) + 1e-12)
cosdec = (Wn[:, chosen].T @ Wn[:, chosen])

# ── pairwise series statistics ──────────────────────────────────────────────
flat = S.reshape(-1, C)
corr = np.corrcoef(flat.T)
nzmask = (S != 0)
cofire = np.zeros((C, C))
for r in range(Rn):
    B = nzmask[r]
    cofire += B.T.astype(float) @ B
cofire /= Rn

# ── edge rate per ordered pair per lag ──────────────────────────────────────
rows = []
for a in range(C):
    for b in range(C):
        if a == b:
            continue
        for tau in (1, 2):
            rate = np.mean([(a, b, tau) in d for d in dets])
            is_gt = (a in mapping and b in mapping
                     and (mapping[a], mapping[b], tau) in gt)
            rows.append(dict(a=a, b=b, tau=tau, rate=rate, is_gt=int(is_gt),
                             pc0prod=load[a] * load[b],
                             projprod=proj[a] * proj[b],
                             abscorr=abs(corr[a, b]),
                             cosdec=abs(cosdec[a, b]),
                             cofire=cofire[a, b] / T,
                             touches_unmatched=int(a in unmatched or b in unmatched)))
R = {k: np.array([r[k] for r in rows]) for k in rows[0]}
n_pairs = len(rows)
print(f"\nOrdered (a,b,tau) slots: {n_pairs}   of which true edges: "
      f"{int(R['is_gt'].sum())}   mean detection rate {R['rate'].mean():.3f}")

print(f"\n{'─'*74}\nFP structure — Spearman rho of DETECTION RATE vs each predictor")
print(f"  (computed over the {int((1-R['is_gt']).sum())} NON-true slots, i.e. "
      f"pure false positives)")
fpm = R["is_gt"] == 0
for k in ("pc0prod", "projprod", "abscorr", "cosdec", "cofire"):
    rho, pv = spearmanr(R[k][fpm], R["rate"][fpm])
    print(f"    {k:<10} rho={rho:+.3f}  p={pv:.2e}")

print(f"\n  detection rate on TRUE vs FALSE slots:")
print(f"    true  slots: {R['rate'][R['is_gt']==1].mean():.3f} "
      f"(n={int(R['is_gt'].sum())})")
print(f"    false slots: {R['rate'][fpm].mean():.3f} (n={int(fpm.sum())})")
rb, pb = pointbiserialr(R["is_gt"], R["rate"])
print(f"    point-biserial r(is_gt, rate) = {rb:+.3f}  p={pb:.2e}")

print(f"\n  FP detection rate by |PC0 loading| product quartile:")
q = np.quantile(R["pc0prod"][fpm], [0, .25, .5, .75, 1.0])
for i in range(4):
    m = fpm & (R["pc0prod"] >= q[i]) & (R["pc0prod"] <= q[i + 1])
    print(f"    Q{i+1} [{q[i]:.4f},{q[i+1]:.4f}]  rate={R['rate'][m].mean():.3f} "
          f"(n={int(m.sum())})")

print(f"\n  FP detection rate by |corr| quartile:")
q = np.quantile(R["abscorr"][fpm], [0, .25, .5, .75, 1.0])
for i in range(4):
    m = fpm & (R["abscorr"] >= q[i]) & (R["abscorr"] <= q[i + 1])
    print(f"    Q{i+1} [{q[i]:.3f},{q[i+1]:.3f}]  rate={R['rate'][m].mean():.3f} "
          f"(n={int(m.sum())})")

print(f"\n  where do the FPs live?")
mu_ = R["rate"][fpm & (R["touches_unmatched"] == 1)].mean()
mm_ = R["rate"][fpm & (R["touches_unmatched"] == 0)].mean()
print(f"    slots touching an UNMATCHED discovered variable: rate={mu_:.3f} "
      f"(n={int((fpm & (R['touches_unmatched']==1)).sum())})")
print(f"    slots between two MATCHED variables            : rate={mm_:.3f} "
      f"(n={int((fpm & (R['touches_unmatched']==0)).sum())})")
tot_fp = lad["res"]["R3b"]["fp"].mean()
print(f"    mean FP per realisation: {tot_fp:.1f} against {len(gt)} true edges")

print(f"\n  saturation check: fraction of all ordered slots detected in "
      f">=50% of realisations: {(R['rate']>=0.5).mean():.3f}")
print(f"  mean |corr| among the 12 chosen (off-diag): "
      f"{np.abs(corr[~np.eye(C,dtype=bool)]).mean():.3f}  "
      f"max {np.abs(corr[~np.eye(C,dtype=bool)]).max():.3f}")

# conditioning of the R3b variable set (CLAUDE.md guardrail 5)
ev = np.linalg.eigvalsh(corr)
print(f"\n  conditioning of the R3b correlation matrix: cond={ev.max()/max(ev.min(),1e-12):.1f}"
      f"  min eig={ev.min():.4f}  max |corr| off-diag="
      f"{np.abs(corr[~np.eye(C,dtype=bool)]).max():.3f}")

np.save(OUT / "fp_structure.npy",
        dict(rows=R, evr=evr, pc0=pc0, load=load, proj=proj, corr=corr,
             cosdec=cosdec, cofire=cofire, chosen=chosen, mapping=mapping),
        allow_pickle=True)
print(f"\nsaved -> {OUT}/fp_structure.npy")
