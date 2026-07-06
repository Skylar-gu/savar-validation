"""
E3 — Upgraded internals-free dynamical arm (litext plan, notes/literature_extension_experiments.md §3).

Block B's impulse-response test recovered a lag-correct HALF of the graph
(F1 0.500); FU1 localized the recall cap to autoregressive rollout damping (a
generative-loop artifact — the representation carries the timescales). Three
arms, ablated separately, all under Block B's exact detection protocol
(W-projected response, pixel-permutation null, pair-level F1 vs gt cross
edges):

  A  baseline   — Block B replica: impulse on last input frame, free rollout.
  B  teacher    — teacher-forced perturbation propagation: at every step the
                  BASE window is the true trajectory; the perturbation delta is
                  carried through the model (pert = model(true+delta) -
                  model(true)) and shifted into the next window. No forecast-
                  error compounding, no rollout blur. (finite-amplitude tangent
                  propagation along the true trajectory)
  C  sustained  — constant forcing: the impulse pattern is added to every
                  input frame (initial window and each newly predicted frame)
                  during free rollout; slow modes integrate the forcing
                  (quasi-steady response ~ 1/(1-phi_j) amplification).

Dose ladder: arms A and C also run at SIGMA x3; per-edge linearity screen
(response-map correlation and slope ratio across doses).

Env: E3_NWIN (240), E3_NSTEPS (12), E3_SIGMA (1.0), E3_NPERM (1000),
     E3_ALPHA (0.01), E3_SPLIT, E3_CKPT
Output: results/litext_e3_dynarm.npy
"""

import sys, os, glob
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.gnn.gnn_forecaster import MeshGNN, K

SPLIT   = os.environ.get("E3_SPLIT", "data/splits_hetdynamics_eqvar/test")
CKPT    = os.environ.get("E3_CKPT", "checkpoints/hetdynamics_eqvar/best.pt")
N_WIN   = int(os.environ.get("E3_NWIN", 240))
N_STEPS = int(os.environ.get("E3_NSTEPS", 12))
SIGMA   = float(os.environ.get("E3_SIGMA", 1.0))
DOSE_HI = 3.0
N_PERM  = int(os.environ.get("E3_NPERM", 1000))
ALPHA   = float(os.environ.get("E3_ALPHA", 0.01))
OUT     = os.environ.get("E3_OUT", "results/litext_e3_dynarm.npy")
PHI     = np.array([0.15, 0.30, 0.42, 0.55, 0.68, 0.78, 0.86, 0.92])
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BS      = 48

rng = np.random.default_rng(0)

ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
model = MeshGNN(50, 50).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
model.adjacency()

paths = sorted(glob.glob(os.path.join(SPLIT, "realisation_*.npz")))
d0 = np.load(paths[0])
W, W_plus = d0["W"].astype(np.float64), d0["W_plus"].astype(np.float64)
N_MODES = W.shape[0]
gt_pairs = {(int(c), int(e)) for c, e, l, v in d0["fine_edges"] if int(c) != int(e)}
gt_lag = {}
for c, e, l, v in d0["fine_edges"]:
    c, e, l = int(c), int(e), int(l)
    if c != e:
        gt_lag.setdefault((c, e), []).append(l)

# transitive closure of the gt graph (dynamical probing sees TOTAL effects:
# a detected 2-hop path is not a model error — score both levels)
closure = set(gt_pairs)
changed = True
while changed:
    changed = False
    for (a, b) in list(closure):
        for (c, d) in list(closure):
            if b == c and a != d and (a, d) not in closure:
                closure.add((a, d)); changed = True

obs_list = [np.load(p)["observations"] for p in paths]
pix_std = float(np.concatenate([o.reshape(len(o), -1)
                                for o in obs_list[:10]]).std())
patterns = W_plus / np.abs(W_plus).max(0, keepdims=True)
print(f"model={CKPT} split={SPLIT} ({len(paths)} reals) pixel std={pix_std:.4f}")
print(f"{N_WIN} windows x {N_MODES} impulses, {N_STEPS} steps, sigma={SIGMA}")

# windows WITH their forward true trajectory (needed by arm B)
win_src = []
ri = 0
while len(win_src) < N_WIN:
    o = obs_list[ri % len(obs_list)]                 # (T, 50, 50) in splits
    t0 = int(rng.integers(0, len(o) - K - N_STEPS))
    win_src.append((ri % len(obs_list), t0))
    ri += 1
# traj[w, s] = true frame at t0+s, s in 0..K-1+N_STEPS
traj = np.stack([obs_list[r][t0:t0 + K + N_STEPS]
                 for r, t0 in win_src]).astype(np.float32)   # (N_WIN, K+S, 50, 50)


@torch.no_grad()
def batched_model(x_np):
    """x_np: (N, K, 50, 50) float32 -> (N, 50, 50) predictions."""
    outs = []
    for i in range(0, len(x_np), BS):
        xb = torch.from_numpy(x_np[i:i + BS]).to(DEVICE)
        outs.append(model(xb)[:, 0].cpu().numpy())
    return np.concatenate(outs, 0)


@torch.no_grad()
def rollout_dX(sigma, sustained):
    """Arms A/C: free rollout, perturbed vs base. Returns (8, S, 2500)."""
    imps = (sigma * pix_std * patterns.T).astype(np.float32)      # (8, L)
    mean_dX = np.zeros((N_MODES, N_STEPS, 2500))
    for i in range(0, N_WIN, BS):
        wb = torch.from_numpy(traj[i:i + BS, :K]).to(DEVICE)      # (b,K,50,50)
        # base rollout
        xs = wb.clone()
        base = []
        for _ in range(N_STEPS):
            y = model(xs)
            base.append(y[:, 0].cpu().numpy())
            xs = torch.cat([xs[:, 1:], y], dim=1)
        base = np.stack(base, 1)                                   # (b,S,50,50)
        for j in range(N_MODES):
            imp = torch.from_numpy(imps[j].reshape(50, 50)).to(DEVICE)
            xp = wb.clone()
            if sustained:
                xp += imp                                          # all K frames
            else:
                xp[:, -1] += imp
            pert = []
            for _ in range(N_STEPS):
                y = model(xp)
                pert.append(y[:, 0].cpu().numpy())
                ynew = y if not sustained else y + imp
                xp = torch.cat([xp[:, 1:], ynew], dim=1)
            pert = np.stack(pert, 1)
            mean_dX[j] += (pert - base).reshape(pert.shape[0], N_STEPS, -1).sum(0)
    return mean_dX / N_WIN


@torch.no_grad()
def teacher_dX(sigma):
    """Arm B: perturbation carried along the TRUE trajectory. (8, S, 2500)."""
    imps = (sigma * pix_std * patterns.T).astype(np.float32)
    mean_dX = np.zeros((N_MODES, N_STEPS, 2500))
    for i in range(0, N_WIN, BS):
        tr = traj[i:i + BS]                                        # (b,K+S,...)
        b = len(tr)
        # delta windows per impulse: (8, b, K, 50, 50)
        delta = np.zeros((N_MODES, b, K, 50, 50), dtype=np.float32)
        for j in range(N_MODES):
            delta[j, :, -1] = imps[j].reshape(50, 50)
        for s in range(N_STEPS):
            base_win = tr[:, s:s + K]                              # true frames
            y_base = batched_model(base_win)                       # (b,50,50)
            for j in range(N_MODES):
                y_pert = batched_model(base_win + delta[j])
                dy = y_pert - y_base                               # (b,50,50)
                mean_dX[j, s] += dy.reshape(b, -1).sum(0)
                delta[j] = np.concatenate(
                    [delta[j, :, 1:], dy[:, None]], axis=1)
    return mean_dX / N_WIN


# impulse Z-amplitudes a_j = W_j . impulse_j (for kernel normalization)
def _impulse_amps(sigma):
    imps = sigma * pix_std * patterns.T                            # (8, L)
    return np.array([W[j] @ imps[j] for j in range(N_MODES)])


def deconv_direct(R, amps):
    """total-effect kernels T[i,j,tau] -> direct kernels B via
    T[tau] = B[tau] + sum_{s<tau} B[s] @ T[tau-s]  (Volterra recursion).
    R[i,j,t] is the response at lag t=index+1; columns normalized by the
    impulse amplitude a_j so T is per unit Z-impulse."""
    S = R.shape[2]
    T = R / amps[None, :, None]                                    # (8,8,S)
    B = np.zeros_like(T)
    for t in range(S):                                             # lag t+1
        acc = np.zeros((N_MODES, N_MODES))
        for s in range(t):                                         # B lag s+1
            acc += B[:, :, s] @ T[:, :, t - s - 1]
        B[:, :, t] = T[:, :, t] - acc
    return B


def score(mean_dX, tag, stat_kind="max", amps=None):
    R = np.einsum("il,jtl->ijt", W, mean_dX)                       # (8,8,S)
    def _stat(Rm):
        if stat_kind == "max":
            return np.abs(Rm).max(2)
        if stat_kind == "int":
            return np.abs(Rm).sum(2)
        return np.abs(deconv_direct(Rm, amps)).max(2)              # "deconv"
    null_s = np.zeros((N_PERM, N_MODES, N_MODES))
    for p in range(N_PERM):
        Wp_ = W[:, rng.permutation(2500)]
        null_s[p] = _stat(np.einsum("il,jtl->ijt", Wp_, mean_dX))
    thresh = np.quantile(null_s, 1 - ALPHA, axis=0)
    stat = _stat(R)
    best_lag = (np.abs(deconv_direct(R, amps)).argmax(2) + 1) \
        if stat_kind == "deconv" else (np.abs(R).argmax(2) + 1)
    det = {(j, i) for i in range(N_MODES) for j in range(N_MODES)
           if i != j and stat[i, j] > thresh[i, j]}
    tp, fp, fn = len(gt_pairs & det), len(det - gt_pairs), len(gt_pairs - det)
    P = tp / (tp + fp) if tp + fp else 0.0
    Rc = tp / (tp + fn) if tp + fn else 0.0
    F1 = 2 * P * Rc / (P + Rc) if P + Rc else 0.0
    # ancestor level: detected influence vs transitive closure
    tpa, fpa = len(closure & det), len(det - closure)
    fna = len(closure - det)
    Pa = tpa / (tpa + fpa) if tpa + fpa else 0.0
    Ra = tpa / (tpa + fna) if tpa + fna else 0.0
    F1a = 2 * Pa * Ra / (Pa + Ra) if Pa + Ra else 0.0
    print(f"\n[{tag}] direct:   TP={tp} FP={fp} FN={fn} -> "
          f"P={P:.3f} R={Rc:.3f} F1={F1:.3f}")
    print(f"[{tag}] ancestor: TP={tpa} FP={fpa} FN={fna} -> "
          f"P={Pa:.3f} R={Ra:.3f} F1={F1a:.3f}")
    print(f"  detected: {sorted(det)}")
    print(f"  missed:   {sorted(gt_pairs - det)}")
    print(f"  false+ (direct):   {sorted(det - gt_pairs)}")
    print(f"  false+ (ancestor): {sorted(det - closure)}")
    lag_rows = []
    for (c, e) in sorted(gt_pairs & det):
        bl, tl = int(best_lag[e, c]), gt_lag[(c, e)]
        lag_rows.append((c, e, bl, tl))
    lag_ok = sum(1 for c, e, bl, tl in lag_rows
                 if any(abs(bl - t) <= 1 for t in tl))
    if lag_rows:
        print(f"  lag-consistent (|peak-true|<=1): {lag_ok}/{len(lag_rows)}")
    # self-response e-folding
    efold = np.zeros(N_MODES)
    for j in range(N_MODES):
        r = np.abs(R[j, j])
        y = np.log(r + 1e-15)
        A_ = np.vstack([np.arange(1, N_STEPS + 1), np.ones(N_STEPS)]).T
        slope = np.linalg.lstsq(A_, y, rcond=None)[0][0]
        efold[j] = -1.0 / slope if slope < 0 else np.inf
    tau_d = -1.0 / np.log(PHI)
    fin = np.isfinite(efold)
    def _sp(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        ra -= ra.mean(); rb -= rb.mean()
        d = np.sqrt((ra**2).sum() * (rb**2).sum())
        return float((ra*rb).sum() / d) if d > 0 else np.nan
    sp = _sp(tau_d[fin], efold[fin]) if fin.sum() > 2 else np.nan
    print(f"  efold: {[f'{e:.1f}' for e in efold]}  Spearman={sp:.3f}")
    return dict(R=R, stat=stat, thresh=thresh, det=sorted(det), tp=tp, fp=fp,
                fn=fn, P=P, recall=Rc, F1=F1, F1_ancestor=F1a, P_ancestor=Pa,
                R_ancestor=Ra, lag_rows=lag_rows,
                lag_ok=lag_ok if lag_rows else 0, efold=efold, spearman=sp)


ARMS = os.environ.get("E3_ARMS", "ABCD")   # subset of A/B/C, D=dose ladder
AMPS = _impulse_amps(SIGMA)
results = {}
dX_A = dX_B = dX_C = None
if "A" in ARMS:
    print("\n=== arm A: baseline (free rollout, impulse) ===")
    dX_A = rollout_dX(SIGMA, sustained=False)
    results["A_baseline"] = score(dX_A, "A baseline s=1", amps=AMPS)
    results["A_baseline_int"] = score(dX_A, "A baseline s=1 INT",
                                      stat_kind="int", amps=AMPS)

if "B" in ARMS:
    print("\n=== arm B: teacher-forced propagation ===")
    dX_B = teacher_dX(SIGMA)
    results["B_teacher"] = score(dX_B, "B teacher s=1", amps=AMPS)
    results["B_teacher_int"] = score(dX_B, "B teacher s=1 INT",
                                     stat_kind="int", amps=AMPS)
    results["B_teacher_deconv"] = score(dX_B, "B teacher s=1 DECONV",
                                        stat_kind="deconv", amps=AMPS)

if "C" in ARMS:
    print("\n=== arm C: sustained forcing (free rollout) ===")
    dX_C = rollout_dX(SIGMA, sustained=True)
    results["C_sustained"] = score(dX_C, "C sustained s=1", amps=AMPS)

if "D" in ARMS and dX_A is not None and dX_C is not None:
    print("\n=== dose ladder (x3) ===")
    dX_A3 = rollout_dX(DOSE_HI, sustained=False)
    results["A_dose3"] = score(dX_A3, "A baseline s=3",
                               amps=_impulse_amps(DOSE_HI))
    dX_C3 = rollout_dX(DOSE_HI, sustained=True)
    results["C_dose3"] = score(dX_C3, "C sustained s=3",
                               amps=_impulse_amps(DOSE_HI))

# per-edge linearity screen across doses (response-map corr + slope ratio)
def linearity(dX1, dX3, tag):
    R1 = np.einsum("il,jtl->ijt", W, dX1)
    R3 = np.einsum("il,jtl->ijt", W, dX3)
    rows = {}
    for (c, e) in sorted(gt_pairs):
        a, b = R1[e, c], R3[e, c]
        cc = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
        k = float(np.abs(b).max() / (np.abs(a).max() + 1e-15))
        rows[(c, e)] = (cc, k)
    med_cc = float(np.nanmedian([v[0] for v in rows.values()]))
    med_k = float(np.nanmedian([v[1] for v in rows.values()]))
    print(f"  [{tag}] median response-shape corr(1x,3x)={med_cc:.3f}, "
          f"median amplitude ratio={med_k:.2f} (linear => 3.0)")
    return rows

lin_A = lin_C = {}
if "D" in ARMS and dX_A is not None and dX_C is not None:
    print("\nlinearity across doses (gt edges):")
    lin_A = linearity(dX_A, dX_A3, "A")
    lin_C = linearity(dX_C, dX_C3, "C")

# union arm: B (max + integral stats) + C detections combined
det_B = set(map(tuple, results["B_teacher"]["det"])) | \
        set(map(tuple, results["B_teacher_int"]["det"])) \
        if "B" in ARMS else set()
det_C = set(map(tuple, results["C_sustained"]["det"])) if "C" in ARMS else set()
det_U = det_B | det_C
tp, fp = len(gt_pairs & det_U), len(det_U - gt_pairs)
fn = len(gt_pairs - det_U)
P = tp / (tp + fp) if tp + fp else 0.0
Rc = tp / (tp + fn) if tp + fn else 0.0
F1 = 2 * P * Rc / (P + Rc) if P + Rc else 0.0
tpa, fpa, fna = len(closure & det_U), len(det_U - closure), len(closure - det_U)
Pa = tpa / (tpa + fpa) if tpa + fpa else 0.0
Ra = tpa / (tpa + fna) if tpa + fna else 0.0
F1a = 2 * Pa * Ra / (Pa + Ra) if Pa + Ra else 0.0
print(f"\n[union B|C] direct: TP={tp} FP={fp} FN={fn} -> "
      f"P={P:.3f} R={Rc:.3f} F1={F1:.3f}  |  ancestor F1={F1a:.3f}")
results["union_BC"] = dict(det=sorted(det_U), tp=tp, fp=fp, fn=fn, P=P,
                           recall=Rc, F1=F1, F1_ancestor=F1a)

os.makedirs("results", exist_ok=True)
np.save(OUT, dict(results=results, lin_A=lin_A, lin_C=lin_C, sigma=SIGMA,
                  dose_hi=DOSE_HI, n_win=N_WIN, n_steps=N_STEPS, alpha=ALPHA,
                  n_perm=N_PERM, ckpt=CKPT, split=SPLIT,
                  anchors=dict(blockB_F1=0.500, pcmci_trueZ=0.823),
                  note="E3 litext: teacher-forced + sustained-forcing upgrades "
                       "of the Hakim-Masanam arm"),
        allow_pickle=True)
print(f"\nsaved -> {OUT}")
