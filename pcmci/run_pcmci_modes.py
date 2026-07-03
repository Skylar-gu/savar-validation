"""
Causal discovery on the INSTANTANEOUS mode states Z (full fine cadence).

Why this script
---------------
The GNN forecaster reads a K-frame window, so any causal discovery run on its
*component* activations is temporally smeared by K (a wider window convolves the
lag structure). The fix is to run discovery on the instantaneous mode states Z
— Z(t) = W @ y(t) is a per-step projection, K-independent — exactly the Phase-6
contract. This script does that on a fine-cadence dataset and scores edge
recovery against the fine ground-truth graph, so it can be compared ACROSS noise
levels (D_y): lower observation noise → cleaner Z → discovery should improve.

Runs PCMCI+ (run_pcmciplus, tau_min=0) at full cadence (stride 1, so no aliasing:
every fine edge stays lagged) with ParCorr, TAU_MAX = max fine lag.

Metrics on cross-mode directed edges, averaged over realisations:
  P / R / F1   directed-edge recovery   (cause, eff, tau)
  sign-acc     fraction of recovered edges with correct coefficient sign

Env:
  PCM_DATA_DIRS  comma list of dataset dirs   (default data/realisations_finecadence)
  PCM_NREAL      realisations per dir         (default 40)
  PCM_PCALPHA    pc_alpha                     (default 0.05)
Output: results/pcmci_modes.npy  (one row dict per dataset dir)
"""
import sys, os
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from pathlib import Path
from tigramite.data_processing import DataFrame
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

DATA_DIRS = os.environ.get("PCM_DATA_DIRS", "data/realisations_finecadence").split(",")
N_REAL    = int(os.environ.get("PCM_NREAL", 40))
PC_ALPHA  = float(os.environ.get("PCM_PCALPHA", 0.05))


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def gt_edges(fine_edges, tau_max):
    """Directed cross edges (cause, eff, tau) and their true sign, from the fine
    graph at full cadence (tau = fine lag)."""
    edges, sign = set(), {}
    for cause, eff, ell, coeff in fine_edges:
        cause, eff, ell = int(cause), int(eff), int(ell)
        if cause == eff:
            continue
        edges.add((cause, eff, ell))
        sign[(cause, eff, ell)] = np.sign(coeff)
        tau_max = max(tau_max, ell)
    return edges, sign, tau_max


def detect(graph, val_matrix):
    """Directed lagged edges (cause, eff, tau>=1) and their detected sign."""
    N, _, T1 = graph.shape
    det, sign = set(), {}
    for c in range(N):
        for e in range(N):
            if c == e:
                continue
            for tau in range(1, T1):
                if graph[c, e, tau] == '-->':
                    det.add((c, e, tau))
                    sign[(c, e, tau)] = np.sign(val_matrix[c, e, tau])
    return det, sign


def run_dir(data_dir):
    paths = sorted(Path(data_dir).glob("realisation_*.npz"))[:N_REAL]
    assert paths, f"no realisations in {data_dir}"
    fine0 = np.load(paths[0])["fine_edges"]
    _, _, tau_max = gt_edges(fine0, 1)
    dy = float(np.load(paths[0])["metadata"][3])

    TP = FP = FN = 0
    sign_ok = sign_tot = 0
    for p in paths:
        d = np.load(p)
        Z = d["latent_states"].astype(np.float64).T          # (T, N)  instantaneous modes
        gt, gsign, _ = gt_edges(d["fine_edges"], tau_max)
        pc = PCMCI(dataframe=DataFrame(Z), cond_ind_test=ParCorr(), verbosity=0)
        res = pc.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=PC_ALPHA)
        det, dsign = detect(res["graph"], res["val_matrix"])
        det = {e for e in det if e[2] >= 1}                   # lagged directed
        TP += len(gt & det); FP += len(det - gt); FN += len(gt - det)
        for e in (gt & det):
            sign_tot += 1
            sign_ok += int(dsign[e] == gsign[e])
    P, R, F1 = prf(TP, FP, FN)
    sign_acc = sign_ok / sign_tot if sign_tot else 0.0
    return dict(data_dir=data_dir, dy=dy, n_real=len(paths), tau_max=tau_max,
                P=P, R=R, F1=F1, sign_acc=sign_acc, tp=TP, fp=FP, fn=FN)


print("PCMCI+ on instantaneous mode states Z (full cadence, no aliasing)")
print("=" * 74)
print(f"  dirs={DATA_DIRS}  n_real={N_REAL}  pc_alpha={PC_ALPHA}\n")
rows = []
for dd in DATA_DIRS:
    r = run_dir(dd.strip())
    rows.append(r)
    print(f"  {os.path.basename(r['data_dir']):34s} D_y={r['dy']:.3f}  "
          f"P={r['P']:.3f} R={r['R']:.3f} F1={r['F1']:.3f}  sign={r['sign_acc']:.3f}  "
          f"(tau_max={r['tau_max']}, n={r['n_real']})")

os.makedirs("results", exist_ok=True)
np.save("results/pcmci_modes.npy", rows)
print("\nsaved -> results/pcmci_modes.npy")
