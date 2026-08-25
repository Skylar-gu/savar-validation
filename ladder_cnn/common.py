"""Shared machinery for the SAVAR SAE->PCMCI+ oracle-ablation ladder.

Protocol frozen in PREREG.md. Index conventions imported unchanged from the
repo: ground truth G[eff, cause, tau-1]; tigramite p_matrix[cause, eff, tau];
edge tuples stored (cause, eff, tau).

Reads data/, checkpoints/, sae_data/ under SAVAR_ROOT (paths.py); writes only to
SAVAR_ROOT/results/ladder_cnn/.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import SAVAR_ROOT as ROOT   # $SAVAR_ROOT or the package dir
OUT = ROOT / "results" / "ladder_cnn"       # ladder outputs (filenames unchanged)
OUT.mkdir(parents=True, exist_ok=True)

TAU_MAX, ALPHA, PC_ALPHA = 2, 0.05, 0.2
INPUT_DIM, N_FEATURES, K_TOPK = 256, 512, 25
N_MODES = 8
NY = NX = 50
L = NY * NX
MATCH_COS_MIN = 0.30
MATCH_R_MIN = 0.10


# ── SAE ───────────────────────────────────────────────────────────────────────
class TopKSAE(nn.Module):
    def __init__(self, d=INPUT_DIM, n=N_FEATURES, k=K_TOPK):
        super().__init__()
        self.k = k
        self.encoder = nn.Linear(d, n)
        self.decoder = nn.Linear(n, d)

    def encode(self, x):
        pre = self.encoder(x)
        v, i = torch.topk(pre, self.k, dim=-1)
        a = torch.zeros_like(pre)
        a.scatter_(-1, i, F.relu(v))
        return a


def load_sae(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sae = TopKSAE()
    sae.load_state_dict(ck["model_state"])
    sae.eval()
    mean = np.asarray(ck["act_mean"], np.float32)
    std = np.asarray(ck["act_std"], np.float32)
    return sae, mean, std


def encode_block(sae, X, mean, std, bs=32768):
    """X (M, 256) float -> (M, 512) float32 codes."""
    Xn = ((X - mean) / (std + 1e-8)).astype(np.float32)
    out = np.empty((len(Xn), N_FEATURES), np.float32)
    with torch.no_grad():
        for i in range(0, len(Xn), bs):
            out[i:i + bs] = sae.encode(torch.from_numpy(Xn[i:i + bs])).numpy()
    return out


# ── metrics: byte-identical conventions to pcmci/run_pcmci.py ─────────────────
def gt_edges(G, cross_only=True):
    N, _, tmax = G.shape
    out = set()
    for eff in range(N):
        for cause in range(N):
            if cross_only and cause == eff:
                continue
            for tau in range(1, tmax + 1):
                if G[eff, cause, tau - 1] != 0:
                    out.add((cause, eff, tau))
    return out


def detected_edges(p_matrix, alpha=ALPHA, cross_only=True):
    N = p_matrix.shape[0]
    tmax = p_matrix.shape[2] - 1
    out = set()
    for cause in range(N):
        for eff in range(N):
            if cross_only and cause == eff:
                continue
            for tau in range(1, tmax + 1):
                if p_matrix[cause, eff, tau] < alpha:
                    out.add((cause, eff, tau))
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    return p, r, (2 * p * r / (p + r) if p + r > 0 else 0.0)


# ── Hungarian-strict scoring, as sae/discover_modes.py::score_candidate ───────
def hungarian_strict_score(det, mapping, gt):
    """det: set of (c,e,tau) on DISCOVERED variable indices.
    mapping: dict discovered_idx -> true_mode_idx (partial).
    Edges touching an unmatched variable are FP; gt edges at unmatched true
    modes are FN (automatic, since they cannot appear in `mapped`)."""
    mapped, fp_unmatched = set(), 0
    for (c, e, tau) in det:
        if c in mapping and e in mapping:
            mapped.add((mapping[c], mapping[e], tau))
        else:
            fp_unmatched += 1
    tp = len(gt & mapped)
    fp = len(mapped - gt) + fp_unmatched
    fn = len(gt - mapped)
    return tp, fp, fn


def map_foot(What, W_true, cos_min=MATCH_COS_MIN):
    """MAP-FOOT: Hungarian on footprint cosine, accept >= cos_min."""
    Wt = W_true / (np.linalg.norm(W_true, axis=1, keepdims=True) + 1e-12)
    Wh = What / (np.linalg.norm(What, axis=1, keepdims=True) + 1e-12)
    M = Wh @ Wt.T
    ri, ci = linear_sum_assignment(-M)
    return {int(a): int(b) for a, b in zip(ri, ci) if M[a, b] >= cos_min}, M


def map_r(series, Zser, r_min=MATCH_R_MIN):
    """MAP-R: Hungarian on |Pearson r| between discovered series and true Z.

    series (T, C) for ONE realisation-concatenation; Zser (T, 8). Both are
    concatenated over the realisations used, so the mapping is global, not
    per-realisation."""
    C = series.shape[1]
    M = np.zeros((C, N_MODES))
    for c in range(C):
        x = series[:, c] - series[:, c].mean()
        sx = np.sqrt((x * x).sum())
        for j in range(N_MODES):
            y = Zser[:, j] - Zser[:, j].mean()
            sy = np.sqrt((y * y).sum())
            M[c, j] = abs(float((x * y).sum() / (sx * sy))) if sx > 0 and sy > 0 else 0.0
    ri, ci = linear_sum_assignment(-M)
    return {int(a): int(b) for a, b in zip(ri, ci) if M[a, b] >= r_min}, M


# ── PCMCI driver ──────────────────────────────────────────────────────────────
def pcmci_one(series):
    """series (T, C) -> set of detected (cause, eff, tau)."""
    from tigramite.data_processing import DataFrame
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI
    pc = PCMCI(dataframe=DataFrame(np.asarray(series, np.float64)),
               cond_ind_test=ParCorr(), verbosity=0)
    res = pc.run_pcmci(tau_min=1, tau_max=TAU_MAX,
                       pc_alpha=PC_ALPHA, alpha_level=ALPHA)
    return detected_edges(res["p_matrix"]), res["p_matrix"], res["val_matrix"]


def _pool_worker(args):
    idx, series = args
    det, pm, vm = pcmci_one(series)
    return idx, sorted(det), vm


def run_ladder_rung(series_per_real, mapping, gt, workers=4, keep_vm=False):
    """series_per_real: list of (T, C) arrays, one per realisation.
    Returns dict with per-realisation P/R/F1 and aggregates."""
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(i, s) for i, s in enumerate(series_per_real)]
    res = {}
    if workers <= 1:
        for j in jobs:
            i, det, vm = _pool_worker(j)
            res[i] = (det, vm)
    else:
        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=_init_worker) as ex:
            for i, det, vm in ex.map(_pool_worker, jobs):
                res[i] = (det, vm)
    P, R, Fs, TP, FP, FN = [], [], [], [], [], []
    dets = []
    vms = []
    for i in range(len(jobs)):
        det, vm = res[i]
        det = set(map(tuple, det))
        dets.append(det)
        if keep_vm:
            vms.append(vm)
        tp, fp, fn = hungarian_strict_score(det, mapping, gt)
        p, r, f = prf(tp, fp, fn)
        P.append(p); R.append(r); Fs.append(f)
        TP.append(tp); FP.append(fp); FN.append(fn)
    return dict(precision=np.array(P), recall=np.array(R), f1=np.array(Fs),
                tp=np.array(TP), fp=np.array(FP), fn=np.array(FN),
                detected=dets, val_matrices=vms, mapping=mapping,
                n_real=len(jobs))


def _init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


# ── nulls ─────────────────────────────────────────────────────────────────────
def phase_randomise(x, rng):
    """Preserve amplitude spectrum (hence autocorrelation), randomise phases."""
    n = len(x)
    Xf = np.fft.rfft(x - x.mean())
    ph = rng.uniform(0, 2 * np.pi, len(Xf))
    ph[0] = 0.0
    if n % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(np.abs(Xf) * np.exp(1j * ph), n) + x.mean()


def circ_shift(x, rng, margin=50):
    return np.roll(x, int(rng.integers(margin, len(x) - margin)))


# ── selection rule SEL-VAR (frozen in PREREG §3) ──────────────────────────────
def sel_var(cand_series, n_max=12, rank_by="variance", dedup=0.90,
            min_nz=0.02):
    """cand_series: (R, T, C) candidate series.
    Returns list of selected candidate column indices."""
    Rn, T, C = cand_series.shape
    flat = cand_series.reshape(-1, C)
    nz = (flat != 0).mean(0)
    var = flat.var(0)
    alive = np.where((nz >= min_nz) & (var > 1e-12))[0]
    if rank_by == "variance":
        score = var[alive]
    elif rank_by == "freq":
        score = nz[alive]
    elif rank_by == "pc1":
        Xz = flat[:, alive]
        Xz = (Xz - Xz.mean(0)) / (Xz.std(0) + 1e-12)
        from sklearn.decomposition import PCA
        p = PCA(n_components=1, random_state=0).fit(Xz)
        score = np.abs(p.components_[0])
    else:
        raise ValueError(rank_by)
    order = alive[np.argsort(-score)]
    Xc = flat
    chosen = []
    for c in order:
        ok = True
        for d in chosen:
            a, b = Xc[:, c], Xc[:, d]
            sa, sb = a.std(), b.std()
            if sa < 1e-12 or sb < 1e-12:
                continue
            r = float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))
            if abs(r) > dedup:
                ok = False
                break
        if ok:
            chosen.append(int(c))
        if len(chosen) >= n_max:
            break
    return chosen


def load_gt():
    d = np.load(ROOT / "data/realisations/realisation_000.npz")
    G = d["ground_truth_graph"].astype(np.float64)
    return G, gt_edges(G, cross_only=True)
