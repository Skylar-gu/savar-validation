"""
E4-v1 — agreement→accuracy calibration (litext plan §3 E4, revised §7).

Two W-free channels per candidate aggregation map W-hat (from E1,
results/litext_e1_discovery.npy "cands"):

  G_int  PCMCI+ on W-hat-pooled pixel series (E1 protocol, ParCorr,
         tau_max=6, alpha=0.05, 24 reals) — but now saving PER-REAL EDGE SETS
         (E1 only kept tp/fp/fn counts). Consensus graph: pair detected in
         >= 50% of reals (lags marginalized: the dyn channel is pair-level).
  G_dyn  E3 arm-B teacher-forced response graph computed THROUGH the
         candidate: impulse patterns from pinv(W-hat) (fully W-free — NOT
         W_plus; documented choice so the channel needs no ground truth),
         responses projected onto W-hat rows, integral statistic
         sum_tau |R| with pixel-permutation null (240 windows, 24 steps,
         alpha=0.01, 1000 perms). Deconv (Volterra) variant kept as a
         secondary zero-FP core.

Scoring conventions (plan §7.2 — MANDATORY):
  * BEHAVIOR-BASED variable matching everywhere: candidate var <-> true mode
    by Hungarian on mean |corr(pooled pixel series, true Z)| across reals,
    match kept iff mean |corr| >= 0.3. (Footprint-cosine matching mislabeled
    shift5: behavior-matched truth-F1 is 0.835, not 0.000.)
  * Truth-F1 reported exact-lag (E1-comparable) and pair-level (agreement-
    comparable), micro-averaged over reals, Hungarian-strict (edges touching
    unmatched vars = FP; gt edges at unmatched modes = FN).

Agreement = pair-level edge-set F1 between G_int consensus and G_dyn, in the
candidate's own variable space (no truth needed). Cross-W-hat cells
(G_int from A vs G_dyn from B, both mapped to mode space via behavior
matching) probe the shared-W-hat confound.

Deliverables: Spearman(agreement, truth-F1) across candidates (bar >= 0.8);
disagreement classification — "in G_int but not G_dyn" should be enriched for
the 3 known model-unimplemented edges (2->0),(2->3),(5->6) on good maps.

Env: E4_STAGE (all|int|dyn|cal), E4_NREAL (24), E4_NWIN (240), E4_NSTEPS (24),
     E4_NPERM (1000), E4_ALPHA_DYN (0.01), E4_MATCH_R (0.3), E4_CONS (0.5)
Output: results/litext_e4_agreement.npy
        (+ partial caches litext_e4_int_partial.npy / litext_e4_dyn_partial.npy)
"""

import sys, os, glob
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import linear_sum_assignment
from scipy import stats

ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = Path(os.environ.get("E4_DATA_DIR", ROOT / "data/realisations_hetdynamics_eqvar"))
SPLIT     = Path(os.environ.get("E4_SPLIT", ROOT / "data/splits_hetdynamics_eqvar/test"))
CKPT      = Path(os.environ.get("E4_CKPT", ROOT / "checkpoints/hetdynamics_eqvar/best.pt"))
CANDS_SRC = Path(os.environ.get("E4_CANDS", ROOT / "results/litext_e1_discovery.npy"))
RES_DIR   = ROOT / "results"
TAG       = os.environ.get("E4_TAG", "")           # suffix for rung reruns
INT_PART  = RES_DIR / f"litext_e4_int_partial{TAG}.npy"
DYN_PART  = RES_DIR / f"litext_e4_dyn_partial{TAG}.npy"
OUT       = RES_DIR / f"litext_e4_agreement{TAG}.npy"

STAGE     = os.environ.get("E4_STAGE", "all")
N_REAL    = int(os.environ.get("E4_NREAL", 24))
N_WIN     = int(os.environ.get("E4_NWIN", 240))
N_STEPS   = int(os.environ.get("E4_NSTEPS", 24))
N_PERM    = int(os.environ.get("E4_NPERM", 1000))
ALPHA_DYN = float(os.environ.get("E4_ALPHA_DYN", 0.01))
PC_ALPHA  = 0.05
MATCH_R   = float(os.environ.get("E4_MATCH_R", 0.3))
CONS_FRAC = float(os.environ.get("E4_CONS", 0.5))
N_MATCH_R = int(os.environ.get("E4_MATCH_NREAL", 6))
SIGMA     = 1.0
K         = 3
NY = NX   = 50
L         = NY * NX

paths = sorted(DATA_DIR.glob("realisation_*.npz"))
d0 = np.load(paths[0])
gt = {(int(c), int(e), int(l)) for c, e, l, _ in d0["fine_edges"] if int(c) != int(e)}
gt_pairs = {(c, e) for c, e, _ in gt}
N_MODES = d0["W"].shape[0]
TAU_MAX = max(l for _, _, l in gt)
UNIMPL = {(2, 0), (2, 3), (5, 6)}       # E3: model-unimplemented edges

src = np.load(CANDS_SRC, allow_pickle=True).item()
CANDS = {k: v.astype(np.float64) for k, v in src["cands"].items()}
E1_F1 = {k: float(v["F1"]) for k, v in src.get("graph", {}).items()}
# E4_ONLY: restrict to a comma list (e.g. the oracle dyn-liveness pre-check)
if os.environ.get("E4_ONLY"):
    _keep = set(os.environ["E4_ONLY"].split(","))
    CANDS = {k: v for k, v in CANDS.items() if k in _keep}
    print(f"[E4_ONLY] restricted to {sorted(CANDS)}")
print(f"candidates: {sorted(CANDS)}  (N_REAL={N_REAL}, stage={STAGE})")

# ── behavior-based matching: candidate var <-> true mode ─────────────────────
def behavior_mapping(What):
    """Hungarian on mean |corr(pooled series, true Z)| over N_MATCH_R reals."""
    C = What.shape[0]
    M = np.zeros((C, N_MODES))
    for ri in range(N_MATCH_R):
        d = np.load(paths[ri])
        obs = d["observations"].astype(np.float64)          # (L, T)
        Z = d["latent_states"].astype(np.float64)           # (N, T)
        P = What @ obs                                       # (C, T)
        Pc = P - P.mean(1, keepdims=True)
        Zc = Z - Z.mean(1, keepdims=True)
        num = Pc @ Zc.T
        den = np.sqrt((Pc**2).sum(1))[:, None] * np.sqrt((Zc**2).sum(1))[None]
        M += np.abs(num / np.maximum(den, 1e-12))
    M /= N_MATCH_R
    ri_, ci_ = linear_sum_assignment(-M)
    pairs = [(int(a), int(b)) for a, b in zip(ri_, ci_) if M[a, b] >= MATCH_R]
    return ({a: b for a, b in pairs},
            float(np.mean([M[a, b] for a, b in pairs])) if pairs else 0.0, M)

print("\n[match] behavior-based variable matching (|corr| >= "
      f"{MATCH_R}, {N_MATCH_R} reals)")
BEH = {}
for name, What in CANDS.items():
    mp, mc, _ = behavior_mapping(What)
    BEH[name] = mp
    print(f"  {name:<10} matched {len(mp)}/{N_MODES} (mean |corr| {mc:.3f})")

# ── stage INT: PCMCI battery with per-real edge sets ─────────────────────────
def detect(graph):
    N, _, T1 = graph.shape
    return sorted((c, e, tau) for c in range(N) for e in range(N) if c != e
                  for tau in range(1, T1) if graph[c, e, tau] == "-->")

def _worker_init():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

def pcmci_one(args):
    ri, series = args
    from tigramite.data_processing import DataFrame
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI
    pc = PCMCI(dataframe=DataFrame(series), cond_ind_test=ParCorr(), verbosity=0)
    res = pc.run_pcmciplus(tau_min=0, tau_max=TAU_MAX, pc_alpha=PC_ALPHA)
    return ri, detect(res["graph"])

def run_int_battery():
    part = np.load(INT_PART, allow_pickle=True).item() if INT_PART.exists() else {}
    for name, What in CANDS.items():
        if name in part:
            print(f"  {name:<10} (cached)")
            continue
        jobs = []
        for ri in range(N_REAL):
            obs = np.load(paths[ri])["observations"].astype(np.float64)
            jobs.append((ri, (What @ obs).T))
        dets = {}
        with ProcessPoolExecutor(max_workers=4, initializer=_worker_init) as ex:
            for ri, det in ex.map(pcmci_one, jobs):
                dets[ri] = det
        part[name] = dets
        np.save(INT_PART, part, allow_pickle=True)
        print(f"  {name:<10} done ({N_REAL} reals)")
    return part

# ── stage DYN: teacher-forced response graph through each candidate ─────────
def run_dyn_battery():
    import torch
    sys.path.insert(0, str(ROOT / "train" / "gnn"))
    from gnn_forecaster import MeshGNN
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BS = 48
    model = MeshGNN(NY, NX).to(DEVICE)
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.eval()

    spaths = sorted(glob.glob(str(SPLIT / "realisation_*.npz")))
    obs_list = [np.load(p)["observations"] for p in spaths]     # (T,50,50)
    pix_std = float(np.concatenate([o.reshape(len(o), -1)
                                    for o in obs_list[:10]]).std())
    rng = np.random.default_rng(0)
    win_src = []
    while len(win_src) < N_WIN:
        r = len(win_src) % len(obs_list)
        t0 = int(rng.integers(0, len(obs_list[r]) - K - N_STEPS))
        win_src.append((r, t0))
    traj = np.stack([obs_list[r][t0:t0 + K + N_STEPS]
                     for r, t0 in win_src]).astype(np.float32)

    @torch.no_grad()
    def batched_model(x_np):
        outs = []
        for i in range(0, len(x_np), BS):
            xb = torch.from_numpy(x_np[i:i + BS]).to(DEVICE)
            outs.append(model(xb)[:, 0].cpu().numpy())
        return np.concatenate(outs, 0)

    @torch.no_grad()
    def teacher_dX(imps):
        """imps: (C, L). Returns mean_dX (C, N_STEPS, L)."""
        C = imps.shape[0]
        mean_dX = np.zeros((C, N_STEPS, L))
        for i in range(0, N_WIN, BS):
            tr = traj[i:i + BS]
            b = len(tr)
            delta = np.zeros((C, b, K, NY, NX), dtype=np.float32)
            for j in range(C):
                delta[j, :, -1] = imps[j].reshape(NY, NX)
            for s in range(N_STEPS):
                base_win = tr[:, s:s + K]
                y_base = batched_model(base_win)
                for j in range(C):
                    dy = batched_model(base_win + delta[j]) - y_base
                    mean_dX[j, s] += dy.reshape(b, -1).sum(0)
                    delta[j] = np.concatenate([delta[j, :, 1:], dy[:, None]],
                                              axis=1)
        return mean_dX / N_WIN

    def deconv_direct(T_, C):
        B = np.zeros_like(T_)
        for t in range(T_.shape[2]):
            acc = np.zeros((C, C))
            for s in range(t):
                acc += B[:, :, s] @ T_[:, :, t - s - 1]
            B[:, :, t] = T_[:, :, t] - acc
        return B

    part = np.load(DYN_PART, allow_pickle=True).item() if DYN_PART.exists() else {}
    import time
    for name, What in CANDS.items():
        if name in part:
            print(f"  {name:<10} (cached)")
            continue
        t0_ = time.time()
        C = What.shape[0]
        # W-free impulse patterns: pinv of the candidate map (col-normalized)
        Wp = np.linalg.pinv(What)                            # (L, C)
        patterns = Wp / np.maximum(np.abs(Wp).max(0, keepdims=True), 1e-12)
        imps = (SIGMA * pix_std * patterns.T).astype(np.float32)   # (C, L)
        amps = np.array([What[j] @ imps[j] for j in range(C)])
        mean_dX = teacher_dX(imps)
        R = np.einsum("il,jtl->ijt", What, mean_dX)          # (C, C, S)
        # permutation null (pixel-permute the readout rows)
        rng_p = np.random.default_rng(1)
        null_int = np.zeros((N_PERM, C, C))
        null_dec = np.zeros((N_PERM, C, C))
        for p in range(N_PERM):
            Rp = np.einsum("il,jtl->ijt", What[:, rng_p.permutation(L)], mean_dX)
            null_int[p] = np.abs(Rp).sum(2)
            null_dec[p] = np.abs(deconv_direct(
                Rp / np.maximum(np.abs(amps), 1e-12)[None, :, None], C)).max(2)
        th_int = np.quantile(null_int, 1 - ALPHA_DYN, axis=0)
        th_dec = np.quantile(null_dec, 1 - ALPHA_DYN, axis=0)
        s_int = np.abs(R).sum(2)
        s_dec = np.abs(deconv_direct(
            R / np.maximum(np.abs(amps), 1e-12)[None, :, None], C)).max(2)
        det_int = sorted((j, i) for i in range(C) for j in range(C)
                         if i != j and s_int[i, j] > th_int[i, j])
        det_dec = sorted((j, i) for i in range(C) for j in range(C)
                         if i != j and s_dec[i, j] > th_dec[i, j])
        part[name] = dict(det_int=det_int, det_dec=det_dec,
                          stat_int=s_int, thresh_int=th_int, amps=amps)
        np.save(DYN_PART, part, allow_pickle=True)
        print(f"  {name:<10} C={C}: |dyn_int|={len(det_int)} "
              f"|dyn_deconv|={len(det_dec)} ({time.time()-t0_:.0f}s)")
    return part

# ── stage CAL: truth-F1 (behavior-matched) + agreement + calibration ─────────
def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)

def truth_score(dets_per_real, mapping, pair_level=False):
    """micro-averaged over reals, Hungarian-strict via behavior mapping."""
    tp = fp = fn = 0
    gt_ref = gt_pairs if pair_level else gt
    for ri, det in dets_per_real.items():
        mapped, fp_un = set(), 0
        for (c, e, tau) in det:
            if c in mapping and e in mapping:
                mapped.add((mapping[c], mapping[e]) if pair_level
                           else (mapping[c], mapping[e], tau))
            else:
                fp_un += 1
        if pair_level:
            fp_un = 0  # recount unmatched at pair level
            pairs_un = {(c, e) for (c, e, _) in det
                        if c not in mapping or e not in mapping}
            fp_un = len(pairs_un)
        tp += len(gt_ref & mapped)
        fp += len(mapped - gt_ref) + fp_un
        fn += len(gt_ref - mapped)
    return prf(tp, fp, fn)

def consensus_pairs(dets_per_real, C):
    cnt = np.zeros((C, C))
    for ri, det in dets_per_real.items():
        for (c, e) in {(c, e) for (c, e, _) in det}:
            cnt[c, e] += 1
    thr = CONS_FRAC * len(dets_per_real)
    return sorted((c, e) for c in range(C) for e in range(C)
                  if c != e and cnt[c, e] >= thr)

def set_f1(A, B):
    A, B = set(map(tuple, A)), set(map(tuple, B))
    tp = len(A & B)
    p = tp / len(B) if B else 0.0
    r = tp / len(A) if A else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0

def to_mode_space(pairs, mapping):
    return {(mapping[c], mapping[e]) for (c, e) in pairs
            if c in mapping and e in mapping}

def calibrate(int_part, dyn_part):
    rows = {}
    print(f"\n[cal] {'candidate':<10} {'F1lag':>6} {'F1pair':>7} "
          f"{'agree':>6} {'|int|':>5} {'|dyn|':>5}  (behavior-matched truth)")
    for name in CANDS:
        if name not in int_part or name not in dyn_part:
            continue
        C = CANDS[name].shape[0]
        mp = BEH[name]
        _, _, f1_lag = truth_score(int_part[name], mp, pair_level=False)
        _, _, f1_pair = truth_score(int_part[name], mp, pair_level=True)
        g_int = consensus_pairs(int_part[name], C)
        g_dyn = dyn_part[name]["det_int"]
        agree = set_f1(g_int, g_dyn)
        rows[name] = dict(f1_lag=f1_lag, f1_pair=f1_pair, agree=agree,
                          g_int=g_int, g_dyn=g_dyn,
                          g_dyn_deconv=dyn_part[name]["det_dec"],
                          n_matched=len(mp), e1_f1=E1_F1.get(name, np.nan))
        print(f"  {name:<10} {f1_lag:>6.3f} {f1_pair:>7.3f} {agree:>6.3f} "
              f"{len(g_int):>5} {len(g_dyn):>5}")

    names = sorted(rows)
    ag = np.array([rows[n]["agree"] for n in names])
    fl = np.array([rows[n]["f1_lag"] for n in names])
    fp_ = np.array([rows[n]["f1_pair"] for n in names])
    sp_lag = stats.spearmanr(ag, fl).statistic
    sp_pair = stats.spearmanr(ag, fp_).statistic
    pr_pair = stats.pearsonr(ag, fp_).statistic
    print(f"\n  Spearman(agree, truth-F1 exact-lag) = {sp_lag:+.3f}")
    print(f"  Spearman(agree, truth-F1 pair)      = {sp_pair:+.3f}   "
          f"(Pearson {pr_pair:+.3f})  [bar >= 0.8]")

    # disagreement classification on good maps (truth pair-F1 >= 0.7)
    print("\n[cal] disagreement (in G_int, not G_dyn), mode space, good maps:")
    dis_counts, dis_total = {}, 0
    for name in names:
        if rows[name]["f1_pair"] < 0.7:
            continue
        mp = BEH[name]
        mi = to_mode_space(rows[name]["g_int"], mp)
        md = to_mode_space(rows[name]["g_dyn"], mp)
        dis = (mi - md) & gt_pairs     # true edges the dyn channel missed
        extra = (mi - md) - gt_pairs
        for e in dis:
            dis_counts[e] = dis_counts.get(e, 0) + 1
        dis_total += len(dis)
        print(f"  {name:<10} int\\dyn true-edges: {sorted(dis)}  "
              f"non-gt: {sorted(extra)}")
    in_unimpl = sum(v for k, v in dis_counts.items() if k in UNIMPL)
    print(f"  -> {in_unimpl}/{dis_total} of true-edge disagreements are the 3 "
          f"model-unimplemented edges {sorted(UNIMPL)} "
          f"(base rate {len(UNIMPL)}/{len(gt_pairs)} = "
          f"{len(UNIMPL)/len(gt_pairs):.2f})")

    # cross-W-hat agreement matrix (mode space) — shared-W confound check
    print("\n[cal] cross-What agreement (G_int rows x G_dyn cols, mode space):")
    Xnames = [n for n in names if rows[n]["n_matched"] >= 4]
    XM = np.full((len(Xnames), len(Xnames)), np.nan)
    for a, na in enumerate(Xnames):
        mi = to_mode_space(rows[na]["g_int"], BEH[na])
        for b, nb in enumerate(Xnames):
            md = to_mode_space(rows[nb]["g_dyn"], BEH[nb])
            XM[a, b] = set_f1(mi, md)
    hdr = " ".join(f"{n[:7]:>7}" for n in Xnames)
    print(f"  {'':<10}{hdr}")
    for a, na in enumerate(Xnames):
        print(f"  {na:<10}" + " ".join(f"{XM[a,b]:>7.3f}"
                                       for b in range(len(Xnames))))
    diag = np.array([XM[i, i] for i in range(len(Xnames))])
    off = XM[~np.eye(len(Xnames), dtype=bool)]
    print(f"  diag mean {np.nanmean(diag):.3f} vs off-diag mean "
          f"{np.nanmean(off):.3f}")

    # ── pool-crossed agreement PX(A) = mean_{B != A} F1(G_int(A), G_dyn(B)) ──
    # THE pre-registered E4-v1 selector (rule fixed on the parent rung:
    # mean over the dyn pool, candidates with < 4 behavior matches -> 0).
    PX = {}
    for n in names:
        if n in Xnames:
            a = Xnames.index(n)
            PX[n] = float(np.mean([XM[a, b] for b in range(len(Xnames))
                                   if b != a]))
        else:
            PX[n] = 0.0
    pxv = np.array([PX[n] for n in names])
    sp_px_pair = stats.spearmanr(pxv, fp_).statistic
    sp_px_lag = stats.spearmanr(pxv, fl).statistic
    pr_px_pair = stats.pearsonr(pxv, fp_).statistic
    print(f"\n[cal] POOL-CROSSED agreement (pre-registered selector):")
    for n in sorted(names, key=lambda x: -rows[x]["f1_pair"]):
        print(f"  {n:<10} truthF1(pair)={rows[n]['f1_pair']:.3f}  "
              f"PX={PX[n]:.3f}  sameW={rows[n]['agree']:.3f}")
    print(f"  Spearman(PX, truth-F1 pair)      = {sp_px_pair:+.3f}  "
          f"(Pearson {pr_px_pair:+.3f})  [bar >= 0.8]")
    print(f"  Spearman(PX, truth-F1 exact-lag) = {sp_px_lag:+.3f}")

    out = dict(rows=rows, beh_mapping=BEH,
               px=PX, spearman_px_pair=float(sp_px_pair),
               spearman_px_lag=float(sp_px_lag),
               pearson_px_pair=float(pr_px_pair),
               spearman_lag=float(sp_lag), spearman_pair=float(sp_pair),
               pearson_pair=float(pr_pair),
               cross_matrix=XM, cross_names=Xnames,
               dis_counts=dis_counts, dis_total=dis_total,
               in_unimpl=in_unimpl,
               config=dict(n_real=N_REAL, n_win=N_WIN, n_steps=N_STEPS,
                           alpha_dyn=ALPHA_DYN, pc_alpha=PC_ALPHA,
                           match_r=MATCH_R, cons_frac=CONS_FRAC,
                           impulse="pinv(What) — fully W-free"),
               note="E4-v1: behavior-matched truth; agreement = pair-level F1 "
                    "G_int(consensus) vs G_dyn(teacher-forced integral)")
    np.save(OUT, out, allow_pickle=True)
    print(f"\nsaved -> {OUT}")

if STAGE in ("all", "int"):
    print(f"\n[int] PCMCI battery, per-real edge sets ({N_REAL} reals)")
    int_part = run_int_battery()
if STAGE in ("all", "dyn"):
    print(f"\n[dyn] teacher-forced response graphs ({N_WIN} win x {N_STEPS} "
          f"steps, integral stat, alpha={ALPHA_DYN})")
    dyn_part = run_dyn_battery()
if STAGE in ("all", "cal"):
    int_part = np.load(INT_PART, allow_pickle=True).item()
    dyn_part = np.load(DYN_PART, allow_pickle=True).item()
    calibrate(int_part, dyn_part)
