"""
E1-MULTIVAR — Unsupervised mode-discovery bake-off, R3 multivariate rung
(channel-aware fork of sae/discover_modes.py).

WHAT CHANGES vs the single-channel parent (everything else — builders, coherence
prune, PCMCI+ battery protocol, Hungarian-strict behavior matching, env knobs,
output conventions — is byte-faithful):

  * Observations are (C, L, T): C coupled observed channels per node. They are
    FLATTENED channel-major to a (C*L, T) pixel field; every pixel-side operator
    (varimax / k-means candidate builders, coherence prune, footprint metrics,
    pooling) runs over that (C*L) field.
  * There are NC = N*C latent modes (channel-major index m = channel*N + node).
    The true aggregation is the BLOCK-DIAGONAL  Ŵ = I_C ⊗ W  of shape (NC, C*L):
    row m=(ch,n) pools footprint W[n] out of channel-block ch of the flat field.
    That block-diagonal Ŵ is the oracle candidate here (was the (8, L) W).
  * Ground truth Φ is the NC-mode graph from `fine_edges` (already channel-major
    stacked: rows (cause_m, effect_m, lag, coeff)); scoring is behavior-based
    against that NC-mode graph exactly as the parent scored the 8-mode graph.

Corrupted footprint anchors (merge/split/shift…) are 8-mode-and-single-channel
specific; they are gated off here (NC != 8) — see note in build_corrupted.

Env knobs & outputs are IDENTICAL to the parent (E1_TAG, E1_BUILDERS,
E1_SKIP_ACTS, E1_NREAL, E1_DISC, E1_C0 …); results ->
results/litext_e1_discovery{TAG}.npy. Run the multivar rung with E1_TAG=_multivar.

Env additions: E1_DATA_DIR defaults to data/realisations_multivar2; E1_CKPT to
checkpoints/multivar/best.pt (the multivar MeshGNN). C, N, NC, tau_max are read
from mv_meta.
"""

import sys, os
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = Path(os.environ.get("E1_DATA_DIR", ROOT / "data/realisations_multivar2"))
SAE_DIR   = Path(os.environ.get("E1_SAE_DIR", ROOT / "sae_data/multivar"))
CKPT      = Path(os.environ.get("E1_CKPT", ROOT / "checkpoints/multivar/best.pt"))
RES_DIR   = ROOT / "results"
E1_TAG    = os.environ.get("E1_TAG", "")           # suffix for rung reruns
PARTIAL   = RES_DIR / f"litext_e1_discovery_partial{E1_TAG}.npy"

N_REAL   = int(os.environ.get("E1_NREAL", 24))
N_DISC   = int(os.environ.get("E1_DISC", 4))
COH_MIN  = float(os.environ.get("E1_COH", 0.25))
STAGE    = os.environ.get("E1_STAGE", "all")
PC_ALPHA = 0.05
K        = 3
NY = NX  = int(os.environ.get("E1_GRID", 50))
HIDDEN   = int(os.environ.get("E1_HIDDEN", 256))
MATCH_COS_MIN = 0.30
SEED     = 0

paths = sorted(DATA_DIR.glob("realisation_*.npz"))
d0 = np.load(paths[0])
# ── multivar layout from mv_meta [N, C, L, NC, tau_max] ──────────────────────
N_S, C, L_CH, NC, TAU_META = (int(x) for x in d0["mv_meta"])
LC       = C * L_CH                                    # flattened (channel × space)
N_MODES  = NC                                          # true mode count (16 for C=2)
L        = LC                                          # builders/prune operate over the C*L field
C0       = int(os.environ.get("E1_C0", NC))            # initial component budget (>= NC so builders can reach NC)

# BLOCK-DIAGONAL true aggregation  Ŵ = I_C ⊗ W  → (NC, C*L)
W_SINGLE = d0["W"].astype(np.float64)                  # (N, L_CH) shared footprints
W_TRUE   = np.zeros((NC, LC), dtype=np.float64)
for ch in range(C):
    W_TRUE[ch * N_S:(ch + 1) * N_S, ch * L_CH:(ch + 1) * L_CH] = W_SINGLE

# ground truth from fine_edges (channel-major stacked (cause_m, effect_m, lag))
gt = {(int(c), int(e), int(l)) for c, e, l, _ in d0["fine_edges"]
      if int(c) != int(e)}
TAU_MAX = max(l for _, _, l in gt)
T_TOTAL = int(d0["observations"].shape[2])             # obs is (C, L, T)
T_EFF   = T_TOTAL - K
DISC_REALS = list(range(len(paths) - N_DISC, len(paths)))
assert max(range(N_REAL)) < min(DISC_REALS), \
    f"need N_REAL({N_REAL}) + N_DISC({N_DISC}) <= #reals({len(paths)})"

# ── step 0: per-node activation scalar field on discovery reals (cached) ─────
os.makedirs(SAE_DIR, exist_ok=True)
SCAL_CACHE = SAE_DIR / "litext_node_scalar_acts.npy"
PC1_CACHE  = SAE_DIR / "litext_channel_pc1.npy"


def _load_frames(path):
    """(C, L, T) obs -> (T, C, NY, NX) float32 frames for the multivar GNN."""
    obs = np.load(path)["observations"].astype(np.float32)          # (C, L, T)
    return np.transpose(obs, (2, 0, 1)).reshape(T_TOTAL, C, NY, NX)


def _windows(frames, starts):
    """starts -> (B, C, K, NY, NX) tensor windows for MeshGNN.forward."""
    import torch
    w = torch.stack([frames[t:t + K] for t in starts])              # (B, K, C, NY, NX)
    return w.permute(0, 2, 1, 3, 4).contiguous()                    # (B, C, K, NY, NX)


def _tile_channels(H):
    """Per-spatial-node hidden H (B, L_CH, hidden) -> channel-tiled (B, C*L_CH,
    hidden) so the BLOCK-DIAGONAL Ŵ (NC, C*L) can pool it into NC modes.
    NOTE (documented limitation): the multivar MeshGNN carries a SINGLE hidden
    vector per SPATIAL node (channels are mixed inside the 256-dim state), so
    co-located modes (ch0,n) and (ch1,n) receive the SAME pooled vector here.
    The activation path therefore cannot separate channels of the same footprint
    without a channel-aware readout — the pixel-side battery (below) does not
    share this limitation. See notes in the deliverable."""
    import torch
    return torch.cat([H] * C, dim=1)                                # (B, C*L_CH, hidden)


def extract_node_scalar():
    import torch
    sys.path.insert(0, str(ROOT / "train" / "gnn"))
    from gnn_forecaster_multivar import MeshGNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MeshGNN(ny=NY, nx=NX, k=K, channels=C).to(device)
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    captured = {}
    model.layers[-1].register_forward_hook(lambda m, i, o: captured.update(act=o))
    BS = 64
    rng = np.random.default_rng(SEED)
    n_samp = min(512, T_EFF)
    # pass 1: channel-space PC1 from sampled tiled-H vectors
    samples = []
    with torch.no_grad():
        for ri in DISC_REALS[:2]:
            frames = torch.from_numpy(_load_frames(paths[ri]))
            for i in range(0, n_samp, BS):
                starts = range(i, min(i + BS, n_samp))
                model(_windows(frames, starts).to(device))
                H = _tile_channels(captured["act"]).cpu().numpy()   # (B, C*L, 256)
                idx = rng.choice(H.shape[0] * LC, min(4000, H.shape[0] * LC),
                                 replace=False)
                samples.append(H.reshape(-1, HIDDEN)[idx])
    S = np.concatenate(samples, 0)
    mu = S.mean(0)
    v1 = PCA(n_components=1).fit(S - mu).components_[0]
    # pass 2: scalar field s(pix,t) = (H - mu) . v1  over the C*L flat field
    out = np.empty((N_DISC, LC, T_EFF), dtype=np.float32)
    mu_t = torch.from_numpy(mu.astype(np.float32)).to(device)
    v1_t = torch.from_numpy(v1.astype(np.float32)).to(device)
    with torch.no_grad():
        for k_r, ri in enumerate(DISC_REALS):
            frames = torch.from_numpy(_load_frames(paths[ri]))
            for i in range(0, T_EFF, BS):
                starts = range(i, min(i + BS, T_EFF))
                model(_windows(frames, starts).to(device))
                H = _tile_channels(captured["act"])                 # (B, C*L, 256)
                s = ((H - mu_t) @ v1_t).cpu().numpy()               # (B, C*L)
                out[k_r, :, i:i + s.shape[0]] = s.T
            print(f"  [scalar extract {k_r+1}/{N_DISC}]")
    np.save(SCAL_CACHE, out)
    np.save(PC1_CACHE, np.stack([mu, v1]))
    return out

# E1_BUILDERS / E1_SKIP_ACTS: identical semantics to the parent.
BUILDER_SET = set(os.environ.get(
    "E1_BUILDERS", "vmax_act,vmax_pix,km_act,km_pix,dmd_act").split(","))
SKIP_ACTS = os.environ.get("E1_SKIP_ACTS", "0") == "1"
NEED_ACT_FIELD = bool(BUILDER_SET & {"vmax_act", "km_act", "dmd_act"})

if NEED_ACT_FIELD:
    if SCAL_CACHE.exists():
        print(f"[0] using cache {SCAL_CACHE}")
        S_ACT = np.load(SCAL_CACHE)                    # (N_DISC, C*L, T_EFF)
    else:
        S_ACT = extract_node_scalar()
else:
    S_ACT = None

# raw pixel field, flattened channel-major to (C*L, T)
S_PIX = np.stack([np.load(paths[ri])["observations"].astype(np.float32)
                  .reshape(LC, T_TOTAL) for ri in DISC_REALS])       # (N_DISC, C*L, T)

# ── step 1: candidate builders (operate over the C*L flat field) ─────────────
def varimax(Phi, gamma=1.0, q=200, tol=1e-8):
    p, k = Phi.shape
    R = np.eye(k); d = 0.0
    for _ in range(q):
        Lm = Phi @ R
        u, s, vt = np.linalg.svd(
            Phi.T @ (Lm ** 3 - (gamma / p) * Lm @ np.diag((Lm ** 2).sum(0))))
        R = u @ vt
        d_new = s.sum()
        if d_new < d * (1 + tol):
            break
        d = d_new
    return Phi @ R

def loading_to_footprint(load):
    if load[np.argmax(np.abs(load))] < 0:
        load = -load
    fp = np.clip(load, 0, None)
    fp[fp < 0.05 * fp.max()] = 0.0
    s = fp.sum()
    return fp / s if s > 0 else fp

def coherence_prune(What, field):
    keep, coh = [], []
    Tc = min(field.shape[-1], 1200)
    F = field[..., :Tc].reshape(field.shape[0], L, Tc)
    for c in range(What.shape[0]):
        y = np.einsum("l,rlt->rt", What[c], F)         # (R, Tc)
        members = np.where(What[c] > 0.01 * What[c].max())[0]
        members = members[np.argsort(-What[c][members])][:40]
        cs = []
        for r in range(F.shape[0]):
            yv = y[r] - y[r].mean()
            for m in members[:15]:
                xv = F[r, m] - F[r, m].mean()
                den = np.sqrt((xv**2).sum() * (yv**2).sum())
                if den > 0:
                    cs.append(abs(float((xv * yv).sum() / den)))
        c_mean = float(np.mean(cs)) if cs else 0.0
        coh.append(c_mean)
        if c_mean >= COH_MIN:
            keep.append(c)
    return What[keep] if keep else What[:0], np.array(coh)

def cand_varimax(field):
    X = field.transpose(0, 2, 1).reshape(-1, L)        # (R*T, C*L)
    X = X - X.mean(0)
    pca = PCA(n_components=C0, random_state=SEED).fit(X)
    loads = (pca.components_ * np.sqrt(pca.explained_variance_)[:, None]).T
    rot = varimax(loads)                                # (C*L, C0)
    What = np.stack([loading_to_footprint(rot[:, c]) for c in range(C0)])
    return coherence_prune(What, field)

def cand_kmeans(field):
    Xn = field.transpose(1, 0, 2).reshape(L, -1)        # (C*L, R*T)
    Xn = (Xn - Xn.mean(1, keepdims=True)) / (Xn.std(1, keepdims=True) + 1e-9)
    emb = PCA(n_components=50, random_state=SEED).fit_transform(Xn)
    km = KMeans(n_clusters=C0, n_init=4, random_state=SEED).fit(emb)
    What = np.stack([(km.labels_ == c).astype(np.float64) for c in range(C0)])
    What = What / np.maximum(What.sum(1, keepdims=True), 1)
    return coherence_prune(What, field)

def cand_dmd(field):
    from pydmd import DMD
    embs = []
    for r in range(field.shape[0]):
        A = field[r].astype(np.float64)
        A = (A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-9)
        dmd = DMD(svd_rank=20)
        dmd.fit(A)
        m = np.abs(dmd.modes)                           # (C*L, r)
        embs.append(m / (np.linalg.norm(m, axis=0, keepdims=True) + 1e-12))
    E = np.concatenate(embs, axis=1)                    # (C*L, R*20)
    km = KMeans(n_clusters=C0, n_init=4, random_state=SEED).fit(E)
    What = np.stack([(km.labels_ == c).astype(np.float64) for c in range(C0)])
    What = What / np.maximum(What.sum(1, keepdims=True), 1)
    return coherence_prune(What, field)

# Corrupted footprint anchors (merge01/coarse4/split7/…): these are hard-wired
# to N_MODES==8 single-channel footprints reshaped on a single NY×NX grid, so
# they are NOT meaningful in the (C*L) block-diagonal setting. They are gated off
# for NC != 8 exactly as the parent gates them (E2's quality axis for the
# multivar rung would need block-diagonal analogues — see deliverable note).

print("[1] building candidates")
CANDS = {}
BUILDERS = [("vmax_act", cand_varimax, S_ACT),
            ("vmax_pix", cand_varimax, S_PIX[..., :T_EFF]),
            ("km_act",   cand_kmeans,  S_ACT),
            ("km_pix",   cand_kmeans,  S_PIX[..., :T_EFF]),
            ("dmd_act",  cand_dmd,     S_ACT)]
BUILDERS = [(n, f, fld) for n, f, fld in BUILDERS if n in BUILDER_SET]
import time
for name, fn, field in BUILDERS:
    t0_ = time.time()
    What, coh = fn(field)
    CANDS[name] = What
    print(f"  {name}: N-hat={What.shape[0]} in {time.time()-t0_:.0f}s "
          f"(coh floor {COH_MIN}; "
          f"coherences {np.sort(coh)[::-1][:What.shape[0]+2].round(2)})")
CANDS["oracle"] = W_TRUE / W_TRUE.sum(1, keepdims=True)
if os.environ.get("E1_ONLY"):
    _keep = set(os.environ["E1_ONLY"].split(","))
    CANDS = {k: v for k, v in CANDS.items() if k in _keep}
    print(f"  [E1_ONLY] restricted to {sorted(CANDS)}")

# ── step 2: footprint metrics (Hungarian cosine vs block-diagonal true Ŵ) ─────
def footprint_metrics(What):
    Wt = W_TRUE / np.linalg.norm(W_TRUE, axis=1, keepdims=True)
    Wh = What / (np.linalg.norm(What, axis=1, keepdims=True) + 1e-12)
    M = Wh @ Wt.T                                       # (C_hat, NC) cosine
    ri, ci = linear_sum_assignment(-M)
    pairs = [(int(a), int(b)) for a, b in zip(ri, ci) if M[a, b] >= MATCH_COS_MIN]
    cos = [float(M[a, b]) for a, b in pairs]
    sup_t = W_TRUE > 0
    sup_h = What > (0.01 * What.max(1, keepdims=True))
    iou = [float((sup_h[a] & sup_t[b]).sum() / max((sup_h[a] | sup_t[b]).sum(), 1))
           for a, b in pairs]
    mapping = {a: b for a, b in pairs}
    return dict(n_hat=What.shape[0], n_matched=len(pairs),
                mean_cos=float(np.mean(cos)) if cos else 0.0,
                mean_iou=float(np.mean(iou)) if iou else 0.0,
                mapping=mapping, cos_matrix=M)

FOOT = {name: footprint_metrics(What) for name, What in CANDS.items()}
print(f"\n[2] {'candidate':<10} {'N^':>3} {'match':>6} {'cos':>6} {'IoU':>6}")
for name, fm in FOOT.items():
    print(f"    {name:<10} {fm['n_hat']:>3} {fm['n_matched']:>4}/{N_MODES} "
          f"{fm['mean_cos']:>6.3f} {fm['mean_iou']:>6.3f}")

if STAGE == "footprints":
    np.save(RES_DIR / f"litext_e1_footprints{E1_TAG}.npy",
            dict(cands={k: v for k, v in CANDS.items()}, foot=FOOT),
            allow_pickle=True)
    print("footprints-only stage done"); sys.exit(0)

# ── step 3: PCMCI battery (flat C*L pixels pooled through each W-hat) ─────────
def detect(graph):
    Nn, _, T1 = graph.shape
    return {(c, e, tau) for c in range(Nn) for e in range(Nn) if c != e
            for tau in range(1, T1) if graph[c, e, tau] == "-->"}

def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)

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

INT_SETS_PATH = RES_DIR / f"litext_e4_int_partial{E1_TAG}.npy"

def score_candidate(What, mapping, series_fn, tag, save_sets=False):
    agg = dict(tp=0, fp=0, fn=0)
    dets_per_real = {}
    jobs = [(ri, series_fn(ri, What)) for ri in range(N_REAL)]
    with ProcessPoolExecutor(max_workers=min(30, max(1, (os.cpu_count() or 6) - 2)), initializer=_worker_init) as ex:
        for ri, det in ex.map(pcmci_one, jobs):
            dets_per_real[ri] = sorted(det)
            mapped = set()
            fp_unmatched = 0
            for (c, e, tau) in det:
                if c in mapping and e in mapping:
                    mapped.add((mapping[c], mapping[e], tau))
                else:
                    fp_unmatched += 1
            agg["tp"] += len(gt & mapped)
            agg["fp"] += len(mapped - gt) + fp_unmatched
            agg["fn"] += len(gt - mapped)
    if save_sets:
        sets_part = (np.load(INT_SETS_PATH, allow_pickle=True).item()
                     if INT_SETS_PATH.exists() else {})
        sets_part[tag] = dets_per_real
        np.save(INT_SETS_PATH, sets_part, allow_pickle=True)
    p, r, f1 = prf(agg["tp"], agg["fp"], agg["fn"])
    print(f"    {tag:<14} F1={f1:.3f} P={p:.2f} R={r:.2f} "
          f"(tp={agg['tp']} fp={agg['fp']} fn={agg['fn']})")
    return dict(P=p, R=r, F1=f1, **agg)

def pix_series(ri, What):
    obs = np.load(paths[ri])["observations"].astype(np.float64).reshape(LC, -1)
    return (What @ obs).T                               # (T, C_hat)

partial = {}
if PARTIAL.exists():
    partial = np.load(PARTIAL, allow_pickle=True).item()
    print(f"[3] resuming: {sorted(partial)} cached")

print(f"\n[3] PCMCI battery ({N_REAL} reals each)")
GRAPH = {}
for name, What in CANDS.items():
    if name in partial:
        GRAPH[name] = partial[name]
        print(f"    {name:<14} (cached) F1={partial[name]['F1']:.3f}")
        continue
    GRAPH[name] = score_candidate(What, FOOT[name]["mapping"], pix_series, name,
                                  save_sets=True)
    partial[name] = GRAPH[name]
    np.save(PARTIAL, partial, allow_pickle=True)

# ── step 4: fully-internal path (pool ACTIVATIONS through block-diagonal Ŵ) ───
if SKIP_ACTS:
    np.save(RES_DIR / f"litext_e1_discovery{E1_TAG}.npy",
            dict(cands={k: v for k, v in CANDS.items()},
                 foot={k: {kk: vv for kk, vv in v.items() if kk != "cos_matrix"}
                       for k, v in FOOT.items()},
                 graph=GRAPH, acts_rows={}, best_internal=None,
                 n_real=N_REAL, disc_reals=DISC_REALS, pc_alpha=PC_ALPHA,
                 tau_max=TAU_MAX, coh_min=COH_MIN, c0=C0,
                 mv_meta=dict(N=N_S, C=C, L=L_CH, NC=NC, tau_max=TAU_META),
                 note="E1-multivar pixel-side only (E1_SKIP_ACTS=1)"),
            allow_pickle=True)
    print(f"\n[4] skipped (E1_SKIP_ACTS=1); saved -> "
          f"results/litext_e1_discovery{E1_TAG}.npy")
    sys.exit(0)

print("\n[4] fully-internal path (pool ACTIVATIONS through W-hat, PC1 readout)")
internal = {k: GRAPH[k]["F1"] for k in ("vmax_act", "km_act", "dmd_act")
            if k in GRAPH and CANDS[k].shape[0] > 0}
best_int = max(internal, key=internal.get) if internal else None
ACT_ROWS = {}

def extract_pooled(What, reals):
    import torch
    sys.path.insert(0, str(ROOT / "train" / "gnn"))
    from gnn_forecaster_multivar import MeshGNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MeshGNN(ny=NY, nx=NX, k=K, channels=C).to(device)
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    captured = {}
    model.layers[-1].register_forward_hook(lambda m, i, o: captured.update(act=o))
    W_t = torch.from_numpy(What.astype(np.float32)).to(device)       # (C_hat, C*L)
    BS = 64
    out = np.empty((len(reals), What.shape[0], T_EFF, HIDDEN), dtype=np.float32)
    with torch.no_grad():
        for k_r, ri in enumerate(reals):
            frames = torch.from_numpy(_load_frames(paths[ri]))
            chunks = []
            for i in range(0, T_EFF, BS):
                starts = range(i, min(i + BS, T_EFF))
                model(_windows(frames, starts).to(device))
                H = _tile_channels(captured["act"])                 # (B, C*L, 256)
                chunks.append(torch.einsum("jl,blc->bjc", W_t, H).cpu().numpy())
            out[k_r] = np.concatenate(chunks, 0).transpose(1, 0, 2)
    return out

for name in ([best_int] if best_int else []) + ["oracle"]:
    What = CANDS[name]
    pooled_disc = extract_pooled(What, DISC_REALS)
    pcs = []
    for c in range(What.shape[0]):
        Xc = pooled_disc[:, c].reshape(-1, HIDDEN)
        p = PCA(n_components=1, random_state=SEED).fit(Xc - Xc.mean(0))
        pcs.append((Xc.mean(0), p.components_[0]))
    pooled_eval = extract_pooled(What, list(range(N_REAL)))

    def act_series(ri, _W, _pe=pooled_eval, _pcs=pcs):
        return np.stack([(_pe[ri, c] - _pcs[c][0]) @ _pcs[c][1]
                         for c in range(len(_pcs))]).T

    tag = f"acts:{name}"
    if tag in partial:
        ACT_ROWS[name] = partial[tag]
        print(f"    {tag:<14} (cached) F1={partial[tag]['F1']:.3f}")
    else:
        ACT_ROWS[name] = score_candidate(What, FOOT[name]["mapping"],
                                         act_series, tag)
        partial[tag] = ACT_ROWS[name]
        np.save(PARTIAL, partial, allow_pickle=True)

os.makedirs(RES_DIR, exist_ok=True)
np.save(RES_DIR / f"litext_e1_discovery{E1_TAG}.npy",
        dict(cands={k: v for k, v in CANDS.items()},
             foot={k: {kk: vv for kk, vv in v.items() if kk != "cos_matrix"}
                   for k, v in FOOT.items()},
             graph=GRAPH, acts_rows=ACT_ROWS, best_internal=best_int,
             n_real=N_REAL, disc_reals=DISC_REALS, pc_alpha=PC_ALPHA,
             tau_max=TAU_MAX, coh_min=COH_MIN, c0=C0,
             mv_meta=dict(N=N_S, C=C, L=L_CH, NC=NC, tau_max=TAU_META),
             note="E1-multivar: unsupervised mode discovery -> block-diagonal "
                  "Ŵ pooling -> PCMCI+; Hungarian-strict behavior edge mapping. "
                  "Acts path pools a per-spatial-node hidden channel-tiled: "
                  "co-located modes share activation geometry (see deliverable)."),
        allow_pickle=True)
print(f"\nsaved -> results/litext_e1_discovery{E1_TAG}.npy")
