"""
E1 — Unsupervised mode-discovery bake-off (litext plan, notes/literature_extension_experiments.md §3).

Question: how much of the oracle-W activation-graph result (FU1: F1 0.855 =
true-Z ceiling) survives when the aggregation map W must be DISCOVERED from the
frozen model's internals, with no access to W or Z?

Stage-1 candidates (fit on held-out "discovery" realisations, disjoint from the
PCMCI eval realisations):
  vmax_act — varimax-rotated PCA on the per-node activation scalar field
             (the classic climate mode-discovery operator, run on INTERNALS)
  vmax_pix — same on raw pixels (control: is discovery easier from data or
             from the representation?)
  km_act   — k-means clustering of z-scored per-node activation time courses
  km_pix   — same on raw pixel time courses
  dmd_act  — k-means on |DMD mode| loadings of the activation field

Corrupted variants of the true W (populate the quality axis for E2):
  merge01, coarse4 (merged), split7, fine16 (split), shift5 (misplaced),
  random8 (misplaced, random disjoint), oracle (anchor).

For every candidate: footprint metrics vs true W (Hungarian cosine / support
IoU, N-hat), then the graph: pool raw pixels through W-hat -> PCMCI+ (ParCorr,
tau_max = max fine lag, alpha 0.05, N_REAL reals; byte-identical protocol to
Block G) -> F1 vs ground-truth Phi under Hungarian-strict variable mapping
(edges touching unmatched variables count as FP; gt edges at unmatched true
modes count as FN).

Fully-internal path (best internal candidate + oracle): pool per-node GNN
activations through W-hat, per-mode PC1 readout (unsupervised — no Z anywhere),
PCMCI+ as above.

Output: results/litext_e1_discovery.npy (+ per-candidate partial cache)
Env: E1_NREAL (24), E1_DISC (4 discovery reals from the tail), E1_C0 (12
initial components), E1_COH (0.25 coherence floor), E1_STAGE (all|footprints)
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
DATA_DIR  = Path(os.environ.get("E1_DATA_DIR", ROOT / "data/realisations_hetdynamics_eqvar"))
SAE_DIR   = Path(os.environ.get("E1_SAE_DIR", ROOT / "sae_data/hetdynamics_eqvar"))
CKPT      = Path(os.environ.get("E1_CKPT", ROOT / "checkpoints/hetdynamics_eqvar/best.pt"))
RES_DIR   = ROOT / "results"
E1_TAG    = os.environ.get("E1_TAG", "")           # suffix for rung reruns
PARTIAL   = RES_DIR / f"litext_e1_discovery_partial{E1_TAG}.npy"

N_REAL   = int(os.environ.get("E1_NREAL", 24))
N_DISC   = int(os.environ.get("E1_DISC", 4))
C0       = int(os.environ.get("E1_C0", 12))
COH_MIN  = float(os.environ.get("E1_COH", 0.25))
STAGE    = os.environ.get("E1_STAGE", "all")
PC_ALPHA = 0.05
K        = 3
N_MODES  = 8
NY = NX  = 50
L        = NY * NX
HIDDEN   = 256
MATCH_COS_MIN = 0.30
SEED     = 0

paths = sorted(DATA_DIR.glob("realisation_*.npz"))
d0 = np.load(paths[0])
W_TRUE = d0["W"].astype(np.float64)                    # (8, 2500)
gt = {(int(c), int(e), int(l)) for c, e, l, _ in d0["fine_edges"]
      if int(c) != int(e)}
TAU_MAX = max(l for _, _, l in gt)
T_TOTAL = int(d0["observations"].shape[1])
T_EFF   = T_TOTAL - K
DISC_REALS = list(range(len(paths) - N_DISC, len(paths)))   # e.g. 96..99
assert max(range(N_REAL)) < min(DISC_REALS)

# ── step 0: per-node activation scalar field on discovery reals (cached) ─────
os.makedirs(SAE_DIR, exist_ok=True)
SCAL_CACHE = SAE_DIR / "litext_node_scalar_acts.npy"
PC1_CACHE  = SAE_DIR / "litext_channel_pc1.npy"

def extract_node_scalar():
    import torch
    sys.path.insert(0, str(ROOT / "train" / "gnn"))
    from gnn_forecaster import MeshGNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MeshGNN(ny=NY, nx=NX, k=K).to(device)
    ckpt = torch.load(CKPT, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    captured = {}
    model.layers[-1].register_forward_hook(lambda m, i, o: captured.update(act=o))
    BS = 64
    rng = np.random.default_rng(SEED)
    # pass 1: channel-space PC1 from sampled H vectors
    samples = []
    with torch.no_grad():
        for ri in DISC_REALS[:2]:
            d = np.load(paths[ri])
            frames = torch.from_numpy(
                d["observations"].astype(np.float32).T.reshape(T_TOTAL, NY, NX))
            for i in range(0, 512, BS):
                w = torch.stack([frames[t:t + K] for t in range(i, i + BS)])
                model(w.to(device))
                H = captured["act"].cpu().numpy()          # (B, L, 256)
                idx = rng.choice(H.shape[0] * L, 4000, replace=False)
                samples.append(H.reshape(-1, HIDDEN)[idx])
    S = np.concatenate(samples, 0)
    mu = S.mean(0)
    v1 = PCA(n_components=1).fit(S - mu).components_[0]
    # pass 2: scalar field s(node,t) = (H - mu) . v1
    out = np.empty((N_DISC, L, T_EFF), dtype=np.float32)
    mu_t = torch.from_numpy(mu.astype(np.float32)).to(device)
    v1_t = torch.from_numpy(v1.astype(np.float32)).to(device)
    with torch.no_grad():
        for k_r, ri in enumerate(DISC_REALS):
            d = np.load(paths[ri])
            frames = torch.from_numpy(
                d["observations"].astype(np.float32).T.reshape(T_TOTAL, NY, NX))
            for i in range(0, T_EFF, BS):
                w = torch.stack([frames[t:t + K]
                                 for t in range(i, min(i + BS, T_EFF))])
                model(w.to(device))
                s = ((captured["act"] - mu_t) @ v1_t).cpu().numpy()  # (B, L)
                out[k_r, :, i:i + s.shape[0]] = s.T
            print(f"  [scalar extract {k_r+1}/{N_DISC}]")
    np.save(SCAL_CACHE, out)
    np.save(PC1_CACHE, np.stack([mu, v1]))
    return out

# E1_BUILDERS: comma list of stage-1 builders to run (default: all five);
# E1_SKIP_ACTS=1 also skips step 4 (fully-internal path). Lets a rung run its
# pixel-side battery before the checkpoint exists, then resume with acts.
BUILDER_SET = set(os.environ.get(
    "E1_BUILDERS", "vmax_act,vmax_pix,km_act,km_pix,dmd_act").split(","))
SKIP_ACTS = os.environ.get("E1_SKIP_ACTS", "0") == "1"
NEED_ACT_FIELD = bool(BUILDER_SET & {"vmax_act", "km_act", "dmd_act"})

if NEED_ACT_FIELD:
    if SCAL_CACHE.exists():
        print(f"[0] using cache {SCAL_CACHE}")
        S_ACT = np.load(SCAL_CACHE)                    # (N_DISC, L, T_EFF)
    else:
        S_ACT = extract_node_scalar()
else:
    S_ACT = None

S_PIX = np.stack([np.load(paths[ri])["observations"].astype(np.float32)
                  for ri in DISC_REALS])               # (N_DISC, L, T)

# ── step 1: candidate builders ────────────────────────────────────────────────
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
    """signed loading (L,) -> nonneg L1-normalized footprint row."""
    if load[np.argmax(np.abs(load))] < 0:
        load = -load
    fp = np.clip(load, 0, None)
    fp[fp < 0.05 * fp.max()] = 0.0
    s = fp.sum()
    return fp / s if s > 0 else fp

def coherence_prune(What, field):
    """keep rows whose member nodes cohere with the pooled series."""
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
    X = field.transpose(0, 2, 1).reshape(-1, L)        # (R*T, L)
    X = X - X.mean(0)
    pca = PCA(n_components=C0, random_state=SEED).fit(X)
    loads = (pca.components_ * np.sqrt(pca.explained_variance_)[:, None]).T
    rot = varimax(loads)                                # (L, C0)
    What = np.stack([loading_to_footprint(rot[:, c]) for c in range(C0)])
    return coherence_prune(What, field)

def cand_kmeans(field):
    # node time-course matrix: concat reals along time
    Xn = field.transpose(1, 0, 2).reshape(L, -1)        # (L, R*T)
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
        m = np.abs(dmd.modes)                           # (L, r)
        embs.append(m / (np.linalg.norm(m, axis=0, keepdims=True) + 1e-12))
    E = np.concatenate(embs, axis=1)                    # (L, R*20)
    km = KMeans(n_clusters=C0, n_init=4, random_state=SEED).fit(E)
    What = np.stack([(km.labels_ == c).astype(np.float64) for c in range(C0)])
    What = What / np.maximum(What.sum(1, keepdims=True), 1)
    return coherence_prune(What, field)

# corrupted variants of the true W
def build_corrupted():
    rng = np.random.default_rng(SEED)
    out = {}
    def norm(M):
        return M / np.maximum(M.sum(1, keepdims=True), 1e-12)
    out["merge01"] = norm(np.vstack([(W_TRUE[0] + W_TRUE[1])[None], W_TRUE[2:]]))
    out["coarse4"] = norm(np.stack([W_TRUE[2*i] + W_TRUE[2*i+1] for i in range(4)]))
    w7 = W_TRUE[7].reshape(NY, NX)
    cols = np.where(w7.sum(0) > 0)[0]
    mid = cols[len(cols)//2]
    lft, rgt = w7.copy(), w7.copy()
    lft[:, mid:] = 0; rgt[:, :mid] = 0
    out["split7"] = norm(np.vstack([W_TRUE[:7], lft.reshape(1, L), rgt.reshape(1, L)]))
    halves = []
    for j in range(N_MODES):
        wj = W_TRUE[j].reshape(NY, NX)
        rows = np.where(wj.sum(1) > 0)[0]
        rmid = rows[len(rows)//2]
        top, bot = wj.copy(), wj.copy()
        top[rmid:, :] = 0; bot[:rmid, :] = 0
        halves += [top.reshape(L), bot.reshape(L)]
    out["fine16"] = norm(np.stack(halves))
    out["shift5"] = norm(np.stack([np.roll(W_TRUE[j].reshape(NY, NX), 5, axis=1)
                                   .reshape(L) for j in range(N_MODES)]))
    # diag8: whole blob lattice shifted diagonally by half a blob (8,8) —
    # each footprint straddles four true blobs at ~25% each
    out["diag8"] = norm(np.stack(
        [np.roll(np.roll(W_TRUE[j].reshape(NY, NX), 8, axis=0), 8, axis=1)
         .reshape(L) for j in range(N_MODES)]))
    # blur: heavily smeared footprints spilling into neighbours
    from scipy.ndimage import gaussian_filter
    out["blur"] = norm(np.stack(
        [gaussian_filter(W_TRUE[j].reshape(NY, NX), sigma=6).reshape(L)
         for j in range(N_MODES)]))
    return out

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
for name, What in build_corrupted().items():
    CANDS[name] = What
    print(f"  {name}: C={What.shape[0]}")
CANDS["oracle"] = W_TRUE / W_TRUE.sum(1, keepdims=True)
# E1_ONLY: restrict the battery to a comma list of candidates (targeted reruns)
if os.environ.get("E1_ONLY"):
    _keep = set(os.environ["E1_ONLY"].split(","))
    CANDS = {k: v for k, v in CANDS.items() if k in _keep}
    print(f"  [E1_ONLY] restricted to {sorted(CANDS)}")

# ── step 2: footprint metrics ─────────────────────────────────────────────────
def footprint_metrics(What):
    Wt = W_TRUE / np.linalg.norm(W_TRUE, axis=1, keepdims=True)
    Wh = What / (np.linalg.norm(What, axis=1, keepdims=True) + 1e-12)
    M = Wh @ Wt.T                                       # (C, 8) cosine
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
    print(f"    {name:<10} {fm['n_hat']:>3} {fm['n_matched']:>4}/8 "
          f"{fm['mean_cos']:>6.3f} {fm['mean_iou']:>6.3f}")

if STAGE == "footprints":
    np.save(RES_DIR / f"litext_e1_footprints{E1_TAG}.npy",
            dict(cands={k: v for k, v in CANDS.items()}, foot=FOOT),
            allow_pickle=True)
    print("footprints-only stage done"); sys.exit(0)

# ── step 3: PCMCI battery (pixels pooled through each W-hat) ─────────────────
def detect(graph):
    N, _, T1 = graph.shape
    return {(c, e, tau) for c in range(N) for e in range(N) if c != e
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
    with ProcessPoolExecutor(max_workers=4, initializer=_worker_init) as ex:
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
        # per-real EDGE SETS in e4_agreement.py INT_PART format (plan §7.2) —
        # lets E4's int stage reuse this battery instead of rerunning PCMCI
        sets_part = (np.load(INT_SETS_PATH, allow_pickle=True).item()
                     if INT_SETS_PATH.exists() else {})
        sets_part[tag] = dets_per_real
        np.save(INT_SETS_PATH, sets_part, allow_pickle=True)
    p, r, f1 = prf(agg["tp"], agg["fp"], agg["fn"])
    print(f"    {tag:<14} F1={f1:.3f} P={p:.2f} R={r:.2f} "
          f"(tp={agg['tp']} fp={agg['fp']} fn={agg['fn']})")
    return dict(P=p, R=r, F1=f1, **agg)

def pix_series(ri, What):
    obs = np.load(paths[ri])["observations"].astype(np.float64)
    return (What @ obs).T                               # (T, C)

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

# ── step 4: fully-internal path for best internal candidate + oracle ─────────
if SKIP_ACTS:
    np.save(RES_DIR / f"litext_e1_discovery{E1_TAG}.npy",
            dict(cands={k: v for k, v in CANDS.items()},
                 foot={k: {kk: vv for kk, vv in v.items() if kk != "cos_matrix"}
                       for k, v in FOOT.items()},
                 graph=GRAPH, acts_rows={}, best_internal=None,
                 n_real=N_REAL, disc_reals=DISC_REALS, pc_alpha=PC_ALPHA,
                 tau_max=TAU_MAX, coh_min=COH_MIN, c0=C0,
                 note="E1 pixel-side only (E1_SKIP_ACTS=1)"),
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
    from gnn_forecaster import MeshGNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MeshGNN(ny=NY, nx=NX, k=K).to(device)
    ckpt = torch.load(CKPT, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    captured = {}
    model.layers[-1].register_forward_hook(lambda m, i, o: captured.update(act=o))
    W_t = torch.from_numpy(What.astype(np.float32)).to(device)
    BS = 64
    out = np.empty((len(reals), What.shape[0], T_EFF, HIDDEN), dtype=np.float32)
    with torch.no_grad():
        for k_r, ri in enumerate(reals):
            frames = torch.from_numpy(np.load(paths[ri])["observations"]
                                      .astype(np.float32).T.reshape(T_TOTAL, NY, NX))
            chunks = []
            for i in range(0, T_EFF, BS):
                w = torch.stack([frames[t:t + K]
                                 for t in range(i, min(i + BS, T_EFF))])
                model(w.to(device))
                chunks.append(torch.einsum("jl,blc->bjc", W_t,
                                           captured["act"]).cpu().numpy())
            out[k_r] = np.concatenate(chunks, 0).transpose(1, 0, 2)
    return out

for name in ([best_int] if best_int else []) + ["oracle"]:
    What = CANDS[name]
    pooled_disc = extract_pooled(What, DISC_REALS)      # fit PC1 per mode
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
             anchors=dict(oracle_pix_blockG=0.853, oracle_acts_fu1=0.855,
                          trueZ=0.853),
             note="E1 litext: unsupervised mode discovery -> pixel/activation "
                  "pooling -> PCMCI+; Hungarian-strict edge mapping"),
        allow_pickle=True)
print(f"\nsaved -> results/litext_e1_discovery{E1_TAG}.npy")
