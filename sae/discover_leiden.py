"""
Unknown-N mode discovery via community detection (litext plan R4 SCALE row;
Stage-1 route "b" in notes/literature_extension_experiments.md:80-91, and the
Ising/Leiden composition of notes/graphcast_interp_design.md §4).

Motivation: on the R4 scale rung the mode count N is UNKNOWN a priori, so the
fixed-N Stage-1 operators in discover_modes.py (varimax / k-means with C0 given)
do not apply.  Community detection estimates N̂ from the data: build a
similarity graph over spatial sites (pixels, or per-node activations), then run
Leiden modularity optimisation — the number of recovered communities IS N̂, and
each community's spatial support IS a footprint map.  Leiden is favoured over
varimax here because it needs no target rank and degrades gracefully as N grows
(notes/literature_extension_experiments.md:276).

BACKENDS (auto-selected, best first):
  1. leidenalg + python-igraph  — true Leiden (RBConfigurationVertexPartition).
     *** NOT installed in the shared .venv as of this writing — coordinator
     should `pip install python-igraph leidenalg` to use it. ***
  2. spectral fallback (scipy + sklearn, always available) — normalised-Laplacian
     eigengap to estimate N̂, then KMeans on the spectral embedding.  Used for
     the CPU smoke test.  Community-detection semantics (unknown-N) preserved.

Output contract — MATCHES what discover_modes.py / sae/e1_discovery expect from
a Stage-1 builder:  What of shape (N̂, L), rows non-negative and L1-normalised
(nonneg footprint maps, `loading_to_footprint` convention), so the downstream
PCMCI battery (pool pixels through What -> PCMCI+) is reused unchanged.

CLI (smoke / standalone):
    python sae/discover_leiden.py \
        --data data/realisations_scale24 --disc 2 --out results/scale24_leiden.npy
Env knobs mirror the discover_modes vocabulary where sensible: LEI_KNN (15),
LEI_RES (1.0), LEI_VAR_FLOOR (pct, 40), LEI_MIN_SIZE (px, 8), LEI_MAX_PIX
(12000 subsample cap), LEI_FIELD (pixels|acts), LEI_SEED (0).
"""

import sys, os, argparse
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from pathlib import Path
from scipy import sparse
from scipy.sparse.csgraph import connected_components, laplacian
from sklearn.cluster import KMeans


# ── backend probe ────────────────────────────────────────────────────────────
def _have_leiden():
    try:
        import igraph          # noqa: F401
        import leidenalg       # noqa: F401
        return True
    except Exception:
        return False

HAVE_LEIDEN = _have_leiden()


# ── step 1: field -> standardised per-site time-course matrix ────────────────
def field_to_sites(field):
    """field (R, L, T) -> X (L, R*T) z-scored per site; plus per-site std (mass
    proxy) and an 'active' mask (sites carrying signal above a variance floor)."""
    R, L, T = field.shape
    X = field.transpose(1, 0, 2).reshape(L, R * T).astype(np.float64)
    mu = X.mean(1, keepdims=True)
    sd = X.std(1, keepdims=True)
    Xz = (X - mu) / (sd + 1e-9)
    return Xz, sd[:, 0], L


def active_mask(site_std, var_floor_pct):
    """keep sites whose temporal std exceeds the `var_floor_pct` percentile —
    isolates blob interiors from near-silent background before graph build."""
    thr = np.percentile(site_std, var_floor_pct)
    return site_std > thr


# ── step 2: symmetric kNN similarity graph over active sites ─────────────────
def knn_graph(Xz, knn, seed, max_pix):
    """cosine-kNN graph on z-scored rows. Returns (edges_i, edges_j, w, keep_idx)
    where keep_idx maps graph vertices back to global site indices."""
    n = Xz.shape[0]
    rng = np.random.default_rng(seed)
    keep = np.arange(n)
    if n > max_pix:                                   # subsample for scalability
        keep = np.sort(rng.choice(n, max_pix, replace=False))
    Xs = Xz[keep]
    Xn = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-12)
    S = Xn @ Xn.T                                     # (m, m) cosine similarity
    np.fill_diagonal(S, -np.inf)
    k = min(knn, S.shape[0] - 1)
    nbr = np.argpartition(-S, k, axis=1)[:, :k]       # top-k neighbours per row
    rows = np.repeat(np.arange(S.shape[0]), k)
    cols = nbr.ravel()
    w = np.clip(S[rows, cols], 0.0, None)             # non-negative edge weights
    # symmetrise (mutual-OR): keep max weight of (i,j)/(j,i)
    A = sparse.coo_matrix((w, (rows, cols)), shape=(S.shape[0], S.shape[0]))
    A = A.maximum(A.T).tocoo()
    return A.row, A.col, A.data, keep


# ── step 3a: Leiden partition (preferred backend) ────────────────────────────
def _partition_leiden(ei, ej, w, m, res, seed):
    import igraph, leidenalg
    mask = ei < ej                                    # undirected: one edge per pair
    g = igraph.Graph(n=m, edges=list(zip(ei[mask].tolist(), ej[mask].tolist())))
    g.es["weight"] = w[mask].tolist()
    part = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        weights="weight", resolution_parameter=res, seed=seed)
    return np.asarray(part.membership)


# ── step 3b: spectral fallback with eigengap N̂ estimation ────────────────────
def _partition_spectral(ei, ej, w, m, seed, n_max=40):
    """Estimate N̂ via the largest eigengap of the normalised Laplacian spectrum,
    then KMeans on the corresponding eigenvectors. Unknown-N, sklearn-only."""
    A = sparse.coo_matrix((w, (ei, ej)), shape=(m, m)).tocsr()
    A = A.maximum(A.T)
    # graph may be disconnected (disjoint blobs) — seed N̂ from components, then
    # refine within the low-frequency subspace.
    n_cc, cc = connected_components(A, directed=False)
    Ln = laplacian(A, normed=True).tocsr()
    kk = min(n_max, m - 2)
    try:
        from scipy.sparse.linalg import eigsh
        vals, vecs = eigsh(Ln, k=kk, which="SM", tol=1e-3, maxiter=5000)
    except Exception:
        dv, dvec = np.linalg.eigh(Ln.toarray())
        vals, vecs = dv[:kk], dvec[:, :kk]
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    gaps = np.diff(vals)
    # candidate N̂ = position of the largest eigengap (≥ #components, ≥2)
    n_hat = max(int(np.argmax(gaps)) + 1, n_cc, 2)
    n_hat = min(n_hat, kk)
    emb = vecs[:, :n_hat]
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    km = KMeans(n_clusters=n_hat, n_init=6, random_state=seed).fit(emb)
    return km.labels_, dict(n_cc=int(n_cc), eigengap_nhat=int(n_hat),
                            eigvals=vals[:min(len(vals), n_hat + 5)])


# ── step 4: communities -> L1-normalised footprint maps ──────────────────────
def communities_to_footprints(labels, keep_idx, site_std, L, min_size):
    """Each community's footprint = its member sites weighted by temporal std
    (blob-mass proxy), embedded back into the full L-vector, L1-normalised.
    Communities smaller than `min_size` (the background/noise crumbs) dropped."""
    rows = []
    for c in np.unique(labels):
        members = keep_idx[labels == c]
        if members.size < min_size:
            continue
        fp = np.zeros(L)
        fp[members] = site_std[members]
        s = fp.sum()
        if s > 0:
            rows.append(fp / s)
    return np.stack(rows) if rows else np.zeros((0, L))


# ── top-level driver ─────────────────────────────────────────────────────────
def discover_leiden(field, knn=15, res=1.0, var_floor_pct=40.0, min_size=8,
                    max_pix=12000, seed=0, verbose=True):
    """field (R, L, T) -> dict(what=(N̂, L), n_hat, backend, ...)."""
    Xz, site_std, L = field_to_sites(field)
    act = active_mask(site_std, var_floor_pct)
    idx_active = np.where(act)[0]
    ei, ej, w, keep_local = knn_graph(Xz[idx_active], knn, seed, max_pix)
    keep_idx = idx_active[keep_local]                 # graph vertex -> global site
    m = keep_idx.size
    extra = {}
    if HAVE_LEIDEN:
        backend = "leidenalg"
        labels = _partition_leiden(ei, ej, w, m, res, seed)
    else:
        backend = "spectral-eigengap (fallback; leidenalg/igraph not installed)"
        labels, extra = _partition_spectral(ei, ej, w, m, seed)
    what = communities_to_footprints(labels, keep_idx, site_std, L, min_size)
    n_hat = what.shape[0]
    if verbose:
        sizes = np.bincount(labels)
        print(f"[discover_leiden] backend={backend}")
        print(f"  active sites {m}/{L} (var floor p{var_floor_pct:g}); "
              f"kNN={knn} edges={ei.size}")
        print(f"  raw communities={len(np.unique(labels))} "
              f"(sizes {np.sort(sizes)[::-1][:8]}...)  "
              f"-> N̂={n_hat} after min_size={min_size} prune")
        if extra:
            print(f"  eigengap N̂={extra['eigengap_nhat']} "
                  f"(connected components={extra['n_cc']})")
    return dict(what=what, n_hat=n_hat, backend=backend, labels=labels,
                keep_idx=keep_idx, **extra)


# ── drop-in E1 (discover_modes.py) Stage-1 builder ───────────────────────────
def cand_leiden(field, **kw):
    """Signature matches discover_modes.py's builders: (field) -> (What, coh).
    Register in the E1 BUILDERS list as:
        ("leiden", discover_leiden.cand_leiden, S_PIX[..., :T_EFF])
    (pixel-side candidate; needs no checkpoint, so runs before acts exist).
    Returns a per-row coherence proxy = fraction of the footprint's L1 mass, so
    E1's `coherence_prune` — if the caller re-applies it — will drop the diffuse
    background community and pull N̂ from ~N+1 to N.  Env knobs: LEI_* as above."""
    res = discover_leiden(
        field,
        knn=int(os.environ.get("LEI_KNN", 15)),
        res=float(os.environ.get("LEI_RES", 1.0)),
        var_floor_pct=float(os.environ.get("LEI_VAR_FLOOR", 40.0)),
        min_size=int(os.environ.get("LEI_MIN_SIZE", 8)),
        max_pix=int(os.environ.get("LEI_MAX_PIX", 12000)),
        seed=int(os.environ.get("LEI_SEED", 0)),
        verbose=True,
    )
    what = res["what"]
    # coherence proxy: spatial concentration (peak share) of each footprint —
    # diffuse background communities score low, tight blobs score high.
    coh = np.array([float(row.max() / (row.sum() + 1e-12)) for row in what]) \
        if what.shape[0] else np.zeros(0)
    return what, coh


# ── loader mirroring discover_modes' discovery-realisation convention ────────
def load_field(data_dir, n_disc, which="pixels"):
    paths = sorted(Path(data_dir).glob("realisation_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no realisation_*.npz under {data_dir}")
    disc = list(range(len(paths) - n_disc, len(paths)))    # tail = discovery reals
    stack = [np.load(paths[ri])["observations"].astype(np.float32) for ri in disc]
    field = np.stack(stack)                                 # (R, L, T)
    return field, disc, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("LEI_DATA", "data/realisations_scale24"))
    ap.add_argument("--disc", type=int, default=int(os.environ.get("LEI_DISC", 2)))
    ap.add_argument("--knn", type=int, default=int(os.environ.get("LEI_KNN", 15)))
    ap.add_argument("--res", type=float, default=float(os.environ.get("LEI_RES", 1.0)))
    ap.add_argument("--var-floor", type=float, default=float(os.environ.get("LEI_VAR_FLOOR", 40.0)))
    ap.add_argument("--min-size", type=int, default=int(os.environ.get("LEI_MIN_SIZE", 8)))
    ap.add_argument("--max-pix", type=int, default=int(os.environ.get("LEI_MAX_PIX", 12000)))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("LEI_SEED", 0)))
    ap.add_argument("--out", default=os.environ.get("LEI_OUT", ""))
    args = ap.parse_args()

    field, disc, paths = load_field(args.data, args.disc)
    print(f"loaded {field.shape[0]} discovery reals {disc} from {args.data}  "
          f"field shape (R,L,T)={field.shape}")
    res = discover_leiden(field, knn=args.knn, res=args.res,
                          var_floor_pct=args.var_floor, min_size=args.min_size,
                          max_pix=args.max_pix, seed=args.seed)

    # footprint-vs-truth sanity (Hungarian cosine) if W present in the data
    d0 = np.load(paths[0])
    if "W" in d0 and res["what"].shape[0] > 0:
        from scipy.optimize import linear_sum_assignment
        Wt = d0["W"].astype(np.float64)
        Wt = Wt / (np.linalg.norm(Wt, axis=1, keepdims=True) + 1e-12)
        Wh = res["what"] / (np.linalg.norm(res["what"], axis=1, keepdims=True) + 1e-12)
        M = Wh @ Wt.T
        ri, ci = linear_sum_assignment(-M)
        cos = M[ri, ci]
        matched = int((cos >= 0.30).sum())
        print(f"  footprint check: N_true={Wt.shape[0]}  N̂={res['what'].shape[0]}  "
              f"matched(cos≥.30)={matched}  mean matched cos={cos[cos>=0.30].mean():.3f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        np.save(args.out, dict(what=res["what"], n_hat=res["n_hat"],
                               backend=res["backend"], disc_reals=disc),
                allow_pickle=True)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
