"""Block A redundancy diagnostics for a VPD run (agent session 2026-07-03).

Given a run dir whose diagnostics/ has been populated by vpd/diagnostics.py
(mean_g_map__*.npy (C,ny,nx), usage_by_mode__*.npy (C,n_modes),
hub_usage__*.npy, ablation_drop__*.npy), compute per decomposed module:

  * participation ratio (PR) of the C gate maps — (sum s^2)^2 / sum s^4 of the
    singular values of the (C, L) map matrix; PR≈1 => one shared spatial pattern
  * median |pairwise Pearson corr| between gate maps (and signed median)
  * blob-usage matrix (C, 8): mean gate over each SAVAR mode blob's support
    (pixels where W[j] > 0), and its per-component CV across the 8 blobs
    (median CV across components; ≈0.04 baseline = fires on all blobs equally)
  * mode segregation: dominant blob per component, count of components whose
    dominant-blob usage exceeds the runner-up by >20% ("mode-preferential")
  * Spearman corr between each component's dominant-mode phi timescale and the
    component's blob-usage entropy (do slow modes attract dedicated comps?)

Baselines to beat (finecadence run p-0625f7dd, notes/results_gnn.md section 5):
PR ~= 1.1/64, blob-usage CV ~= 0.04, median pairwise corr 0.79 (layer 0).

Usage:
  python3 vpd/analyze_redundancy.py <run_dir> <split_train_dir> [out_npy]
"""

import glob
import os
import sys

import numpy as np

PHI = np.array([0.15, 0.30, 0.42, 0.55, 0.68, 0.78, 0.86, 0.92])
TAU = -1.0 / np.log(PHI)  # decorrelation timescale per mode


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def participation_ratio(maps_cl, center=True):
    X = maps_cl - maps_cl.mean(0, keepdims=True) if center else maps_cl
    s = np.linalg.svd(X, compute_uv=False)
    s2 = s ** 2
    return float(s2.sum() ** 2 / (s2 ** 2).sum())


def analyse(run_dir, split_train_dir, out_npy=None):
    diag = os.path.join(run_dir, "diagnostics")
    map_files = sorted(glob.glob(os.path.join(diag, "mean_g_map__*.npy")))
    assert map_files, f"no mean_g_map__*.npy under {diag} — run vpd/diagnostics.py first"

    W = np.load(sorted(glob.glob(os.path.join(split_train_dir, "realisation_*.npz")))[0])["W"]
    n_modes, L = W.shape
    blob_masks = [W[j] > 0 for j in range(n_modes)]  # disjoint blob supports

    results = {}
    print(f"run: {run_dir}\nphi={PHI.tolist()}  tau={np.round(TAU, 2).tolist()}\n")
    hdr = (f"{'module':<18} {'C':>3} {'PR':>6} {'PR_unc':>7} {'med|r|':>7} {'med r':>7} "
           f"{'blobCV':>7} {'#pref':>6} {'sp(phi,ent)':>11}")
    print(hdr)
    print("-" * len(hdr))

    for f in map_files:
        tag = os.path.basename(f)[len("mean_g_map__"):-len(".npy")]
        maps = np.load(f)                       # (C, ny, nx)
        C = maps.shape[0]
        X = maps.reshape(C, -1)                 # (C, L)

        pr = participation_ratio(X, center=True)
        pr_unc = participation_ratio(X, center=False)

        # pairwise Pearson between component maps
        Xc = X - X.mean(1, keepdims=True)
        nrm = np.linalg.norm(Xc, axis=1) + 1e-12
        R = (Xc @ Xc.T) / np.outer(nrm, nrm)
        iu = np.triu_indices(C, 1)
        med_abs_r = float(np.median(np.abs(R[iu])))
        med_r = float(np.median(R[iu]))

        # blob usage (C, 8): mean gate over each blob's support
        blob = np.stack([X[:, m].mean(1) for m in blob_masks], axis=1)  # (C, 8)
        cv = blob.std(1) / (blob.mean(1) + 1e-12)                       # per-component
        med_cv = float(np.median(cv))

        # mode-preferential components: dominant blob >20% above runner-up
        srt = np.sort(blob, axis=1)
        pref = (srt[:, -1] > 1.2 * srt[:, -2])
        n_pref = int(pref.sum())
        dom = blob.argmax(1)                                            # (C,)

        # entropy of blob-usage profile (low entropy = specialized)
        p = blob / (blob.sum(1, keepdims=True) + 1e-12)
        ent = -(p * np.log(p + 1e-12)).sum(1)
        sp = spearman(TAU[dom], -ent)  # slow-dominant comps more specialized?

        dom_counts = np.bincount(dom, minlength=n_modes)
        results[tag] = dict(C=C, pr=pr, pr_uncentered=pr_unc, med_abs_r=med_abs_r,
                            med_r=med_r, blob_usage=blob, blob_cv=cv,
                            med_blob_cv=med_cv, n_mode_pref=n_pref,
                            dominant_mode=dom, dominant_mode_counts=dom_counts,
                            blob_entropy=ent, spearman_tau_specialization=sp)
        print(f"{tag:<18} {C:>3} {pr:>6.2f} {pr_unc:>7.2f} {med_abs_r:>7.3f} {med_r:>7.3f} "
              f"{med_cv:>7.3f} {n_pref:>6} {sp:>11.3f}")
        print(f"{'':<18}     dominant-mode counts: {dom_counts.tolist()}")

    if out_npy:
        np.save(out_npy, results, allow_pickle=True)
        print(f"\nsaved -> {out_npy}")
    return results


if __name__ == "__main__":
    analyse(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
