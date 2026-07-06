"""T6b — Block-A redundancy diagnostics for a VPD run on the MOVMECH net.

analyze_redundancy.py assumes a FIXED mode layout (loads W once from the first
train realisation). In movmech the blob centres move per realisation, so mean
gate maps averaged over realisations smear over blob positions and the fixed-W
blob-usage stats are meaningless. This script:

  1. collects the mean CI gate PER VAL REALISATION (grouping the val dataset's
     windows by segment), giving g[r] (L, C) per module;
  2. computes blob usage against THAT realisation's W[r] (val npz), then
     averages the (C, 8) blob-usage matrices over realisations;
  3. reports the same summary columns as analyze_redundancy.py (PR + med|r| of
     the all-realisation mean map for direct comparison with the eqvar runs,
     plus per-realisation-mean PR, blob CV, #mode-preferential, dominant-mode
     counts, spearman(tau, specialization));
  4. adds the pattern-PR / substantive-component decomposition from the
     Follow-up-2 writeup (row-normalized maps, centered-map norm > 1).

Run with the param-decomp venv:
  PARAM_DECOMP_OUT_DIR=/home/ec2-user/savar-project/vpd_out \
  param-decomp/.venv/bin/python vpd/analyze_redundancy_movmech.py \
      vpd_out/runs/<id> data/splits_movmech_place [out.npy]
"""

import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = "/home/ec2-user/savar-project"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from param_decomp_lab.component_model_io import load_component_model

from train.gnn_forecaster import MultiRealisationDataset
from vpd.run_gnn_vpd import GNNExperimentConfig, build_target, run_batch_gnn
from vpd.analyze_redundancy import PHI, TAU, spearman, participation_ratio

N_MODES = 8
N_REALS = 40          # val realisations to analyse
N_WINDOWS = 32        # windows per realisation (4 batches of 8)
BATCH = 8


@torch.no_grad()
def per_realisation_gates(comp_model, ds, device, n_reals, n_windows):
    """dict module -> (n_reals, L, C) mean lower-leaky CI gate per realisation."""
    by_seg = {}
    for s, i in ds.index:
        by_seg.setdefault(s, []).append(i)
    segs = sorted(by_seg)[:n_reals]
    out = {}
    for r, s in enumerate(segs):
        idxs = by_seg[s][:n_windows]
        seg = ds.segs[s]
        sums, n = {}, 0
        for b0 in range(0, len(idxs), BATCH):
            bi = idxs[b0:b0 + BATCH]
            x = torch.stack([seg[i:i + ds.k] for i in bi]).to(device)
            y = torch.stack([seg[i + ds.k].unsqueeze(0) for i in bi]).to(device)
            cache = comp_model((x, y), cache_type="input").cache
            ci = comp_model.calc_causal_importances(cache, sampling="continuous")
            for mod, g in ci.lower_leaky.items():          # (B, L, C)
                s_ = g.sum(dim=0)
                sums[mod] = s_ if mod not in sums else sums[mod] + s_
            n += x.shape[0]
        for mod, s_ in sums.items():
            out.setdefault(mod, []).append((s_ / n).cpu().numpy())   # (L, C)
    return {mod: np.stack(v) for mod, v in out.items()}, segs


def analyse(run_dir, split_dir, out_npy=None):
    run_dir = run_dir if os.path.isabs(run_dir) else os.path.join(PROJECT_ROOT, run_dir)
    split_dir = split_dir if os.path.isabs(split_dir) else os.path.join(PROJECT_ROOT, split_dir)
    ckpt = sorted(glob.glob(os.path.join(run_dir, "model_*.pth")))[-1]
    cfg = GNNExperimentConfig.from_file(os.path.join(run_dir, "experiment_config.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target = build_target(cfg.target).to(device)
    target.adjacency()
    comp_model = load_component_model(cfg.pd, Path(ckpt), target, run_batch_gnn)
    comp_model.to(device)

    val_dir = os.path.join(split_dir, "val")
    ds = MultiRealisationDataset(val_dir, k=cfg.target.k)
    val_files = sorted(glob.glob(os.path.join(val_dir, "realisation_*.npz")))

    gates, segs = per_realisation_gates(comp_model, ds, device, N_REALS, N_WINDOWS)
    Ws = [np.load(val_files[s])["W"] for s in segs]        # each (8, L)
    masks = [[w[j] > 0 for j in range(N_MODES)] for w in Ws]

    results = {}
    print(f"run: {run_dir}  ({len(segs)} val realisations x {N_WINDOWS} windows)")
    hdr = (f"{'module':<18} {'C':>3} {'PR':>6} {'PRreal':>7} {'med|r|':>7} "
           f"{'blobCV':>7} {'#pref':>6} {'sp(phi,ent)':>11} {'PRpat':>7} {'nsub':>5} {'#prefS':>7}")
    print(hdr)
    print("-" * len(hdr))

    for mod in sorted(gates):
        G = gates[mod]                                     # (R, L, C)
        R_, L, C = G.shape
        mean_map = G.mean(0).T                             # (C, L) — eqvar-comparable object
        pr = participation_ratio(mean_map, center=True)
        pr_real = float(np.mean([participation_ratio(G[r].T, center=True) for r in range(R_)]))

        Xc = mean_map - mean_map.mean(1, keepdims=True)
        nrm = np.linalg.norm(Xc, axis=1) + 1e-12
        Rm = (Xc @ Xc.T) / np.outer(nrm, nrm)
        iu = np.triu_indices(C, 1)
        med_abs_r = float(np.median(np.abs(Rm[iu])))

        # per-realisation blob usage against that realisation's OWN W
        blob = np.mean([np.stack([G[r][m].mean(0) for m in masks[r]], axis=1)
                        for r in range(R_)], axis=0)       # (C, 8)
        cv = blob.std(1) / (blob.mean(1) + 1e-12)
        med_cv = float(np.median(cv))
        srt = np.sort(blob, axis=1)
        pref = srt[:, -1] > 1.2 * srt[:, -2]
        n_pref = int(pref.sum())
        dom = blob.argmax(1)
        p = blob / (blob.sum(1, keepdims=True) + 1e-12)
        ent = -(p * np.log(p + 1e-12)).sum(1)
        sp = spearman(TAU[dom], -ent)
        dom_counts = np.bincount(dom, minlength=N_MODES)

        # pattern-PR among substantive components (Follow-up 2 decomposition)
        cnorm = np.linalg.norm(Xc, axis=1)
        subst = cnorm > 1.0
        n_sub = int(subst.sum())
        if n_sub >= 2:
            rows = mean_map[subst] / (np.linalg.norm(mean_map[subst], axis=1, keepdims=True) + 1e-12)
            pr_pat = participation_ratio(rows, center=True)
            n_pref_sub = int((pref & subst).sum())
        else:
            pr_pat, n_pref_sub = float("nan"), 0

        results[mod] = dict(C=C, pr=pr, pr_per_real=pr_real, med_abs_r=med_abs_r,
                            blob_usage=blob, med_blob_cv=med_cv, n_mode_pref=n_pref,
                            dominant_mode_counts=dom_counts, blob_entropy=ent,
                            spearman_tau_specialization=sp, pr_pattern=pr_pat,
                            n_substantive=n_sub, n_pref_substantive=n_pref_sub)
        print(f"{mod:<18} {C:>3} {pr:>6.2f} {pr_real:>7.2f} {med_abs_r:>7.3f} "
              f"{med_cv:>7.3f} {n_pref:>6} {sp:>11.3f} {pr_pat:>7.2f} {n_sub:>5} {n_pref_sub:>7}")
        print(f"{'':<18}     dominant-mode counts: {dom_counts.tolist()}")

    if out_npy:
        np.save(out_npy, results, allow_pickle=True)
        print(f"\nsaved -> {out_npy}")
    return results


if __name__ == "__main__":
    analyse(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
