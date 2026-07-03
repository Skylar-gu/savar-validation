"""VPD diagnostics for the SAVAR MeshGNN run (note §7).

Loads a decomposed `ComponentModel` (e.g. vpd_out/runs/<id>/model_4000.pth), then for
each decomposed shared MP-MLP computes:

  * mean_g_map      (C, ny, nx)  — mean CI gate per node, reshaped: the spatial usage map
  * hub/nonhub      (C,)         — mesh-structural usage (hubs = stride-5 lattice)
  * usage_by_mode   (C, n_modes) — content alignment (corr of the map with each SAVAR mode W[j])
  * ablation_drop   (C,)         — forecast-MSE rise when component c is fully masked

Then the §0 grid-lock test: a component is grid-locked if its gate is explained by
hub-ness, not by any mode; content if the reverse. Crossed with ablation_drop this
separates *used* grid-lock from grid-locked-but-epiphenomenal (the SAE finding, in one
object).

Run:
  PARAM_DECOMP_OUT_DIR=/home/ec2-user/savar-project/vpd_out \
  param-decomp/.venv/bin/python vpd/diagnostics.py vpd_out/runs/p-f77f0dca
"""

import glob
import os
import sys
from pathlib import Path

import fire
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = "/home/ec2-user/savar-project"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from param_decomp.masks import make_mask_infos
from param_decomp_lab.component_model_io import load_component_model

from train.gnn_forecaster import HUB_STRIDE, MultiRealisationDataset
from vpd.run_gnn_vpd import GNNExperimentConfig, MeshGNNNodeOut, build_target, run_batch_gnn


def hub_node_ids(ny: int, nx: int, hub_stride: int = HUB_STRIDE) -> np.ndarray:
    """Node ids on the coarse hub lattice — the high-degree nodes that carry mesh heterogeneity."""
    return np.array(
        [r * nx + c for r in range(0, ny, hub_stride) for c in range(0, nx, hub_stride)],
        dtype=np.int64,
    )


def load_mode_maps(split_train_dir: str) -> np.ndarray:
    """SAVAR mode spatial maps W: (n_modes, L). Same across realisations (fixed layout)."""
    path = sorted(glob.glob(os.path.join(split_train_dir, "realisation_*.npz")))[0]
    return np.load(path)["W"].astype(np.float32)


@torch.no_grad()
def collect_mean_gate(comp_model, loader, device, *, max_batches: int) -> dict[str, np.ndarray]:
    """Mean lower-leaky CI gate per decomposed module, averaged over (batch, samples) -> (L, C)."""
    sums: dict[str, torch.Tensor] = {}
    n = 0
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        cache = comp_model(batch, cache_type="input").cache
        ci = comp_model.calc_causal_importances(cache, sampling="continuous")
        for mod, g in ci.lower_leaky.items():            # g: (B, L, C)
            s = g.sum(dim=0)                              # (L, C)
            sums[mod] = s if mod not in sums else sums[mod] + s
        n += next(iter(ci.lower_leaky.values())).shape[0]
    return {mod: (s / n).cpu().numpy() for mod, s in sums.items()}  # (L, C)


@torch.no_grad()
def ablation_drops(comp_model, batch, device) -> dict[str, np.ndarray]:
    """Per-component forecast-MSE rise when that component is fully masked (others kept).

    Baseline = all components present + delta (reconstructs the frozen forecast). For each
    c, zero its column of the component mask and remeasure MSE to the frozen target."""
    target_out = run_batch_gnn(comp_model.target_model, batch).detach()  # frozen (B, L, 1)
    cache = comp_model(batch, cache_type="input").cache
    ci = comp_model.calc_causal_importances(cache, sampling="continuous")
    weight_deltas = comp_model.calc_weight_deltas()

    def mse(masks_by_mod: dict[str, torch.Tensor]) -> float:
        wd = {k: (weight_deltas[k], torch.ones(masks_by_mod[k].shape[:-1], device=device))
              for k in masks_by_mod}
        mask_infos = make_mask_infos(masks_by_mod, routing_masks="all", weight_deltas_and_masks=wd)
        out = comp_model(batch, mask_infos=mask_infos)
        return ((out - target_out) ** 2).mean().item()

    ones = {mod: torch.ones_like(g) for mod, g in ci.lower_leaky.items()}  # (B, L, C)
    base = mse(ones)

    out: dict[str, np.ndarray] = {}
    for mod, g in ci.lower_leaky.items():
        C = g.shape[-1]
        drops = np.empty(C, dtype=np.float32)
        for c in range(C):
            m = {k: v.clone() for k, v in ones.items()}
            m[mod][..., c] = 0.0
            drops[c] = mse(m) - base
        out[mod] = drops
    return out


def analyse(run_dir: str, *, max_batches: int = 20, ablate: bool = True) -> None:
    run_dir = run_dir if os.path.isabs(run_dir) else os.path.join(PROJECT_ROOT, run_dir)
    ckpt = sorted(glob.glob(os.path.join(run_dir, "model_*.pth")))[-1]
    cfg = GNNExperimentConfig.from_file(os.path.join(run_dir, "experiment_config.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ny, nx = cfg.target.ny, cfg.target.nx
    L = ny * nx

    target = build_target(cfg.target).to(device)
    target.adjacency()
    comp_model = load_component_model(cfg.pd, Path(ckpt), target, run_batch_gnn)
    comp_model.to(device)

    split = cfg.data.split_dir
    split = split if os.path.isabs(split) else os.path.join(PROJECT_ROOT, split)
    val_ds = MultiRealisationDataset(os.path.join(split, "val"), k=cfg.target.k)
    loader = DataLoader(val_ds, batch_size=cfg.pd.batch_size, shuffle=False, drop_last=True)

    print(f"run={run_dir}\n ckpt={os.path.basename(ckpt)}  val_windows={len(val_ds)}  "
          f"targets={list(comp_model.components.keys())}")

    mean_gate = collect_mean_gate(comp_model, loader, device, max_batches=max_batches)  # (L,C)
    W = load_mode_maps(os.path.join(split, "train"))                                    # (n_modes, L)
    hubs = hub_node_ids(ny, nx)
    is_hub = np.zeros(L, dtype=bool); is_hub[hubs] = True

    drops = ablation_drops(comp_model, next(iter(loader)), device) if ablate else {}

    out_dir = os.path.join(run_dir, "diagnostics")
    os.makedirs(out_dir, exist_ok=True)

    for mod, g_lc in mean_gate.items():                  # g_lc: (L, C)
        C = g_lc.shape[1]
        g_cmap = g_lc.T.reshape(C, ny, nx)               # (C, ny, nx)
        hub_usage = g_lc[is_hub].mean(0)                 # (C,)
        nonhub_usage = g_lc[~is_hub].mean(0)             # (C,)
        hub_ratio = hub_usage / (nonhub_usage + 1e-8)

        gc = g_lc - g_lc.mean(0, keepdims=True)          # center over nodes
        Wc = W - W.mean(1, keepdims=True)
        denom = (np.linalg.norm(gc, axis=0)[None, :] * np.linalg.norm(Wc, axis=1)[:, None] + 1e-8)
        mode_corr = (Wc @ gc) / denom                    # (n_modes, C)
        best_mode = mode_corr.argmax(0)
        best_mode_corr = mode_corr.max(0)

        tag = mod.replace(".", "_")
        np.save(os.path.join(out_dir, f"mean_g_map__{tag}.npy"), g_cmap)
        np.save(os.path.join(out_dir, f"hub_usage__{tag}.npy"), hub_usage)
        np.save(os.path.join(out_dir, f"usage_by_mode__{tag}.npy"), mode_corr.T)  # (C, n_modes)
        if mod in drops:
            np.save(os.path.join(out_dir, f"ablation_drop__{tag}.npy"), drops[mod])

        # active components only: those whose gate is meaningfully on somewhere
        active = g_lc.max(0) > 0.05
        print(f"\n=== {mod}  (C={C}, active={int(active.sum())}) ===")
        order = np.argsort(-hub_ratio)
        print(" top hub-locked (hub/nonhub gate ratio):")
        for c in order[:6]:
            if not active[c]:
                continue
            d = f"{drops[mod][c]:+.2e}" if mod in drops else "n/a"
            kind = "GRID-LOCK" if best_mode_corr[c] < 0.2 else f"mode{best_mode[c]}({best_mode_corr[c]:.2f})"
            print(f"   c{c:02d}  hub_ratio={hub_ratio[c]:5.2f}  best={kind:>16}  abl_drop={d}")
        morder = np.argsort(-best_mode_corr)
        print(" top mode-aligned (content):")
        for c in morder[:6]:
            if not active[c]:
                continue
            d = f"{drops[mod][c]:+.2e}" if mod in drops else "n/a"
            print(f"   c{c:02d}  mode{best_mode[c]} corr={best_mode_corr[c]:.2f}  "
                  f"hub_ratio={hub_ratio[c]:5.2f}  abl_drop={d}")
        if mod in drops:
            # the payoff cross: hub-locked AND inert vs hub-locked AND used
            gl = active & (hub_ratio > 1.3) & (best_mode_corr < 0.2)
            if gl.any():
                used = gl & (drops[mod] > np.quantile(drops[mod][active], 0.75))
                print(f" grid-locked components: {int(gl.sum())}  "
                      f"(of which forecast-used: {int(used.sum())}, epiphenomenal: {int((gl & ~used).sum())})")

    print(f"\nsaved arrays -> {out_dir}")


if __name__ == "__main__":
    fire.Fire(analyse)
