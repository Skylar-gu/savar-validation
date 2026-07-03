"""VPD causal evaluation (note §5/§8 — milestone M5).

Two tests on a decomposed run, both tying VPD mechanisms back to the ground-truth
causal graph Phi (the SAVAR mode VAR), read from the realisation npz `ground_truth_graph`
(`[cause, eff, lag]`) / `fine_edges`.

A. CENTRALITY TIE-IN (Phase-8 analogue). Per mode: causal out-degree (and other
   centralities) from Phi vs VPD forecast importance — the summed ablation_drop of the
   components whose usage map aligns to that mode (component->mode by argmax mode-corr).
   Hypothesis: causally-central modes carry more of the forecasting computation.

B. PCMCI BRIDGE. Build per-component activation time series (mean_node |V^T x| over a
   realisation), aggregate components->modes, run PCMCI on the (T, 8) mode series, and
   score the recovered lagged graph against Phi's cross-edges. Tests whether VPD
   mechanism activations carry the causal structure (feeding the discovery program with
   mechanisms instead of SAE features).

Run:
  PARAM_DECOMP_OUT_DIR=/home/ec2-user/savar-project/vpd_out \
  param-decomp/.venv/bin/python vpd/causal_eval.py vpd_out/runs/p-f77f0dca
"""

import glob
import os
import sys
from pathlib import Path

import fire
import networkx as nx
import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader

PROJECT_ROOT = "/home/ec2-user/savar-project"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from param_decomp_lab.component_model_io import get_all_component_acts, load_component_model
from tigramite.data_processing import DataFrame
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

from train.gnn_forecaster import K, MultiRealisationDataset
from vpd.run_gnn_vpd import GNNExperimentConfig, build_target, run_batch_gnn

N_MODES = 8
PC_ALPHA = 0.05


# ── ground truth ───────────────────────────────────────────────────────────────
def load_phi(train_dir: str):
    """Cross-edges of Phi: (fine_edges array, out-degree-style centralities per mode)."""
    d = np.load(sorted(glob.glob(os.path.join(train_dir, "realisation_*.npz")))[0])
    fine = d["fine_edges"]                                  # (E, 4): cause, eff, fine_lag, coeff
    G = nx.DiGraph(); G.add_nodes_from(range(N_MODES))
    for cause, eff, _ell, coeff in fine:
        cause, eff = int(cause), int(eff)
        if cause == eff:
            continue
        w = abs(float(coeff))
        if G.has_edge(cause, eff):
            G[cause][eff]["weight"] += w
        else:
            G.add_edge(cause, eff, weight=w)
    nodes = list(range(N_MODES))
    cent = {
        "out_degree": np.array([G.out_degree(n) for n in nodes], float),
        "out_strength": np.array([G.out_degree(n, weight="weight") for n in nodes], float),
        "total_degree": np.array([G.degree(n) for n in nodes], float),
        "betweenness": np.array(list(nx.betweenness_centrality(G, normalized=True).values())),
        "descendants": np.array([len(nx.descendants(G, n)) for n in nodes], float),
    }
    return fine, cent


def mode_variance(train_dir: str) -> np.ndarray:
    """Per-mode latent variance (the input-ablation amplitude confound)."""
    Z = np.concatenate([np.load(f)["latent_states"] for f in
                        sorted(glob.glob(os.path.join(train_dir, "realisation_*.npz")))[:10]], 0)
    return Z.var(0)


# ── component -> mode assignment from the saved diagnostics ──────────────────────
def assignments_from_diagnostics(run_dir: str, *, corr_thresh: float = 0.15):
    """Per decomposed module: (assigned_mode (C,), best_corr (C,), ablation_drop (C,)).

    A component is assigned to its argmax-correlated mode only if that corr exceeds
    `corr_thresh`; otherwise -1 (grid/hub component, excluded from the content test)."""
    diag = os.path.join(run_dir, "diagnostics")
    out = {}
    for f in sorted(glob.glob(os.path.join(diag, "usage_by_mode__*.npy"))):
        tag = os.path.basename(f)[len("usage_by_mode__"):-4]
        ubm = np.load(f)                                   # (C, n_modes)
        best_mode = ubm.argmax(1); best_corr = ubm.max(1)
        assigned = np.where(best_corr >= corr_thresh, best_mode, -1)
        adrop_f = os.path.join(diag, f"ablation_drop__{tag}.npy")
        adrop = np.load(adrop_f) if os.path.exists(adrop_f) else None
        out[tag] = dict(assigned=assigned, best_corr=best_corr, ablation_drop=adrop)
    return out


def centrality_test(run_dir: str, cent: dict, var: np.ndarray) -> None:
    asg = assignments_from_diagnostics(run_dir)
    # per-mode VPD forecast importance = summed ablation_drop of components assigned there
    imp = np.zeros(N_MODES); ncomp = np.zeros(N_MODES, int)
    for tag, a in asg.items():
        if a["ablation_drop"] is None:
            continue
        for c, m in enumerate(a["assigned"]):
            if m >= 0:
                imp[m] += max(a["ablation_drop"][c], 0.0)
                ncomp[m] += 1

    print("\n" + "=" * 72)
    print("A. CENTRALITY TIE-IN — causal out-degree vs VPD forecast importance")
    print("=" * 72)
    print(f" {'mode':>4} {'out_deg':>7} {'out_str':>7} {'var(Z)':>7} {'#comp':>5} {'VPD_imp':>10}")
    for m in range(N_MODES):
        print(f" X{m:>3} {cent['out_degree'][m]:>7.0f} {cent['out_strength'][m]:>7.2f} "
              f"{var[m]:>7.3f} {ncomp[m]:>5d} {imp[m]:>10.2e}")

    print(f"\n {'centrality':>14} {'Pearson':>8} {'Spearman':>9} {'partial|var':>12}")
    def partial(x, y, z):
        rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
        res = lambda a, b: a - np.vstack([np.ones_like(b), b]).T @ np.linalg.lstsq(
            np.vstack([np.ones_like(b), b]).T, a, rcond=None)[0]
        return stats.pearsonr(res(rx, rz), res(ry, rz))[0]
    for k, v in cent.items():
        if np.allclose(v, v[0]):
            continue
        pe = stats.pearsonr(v, imp)[0]; sp = stats.spearmanr(v, imp)[0]; pv = partial(v, imp, var)
        print(f" {k:>14} {pe:>+8.3f} {sp:>+9.3f} {pv:>+12.3f}")
    print(" (positive => causally-central modes carry more forecast computation)")


# ── Part B: component activation time series -> PCMCI vs Phi ─────────────────────
@torch.no_grad()
def mode_series_for_realisation(comp_model, obs, asg, device, *, k=K, batch=128):
    """(T', N_MODES) per-mode activation series = sum of |component act| over components
    assigned to each mode, where component act = mean_node |V^T x| at each timestep."""
    T = obs.shape[0]
    xs = torch.from_numpy(np.stack([obs[t:t + k] for t in range(T - k)])).float()  # (T',k,ny,nx)
    series = np.zeros((xs.shape[0], N_MODES), dtype=np.float64)
    for i in range(0, xs.shape[0], batch):
        xb = xs[i:i + batch].to(device)
        cache = comp_model((xb, None), cache_type="input").cache
        cacts = get_all_component_acts(comp_model, cache)             # {tag-ish: (B,L,C)}
        for mod, act in cacts.items():
            tag = mod.replace(".", "_")
            assigned = asg[tag]["assigned"]
            comp_mag = act.abs().mean(dim=1).cpu().numpy()            # (B, C)
            for c, m in enumerate(assigned):
                if m >= 0:
                    series[i:i + xb.shape[0], m] += comp_mag[:, c]
    return series


def pcmci_bridge(comp_model, train_dir, asg, device, *, n_real=5, tau_max=6) -> None:
    paths = sorted(glob.glob(os.path.join(train_dir, "realisation_*.npz")))[:n_real]
    fine, _ = load_phi(train_dir)
    gt = {(int(c), int(e), int(l)) for c, e, l, _ in fine if int(c) != int(e)}  # lagged cross-edges

    tp = fp = fn = 0
    for p in paths:
        obs = np.load(p)["observations"].astype(np.float32)
        series = mode_series_for_realisation(comp_model, obs, asg, device)
        if series.std(0).min() < 1e-8:                      # a mode with no assigned comps -> jitter
            series = series + 1e-6 * np.random.randn(*series.shape)
        pc = PCMCI(dataframe=DataFrame(series), cond_ind_test=ParCorr(), verbosity=0)
        res = pc.run_pcmci(tau_min=1, tau_max=tau_max, pc_alpha=PC_ALPHA, alpha_level=PC_ALPHA)
        det = {(c, e, tau) for c in range(N_MODES) for e in range(N_MODES) if c != e
               for tau in range(1, tau_max + 1) if res["p_matrix"][c, e, tau] < PC_ALPHA}
        tp += len(gt & det); fp += len(det - gt); fn += len(gt - det)

    prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    print("\n" + "=" * 72)
    print(f"B. PCMCI BRIDGE — recover Phi cross-edges from VPD mode-activation series "
          f"({len(paths)} realisations)")
    print("=" * 72)
    print(f" ground-truth lagged cross-edges: {len(gt)}")
    print(f" detected TP={tp}  FP={fp}  FN={fn}   precision={prec:.2f}  recall={rec:.2f}  F1={f1:.2f}")
    print(" (PCMCI on per-mode VPD-mechanism activity, scored vs Phi's directed lagged edges)")


def main(run_dir: str, *, pcmci: bool = True, n_real: int = 5) -> None:
    run_dir = run_dir if os.path.isabs(run_dir) else os.path.join(PROJECT_ROOT, run_dir)
    cfg = GNNExperimentConfig.from_file(os.path.join(run_dir, "experiment_config.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = cfg.data.split_dir
    split = split if os.path.isabs(split) else os.path.join(PROJECT_ROOT, split)
    train_dir = os.path.join(split, "train")

    _, cent = load_phi(train_dir)
    var = mode_variance(train_dir)
    centrality_test(run_dir, cent, var)

    if pcmci:
        ckpt = sorted(glob.glob(os.path.join(run_dir, "model_*.pth")))[-1]
        target = build_target(cfg.target).to(device); target.adjacency()
        comp_model = load_component_model(cfg.pd, Path(ckpt), target, run_batch_gnn).to(device)
        asg = assignments_from_diagnostics(run_dir)
        pcmci_bridge(comp_model, train_dir, asg, device, n_real=n_real)


if __name__ == "__main__":
    fire.Fire(main)
