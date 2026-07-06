"""
Block C — SynthSAEBench-style metric suite for existing SAE artifacts.

Literature-standard global metrics instead of per-mode best-|r| scalars
(SynthSAEBench arXiv 2602.14687; "Are SAE benchmarks reliable?" 2605.18229):

  (a) Hungarian-matched MCC  — linear_sum_assignment on the feature x mode
      |Pearson| matrix (globally distinct features, no double counting);
      report mean matched |r|.
  (b) per-mode matched F1    — binarize feature activity (in TopK, code > 0)
      and mode activity (|Z_j| above per-mode median); F1 of matched pairs.
  (c) feature uniqueness     — per matched feature, margin between its
      matched-mode |r| and its best other-mode |r|.

Variants evaluated per data dir:
  mixed    — single shared SAE (sae_mixed.pt), features correlated with every
             mode's stream: M[f, j] = |r(code_j[:, f], Z_j)|.
  per-mode — 8 SAEs; global feature pool = union of all 8 SAEs' features, each
             SAE encoding EVERY mode stream (its own normalisation), so the
             Hungarian match can reject features that fire on all modes.

Correlations computed on the full concatenated streams (same convention as
eval_sae_per_mode.py / eval_sae_mixed.py so old best-|r| numbers are directly
comparable side-by-side).

Usage:
  python3 sae/eval_sae_metrics.py --datadir sae_data/hetdynamics_eqvar \
      [--mixed-ckpt sae_mixed.pt] [--variants mixed,permode] [--tag eqvar]
Prints the table; returns a dict for the aggregator (results/sae_metrics_suite.npy
is written by running this over all dirs via --aggregate, see __main__).
"""

import sys, argparse
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy.optimize import linear_sum_assignment

INPUT_DIM  = 256
N_FEATURES = 512
K_TOPK     = 25
N_MODES    = 8
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KANGate(nn.Module):
    """Mirror of train_sae_mixed.KANGate (per-feature 1-D RBF spline gate)."""
    def __init__(self, n_features, n_basis=8, x_range=4.0):
        super().__init__()
        self.register_buffer("centers", torch.linspace(-x_range, x_range, n_basis))
        self.h = 2.0 * x_range / (n_basis - 1)
        self.coef = nn.Parameter(torch.zeros(n_features, n_basis))
        self.base = nn.Parameter(torch.ones(n_features))

    def forward(self, pre):
        phi = torch.exp(-((pre.unsqueeze(-1) - self.centers) / self.h) ** 2)
        return self.base * pre + (phi * self.coef).sum(-1)


class TopKSAE(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, n_features=N_FEATURES, k=K_TOPK, arch="topk"):
        super().__init__()
        self.k = k
        self.encoder = nn.Linear(input_dim, n_features, bias=True)
        self.gate = KANGate(n_features) if arch == "kan" else None
        self.decoder = nn.Linear(n_features, input_dim, bias=True)

    def encode(self, x):
        pre = self.encoder(x)
        if self.gate is not None:
            pre = self.gate(pre)
        topk_vals, topk_idx = torch.topk(pre, self.k, dim=-1)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, topk_idx, F.relu(topk_vals))
        return acts


def load_sae(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    sae = TopKSAE(ckpt.get("input_dim", INPUT_DIM),
                  ckpt.get("n_features", N_FEATURES),
                  ckpt.get("k", K_TOPK),
                  arch=ckpt.get("arch", "topk")).to(DEVICE)
    sae.load_state_dict(ckpt["model_state"])
    sae.eval()
    return sae, ckpt["act_mean"], ckpt["act_std"]


@torch.no_grad()
def encode(sae, X_np, mean, std, bs=8192):
    X = torch.from_numpy(((X_np - mean) / std).astype(np.float32))
    out = []
    for i in range(0, len(X), bs):
        out.append(sae.encode(X[i:i+bs].to(DEVICE)).cpu().numpy())
    return np.concatenate(out, 0)


def pearson_cols(A, b):
    """|r| of every column of A (n, F) with b (n,)."""
    A = A.astype(np.float64); b = b.astype(np.float64)
    Ac = A - A.mean(0); bc = b - b.mean()
    return (Ac.T @ bc) / ((np.linalg.norm(Ac, axis=0) + 1e-12) * (np.linalg.norm(bc) + 1e-12))


def f1_binary(act_f, act_m):
    tp = float(np.sum(act_f & act_m)); fp = float(np.sum(act_f & ~act_m))
    fn = float(np.sum(~act_f & act_m))
    p = tp / (tp + fp + 1e-12); r = tp / (tp + fn + 1e-12)
    return 2 * p * r / (p + r + 1e-12)


def metric_suite(codes_per_mode, Z_per_mode):
    """codes_per_mode: list of 8 arrays (n_j, F_total) — the global feature pool
    evaluated on each mode's stream. Z_per_mode: list of 8 (n_j,) latents."""
    F_total = codes_per_mode[0].shape[1]
    M = np.zeros((F_total, N_MODES))
    for j in range(N_MODES):
        M[:, j] = np.abs(pearson_cols(codes_per_mode[j], Z_per_mode[j]))
    # Hungarian: maximize sum of matched |r| over 8 distinct features
    rows, cols = linear_sum_assignment(-M)          # rows: feature ids, cols: modes
    match = {int(c): int(r) for r, c in zip(rows, cols)}
    matched_r = np.array([M[match[j], j] for j in range(N_MODES)])
    mcc = float(matched_r.mean())

    per_mode = {}
    for j in range(N_MODES):
        f = match[j]
        r_j = float(M[f, j])
        other = np.delete(M[f], j)
        uniq = r_j - float(other.max())
        act_f = codes_per_mode[j][:, f] > 0
        z = Z_per_mode[j]
        act_m = np.abs(z) > np.median(np.abs(z))
        f1 = f1_binary(act_f, act_m)
        per_mode[j] = dict(feature=f, matched_r=r_j, uniqueness=uniq, f1=f1,
                           best_unmatched_r=float(M[:, j].max()))
    return dict(mcc=mcc, matched_r=matched_r, match=match, per_mode=per_mode,
                mean_uniqueness=float(np.mean([p["uniqueness"] for p in per_mode.values()])),
                mean_f1=float(np.mean([p["f1"] for p in per_mode.values()])))


def eval_dir(datadir, variants=("mixed", "permode"), mixed_ckpt="sae_mixed.pt"):
    datadir = Path(datadir)
    acts_full = np.load(datadir / "activations_full.npy")   # (100, 8, T, 256)
    Z_full = np.load(datadir / "Z_full.npy")                # (100, 8, T)
    streams = [acts_full[:, j].reshape(-1, INPUT_DIM) for j in range(N_MODES)]
    Zs = [Z_full[:, j].reshape(-1) for j in range(N_MODES)]

    out = {}
    if "mixed" in variants and (datadir / mixed_ckpt).exists():
        sae, m, s = load_sae(datadir / mixed_ckpt)
        codes = [encode(sae, streams[j], m, s) for j in range(N_MODES)]
        out["mixed"] = metric_suite(codes, Zs)
    if "permode" in variants and (datadir / "sae_mode_0.pt").exists():
        # global pool: all 8 SAEs' features, each encoding every stream
        codes = [[] for _ in range(N_MODES)]
        for k in range(N_MODES):
            sae, m, s = load_sae(datadir / f"sae_mode_{k}.pt")
            for j in range(N_MODES):
                codes[j].append(encode(sae, streams[j], m, s))
        codes = [np.concatenate(c, 1) for c in codes]        # (n, 8*512)
        out["permode"] = metric_suite(codes, Zs)
    return out


def print_suite(tag, variant, r):
    print(f"\n=== {tag} / {variant}:  MCC={r['mcc']:.4f}  "
          f"mean_uniq={r['mean_uniqueness']:+.4f}  mean_F1={r['mean_f1']:.4f}")
    print(f"  {'mode':<5} {'feat':>6} {'match|r|':>9} {'best|r|':>8} {'uniq':>7} {'F1':>6}")
    for j in range(N_MODES):
        p = r["per_mode"][j]
        print(f"  X{j:<4} {p['feature']:>6} {p['matched_r']:>9.4f} "
              f"{p['best_unmatched_r']:>8.4f} {p['uniqueness']:>+7.3f} {p['f1']:>6.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", required=True)
    ap.add_argument("--variants", default="mixed,permode")
    ap.add_argument("--mixed-ckpt", default="sae_mixed.pt")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None, help="save dict .npy")
    a = ap.parse_args()
    tag = a.tag or Path(a.datadir).name
    res = eval_dir(a.datadir, tuple(a.variants.split(",")), a.mixed_ckpt)
    for v, r in res.items():
        print_suite(tag, v, r)
    if a.out:
        np.save(a.out, {tag: res}, allow_pickle=True)
