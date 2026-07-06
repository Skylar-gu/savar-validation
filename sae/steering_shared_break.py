"""
Follow-up 3 — Steering leakage rerun for the shared-subspace-break SAEs.

Same protocol as sae/steering_dose_response.py (Block F): steer the frozen eqvar
GNN at the last MP layer, H <- H + alpha * d_H (all nodes), alpha in
{-3,-1,+1,+3} sigma_f; measure on-target slope/linearity and off-target leakage.

Differences vs Block F:
  * SAE may be trained on PREPROCESSED pooled activations. Mapping the decoder
    direction d_norm (normalized preprocessed space) back to H-space:
      raw    : d_H = d_norm * std_g
      proj   : d_H = d_norm * std_g              (already in projected subspace;
               minimal-norm preimage of the projection)
      whiten : d_H = (d_norm * std_g) @ Wm_inv   (unwhiten)
    and encoding pooled H applies the forward preprocess first.
  * targets (mode -> feature) come from the variant's own Hungarian match.

Usage:
  python3 sae/steering_shared_break.py --datadir sae_data_hetdynamics_eqvar_whiten \
      --ckpt sae_mixed_topk_seed0_final.pt --targets 5:f5,6:f6,7:f7 --tag whiten
Output: results/steering_shared_break_<tag>.npy
"""

import sys, os, glob, argparse
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train.gnn_forecaster import MeshGNN, K
from eval_sae_metrics import INPUT_DIM, N_MODES, load_sae, encode

ap = argparse.ArgumentParser()
ap.add_argument("--datadir", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--targets", required=True, help="e.g. 5:216,6:475,7:143")
ap.add_argument("--tag", required=True)
a = ap.parse_args()
DATADIR = Path(a.datadir)
TARGETS = {int(kv.split(":")[0]): int(kv.split(":")[1])
           for kv in a.targets.split(",")}

SPLIT = os.environ.get("ST_SPLIT", "data/splits_hetdynamics_eqvar/test")
CKPT = os.environ.get("ST_CKPT", "checkpoints/hetdynamics_eqvar/best.pt")
N_WIN = int(os.environ.get("ST_NWIN", 240))
ALPHAS_SIG = np.array([-3.0, -1.0, 1.0, 3.0])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(0)

# ── preprocess forward / inverse maps ────────────────────────────────────────
wp = DATADIR / "whiten_params.npz"
sd_file = DATADIR / "shared_direction.npy"
if wp.exists():
    z = np.load(wp)
    mu, Wm, Wm_inv = z["mu"], z["Wm"], z["Wm_inv"]
    pre_fwd = lambda x: (x - mu) @ Wm
    dir_to_H = lambda d: d @ Wm_inv
elif sd_file.exists():
    w = np.load(sd_file)
    pre_fwd = lambda x: x - np.outer(x @ w, w)
    dir_to_H = lambda d: d
else:
    pre_fwd = lambda x: x
    dir_to_H = lambda d: d

# ── model + SAE ──────────────────────────────────────────────────────────────
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
model = MeshGNN(50, 50).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
sae, act_mu, act_sd = load_sae(DATADIR / a.ckpt)

acts_full = np.load(DATADIR / "activations_full.npy", mmap_mode="r")
sigma_f, act_rate = {}, {}
for j, f in TARGETS.items():
    stream = np.asarray(acts_full[:, j]).reshape(-1, INPUT_DIM)
    sub = stream[rng.choice(len(stream), min(60000, len(stream)), replace=False)]
    c = encode(sae, sub, act_mu, act_sd)[:, f]
    sigma_f[j] = float(c.std())
    act_rate[j] = float((c > 0).mean())
    print(f"mode X{j}: feature {f}  sigma_f={sigma_f[j]:.4f}  "
          f"active {act_rate[j]:.1%}")

paths = sorted(glob.glob(os.path.join(SPLIT, "realisation_*.npz")))
d0 = np.load(paths[0])
W = d0["W"].astype(np.float64)
obs_list = [np.load(p)["observations"] for p in paths]
windows = []
ri = 0
while len(windows) < N_WIN:
    o = obs_list[ri % len(obs_list)]
    t0 = rng.integers(0, len(o) - K - 1)
    windows.append(o[t0:t0 + K])
    ri += 1
windows = np.stack(windows).astype(np.float32)

_cap = {}
model.layers[-1].register_forward_hook(lambda m, i, o: _cap.__setitem__("H", o))

@torch.no_grad()
def decode_nodes(H):
    return model.decoder(H).squeeze(-1)

results = {"targets": TARGETS, "sigma_f": sigma_f, "act_rate": act_rate,
           "alphas_sigma": ALPHAS_SIG.tolist(), "n_win": N_WIN,
           "tag": a.tag, "modes": {}}
W_t = torch.from_numpy(W.astype(np.float32)).to(DEVICE)
BS = 48
for j, f in TARGETS.items():
    d_norm = sae.decoder.weight[:, f].detach().cpu().numpy()
    d_H = dir_to_H(d_norm * act_sd).astype(np.float32)
    d_H_t = torch.from_numpy(d_H).to(DEVICE)
    resp = np.zeros((len(ALPHAS_SIG), N_MODES, N_WIN))
    code_f = np.zeros(N_WIN)
    with torch.no_grad():
        for b0 in range(0, N_WIN, BS):
            xb = torch.from_numpy(windows[b0:b0 + BS]).to(DEVICE)
            model(xb)
            H = _cap["H"]
            base = decode_nodes(H)
            pooled = torch.einsum("l,blc->bc", W_t[j], H).cpu().numpy()
            code_f[b0:b0 + len(xb)] = encode(
                sae, pre_fwd(pooled.astype(np.float64)).astype(np.float32),
                act_mu, act_sd)[:, f]
            for ai, a_sig in enumerate(ALPHAS_SIG):
                alpha = a_sig * sigma_f[j]
                pert = decode_nodes(H + alpha * d_H_t)
                dpred = pert - base
                r = torch.einsum("il,bl->ib", W_t, dpred).cpu().numpy()
                resp[ai, :, b0:b0 + len(xb)] = r

    alphas = ALPHAS_SIG * sigma_f[j]
    on = resp[:, j, :]
    A_rep = np.repeat(alphas, N_WIN)
    y = on.ravel()
    slope = float(np.polyfit(A_rep, y, 1)[0])
    yhat = np.poly1d(np.polyfit(A_rep, y, 1))(A_rep)
    ss = 1 - ((y - yhat) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-15)
    off = np.abs(np.delete(resp, j, axis=1)).mean(axis=(1, 2))
    onm = np.abs(on).mean(1)
    leak = float((off / (onm + 1e-15)).mean())
    results["modes"][j] = dict(feature=f, slope=slope, linearity_r2=float(ss),
                               leakage_ratio=leak,
                               mean_on_target=on.mean(1).tolist(),
                               mean_abs_off_target=off.tolist(),
                               resp_mean_by_mode=resp.mean(2))
    print(f"X{j} (feat {f}): slope={slope:+.4f} linR2={ss:.3f} "
          f"leakage={leak:.3f}")

os.makedirs("results", exist_ok=True)
np.save(f"results/steering_shared_break_{a.tag}.npy", results, allow_pickle=True)
print(f"saved -> results/steering_shared_break_{a.tag}.npy")
