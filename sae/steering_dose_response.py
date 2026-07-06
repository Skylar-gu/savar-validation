"""
Block F — Steering dose–response: causal validation of mixed-SAE features
(CFD-GNN SAE line, arXiv 2604.04946).

For each slow mode j ∈ {5,6,7} (eqvar), take its Hungarian-matched mixed-SAE
feature f (from Block C: X5→216, X6→475, X7→143) and steer the frozen GNN at
the last MP layer for ALL nodes:

    H ← H + α · d_raw,   d_raw = decoder_dir_f ⊙ act_std   (denormalized),
    α ∈ {−3σ_f, −1σ_f, +1σ_f, +3σ_f},  σ_f = feature code std on stream j.

Because the decoder is a per-node MLP on H, the steered prediction is exactly
decoder(H + α·d_raw) — message passing needn't be re-run. Note the pooling
identity: every W row is L1-normalized, so a uniform per-node shift moves ALL
modes' pooled features by the same α·d — any on-target spatial specificity in
Δpred must come from the decoder's state-dependent Jacobian, which is the test.

Measured per (mode j, α):
  * on-target response  R_j(α)  = W[j] · mean Δpred     (dose–response)
  * slope + linearity R² of R_j vs α (per-window regression pooled)
  * off-target leakage  mean_{i≠j} |R_i(α)| / |R_j(α)|
  * time-aware variant: TopK-ReLU codes are nonnegative, so "modulate by the
    feature's current activation sign" reduces to gating on code_f > 0 —
    we report slope/linearity separately for windows where the feature is
    naturally active vs inactive.

Output: results/steering_dose_response.npy
Env: ST_SPLIT (data/splits_hetdynamics_eqvar/test), ST_NWIN (240)
"""

import sys, os, glob
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train.gnn_forecaster import MeshGNN, K
from eval_sae_metrics import INPUT_DIM, N_MODES, load_sae, encode

SPLIT  = os.environ.get("ST_SPLIT", "data/splits_hetdynamics_eqvar/test")
CKPT   = os.environ.get("ST_CKPT", "checkpoints/hetdynamics_eqvar/best.pt")
N_WIN  = int(os.environ.get("ST_NWIN", 240))
TARGETS = {5: 216, 6: 475, 7: 143}          # mode -> matched feature (Block C)
ALPHAS_SIG = np.array([-3.0, -1.0, 1.0, 3.0])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(0)

# ── model + SAE ───────────────────────────────────────────────────────────────
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
model = MeshGNN(50, 50).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
sae, act_mu, act_sd = load_sae("sae_data_hetdynamics_eqvar/sae_mixed.pt")

# σ_f from the training-distribution codes (mode-j stream)
acts_full = np.load("sae_data_hetdynamics_eqvar/activations_full.npy")
sigma_f, act_rate = {}, {}
for j, f in TARGETS.items():
    stream = acts_full[:, j].reshape(-1, INPUT_DIM)
    sub = stream[rng.choice(len(stream), min(60000, len(stream)), replace=False)]
    c = encode(sae, sub, act_mu, act_sd)[:, f]
    sigma_f[j] = float(c.std())
    act_rate[j] = float((c > 0).mean())
    print(f"mode X{j}: feature {f}  sigma_f={sigma_f[j]:.4f}  "
          f"active {act_rate[j]:.1%} of samples")

# ── windows + W ───────────────────────────────────────────────────────────────
paths = sorted(glob.glob(os.path.join(SPLIT, "realisation_*.npz")))
d0 = np.load(paths[0])
W = d0["W"].astype(np.float64)                      # (8, 2500), L1-normalized rows
obs_list = [np.load(p)["observations"] for p in paths]
windows = []
ri = 0
while len(windows) < N_WIN:
    o = obs_list[ri % len(obs_list)]              # (T, 50, 50) in split dirs
    t0 = rng.integers(0, len(o) - K - 1)
    windows.append(o[t0:t0 + K])
    ri += 1
windows = np.stack(windows).astype(np.float32)      # (N_WIN, K, 50, 50)

# capture H at the last MP layer
_cap = {}
model.layers[-1].register_forward_hook(lambda m, i, o: _cap.__setitem__("H", o))

@torch.no_grad()
def decode_nodes(H):
    return model.decoder(H).squeeze(-1)             # (B, L)

results = {"targets": TARGETS, "sigma_f": sigma_f, "act_rate": act_rate,
           "alphas_sigma": ALPHAS_SIG.tolist(), "n_win": N_WIN, "modes": {}}
W_t = torch.from_numpy(W.astype(np.float32)).to(DEVICE)

BS = 48
for j, f in TARGETS.items():
    d_norm = sae.decoder.weight[:, f].detach()      # (256,) normalized space
    d_raw = (d_norm * torch.from_numpy(act_sd.astype(np.float32)).to(DEVICE))
    resp = np.zeros((len(ALPHAS_SIG), N_MODES, N_WIN))   # R_i(alpha, window)
    code_f = np.zeros(N_WIN)
    with torch.no_grad():
        for b0 in range(0, N_WIN, BS):
            xb = torch.from_numpy(windows[b0:b0 + BS]).to(DEVICE)
            model(xb)
            H = _cap["H"]                            # (B, L, 256)
            base = decode_nodes(H)                   # (B, L)
            # current feature activation on the steered mode's pooled stream
            pooled = torch.einsum("l,blc->bc", W_t[j], H).cpu().numpy()
            code_f[b0:b0 + len(xb)] = encode(sae, pooled, act_mu, act_sd)[:, f]
            for ai, a_sig in enumerate(ALPHAS_SIG):
                alpha = a_sig * sigma_f[j]
                pert = decode_nodes(H + alpha * d_raw)
                dpred = (pert - base)                # (B, L)
                r = torch.einsum("il,bl->ib", W_t, dpred).cpu().numpy()
                resp[ai, :, b0:b0 + len(xb)] = r

    alphas = ALPHAS_SIG * sigma_f[j]
    on = resp[:, j, :]                               # (A, N_WIN)
    # pooled per-window regression: on-target response vs alpha
    A_rep = np.repeat(alphas, N_WIN)
    y = on.ravel()
    slope = float(np.polyfit(A_rep, y, 1)[0])
    yhat = np.poly1d(np.polyfit(A_rep, y, 1))(A_rep)
    ss = 1 - ((y - yhat) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-15)
    off = np.abs(np.delete(resp, j, axis=1)).mean(axis=(1, 2))   # (A,)
    onm = np.abs(on).mean(1)
    leak = float((off / (onm + 1e-15)).mean())

    active = code_f > 0
    def fitsub(mask):
        if mask.sum() < 8:
            return float("nan"), float("nan")
        ys = on[:, mask].ravel(); As = np.repeat(alphas, mask.sum())
        sl = np.polyfit(As, ys, 1)[0]
        yh = np.poly1d(np.polyfit(As, ys, 1))(As)
        r2 = 1 - ((ys - yh) ** 2).sum() / (((ys - ys.mean()) ** 2).sum() + 1e-15)
        return float(sl), float(r2)
    sl_act, r2_act = fitsub(active)
    sl_ina, r2_ina = fitsub(~active)

    results["modes"][j] = dict(
        feature=f, alphas=alphas.tolist(),
        mean_on_target=on.mean(1).tolist(),
        mean_abs_off_target=off.tolist(),
        slope=slope, linearity_r2=float(ss), leakage_ratio=leak,
        n_active=int(active.sum()),
        slope_active=sl_act, r2_active=r2_act,
        slope_inactive=sl_ina, r2_inactive=r2_ina,
        resp_mean_by_mode=resp.mean(2))              # (A, N_MODES)

    print(f"\nX{j} (feat {f}): on-target R(α) = "
          + "  ".join(f"{a:+.1f}σ:{v:+.4f}" for a, v in
                      zip(ALPHAS_SIG, on.mean(1)))
          + f"\n  slope={slope:+.4f}  linearity R²={ss:.3f}  "
          f"leakage(off/on)={leak:.3f}"
          f"\n  time-aware: active({int(active.sum())}w) slope={sl_act:+.4f} "
          f"R²={r2_act:.3f} | inactive slope={sl_ina:+.4f} R²={r2_ina:.3f}")

os.makedirs("results", exist_ok=True)
np.save("results/steering_dose_response.npy", results, allow_pickle=True)
print("\nsaved -> results/steering_dose_response.npy")
