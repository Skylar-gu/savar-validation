"""
Phase 7.2 — Train a TopK Sparse Autoencoder on CNN res3 activations.

Architecture
------------
  input (256,) → Linear(256, 512) + bias → TopK(k=25) → features (512,)
                                           → Linear(512, 256) + bias → reconstruction (256,)

Key choices vs L1 SAE
----------------------
  - TopK activation: exactly K features active per forward pass; no shrinkage of
    active feature magnitudes (unlike L1 which attenuates all nonzero activations).
  - Decoder columns kept unit-norm after each step.
  - Dead feature resampling: features inactive for DEAD_WINDOW consecutive steps
    are reinitialised to unit-normed residual vectors of high-loss examples.

Outputs (written to sae_data/)
-------------------------------
  sae_best.pt       checkpoint with lowest val reconstruction MSE
  sae_history.npy   per-epoch training log
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

DATA_DIR = Path("sae_data")
OUT_DIR  = Path("sae_data")

# ── hyperparameters ───────────────────────────────────────────────────────────

INPUT_DIM  = 256
N_FEATURES = 512    # 2× expansion
K_TOPK     = 25     # active features per forward pass (targets L0 ≈ 20–30)
LR         = 1e-3
EPOCHS     = 60
BATCH_SIZE = 256

RESAMPLE_INTERVAL = 2000   # steps between dead-feature checks
DEAD_WINDOW       = 1000   # steps of inactivity → feature considered dead

VAL_SPLIT  = 0.15          # last 15% of realisations held out for val monitoring
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── load activations ──────────────────────────────────────────────────────────

acts_full = np.load(DATA_DIR / "activations_full.npy")   # (100, 8, 497, 256)
n_real, n_modes, t_eff, d = acts_full.shape
assert d == INPUT_DIM

# Split by realisation; flatten (modes × time) into the sample dimension
n_val   = max(1, int(n_real * VAL_SPLIT))
n_train = n_real - n_val

acts_train = acts_full[:n_train].reshape(-1, d).astype(np.float32)  # (~332K, 256)
acts_val   = acts_full[n_train:].reshape(-1, d).astype(np.float32)  # (~60K,  256)

print(f"Activations  train={len(acts_train):,}  val={len(acts_val):,}  dim={d}")

# ── normalise (fit on train, apply to all) ────────────────────────────────────

act_mean = acts_train.mean(0)          # (256,)
act_std  = acts_train.std(0) + 1e-8   # (256,)

np.save(OUT_DIR / "act_norm.npy", {"mean": act_mean, "std": act_std})

def normalise(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy((x - act_mean) / act_std).to(DEVICE)

X_train = normalise(acts_train)   # (N_train, 256) on device
X_val   = normalise(acts_val)     # (N_val,   256) on device

print(f"Normalised   mean≈{X_train.mean().item():.3f}  std≈{X_train.std().item():.3f}\n")


# ── model ─────────────────────────────────────────────────────────────────────

class TopKSAE(nn.Module):
    def __init__(self, input_dim: int, n_features: int, k: int):
        super().__init__()
        self.k         = k
        self.n_features = n_features
        self.encoder   = nn.Linear(input_dim, n_features, bias=True)
        self.decoder   = nn.Linear(n_features, input_dim, bias=True)
        self._init_weights()

    def _init_weights(self):
        # Decoder columns unit-norm; encoder tied to decoder transpose
        nn.init.normal_(self.decoder.weight, std=1.0 / INPUT_DIM ** 0.5)
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)
            self.encoder.weight.data = self.decoder.weight.data.T.clone()
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre  = self.encoder(x)                                 # (B, M)
        topk_vals, topk_idx = torch.topk(pre, self.k, dim=-1)  # both (B, K)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, topk_idx, F.relu(topk_vals))
        return acts                                            # (B, M) sparse

    def forward(self, x: torch.Tensor):
        acts  = self.encode(x)
        recon = self.decoder(acts)
        return acts, recon

    @torch.no_grad()
    def normalise_decoder(self):
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)


sae = TopKSAE(INPUT_DIM, N_FEATURES, K_TOPK).to(DEVICE)

n_params = sum(p.numel() for p in sae.parameters())
print(f"TopK SAE  input={INPUT_DIM}  features={N_FEATURES}  k={K_TOPK}  params={n_params:,}")
print(f"  device={DEVICE}  epochs={EPOCHS}  batch={BATCH_SIZE}  lr={LR}\n")

optimizer = torch.optim.Adam(sae.parameters(), lr=LR)


# ── dead feature tracking ─────────────────────────────────────────────────────

last_fired   = torch.zeros(N_FEATURES, dtype=torch.long, device=DEVICE)  # step of last fire
global_step  = 0


def resample_dead_features(dead_mask: torch.Tensor):
    """
    Reinitialise dead feature encoder/decoder weights to unit-normed residual
    vectors from high-loss training examples. Resets Adam state for those rows.
    """
    n_dead = dead_mask.sum().item()
    if n_dead == 0:
        return 0

    dead_idx = dead_mask.nonzero(as_tuple=True)[0]   # (n_dead,)

    # Sample a batch and find high-loss examples
    sample_idx = torch.randperm(len(X_train), device=DEVICE)[:min(2048, len(X_train))]
    X_sample   = X_train[sample_idx]
    with torch.no_grad():
        _, recon  = sae(X_sample)
        per_sample_loss = F.mse_loss(recon, X_sample, reduction="none").mean(dim=1)  # (N,)
        residuals = X_sample - recon  # (N, D)

    # Sort by loss descending
    order     = per_sample_loss.argsort(descending=True)
    residuals = residuals[order]

    # Pad if fewer high-loss examples than dead features
    if len(residuals) < n_dead:
        reps = (n_dead + len(residuals) - 1) // len(residuals)
        residuals = residuals.repeat(reps, 1)[:n_dead]
    else:
        residuals = residuals[:n_dead]

    residuals_normed = F.normalize(residuals, dim=1)  # (n_dead, D)

    with torch.no_grad():
        sae.encoder.weight.data[dead_idx] = residuals_normed
        sae.decoder.weight.data[:, dead_idx] = residuals_normed.T
        sae.encoder.bias.data[dead_idx] = 0.0

    # Reset Adam state for reinitialised rows
    enc_w_state = optimizer.state.get(sae.encoder.weight)
    dec_w_state = optimizer.state.get(sae.decoder.weight)
    if enc_w_state:
        enc_w_state["exp_avg"][dead_idx]    = 0.0
        enc_w_state["exp_avg_sq"][dead_idx] = 0.0
    if dec_w_state:
        dec_w_state["exp_avg"][:, dead_idx]    = 0.0
        dec_w_state["exp_avg_sq"][:, dead_idx] = 0.0

    return n_dead


# ── training loop ─────────────────────────────────────────────────────────────

steps_per_epoch = len(X_train) // BATCH_SIZE
best_val_mse    = float("inf")
history         = []
total_resampled = 0

print(f"  {'Ep':>4}  {'Train MSE':>10}  {'Val MSE':>9}  {'L0':>5}  {'Dead':>5}  {'Resampled':>10}")
print(f"  {'─'*58}")

for epoch in range(1, EPOCHS + 1):
    sae.train()
    perm = torch.randperm(len(X_train), device=DEVICE)

    epoch_mse = epoch_l0 = 0.0

    for i in range(steps_per_epoch):
        idx   = perm[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        batch = X_train[idx]

        acts, recon = sae(batch)
        loss        = F.mse_loss(recon, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sae.normalise_decoder()

        # track firing
        fired = (acts.detach() > 0).any(dim=0)   # (M,)
        last_fired[fired] = global_step
        global_step += 1

        epoch_mse += loss.item()
        epoch_l0  += (acts.detach() > 0).float().sum(dim=1).mean().item()

        # dead feature resampling
        if global_step % RESAMPLE_INTERVAL == 0:
            dead_mask = (global_step - last_fired) > DEAD_WINDOW
            n = resample_dead_features(dead_mask)
            total_resampled += n

    train_mse = epoch_mse / steps_per_epoch
    train_l0  = epoch_l0  / steps_per_epoch

    # validation
    sae.eval()
    with torch.no_grad():
        _, val_recon = sae(X_val)
        val_mse  = F.mse_loss(val_recon, X_val).item()
        dead_now = ((global_step - last_fired) > DEAD_WINDOW).sum().item()

    if val_mse < best_val_mse:
        best_val_mse = val_mse
        torch.save({
            "epoch": epoch,
            "model_state":  sae.state_dict(),
            "n_features":   N_FEATURES,
            "k":            K_TOPK,
            "input_dim":    INPUT_DIM,
            "val_mse":      val_mse,
            "act_mean":     act_mean,
            "act_std":      act_std,
        }, OUT_DIR / "sae_best.pt")

    row = {
        "epoch": epoch, "train_mse": train_mse, "val_mse": val_mse,
        "l0": train_l0, "dead": dead_now, "total_resampled": total_resampled,
    }
    history.append(row)

    print(f"  {epoch:>4}  {train_mse:>10.5f}  {val_mse:>9.5f}  "
          f"{train_l0:>5.1f}  {dead_now:>5}  {total_resampled:>10}")

    # warn if feature-feature redundancy is building up (check every 10 epochs)
    if epoch % 10 == 0:
        with torch.no_grad():
            W = F.normalize(sae.encoder.weight.detach(), dim=1)  # (M, D)
            # sample 256 features to keep this O(256^2) not O(512^2)
            idx_s  = torch.randperm(N_FEATURES, device=DEVICE)[:256]
            W_s    = W[idx_s]
            C_feat = (W_s @ W_s.T).abs()                          # (256, 256)
            C_feat.fill_diagonal_(0.0)
            max_corr = C_feat.max().item()
        print(f"         feature–feature max |cos_sim| = {max_corr:.3f}"
              + ("  *** REDUNDANCY WARNING" if max_corr > 0.9 else ""))

np.save(OUT_DIR / "sae_history.npy", history)
print(f"\nBest val MSE : {best_val_mse:.5f}")
print(f"Checkpoint   → sae_data/sae_best.pt")
print(f"History      → sae_data/sae_history.npy")
