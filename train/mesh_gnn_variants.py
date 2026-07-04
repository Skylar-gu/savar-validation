"""
Architecture sweep for the moving-mechanism test (spec v2 §"network upgrades").

Flags on the existing MeshGNN (train/gnn_forecaster.py); the PLAIN net stays the
default everywhere else. Each variant addresses one hypothesis for why a plain
smush-to-one-vector net fails the left-vs-right test (P1) on MOVE=place:

  GNN_VARIANT=plain     (baseline, == MeshGNN)
  GNN_VARIANT=blurpool  A0 — anti-aliased downsampling (Zhang 1904.11486).
      A fixed Gaussian low-pass on the input frames BEFORE the encoder, so a
      moving blob is band-limited before the stride-5 hub lattice / W-pool
      samples it. Nearly free; may be the whole fix.
  GNN_VARIANT=refframe  A1 — features relative to the content's own frame
      (lightweight Invariant Slot Attention, Biza 2302.04973). Each node is
      given its offset from the per-sample activity centroid (|input|-weighted
      mean position) as extra input channels — a translation-equivariant
      coordinate that MOVES WITH the blob, giving a location-invariant "what"
      code without a fixed absolute-position table.
  GNN_VARIANT=slot      A2 — a small Slot-Attention head (Locatello; Mansouri
      2310.19054) after message passing: M slots compete (softmax-over-slots)
      to bind nodes, node states are refined from their bound slot, then the
      usual decoder runs. Heaviest; only if A0/A1 don't pass P1.

Everything else — mesh, MP layers, dataset contract, metrics, checkpoint logic —
is inherited so the variants are directly comparable to the plain net and a
plain MeshGNN checkpoint is a strict subset (extra params only).

Run:
  GNN_VARIANT=blurpool GNN_EPOCHS=12 GNN_BATCH=64 \
    GNN_CKPT_DIR=checkpoints_movmech_place_blurpool \
    GNN_SPLIT_DIR=data/splits_movmech_place \
    python3 train/mesh_gnn_variants.py
"""

import os, sys, time
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnn_forecaster import (
    MeshGNN, MPLayer, GraphCastMPLayer, build_mesh, normalized_adjacency,
    MultiRealisationDataset, run_epoch, K, HIDDEN, N_MP, EMB_DIM, MP_MODE,
    BATCH_SIZE, LR, EPOCHS, DEVICE,
)

VARIANT   = os.environ.get("GNN_VARIANT", "plain")
BLUR_SIGMA = float(os.environ.get("GNN_BLUR_SIGMA", 1.0))
N_SLOTS    = int(os.environ.get("GNN_N_SLOTS", 8))
SLOT_ITERS = int(os.environ.get("GNN_SLOT_ITERS", 3))
CKPT_DIR   = os.environ.get("GNN_CKPT_DIR", f"checkpoints_movmech_place_{VARIANT}")
SPLIT_DIR  = os.environ.get("GNN_SPLIT_DIR", "data/splits_movmech_place")


def gaussian_kernel2d(sigma, radius=None):
    if radius is None:
        radius = max(1, int(round(3 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    g1 = torch.exp(-(x ** 2) / (2 * sigma ** 2)); g1 /= g1.sum()
    k = torch.outer(g1, g1)
    return k / k.sum(), radius


class SlotAttention(nn.Module):
    """Minimal Slot Attention (Locatello et al. 2020)."""
    def __init__(self, n_slots, dim, iters=3, eps=1e-8):
        super().__init__()
        self.n_slots, self.dim, self.iters, self.eps = n_slots, dim, iters, eps
        self.scale = dim ** -0.5
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.slots_logsig = nn.Parameter(torch.zeros(1, 1, dim))
        self.to_q = nn.Linear(dim, dim); self.to_k = nn.Linear(dim, dim); self.to_v = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.norm_in = nn.LayerNorm(dim); self.norm_slots = nn.LayerNorm(dim); self.norm_mlp = nn.LayerNorm(dim)

    def forward(self, inputs):
        B, Ln, D = inputs.shape
        inputs = self.norm_in(inputs)
        k = self.to_k(inputs); v = self.to_v(inputs)
        mu = self.slots_mu.expand(B, self.n_slots, -1)
        sig = self.slots_logsig.exp().expand(B, self.n_slots, -1)
        slots = mu + sig * torch.randn_like(mu)
        for _ in range(self.iters):
            q = self.to_q(self.norm_slots(slots)) * self.scale
            attn = torch.softmax(torch.einsum('bnd,bmd->bnm', k, q), dim=-1)  # softmax over slots
            attn = attn + self.eps
            attn = attn / attn.sum(dim=1, keepdim=True)                        # weighted mean per slot
            updates = torch.einsum('bnm,bnd->bmd', attn, v)
            slots = self.gru(updates.reshape(-1, D), slots.reshape(-1, D)).reshape(B, self.n_slots, D)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots  # (B, M, D)


class VariantMeshGNN(MeshGNN):
    def __init__(self, ny, nx, variant="plain", **kw):
        extra_in = 2 if variant == "refframe" else 0
        super().__init__(ny=ny, nx=nx, **kw)
        self.variant = variant
        if extra_in:  # widen the encoder's first linear to take (k + 2) inputs
            enc0 = self.encoder[0]
            new0 = nn.Linear(enc0.in_features + extra_in, enc0.out_features)
            with torch.no_grad():
                new0.weight[:, :enc0.in_features] = enc0.weight
                new0.weight[:, enc0.in_features:] = 0.0
                new0.bias.copy_(enc0.bias)
            self.encoder[0] = new0
        if variant == "blurpool":
            k2d, r = gaussian_kernel2d(BLUR_SIGMA)
            self.register_buffer("blur_k", k2d.view(1, 1, *k2d.shape))
            self.blur_r = r
        if variant == "slot":
            self.slot_attn = SlotAttention(N_SLOTS, HIDDEN, iters=SLOT_ITERS)
            self.slot_to_node = nn.Linear(HIDDEN, HIDDEN)
            self.slot_gate = nn.Linear(2 * HIDDEN, HIDDEN)
        # precompute node grid coords for refframe
        yy, xx = np.mgrid[0:ny, 0:nx]
        self.register_buffer("coords", torch.tensor(
            np.stack([yy.ravel(), xx.ravel()], 1), dtype=torch.float32))  # (L, 2)

    def forward(self, x):
        B = x.size(0)
        if self.variant == "blurpool":
            xb = x.reshape(B * x.size(1), 1, self.ny, self.nx)
            xb = F.conv2d(xb, self.blur_k, padding=self.blur_r)
            x = xb.reshape(B, x.size(1), self.ny, self.nx)
        feats = x.reshape(B, x.size(1), self.L).permute(0, 2, 1)   # (B, L, k)
        if self.variant == "refframe":
            amp = x.abs().mean(1).reshape(B, self.L)               # (B, L) activity
            w = amp / (amp.sum(1, keepdim=True) + 1e-6)
            centroid = torch.einsum('bl,ld->bd', w, self.coords)   # (B, 2)
            rel = (self.coords.unsqueeze(0) - centroid.unsqueeze(1)) / self.ny  # (B, L, 2)
            feats = torch.cat([feats, rel], dim=-1)
        H = self.encoder(feats)
        if self.mp_mode == "graphcast":
            for layer in self.layers:
                H = layer(H, self.edge_src, self.edge_dst)
        else:
            A = self.adjacency()
            for layer in self.layers:
                H = layer(H, A)
        if self.variant == "slot":
            slots = self.slot_attn(H)                              # (B, M, D)
            q = self.slot_to_node(H)                               # (B, L, D)
            attn = torch.softmax(torch.einsum('bld,bmd->blm', q, slots) * (HIDDEN ** -0.5), dim=-1)
            bound = torch.einsum('blm,bmd->bld', attn, slots)      # slot content per node
            H = H + torch.tanh(self.slot_gate(torch.cat([H, bound], -1)))
        out = self.decoder(H).squeeze(-1)
        return out.reshape(B, 1, self.ny, self.nx)


def main():
    train_ds = MultiRealisationDataset(os.path.join(SPLIT_DIR, "train"), K)
    val_ds   = MultiRealisationDataset(os.path.join(SPLIT_DIR, "val"),   K)
    ny, nx = train_ds.segs[0].shape[1], train_ds.segs[0].shape[2]
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    model = VariantMeshGNN(ny=ny, nx=nx, variant=VARIANT, k=K).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nVariantMeshGNN variant={VARIANT}  params={n_params:,}  "
          f"epochs={EPOCHS} batch={BATCH_SIZE}  ckpt={CKPT_DIR}")
    if VARIANT == "blurpool": print(f"  blur sigma={BLUR_SIGMA} radius={model.blur_r}")
    if VARIANT == "slot":     print(f"  slots={N_SLOTS} iters={SLOT_ITERS}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR / 20)
    os.makedirs(CKPT_DIR, exist_ok=True)
    best = float("inf"); history = []
    print(f"{'Ep':>4} {'TrRMSE':>8} {'VlRMSE':>8} {'VlCorr':>8}  Time")
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, opt)
        vl = run_epoch(model, val_loader)
        sched.step()
        history.append({"epoch": ep, **{f"train_{k}": v for k, v in tr.items()},
                        **{f"val_{k}": v for k, v in vl.items()}})
        print(f"{ep:>4} {tr['rmse']:>8.4f} {vl['rmse']:>8.4f} {vl['corr']:>8.4f}  {time.time()-t0:.1f}s")
        if vl["rmse"] < best:
            best = vl["rmse"]
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "val_rmse": best, "variant": VARIANT}, os.path.join(CKPT_DIR, "best.pt"))
    np.save(os.path.join(CKPT_DIR, "history.npy"), history)
    print(f"\nBest val RMSE : {best:.4f}   ->  {CKPT_DIR}/")


if __name__ == "__main__":
    main()
