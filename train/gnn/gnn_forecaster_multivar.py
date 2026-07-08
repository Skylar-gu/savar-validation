"""
R3 MULTIVARIATE MeshGNN forecaster — channel-aware fork of gnn_forecaster.py.

This is the forecaster under test for R3 (the overlap02/hetdynamics checkpoints
are MeshGNNs, not CNNs). The ONLY change vs gnn_forecaster.py is the per-node
feature/target dimension: each grid cell now carries C coupled observed channels
instead of one scalar field, so the aggregation Ŵ becomes (channel × space).

  per-node input  : C channels × k past frames  (was k)         → C*k features
  per-node output : C channels                    (was 1)
  Input  : (B, C, k, ny, nx)
  Output : (B, C, ny, nx)

Everything else — heterogeneous multi-scale mesh, MP layers (gcn / graphcast),
optional per-node embeddings and static inputs, metrics, checkpoint logic — is
IDENTICAL to the single-channel MeshGNN. C is inferred from the data.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os, sys, time, glob
sys.stdout.reconfigure(line_buffering=True)

# ── config ───────────────────────────────────────────────────────────────────
K          = 3
HIDDEN     = 256
N_MP       = 4          # message-passing layers
EMB_DIM    = int(os.environ.get("GNN_EMB_DIM", 0))  # per-node emb width (0=none, faithful; >0=confounded control)
STATIC_IN  = int(os.environ.get("GNN_STATIC_INPUTS", 0))  # R2 rung: fixed sin/cos coords + hub flag as input channels
MP_MODE    = os.environ.get("GNN_MP_MODE", "gcn")   # "gcn" (fixed Â aggregation) | "graphcast" (edge MLP + node MLP)
HUB_STRIDE = 5          # coarse-mesh spacing (→ heterogeneous hub degree)
HUB_RADIUS = 2          # hub-lattice connection radius
BATCH_SIZE = int(os.environ.get("GNN_BATCH", 32))
LR         = 3e-4
EPOCHS     = int(os.environ.get("GNN_EPOCHS", 40))
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR   = os.environ.get("GNN_CKPT_DIR", "checkpoints/multivar")
SPLIT_DIR  = os.environ.get("GNN_SPLIT_DIR", os.path.join("data", "splits_multivar2"))


# ── dataset (channel-aware; obs stored (T_split, C, ny, nx)) ───────────────────
class MultiRealisationDataset(Dataset):
    def __init__(self, split_dir, k=K):
        self.k = k
        self.segs = []
        for path in sorted(glob.glob(os.path.join(split_dir, "realisation_*.npz"))):
            obs = np.load(path)["observations"]            # (T_split, C, ny, nx)
            self.segs.append(torch.from_numpy(obs).float())
        self.C = self.segs[0].shape[1]
        self.index = [(s, i) for s, seg in enumerate(self.segs)
                      for i in range(len(seg) - k)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        s, i = self.index[idx]
        seg = self.segs[s]
        x = seg[i : i + self.k].permute(1, 0, 2, 3)  # (C, k, ny, nx)
        y = seg[i + self.k]                           # (C, ny, nx)
        return x, y


# ── heterogeneous multi-scale mesh ─────────────────────────────────────────────
def build_mesh(ny, nx, hub_stride=HUB_STRIDE, hub_radius=HUB_RADIUS):
    """Return (edge_src, edge_dst) for a grid with local 8-neighbour edges plus
    long-range hub-hub edges. Hubs sit on a coarse lattice and get high degree."""
    def nid(r, c): return r * nx + c
    edges = set()
    # local 8-neighbourhood
    for r in range(ny):
        for c in range(nx):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < ny and 0 <= cc < nx:
                        edges.add((nid(r, c), nid(rr, cc)))
    # coarse hub lattice + long-range hub-hub edges (the heterogeneity)
    hubs = [(r, c) for r in range(0, ny, hub_stride) for c in range(0, nx, hub_stride)]
    for (r, c) in hubs:
        for (r2, c2) in hubs:
            if (r2, c2) == (r, c):
                continue
            if abs((r2 - r) // hub_stride) <= hub_radius and abs((c2 - c) // hub_stride) <= hub_radius:
                edges.add((nid(r, c), nid(r2, c2)))
    src = np.array([s for s, _ in edges], dtype=np.int64)
    dst = np.array([d for _, d in edges], dtype=np.int64)
    return src, dst, len(hubs)


def normalized_adjacency(src, dst, L, device):
    """Symmetric-normalised sparse adjacency  Â = D^-1/2 (A + I) D^-1/2."""
    self_idx = np.arange(L)
    s = np.concatenate([src, self_idx])
    d = np.concatenate([dst, self_idx])
    idx = torch.tensor(np.stack([d, s]), dtype=torch.long)        # (2, E) dst<-src
    deg = np.bincount(d, minlength=L).astype(np.float32)
    dinv = 1.0 / np.sqrt(deg)
    vals = torch.tensor(dinv[d] * dinv[s], dtype=torch.float32)
    A = torch.sparse_coo_tensor(idx, vals, (L, L)).coalesce().to(device)
    return A


# ── model ──────────────────────────────────────────────────────────────────────
class MPLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, H, A):
        B, L, Hd = H.shape
        agg = torch.sparse.mm(A, H.permute(1, 0, 2).reshape(L, B * Hd)).reshape(L, B, Hd).permute(1, 0, 2)
        upd = self.mlp(torch.cat([H, agg], dim=-1))
        return self.norm(H + upd)


class GraphCastMPLayer(nn.Module):
    """GraphCast-faithful message passing: a LEARNED, DIRECTIONAL edge MLP produces
    a message per edge from (sender, receiver) features, messages are summed at the
    receiver, and a node MLP updates each node from (its state, aggregated messages)."""
    def __init__(self, hidden):
        super().__init__()
        self.edge_mlp = nn.Sequential(            # message function  (sender, receiver) -> msg
            nn.Linear(2 * hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.node_mlp = nn.Sequential(            # update function   (state, agg msgs) -> update
            nn.Linear(2 * hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, H, src, dst):
        B, L, Hd = H.shape
        h_src = H[:, src, :]                              # (B, E, Hd)
        h_dst = H[:, dst, :]                              # (B, E, Hd)
        msg = self.edge_mlp(torch.cat([h_src, h_dst], dim=-1))   # (B, E, Hd)
        agg = H.new_zeros(B, L, Hd)
        agg.index_add_(1, dst, msg)                      # sum messages at receivers
        upd = self.node_mlp(torch.cat([H, agg], dim=-1))
        return self.norm(H + upd)


class MeshGNN(nn.Module):
    def __init__(self, ny, nx, k=K, channels=2, hidden=HIDDEN, n_mp=N_MP, emb_dim=EMB_DIM,
                 mp_mode=MP_MODE, static_in=STATIC_IN):
        super().__init__()
        self.ny, self.nx, self.L = ny, nx, ny * nx
        self.C = channels                       # observed channels (was implicit 1)
        self.k = k
        self.emb_dim = emb_dim
        self.mp_mode = mp_mode
        self.static_in = static_in
        self.node_emb = nn.Embedding(self.L, emb_dim) if emb_dim > 0 else None  # confound; off by default
        n_static = 5 if static_in else 0
        if static_in:
            # fixed static channels (GraphCast lat/lon/orography analogue)
            rr, cc = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
            hub = ((rr % HUB_STRIDE == 0) & (cc % HUB_STRIDE == 0)).astype(np.float32)
            feats = np.stack([np.sin(2 * np.pi * cc / nx), np.cos(2 * np.pi * cc / nx),
                              np.sin(2 * np.pi * rr / ny), np.cos(2 * np.pi * rr / ny),
                              hub], axis=-1).reshape(self.L, 5).astype(np.float32)
            self.register_buffer("static_feats", torch.from_numpy(feats), persistent=False)
        # per-node input is C channels × k frames (was k); output is C channels (was 1)
        self.encoder  = nn.Sequential(
            nn.Linear(channels * k + emb_dim + n_static, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        if mp_mode == "graphcast":
            self.layers = nn.ModuleList([GraphCastMPLayer(hidden) for _ in range(n_mp)])
        elif mp_mode == "gcn":
            self.layers = nn.ModuleList([MPLayer(hidden) for _ in range(n_mp)])
        else:
            raise ValueError(f"unknown GNN_MP_MODE={mp_mode!r} (use 'gcn' or 'graphcast')")
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, channels),
        )
        src, dst, n_hubs = build_mesh(ny, nx)
        self.n_hubs = n_hubs
        self.register_buffer("_dummy", torch.zeros(1), persistent=False)
        self.register_buffer("edge_src", torch.from_numpy(src), persistent=False)
        self.register_buffer("edge_dst", torch.from_numpy(dst), persistent=False)
        self._A = None
        self._edges = (src, dst)

    def adjacency(self):
        if self._A is None:
            src, dst = self._edges
            self._A = normalized_adjacency(src, dst, self.L, self._dummy.device)
        return self._A

    def forward(self, x):
        # x: (B, C, k, ny, nx) → node features (B, L, C*k)
        B = x.size(0)
        feats = x.reshape(B, self.C * self.k, self.L).permute(0, 2, 1)   # (B, L, C*k)
        if self.node_emb is not None:
            ids = torch.arange(self.L, device=x.device)
            emb = self.node_emb(ids).unsqueeze(0).expand(B, -1, -1)      # (B, L, emb)
            feats = torch.cat([feats, emb], dim=-1)
        if self.static_in:
            feats = torch.cat([feats, self.static_feats.unsqueeze(0).expand(B, -1, -1)], dim=-1)
        H = self.encoder(feats)                                          # (B, L, hidden)
        if self.mp_mode == "graphcast":
            for layer in self.layers:
                H = layer(H, self.edge_src, self.edge_dst)
        else:
            A = self.adjacency()
            for layer in self.layers:
                H = layer(H, A)
        out = self.decoder(H)                                           # (B, L, C)
        return out.permute(0, 2, 1).reshape(B, self.C, self.ny, self.nx)


# ── metrics / loops (same as gnn_forecaster) ───────────────────────────────────
def forecast_corr(pred, target):
    p = pred.flatten(1); t = target.flatten(1)
    p = p - p.mean(1, keepdim=True); t = t - t.mean(1, keepdim=True)
    num = (p * t).sum(1)
    den = p.norm(dim=1) * t.norm(dim=1) + 1e-8
    return (num / den).mean().item()


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    tot_mse = tot_corr = n = 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            if training:
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            bs = x.size(0)
            tot_mse += loss.item() * bs
            tot_corr += forecast_corr(pred, y) * bs
            n += bs
    mse = tot_mse / n
    return {"mse": mse, "rmse": mse ** 0.5, "corr": tot_corr / n}


def main():
    train_ds = MultiRealisationDataset(os.path.join(SPLIT_DIR, "train"), K)
    val_ds   = MultiRealisationDataset(os.path.join(SPLIT_DIR, "val"),   K)
    C  = train_ds.segs[0].shape[1]                     # (T, C, ny, nx)
    ny = train_ds.segs[0].shape[2]; nx = train_ds.segs[0].shape[3]

    n_workers = int(os.environ.get("GNN_WORKERS", 4))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=n_workers, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=n_workers, pin_memory=True, persistent_workers=True)

    model = MeshGNN(ny=ny, nx=nx, k=K, channels=C).to(DEVICE)
    init_from = os.environ.get("GNN_INIT_FROM", "")
    if init_from and os.path.exists(init_from):
        ck = torch.load(init_from, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model_state"])
        print(f"  warm-start from {init_from} (epoch {ck.get('epoch')}, "
              f"val_rmse {ck.get('val_rmse'):.4f})")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: MeshGNN (heterogeneous multi-scale mesh, multivariate)")
    print(f"  Device     : {DEVICE}")
    print(f"  Parameters : {n_params:,}  ({n_params/1e6:.2f}M)")
    print(f"  Channels C : {C}")
    print(f"  Mesh       : {ny}x{nx} grid, {model.n_hubs} hubs (stride {HUB_STRIDE})")
    print(f"  MP mode    : {MP_MODE}  ({'edge MLP + node MLP' if MP_MODE=='graphcast' else 'fixed Â aggregation + node MLP'})")
    print(f"  MP layers  : {N_MP}   hidden {HIDDEN}   node-emb {EMB_DIM} "
          f"({'OFF — grid-lock must emerge from mesh' if EMB_DIM == 0 else 'ON — confounded control'})")
    print(f"  Train/Val  : {len(train_ds)} / {len(val_ds)} windows\n")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR / 20)

    best = float("inf"); history = []
    print(f"{'Ep':>4} {'TrRMSE':>8} {'VlRMSE':>8} {'VlCorr':>8} {'LR':>9}  Time")
    print("-" * 50)
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, opt)
        vl = run_epoch(model, val_loader)
        sched.step()
        history.append({"epoch": ep, **{f"train_{k}": v for k, v in tr.items()},
                        **{f"val_{k}": v for k, v in vl.items()}})
        print(f"{ep:>4} {tr['rmse']:>8.4f} {vl['rmse']:>8.4f} {vl['corr']:>8.4f} "
              f"{sched.get_last_lr()[0]:>9.2e}  {time.time()-t0:.1f}s")
        if vl["rmse"] < best:
            best = vl["rmse"]
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "val_rmse": best, "channels": C}, os.path.join(CKPT_DIR, "best.pt"))
    np.save(os.path.join(CKPT_DIR, "history.npy"), history)
    print(f"\nBest val RMSE : {best:.4f}   →  {CKPT_DIR}/")


if __name__ == "__main__":
    os.makedirs(CKPT_DIR, exist_ok=True)
    main()
