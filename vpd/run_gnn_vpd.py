"""VPD on the SAVAR MeshGNN forecaster — experiment module (mirrors resid_mlp/run.py).

Decomposes the frozen MeshGNN's shared message-passing MLPs (and, for the M1 smoke
test, just `decoder.0`) with param-decomp. The per-node CI gate is the spatial usage
map; see notes/vpd_on_savar_gnn.md.

Run with the param-decomp venv:
  PARAM_DECOMP_OUT_DIR=/home/ec2-user/savar-project/vpd_out \
  param-decomp/.venv/bin/python vpd/run_gnn_vpd.py vpd/config_gnn_smoke.yaml
"""

import os
import sys
from pathlib import Path

import fire
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = "/home/ec2-user/savar-project"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from param_decomp.base_config import BaseConfig
from param_decomp.log import logger
from param_decomp.optimize import Trainer
from param_decomp_lab.batch_and_loss_fns import recon_loss_mse
from param_decomp_lab.distributed import get_device
from param_decomp_lab.experiments.utils import ExperimentConfig, init_pd_run
from param_decomp_lab.seed import set_seed

from train.gnn_forecaster import MeshGNN, MultiRealisationDataset


class MeshGNNNodeOut(MeshGNN):
    """MeshGNN whose forecast is returned node-shaped `(B, L, 1)` instead of `(B,1,ny,nx)`.

    param-decomp derives the per-position adversarial/mask axis from the model OUTPUT
    (`persistent_pgd_recon.py`: `batch_dims = target_out.shape[:-1]`), assuming output
    positions == the CI/mask positions (true for an LM's per-token output). The CI gate
    here is per mesh NODE, shape `(B, L, C)`, so the output must expose the same `(B, L)`
    leading dims. Parameters are identical to `MeshGNN`; only the final reshape differs,
    so a `MeshGNN` checkpoint loads unchanged. Recon MSE is elementwise, so equivalent."""

    def forward(self, x):
        out = super().forward(x)            # (B, 1, ny, nx)
        return out.reshape(out.size(0), self.L, 1)


def run_batch_gnn(model, batch):
    """Run the frozen GNN on x = batch[0], moving it to the model's device.

    Our map-style DataLoader yields CPU tensors and this code path does not auto-move
    them, so do it here (the GNN's reference output is computed by the same fn)."""
    x = batch[0]
    device = next(model.parameters()).device
    return model(x.to(device))


class GNNTargetConfig(BaseConfig):
    """Frozen MeshGNN target. `checkpoint_path: null` ⇒ untrained model (M1 plumbing test)."""

    checkpoint_path: str | None = None
    ny: int = 50
    nx: int = 50
    k: int = 3


class GNNDataConfig(BaseConfig):
    split_dir: str = "data/splits_finecadence"


class GNNExperimentConfig(ExperimentConfig[GNNTargetConfig, GNNDataConfig]):
    pass


def build_target(target_cfg: GNNTargetConfig) -> MeshGNN:
    model = MeshGNNNodeOut(ny=target_cfg.ny, nx=target_cfg.nx, k=target_cfg.k)
    if target_cfg.checkpoint_path:
        ckpt = torch.load(target_cfg.checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        logger.info(f"loaded checkpoint {target_cfg.checkpoint_path} (val_rmse={ckpt.get('val_rmse')})")
    else:
        logger.info("no checkpoint — using UNTRAINED MeshGNN (M1 plumbing smoke test)")
    model.eval()
    model.requires_grad_(False)
    return model


def build_loader(target_cfg: GNNTargetConfig, data_cfg: GNNDataConfig, *, batch_size: int) -> DataLoader:
    split = data_cfg.split_dir
    if not os.path.isabs(split):
        split = os.path.join(PROJECT_ROOT, split)
    ds = MultiRealisationDataset(os.path.join(split, "train"), k=target_cfg.k)
    logger.info(f"train windows: {len(ds)}  (split_dir={split})")
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2,
                      pin_memory=True, drop_last=True, persistent_workers=True)


def main(config_path: str | Path, *, group: str | None = None, tags: str | None = None) -> None:
    cfg = GNNExperimentConfig.from_file(config_path)
    set_seed(cfg.pd.seed)
    device = get_device()
    logger.info(f"device: {device}")
    cfg = cfg.model_copy(update={"runtime": cfg.runtime.model_copy(update={"device": device})})

    target_model = build_target(cfg.target).to(device)
    target_model.adjacency()  # build the cached sparse adjacency on-device before wrapping
    train_loader = build_loader(cfg.target, cfg.data, batch_size=cfg.pd.batch_size)

    sink = init_pd_run(cfg, group=group, tags=tags)
    try:
        trainer = Trainer(
            target_model=target_model,
            run_batch=run_batch_gnn,
            reconstruction_loss=recon_loss_mse,
            pd_config=cfg.pd,
            runtime_config=cfg.runtime,
        )
        trainer.run(train_loader, sink, cfg.cadence, None)
    finally:
        sink.finish()


if __name__ == "__main__":
    fire.Fire(main)
