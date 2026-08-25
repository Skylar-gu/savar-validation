"""Single root for every path in the SAVAR package.

    SAVAR_ROOT   env var; defaults to the directory containing this file.

Everything the pipeline reads (data/, checkpoints/, sae_data/) and writes
(results/, results/ladder_cnn/, results/ladder_gnn/) hangs off this root, so the
package can live inside another repository (e.g. causal-graphcast-repro/savar/) or
be pointed at an existing experiment tree without editing any script.
Scripts that use bare relative paths ("data/realisations", "checkpoints/base")
must be run from SAVAR_ROOT as the working directory, as before.
"""
import os
from pathlib import Path

SAVAR_ROOT = Path(os.environ.get("SAVAR_ROOT", Path(__file__).resolve().parent)).resolve()
DATA_DIR = SAVAR_ROOT / "data"
CKPT_DIR = SAVAR_ROOT / "checkpoints"
SAE_DATA_DIR = SAVAR_ROOT / "sae_data"
RESULTS_DIR = SAVAR_ROOT / "results"
