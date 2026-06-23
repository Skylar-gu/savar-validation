"""
DYNOTEARS adapter (main env).

Prepares a stacked (R, T, N) input, shells out to dynotears_worker.py running under
the isolated causalnex venv, and returns signed score matrices (R, N, N, tau_max+1)
indexed [realisation, cause, eff, tau]. No causalnex import happens in this process.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]          # project root
_VENV_PY = _ROOT / ".venv_causalnex" / "bin" / "python"
_WORKER = Path(__file__).resolve().parent / "dynotears_worker.py"


def venv_available():
    return _VENV_PY.exists()


def discover_all(data_RTN, tau_max=2, lambda_w=0.05, lambda_a=0.05,
                 w_threshold=0.0, max_iter=100, verbose=True):
    """data_RTN: (R, T, N) float array. Returns scores (R, N, N, tau_max+1).

    Raises RuntimeError with the worker's stderr if the subprocess fails.
    """
    if not venv_available():
        raise RuntimeError(
            f"causalnex venv not found at {_VENV_PY}. Create it with:\n"
            f"  python3.9 -m venv .venv_causalnex && "
            f".venv_causalnex/bin/pip install causalnex"
        )
    data_RTN = np.ascontiguousarray(np.asarray(data_RTN, dtype=np.float64))
    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / "in.npy"
        out = Path(td) / "out.npy"
        np.save(inp, data_RTN)
        cmd = [
            str(_VENV_PY), str(_WORKER),
            "--in", str(inp), "--out", str(out),
            "--tau_max", str(tau_max),
            "--lambda_w", str(lambda_w), "--lambda_a", str(lambda_a),
            "--w_threshold", str(w_threshold), "--max_iter", str(max_iter),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "DYNOTEARS worker failed:\n"
                f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
            )
        if verbose and proc.stdout:
            sys.stdout.write(proc.stdout)
        return np.load(out)
