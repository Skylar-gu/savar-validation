"""Pytest path setup so tests run from the repo root the same way the scripts do
(savar / instantiate_model / cnn_forecaster resolvable). Mirrors the root-relative
import convention used across the pipeline."""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
for sub in ("", "data_gen", "train", "sae"):
    p = os.path.join(ROOT, sub) if sub else ROOT
    if p not in sys.path:
        sys.path.insert(0, p)

_savar = os.path.join(ROOT, "savar")
if os.path.isdir(_savar) and _savar not in sys.path:
    sys.path.insert(0, _savar)

import pytest


@pytest.fixture(scope="session")
def model():
    """The instantiated SAVAR ground-truth config (skips if upstream savar absent)."""
    pytest.importorskip("savar")
    import importlib
    return importlib.import_module("instantiate_model")
