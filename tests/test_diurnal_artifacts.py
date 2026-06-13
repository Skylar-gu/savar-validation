"""Checks on a generated diurnal realisation (skipped if data not generated).

Turns the manual post-generation verification into a test: shape, recoverable
cycle metadata, and an actual diurnal signal present in every mode.
"""
import numpy as np
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NPZ = ROOT / "data" / "realisations_diurnal" / "realisation_000.npz"

pytestmark = pytest.mark.skipif(
    not NPZ.exists(),
    reason="diurnal dataset not generated (run data_gen/generate_diurnal.py)",
)


def _load():
    return np.load(NPZ)


def test_shapes_and_cycle_meta():
    d = _load()
    assert d["observations"].shape == (2500, 2920)   # (L, T=2 years @ 6h)
    assert d["latent_states"].shape == (8, 2920)
    cm = d["cycle_meta"]                              # [dt, P_D, P_A, het]
    assert int(cm[0]) == 6 and int(cm[1]) == 4 and int(cm[2]) == 1461
    assert cm[3] == pytest.approx(0.6, abs=1e-5)


def test_forcing_recoverable():
    d = _load()
    assert d["forcing_latent"].shape == (8, 2920)
    assert d["diurnal_amp"].shape == (8,)
    assert d["diurnal_phase"].shape == (8,)
    # southern modes (6,7) flipped in annual phase
    assert d["annual_phase"][6] == pytest.approx(np.pi, abs=1e-5)
    assert d["annual_phase"][7] == pytest.approx(np.pi, abs=1e-5)


def test_diurnal_signal_present_in_all_modes():
    d = _load()
    Z = d["latent_states"]
    P_D = 4
    ph = 2 * np.pi * np.arange(Z.shape[1]) / P_D
    amp = np.array([2 * np.abs(np.mean(Z[j] * np.exp(-1j * ph))) for j in range(8)])
    assert np.all(amp > 0.0)        # a 4-step diurnal component exists everywhere
    assert amp.mean() > 0.1         # and it is non-trivial


def test_no_nans_and_finite():
    d = _load()
    assert np.isfinite(d["observations"]).all()
    assert np.isfinite(d["latent_states"]).all()
