"""Cycle math for the 6h diurnal/annual dataset — pure numpy, no deps.

Locks the sampling/phasing decisions made in data_gen/generate_diurnal.py:
  - diurnal period (4 steps), annual period (1461 steps) at dt=6h
  - the 12h semidiurnal tide sits exactly at Nyquist (why it was dropped)
  - per-mode diurnal phase from longitude, annual phase from hemisphere
  - ensemble-mean deseasonalization removes a shared cycle (the PCMCI/SAE fix)
"""
import numpy as np


DT_HOURS = 6
NY = NX = 50
# the 8 modes' blob centres (3x3 grid, 9th slot empty)
X_CENTER = np.array([8, 24, 40, 8, 24, 40, 8, 24])
Y_CENTER = np.array([8, 8, 8, 24, 24, 24, 40, 40])


def test_diurnal_period_is_four_steps():
    assert 24 // DT_HOURS == 4


def test_annual_period():
    assert int(round(365.25 * 24 / DT_HOURS)) == 1461


def test_semidiurnal_sits_at_nyquist():
    # Nyquist period = 2*dt = 12h → a 12h tide is unresolvable at 6h sampling,
    # which is why it is dropped (ERA5/GraphCast can't see it either).
    nyquist_period_h = 2 * DT_HOURS
    assert nyquist_period_h == 12
    assert (12 / DT_HOURS) == 2            # exactly 2 samples/cycle = Nyquist
    assert (24 / DT_HOURS) > 2             # diurnal is above Nyquist → resolvable


def test_diurnal_phase_has_three_longitude_groups():
    phi_d = -2 * np.pi * X_CENTER / NX
    assert len(np.unique(np.round(phi_d, 6))) == 3   # x in {8,24,40}


def test_annual_phase_hemispheric_flip():
    phi_a = np.where(Y_CENTER > NY / 2, np.pi, 0.0)
    # only the southern modes (6,7 at y=40) are flipped to pi
    assert phi_a[6] == np.pi and phi_a[7] == np.pi
    assert np.all(phi_a[:6] == 0.0)


def test_ensemble_mean_removes_shared_cycle():
    """Subtracting the ensemble mean over realisations removes a shared
    deterministic cycle and leaves anomalies uncorrelated with it."""
    rng = np.random.default_rng(0)
    T, R, P_D, P_A = 800, 60, 4, 1461
    t = np.arange(T)
    cycle = np.sin(2 * np.pi * t / P_D) + 0.5 * np.sin(2 * np.pi * t / P_A)  # shared
    Z = np.stack([cycle + 0.3 * rng.standard_normal(T) for _ in range(R)])    # (R, T)

    clim = Z.mean(axis=0)
    assert np.corrcoef(clim, cycle)[0, 1] > 0.99        # ensemble mean ≈ the cycle

    anom = Z - clim
    flat_anom = anom.reshape(-1)
    flat_cycle = np.tile(cycle, R)
    assert abs(np.corrcoef(flat_anom, flat_cycle)[0, 1]) < 0.05   # cycle removed
