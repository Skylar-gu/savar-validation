"""Lock the tigramite p_matrix index convention (cause, eff, tau).

This was verified empirically during development and the whole PCMCI evaluation
(run_pcmci*.py) depends on it: p_matrix[cause, eff, tau] is the p-value for the
link X_cause(t-tau) -> X_eff(t). A tigramite version bump that flips this would
silently invert precision/recall, so we pin it with a known one-way VAR.
"""
import numpy as np
import pytest

pytest.importorskip("tigramite")
from tigramite.data_processing import DataFrame
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI


def test_pmatrix_cause_eff_tau_convention():
    rng = np.random.default_rng(0)
    T = 800
    x0 = rng.standard_normal(T)
    x1 = np.zeros(T)
    for t in range(1, T):                      # X0(t-1) -> X1(t), one direction only
        x1[t] = 0.7 * x0[t - 1] + 0.1 * rng.standard_normal()
    Z = np.column_stack([x0, x1])

    pc = PCMCI(dataframe=DataFrame(Z), cond_ind_test=ParCorr(), verbosity=0)
    res = pc.run_pcmci(tau_min=1, tau_max=2, pc_alpha=0.2, alpha_level=0.05)
    p = res["p_matrix"]
    v = res["val_matrix"]         # partial correlation strength

    # convention: [cause, eff, tau]. The true edge X0(t-1) -> X1(t) is detected,
    # and its partial correlation is strong ONLY in this orientation — if the
    # convention were [eff, cause, tau] the strength would sit at [1, 0, 1].
    assert p[0, 1, 1] < 0.05
    assert abs(v[0, 1, 1]) > 0.3
    assert abs(v[0, 1, 1]) > 3 * abs(v[1, 0, 1])
