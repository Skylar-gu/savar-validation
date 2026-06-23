"""Adapter conventions for the alternative causal discoverers (DYNOTEARS, TSCI).

Gates the (cause, eff, tau) index/sign mapping on a tiny planted VAR(2) before any
full run. DYNOTEARS is skipped if the isolated causalnex venv is absent.

Planted system (N=3):
    X0(t) = 0.4 X0(t-1) + e
    X1(t) = 0.8 X0(t-1) + e        # X0 -> X1 at lag 1, positive
    X2(t) = -0.7 X1(t-2) + e       # X1 -> X2 at lag 2, negative
True cross-edges: (0,1,1)=+, (1,2,2)=-.
"""
import os, sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pcmci"))


def _make_var(R=2, T=1500, seed=0):
    rng = np.random.default_rng(seed)
    data = np.zeros((R, T, 3))
    for r in range(R):
        x = np.zeros((T, 3))
        for t in range(2, T):
            x[t, 0] = 0.4 * x[t - 1, 0] + 0.3 * rng.standard_normal()
            x[t, 1] = 0.8 * x[t - 1, 0] + 0.3 * rng.standard_normal()
            x[t, 2] = -0.7 * x[t - 2, 1] + 0.3 * rng.standard_normal()
        data[r] = x
    return data


# ── eval_common sanity ─────────────────────────────────────────────────────────

def test_summary_collapse_and_gt():
    from eval_common import summary_gt_matrix, summary_score_matrix
    G = np.zeros((3, 3, 2))
    G[1, 0, 0] = 0.8       # X0(t-1)->X1
    G[2, 1, 1] = -0.7      # X1(t-2)->X2
    A = summary_gt_matrix(G)
    assert A[0, 1] == 1 and A[1, 2] == 1 and A.sum() == 2
    score = np.zeros((3, 3, 3))
    score[0, 1, 1] = 0.5; score[1, 2, 2] = -0.9
    S = summary_score_matrix(score)
    assert S[0, 1] == pytest.approx(0.5) and S[1, 2] == pytest.approx(0.9)
    assert np.allclose(np.diag(S), 0)


# ── DYNOTEARS ───────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not (__import__("pathlib").Path(ROOT) / ".venv_causalnex" / "bin" / "python").exists(),
    reason="causalnex venv (.venv_causalnex) not present",
)
def test_dynotears_edges_lags_signs():
    from methods import dynotears
    data = _make_var()
    sc = dynotears.discover_all(data, tau_max=2, w_threshold=0.0, verbose=False)
    assert sc.shape == (2, 3, 3, 3)
    s = sc[0]
    assert s[0, 1, 1] > 0.1, "X0->X1 lag1 should be a clear positive edge"
    assert s[1, 2, 2] < -0.1, "X1->X2 lag2 should be a clear negative edge"
    assert abs(s[0, 2, 1]) < 0.1 and abs(s[2, 0, 1]) < 0.1, "non-edges ~0"


# ── TSCI ─────────────────────────────────────────────────────────────────────────

def test_tsci_summary_shape_and_diag():
    from methods import tsci_adapter
    data = _make_var(R=1, T=800)
    summ = tsci_adapter.discover_all(data, tau=1, Q=3, verbose=False)
    assert summ.shape == (1, 3, 3)
    assert np.allclose(np.diag(summ[0]), 0), "diagonal (self) must be zero"


def test_tsci_direction_on_unidirectional_coupling():
    # X0 drives X1, X1 drives X2; the reverse links are absent. TSCI directional
    # score should favour the true direction over its reverse for the driven pairs.
    from methods import tsci_adapter
    data = _make_var(R=3, T=1500)
    summ = tsci_adapter.discover_all(data, tau=1, Q=3, verbose=False).mean(0)
    assert summ[0, 1] > summ[1, 0], "X0->X1 should beat X1->X0"
