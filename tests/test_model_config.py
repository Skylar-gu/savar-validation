"""Ground-truth SAVAR config invariants (requires upstream savar library).

These lock the dataset's structural guarantees that the whole validation pipeline
depends on: mode weights, pseudo-inverse, the causal graph shape/signs, the
hub/sink structure, and stationarity (spectral radius < 1).
"""
import numpy as np


def test_dimensions(model):
    assert model.N == 8
    assert model.ny == 50 and model.nx == 50
    assert model.L == 2500
    assert model.W_flat.shape == (8, 2500)
    assert model.W_plus.shape == (2500, 8)


def test_mode_weights_l1_normalised(model):
    # each mode's blob sums to 1 (L1-normalised)
    assert np.allclose(model.W_flat.sum(axis=1), 1.0)


def test_pseudo_inverse(model):
    assert np.allclose(model.W_flat @ model.W_plus, np.eye(8), atol=1e-6)


def test_graph_shape(model):
    assert model.G.shape == (8, 8, 2)   # (eff, cause, tau_max)


def test_negative_edges(model):
    # G[eff, cause, tau-1]; these three edges are negative by construction
    assert model.G[3, 2, 0] < 0   # X2 -> X3 (lag 1)
    assert model.G[5, 0, 1] < 0   # X0 -> X5 (lag 2)
    assert model.G[7, 3, 1] < 0   # X3 -> X7 (lag 2)


def test_X7_is_a_sink(model):
    # X7 has zero cross-mode out-degree (only appears as an effect)
    out_of_7 = np.delete(model.G, 7, axis=0)[:, 7, :]   # eff != 7, cause = 7
    assert np.allclose(out_of_7, 0.0)


def test_X0_is_a_hub(model):
    # X0 drives multiple modes (X1, X3, X5)
    out_of_0 = np.delete(model.G, 0, axis=0)[:, 0, :]   # eff != 0, cause = 0
    assert np.count_nonzero(out_of_0) >= 3


def test_stationarity_spectral_radius(model):
    from savar.functions import create_graph
    g = create_graph(model.links_coeffs, return_lag=False)
    p = g.shape[2]
    top = np.hstack([g[:, :, i] for i in range(p)])
    bot = np.hstack([np.eye(model.N * (p - 1)), np.zeros((model.N * (p - 1), model.N))])
    sr = float(np.max(np.abs(np.linalg.eigvals(np.vstack([top, bot])))))
    assert sr < 1.0           # stationary VAR
    assert sr == __import__("pytest").approx(0.7794, abs=1e-3)
