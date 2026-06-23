"""
TSCI core — vendored & trimmed from KurtButler/tangentspaces (MIT). See ./LICENSE.

Only the nearest-neighbour TSCI estimator (`tsci_nn`) and the embedding helpers it
needs are kept. jaxtyping / tqdm / statsmodels dependencies are removed so this runs
on the project's main numpy-2 / torch-2.6 environment with no extra installs. The
optional mutual-information score path imports `bmi` lazily (not installed by default).
"""

import warnings
import numpy as np
import scipy
import scipy.spatial
import scipy.linalg
import scipy.signal


# ── delay embedding & derivatives ─────────────────────────────────────────────

def delay_embed(x, lag, embed_dim):
    """Delay embedding of x (T, 1) with the given lag and embedding dimension."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    num_x = x.shape[0] - (embed_dim - 1) * lag
    embed_list = []
    for i in range(embed_dim):
        start = (embed_dim - 1) * lag - (i * lag)
        embed_list.append(x[start:start + num_x].reshape(-1, x.shape[1]))
    return np.concatenate(embed_list, axis=-1)


def discrete_velocity(x, smooth=False):
    """Discrete derivative of a time series (2nd-order central differences)."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if smooth:
        return scipy.signal.savgol_filter(x, 5, 2, deriv=1, axis=0)
    return np.gradient(x, axis=0)


# ── autocorrelation-based lag selection (numpy; replaces statsmodels.acf) ──────

def _acf(x, nlags=None):
    """Normalised autocorrelation function via FFT. Returns acf[0..nlags]."""
    x = np.asarray(x, dtype=float).ravel()
    x = x - x.mean()
    n = len(x)
    if nlags is None:
        nlags = n - 1
    nfft = 1
    while nfft < 2 * n:
        nfft <<= 1
    f = np.fft.fft(x, n=nfft)
    acf = np.fft.ifft(f * np.conjugate(f))[:n].real
    if acf[0] == 0:
        return np.zeros(nlags + 1)
    acf /= acf[0]
    return acf[:nlags + 1]


def lag_select(x, theta=0.5, max_tau=100):
    """First lag at which the ACF drops below theta (Takens lag heuristic)."""
    acf = _acf(x, nlags=min(max_tau, len(x) - 1))
    if np.all(acf >= theta):
        return max_tau
    tau = int(np.argmax(acf < theta))
    return max_tau if tau == 0 else tau


def false_nearest_neighbors(y, tau, fnn_tol=0.01, Q_max=20, rho=17.0):
    """Heuristic embedding dimension via the false-nearest-neighbours algorithm."""
    Q = 1
    while True:
        Q += 1
        if Q > Q_max:
            warnings.warn("FNN did not converge.")
            return Q_max
        M1 = delay_embed(y, tau, Q)
        M2 = delay_embed(y, tau, Q + 1)
        M1 = M1[: M2.shape[0]]
        fnn = np.zeros(M1.shape[0])
        kdtree = scipy.spatial.KDTree(M1)
        for n in range(M1.shape[0]):
            _, ids = kdtree.query(M1[n, :], 2)
            Rd = np.linalg.norm(M1[ids[1], :] - M1[n, :], 2) / np.sqrt(Q)
            fnn[n] = np.linalg.norm(M2[n, :] - M2[ids[1], :], 2) > rho * Rd
        if np.mean(fnn) < fnn_tol:
            return Q


# ── the TSCI nearest-neighbour estimator ──────────────────────────────────────

def tsci_nn(x_state, y_state, dx_state, dy_state,
            fraction_train=0.8, lib_length=-1, use_mutual_info=False):
    """Tangent Space Causal Inference (nearest-neighbour Jacobian estimator).

    Args mirror the upstream function: delay embeddings `x_state`/`y_state` and their
    delay-embedded velocity fields `dx_state`/`dy_state`. Returns per-test-point
    score arrays (score_x2y, score_y2x) for the causal directions X->Y and Y->X.
    Aggregate (e.g. np.mean) for a scalar directional strength.
    """
    Q_x = dx_state.shape[1]
    Q_y = dy_state.shape[1]
    N_samples = dx_state.shape[0]
    N_train = int(fraction_train * N_samples)
    if lib_length < 0:
        lib_length = N_train

    # X -> Y pushforward
    x_pushforward = np.zeros_like(dy_state[N_train:])
    K = 3 * Q_x
    kdtree = scipy.spatial.KDTree(x_state[:lib_length])
    for n in range(N_train, x_state.shape[0]):
        _, ids = kdtree.query(x_state[n, :], K)
        x_tangents = x_state[ids, :] - x_state[n, :]
        y_tangents = y_state[ids, :] - y_state[n, :]
        J = scipy.linalg.lstsq(x_tangents, y_tangents)[0]
        x_pushforward[n - N_train, :] = dx_state[n, :] @ J

    # Y -> X pushforward
    y_pushforward = np.zeros_like(dx_state[N_train:])
    K = 3 * Q_y
    kdtree = scipy.spatial.KDTree(y_state[:lib_length])
    for n in range(N_train, y_state.shape[0]):
        _, ids = kdtree.query(y_state[n, :], K)
        x_tangents = x_state[ids, :] - x_state[n, :]
        y_tangents = y_state[ids, :] - y_state[n, :]
        J = scipy.linalg.lstsq(y_tangents, x_tangents)[0]
        y_pushforward[n - N_train, :] = dy_state[n, :] @ J

    if use_mutual_info:
        from bmi.estimators import KSGEnsembleFirstEstimator   # lazy: optional dep
        score_x2y = KSGEnsembleFirstEstimator(neighborhoods=(10,)).estimate(
            dx_state[N_train:], y_pushforward)
        score_y2x = KSGEnsembleFirstEstimator(neighborhoods=(10,)).estimate(
            dy_state[N_train:], x_pushforward)
    else:
        dotprods = np.sum(dx_state[N_train:] * y_pushforward, axis=1)
        mags1 = np.sum(dx_state[N_train:] ** 2, axis=1)
        mags2 = np.sum(y_pushforward ** 2, axis=1)
        score_x2y = dotprods / np.sqrt(mags1 * mags2 + 1e-16)

        dotprods = np.sum(dy_state[N_train:] * x_pushforward, axis=1)
        mags1 = np.sum(dy_state[N_train:] ** 2, axis=1)
        mags2 = np.sum(x_pushforward ** 2, axis=1)
        score_y2x = dotprods / np.sqrt(mags1 * mags2 + 1e-16)

    return score_x2y, score_y2x
