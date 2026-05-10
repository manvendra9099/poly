from __future__ import annotations

"""
Continuous Ranked Probability Score (CRPS) — sample-based implementation.

Formula (Gneiting & Raftery 2007):
  CRPS(F, y) = E_F|X - y| - (1/2) * E_F|X - X'|

With a finite ensemble {x_i}_{i=1}^N and observation y:
  Term 1 = (1/N) Σ_i |x_i - y|
  Term 2 = (1/N²) Σ_i Σ_j |x_i - x_j| / 2
         = (1/N²) Σ_i x_{(i)} * (2i - N + 1)    [sorted, 0-indexed]

The sorted formula is O(N log N) vs O(N²) for the naive double sum.

All functions are pure, accept numpy arrays, and are tested against the
analytic CRPS for a Gaussian predictive distribution:
  CRPS(N(μ,σ), y) = σ * [z*(2Φ(z)-1) + 2φ(z) - 1/√π]
  where z = (y - μ)/σ.
"""

import numpy as np
from scipy import stats as scipy_stats


def crps_sample(ensemble: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Sample-based CRPS using the sorted O(N log N) formula.

    Parameters
    ----------
    ensemble : shape (N, ...)  — ensemble members along axis 0
    y        : shape (...)     — verifying observations

    Returns
    -------
    np.ndarray, shape (...)    — CRPS for each grid point / lead time
    """
    N = ensemble.shape[0]
    ensemble = np.asarray(ensemble, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Term 1: E|X - y|
    term1 = np.mean(np.abs(ensemble - y[None, ...]), axis=0)

    # Term 2: (1/2) E|X - X'|  via sorted formula
    sorted_ens = np.sort(ensemble, axis=0)              # (N, ...)
    idx = np.arange(N, dtype=np.float64)
    # weights shape: (N, 1, 1, ...) to broadcast over trailing dims
    shape = (N,) + (1,) * (ensemble.ndim - 1)
    weights = (2 * idx - N + 1).reshape(shape)
    term2 = np.sum(weights * sorted_ens, axis=0) / (N * N)

    return term1 - term2


def crps_gaussian_analytic(mu: float, sigma: float, y: float) -> float:
    """
    Analytic CRPS for a Gaussian predictive distribution N(mu, sigma).

    Used as reference in unit tests.
    """
    z = (y - mu) / sigma
    return sigma * (
        z * (2 * scipy_stats.norm.cdf(z) - 1)
        + 2 * scipy_stats.norm.pdf(z)
        - 1.0 / np.sqrt(np.pi)
    )


def crps_per_lead(
    ensemble: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    CRPS at each lead time independently.

    Parameters
    ----------
    ensemble : (N, H) — N ensemble members, H lead times
    y        : (H,)   — observed log-return path

    Returns
    -------
    np.ndarray, shape (H,)
    """
    return crps_sample(ensemble, y)


def crps_terminal(ensemble: np.ndarray, y: np.ndarray) -> float:
    """
    CRPS for the terminal log-return r̄ = Σ_i r_i.

    Parameters
    ----------
    ensemble : (N, H)
    y        : (H,)

    Returns
    -------
    float — scalar CRPS for the terminal sum
    """
    ens_sum = ensemble.sum(axis=1)      # (N,)
    y_sum = y.sum()
    return float(crps_sample(ens_sum[:, None], np.array([y_sum]))[0])


def mean_crps(
    ensembles: np.ndarray,
    ys: np.ndarray,
    axis: int = 0,
) -> np.ndarray:
    """
    Mean CRPS over multiple forecast times.

    Parameters
    ----------
    ensembles : (T, N, H) — T forecast issue times
    ys        : (T, H)

    Returns
    -------
    np.ndarray, shape (H,) — mean CRPS per lead time
    """
    scores = np.stack(
        [crps_per_lead(ensembles[i], ys[i]) for i in range(len(ys))],
        axis=0,
    )  # (T, H)
    return scores.mean(axis=0)  # (H,)
