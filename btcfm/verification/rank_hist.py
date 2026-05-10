from __future__ import annotations

"""
Rank histogram (Talagrand diagram) for ensemble calibration.

For each forecast time t:
  1. Sort the N ensemble members: x_{(1)} ≤ ... ≤ x_{(N)}.
  2. The rank of the observation y is the number of ensemble members < y,
     giving a value in {0, 1, ..., N}  (N+1 possible values).
  3. Under perfect calibration, rank is Uniform({0, ..., N}).

Shapes of the histogram:
  - Flat    : well calibrated
  - U-shape : underdispersion (ensemble too narrow)
  - Dome    : overdispersion (ensemble too wide)
  - Skewed  : ensemble bias
"""

import numpy as np


def rank_histogram(
    ensemble: np.ndarray,
    y: np.ndarray,
    *,
    normalise: bool = True,
) -> np.ndarray:
    """
    Compute the rank histogram over T forecast times.

    Parameters
    ----------
    ensemble  : (T, N) — ensemble members (single lead time)
    y         : (T,)   — verifying observations
    normalise : if True, return counts / T so values sum to 1

    Returns
    -------
    np.ndarray, shape (N+1,) — histogram counts (or frequencies)
    """
    ensemble = np.asarray(ensemble, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    T, N = ensemble.shape

    # rank[t] = number of ensemble members strictly less than y[t]
    ranks = (ensemble < y[:, None]).sum(axis=1)  # (T,), values in 0..N

    counts = np.bincount(ranks, minlength=N + 1).astype(np.float64)
    if normalise:
        counts /= T
    return counts


def rank_histogram_per_lead(
    ensembles: np.ndarray,
    ys: np.ndarray,
    *,
    normalise: bool = True,
) -> np.ndarray:
    """
    Compute rank histograms for each lead time independently.

    Parameters
    ----------
    ensembles : (T, N, H)
    ys        : (T, H)

    Returns
    -------
    np.ndarray, shape (H, N+1)
    """
    T, N, H = ensembles.shape
    hists = np.stack(
        [rank_histogram(ensembles[:, :, h], ys[:, h], normalise=normalise)
         for h in range(H)],
        axis=0,
    )  # (H, N+1)
    return hists


def flatness_score(hist: np.ndarray) -> float:
    """
    Kolmogorov-Smirnov statistic comparing the rank histogram to uniform.

    Returns a value in [0, 1]; smaller is better (more uniform).
    """
    N_bins = len(hist)
    observed_cdf = np.cumsum(hist) / hist.sum()
    uniform_cdf = np.linspace(1 / N_bins, 1.0, N_bins)
    return float(np.max(np.abs(observed_cdf - uniform_cdf)))
