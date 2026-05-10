from __future__ import annotations

"""
Reliability diagram for binary probabilistic forecasts.

For a binary event E (e.g. "terminal log-return > 0", "price touches +k bps"):
  1. For each forecast, compute the ensemble frequency p̂ = P̂(E).
  2. Bin the forecasts by p̂ into B equal-width bins over [0, 1].
  3. For each bin, compute: mean forecast probability, observed relative frequency.
  4. A well-calibrated model lies on the diagonal (forecast prob ≈ obs freq).

Bootstrap confidence bands are computed by resampling forecast times.
"""

import numpy as np
from typing import Callable


def reliability_diagram(
    ensemble: np.ndarray,
    event_fn: Callable[[np.ndarray], np.ndarray],
    ys: np.ndarray,
    *,
    n_bins: int = 10,
    n_bootstrap: int = 200,
    ci_level: float = 0.9,
) -> dict:
    """
    Compute reliability diagram data for a binary event.

    Parameters
    ----------
    ensemble  : (T, N, H) — T forecasts, N members, H lead times
    event_fn  : maps ensemble path (N, H) → bool (N,) — defines the event.
                Example: lambda ens: ens.sum(axis=1) > 0  (terminal > 0)
    ys        : (T, H) — verifying observations
    n_bins    : number of probability bins
    n_bootstrap: bootstrap resamples for confidence bands
    ci_level  : width of the confidence interval (e.g. 0.9 → 5th–95th %)

    Returns
    -------
    dict with keys:
      bin_centers   : (n_bins,) — midpoints of probability bins
      mean_forecast : (n_bins,) — mean forecast probability per bin
      obs_frequency : (n_bins,) — observed event frequency per bin
      ci_lower      : (n_bins,) — lower CI on obs_frequency
      ci_upper      : (n_bins,) — upper CI on obs_frequency
      counts        : (n_bins,) — number of forecasts in each bin
    """
    ensemble = np.asarray(ensemble, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    T = ensemble.shape[0]

    # Ensemble probability for each forecast time
    p_hat = np.array([event_fn(ensemble[i]).mean() for i in range(T)])   # (T,)
    # Observed event for each forecast time
    obs = np.array([event_fn(ys[i:i+1]).item() for i in range(T)])       # (T,)
    # (ys[i] treated as a single "observation path")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_ids = np.digitize(p_hat, bin_edges[1:-1])  # 0-indexed bin

    mean_forecast = np.full(n_bins, np.nan)
    obs_frequency = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = bin_ids == b
        counts[b] = mask.sum()
        if counts[b] > 0:
            mean_forecast[b] = p_hat[mask].mean()
            obs_frequency[b] = obs[mask].mean()

    # Bootstrap CI on obs_frequency
    rng = np.random.default_rng(0)
    boot_obs_freq = np.full((n_bootstrap, n_bins), np.nan)
    for s in range(n_bootstrap):
        idx = rng.integers(0, T, size=T)
        p_b = p_hat[idx]
        o_b = obs[idx]
        bid_b = np.digitize(p_b, bin_edges[1:-1])
        for b in range(n_bins):
            mask = bid_b == b
            if mask.sum() > 0:
                boot_obs_freq[s, b] = o_b[mask].mean()

    alpha = (1 - ci_level) / 2
    ci_lower = np.nanpercentile(boot_obs_freq, 100 * alpha, axis=0)
    ci_upper = np.nanpercentile(boot_obs_freq, 100 * (1 - alpha), axis=0)

    return {
        "bin_centers": bin_centers,
        "mean_forecast": mean_forecast,
        "obs_frequency": obs_frequency,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "counts": counts,
    }
