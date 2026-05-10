from __future__ import annotations

"""
Spread–skill verification.

For a well-calibrated ensemble:
  ensemble_std ≈ RMSE(ensemble_mean, observation)

equivalently, the spread–skill diagram lies on y = x.

We also compute pinball (quantile) loss at the standard quantile levels
used in NWP ensemble verification (5 / 25 / 50 / 75 / 95th percentiles).
"""

import numpy as np


QUANTILE_LEVELS = np.array([0.05, 0.25, 0.50, 0.75, 0.95])


# ---------------------------------------------------------------------------
# Spread–skill
# ---------------------------------------------------------------------------

def spread_skill(
    ensembles: np.ndarray,
    ys: np.ndarray,
    *,
    n_bins: int = 10,
) -> dict:
    """
    Compute spread–skill relation at a single lead time.

    Parameters
    ----------
    ensembles : (T, N) — T forecast times, N ensemble members
    ys        : (T,)   — verifying observations
    n_bins    : number of spread bins

    Returns
    -------
    dict with keys:
      bin_spread : (n_bins,) — mean ensemble spread within each bin
      bin_rmse   : (n_bins,) — mean |ensemble_mean - y| within each bin
      counts     : (n_bins,) — number of forecasts per bin
      mean_spread: float — overall mean spread
      mean_rmse  : float — overall RMSE of ensemble mean
    """
    ensembles = np.asarray(ensembles, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    spread = ensembles.std(axis=1)                # (T,)
    error = np.abs(ensembles.mean(axis=1) - ys)  # (T,)

    # Bin by spread quantile
    bin_edges = np.quantile(spread, np.linspace(0, 1, n_bins + 1))
    bin_edges[0] -= 1e-10  # ensure lowest value is included
    bin_ids = np.digitize(spread, bin_edges[1:])  # 0-indexed

    bin_spread = np.full(n_bins, np.nan)
    bin_rmse = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = bin_ids == b
        counts[b] = mask.sum()
        if counts[b] > 0:
            bin_spread[b] = spread[mask].mean()
            bin_rmse[b] = error[mask].mean()

    return {
        "bin_spread": bin_spread,
        "bin_rmse": bin_rmse,
        "counts": counts,
        "mean_spread": float(spread.mean()),
        "mean_rmse": float(error.mean()),
    }


def spread_skill_per_lead(
    ensembles: np.ndarray,
    ys: np.ndarray,
    *,
    n_bins: int = 10,
) -> list[dict]:
    """
    spread_skill computed independently for each lead time.

    Parameters
    ----------
    ensembles : (T, N, H)
    ys        : (T, H)

    Returns
    -------
    list of H dicts (one per lead time)
    """
    H = ensembles.shape[2]
    return [
        spread_skill(ensembles[:, :, h], ys[:, h], n_bins=n_bins)
        for h in range(H)
    ]


# ---------------------------------------------------------------------------
# Quantile (pinball) loss
# ---------------------------------------------------------------------------

def pinball_loss(
    ensemble: np.ndarray,
    y: np.ndarray,
    quantiles: np.ndarray = QUANTILE_LEVELS,
) -> np.ndarray:
    """
    Pinball loss at specified quantile levels for a single lead time.

    Parameters
    ----------
    ensemble  : (T, N) — T forecast times, N ensemble members
    y         : (T,)   — observations
    quantiles : (Q,)   — quantile levels in (0, 1)

    Returns
    -------
    np.ndarray, shape (Q,) — mean pinball loss per quantile level
    """
    ensemble = np.asarray(ensemble, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    T, N = ensemble.shape

    q_hat = np.quantile(ensemble, quantiles, axis=1).T  # (T, Q)
    y_rep = y[:, None]  # (T, 1)

    errors = y_rep - q_hat  # (T, Q)
    loss = np.where(
        errors >= 0,
        quantiles[None, :] * errors,
        (quantiles[None, :] - 1.0) * errors,
    )  # (T, Q)
    return loss.mean(axis=0)  # (Q,)


def pinball_loss_per_lead(
    ensembles: np.ndarray,
    ys: np.ndarray,
    quantiles: np.ndarray = QUANTILE_LEVELS,
) -> np.ndarray:
    """
    Pinball loss at each lead time.

    Parameters
    ----------
    ensembles : (T, N, H)
    ys        : (T, H)

    Returns
    -------
    np.ndarray, shape (H, Q)
    """
    H = ensembles.shape[2]
    return np.stack(
        [pinball_loss(ensembles[:, :, h], ys[:, h], quantiles) for h in range(H)],
        axis=0,
    )  # (H, Q)
