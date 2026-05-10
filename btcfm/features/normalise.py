from __future__ import annotations

"""
Rolling z-score normalisation for bar features.

Statistics are computed on a trailing window that **excludes** the forecast
window — no future leakage. Normalisation state (mean, std) is persisted
alongside model checkpoints.

Convention: if std == 0 (constant feature window), the output is 0.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl

from btcfm.features.builders import FEATURE_COLS

logger = logging.getLogger(__name__)


@dataclass
class NormStats:
    """
    Normalisation statistics for a single feature column.

    Attributes
    ----------
    mean : float
    std  : float   — never zero (clamped to 1e-8 if the window is constant)
    """
    mean: float
    std: float

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / max(self.std, 1e-8)

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return z * self.std + self.mean


@dataclass
class NormState:
    """
    Collection of NormStats, one per feature column.

    Saved alongside the model checkpoint so that inference normalises
    identically to training.
    """
    stats: dict[str, NormStats] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        data = {col: {"mean": s.mean, "std": s.std} for col, s in self.stats.items()}
        import json
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> NormState:
        import json
        data = json.loads(path.read_text())
        return cls(stats={col: NormStats(**v) for col, v in data.items()})

    def transform(self, X: np.ndarray, cols: Sequence[str]) -> np.ndarray:
        """
        Apply z-score normalisation to a (N, L, F) or (L, F) array.

        Parameters
        ----------
        X  : shape (..., F)
        cols : feature column names in order (len == F)
        """
        out = X.copy().astype(np.float32)
        for i, col in enumerate(cols):
            if col in self.stats:
                out[..., i] = self.stats[col].apply(out[..., i])
        return out


def compute_rolling_norm(
    features: pl.DataFrame,
    norm_window: int = 1440,
    cols: Sequence[str] | None = None,
    min_periods: int = 30,
) -> pl.DataFrame:
    """
    Compute rolling z-score normalisation in-place on a feature DataFrame.

    For each column in ``cols``, the rolling mean and std over the trailing
    ``norm_window`` rows are used to z-score that column.  The rolling
    statistics at position ``i`` use rows ``[i - norm_window, i)`` (strictly
    past data — the current row is excluded from the statistic computation
    via a one-step shift, so there is no leakage).

    Parameters
    ----------
    features  : output of build_features (contains FEATURE_COLS + "ts")
    norm_window : trailing window in minutes for computing mean / std
    cols      : subset of FEATURE_COLS to normalise (default: all)
    min_periods : minimum non-null observations before emitting a value

    Returns
    -------
    pl.DataFrame with the same schema, values replaced by z-scores.
    """
    if cols is None:
        cols = FEATURE_COLS
    cols = [c for c in cols if c in features.columns]

    df = features.clone()
    for col in cols:
        mu = (
            pl.col(col)
            .shift(1)
            .rolling_mean(window_size=norm_window, min_samples=min_periods)
        )
        sigma = (
            pl.col(col)
            .shift(1)
            .rolling_std(window_size=norm_window, min_samples=min_periods)
        )
        df = df.with_columns(
            pl.when(sigma.abs() > 1e-8)
            .then((pl.col(col) - mu) / sigma)
            .otherwise(pl.lit(0.0))
            .alias(col)
        )

    return df


def fit_norm_stats(
    features: pl.DataFrame,
    cols: Sequence[str] | None = None,
) -> NormState:
    """
    Fit global NormStats from an entire (training) feature DataFrame.

    Used when the caller wants a single fixed normalisation (e.g. for
    synthetic data or when the rolling approach isn't needed).

    Parameters
    ----------
    features : shape (T, F+1) — includes "ts" column
    cols     : feature columns to fit

    Returns
    -------
    NormState
    """
    if cols is None:
        cols = FEATURE_COLS
    stats: dict[str, NormStats] = {}
    for col in cols:
        if col not in features.columns:
            continue
        arr = features[col].drop_nulls().to_numpy()
        stats[col] = NormStats(mean=float(arr.mean()), std=float(arr.std()))
    return NormState(stats=stats)
