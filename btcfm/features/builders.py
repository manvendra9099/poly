from __future__ import annotations

"""
Feature construction on the 1-minute bar grid.

All output features are stationary by construction (returns, normalised
volumes, periodic encodings). Any drift with price level is a bug.

Feature columns produced (one value per minute):

  r_1m          : 1-minute log-return
  r_5m          : 5-minute log-return, forward-filled within window
  r_15m         : 15-minute log-return, forward-filled within window
  rv30_cc       : rolling 30-min realised vol, close-to-close
  rv60_cc       : rolling 60-min realised vol, close-to-close
  rv240_cc      : rolling 240-min realised vol, close-to-close
  rv30_pk       : rolling 30-min realised vol, Parkinson
  rv60_pk       : rolling 60-min realised vol, Parkinson
  rv240_pk      : rolling 240-min realised vol, Parkinson
  vol_imbal     : signed-volume imbalance (buy-sell)/(buy+sell)
  cnt_imbal     : trade-count imbalance
  sin_tod       : sin(2π * minute_of_day / 1440)
  cos_tod       : cos(2π * minute_of_day / 1440)
  sin_dow       : sin(2π * day_of_week / 7)
  cos_dow       : cos(2π * day_of_week / 7)

The caller is responsible for z-scoring via btcfm.features.normalise.
"""

import logging
import math
from typing import Sequence

import numpy as np
import polars as pl

from btcfm.data.schema import validate_bar_schema

logger = logging.getLogger(__name__)

_4LN2 = 4 * math.log(2)

FEATURE_COLS = [
    "r_1m",
    "r_5m",
    "r_15m",
    "rv30_cc",
    "rv60_cc",
    "rv240_cc",
    "rv30_pk",
    "rv60_pk",
    "rv240_pk",
    "vol_imbal",
    "cnt_imbal",
    "sin_tod",
    "cos_tod",
    "sin_dow",
    "cos_dow",
]


def build_features(bars: pl.DataFrame) -> pl.DataFrame:
    """
    Compute features on a bar DataFrame sorted ascending by ``ts``.

    Parameters
    ----------
    bars:
        Conforms to BAR_SCHEMA, sorted ascending by ``ts``.
        Must contain at least ``min_context_length`` rows for rolling
        estimates to be valid.

    Returns
    -------
    pl.DataFrame
        Same length as ``bars``, columns = FEATURE_COLS + ["ts"].
        Early rows where rolling windows cannot be filled are NaN; the
        caller must handle or strip these rows.

    Raises
    ------
    ValueError
        If ``bars`` is missing required columns.
    """
    validate_bar_schema(bars)

    df = bars.sort("ts")

    # ------------------------------------------------------------------
    # 1-minute log-return
    # ------------------------------------------------------------------
    df = df.with_columns(
        (pl.col("close").log() - pl.col("close").shift(1).log()).alias("r_1m")
    )

    # ------------------------------------------------------------------
    # Multi-resolution returns: 5-min and 15-min
    # r_5m[t]  = log(close[t] / close[t-5])  (last obs within 5-min window)
    # Forward-filled so every minute has a value.
    # ------------------------------------------------------------------
    df = df.with_columns(
        (pl.col("close").log() - pl.col("close").shift(5).log())
        .forward_fill()
        .alias("r_5m"),
        (pl.col("close").log() - pl.col("close").shift(15).log())
        .forward_fill()
        .alias("r_15m"),
    )

    # ------------------------------------------------------------------
    # Realised volatility — close-to-close: sqrt(mean(r^2)) over window
    # ------------------------------------------------------------------
    for w in (30, 60, 240):
        df = df.with_columns(
            (pl.col("r_1m").pow(2).rolling_mean(window_size=w).sqrt())
            .alias(f"rv{w}_cc")
        )

    # ------------------------------------------------------------------
    # Parkinson volatility: (ln(H/L))^2 / (4 ln 2)  per bar;
    # RV_park = sqrt(mean over window)
    # ------------------------------------------------------------------
    df = df.with_columns(
        ((pl.col("high") / pl.col("low")).log().pow(2) / _4LN2).alias("_pk_sq")
    )
    for w in (30, 60, 240):
        df = df.with_columns(
            pl.col("_pk_sq").rolling_mean(window_size=w).sqrt().alias(f"rv{w}_pk")
        )
    df = df.drop("_pk_sq")

    # ------------------------------------------------------------------
    # Signed-volume imbalance
    # Edge: if total_vol == 0 (synthetic bar), set to 0.
    # ------------------------------------------------------------------
    total_vol = pl.col("buy_volume") + pl.col("sell_volume")
    df = df.with_columns(
        pl.when(total_vol == 0)
        .then(pl.lit(0.0))
        .otherwise((pl.col("buy_volume") - pl.col("sell_volume")) / total_vol)
        .alias("vol_imbal")
    )

    total_cnt = pl.col("buy_count") + pl.col("sell_count")
    df = df.with_columns(
        pl.when(total_cnt == 0)
        .then(pl.lit(0.0))
        .otherwise(
            (pl.col("buy_count") - pl.col("sell_count")).cast(pl.Float64) / total_cnt.cast(pl.Float64)
        )
        .alias("cnt_imbal")
    )

    # ------------------------------------------------------------------
    # Time-of-day and day-of-week encodings (stationary, bounded, periodic)
    # ------------------------------------------------------------------
    minute_of_day = (
        pl.col("ts").dt.hour() * 60 + pl.col("ts").dt.minute()
    ).cast(pl.Float64)
    day_of_week = pl.col("ts").dt.weekday().cast(pl.Float64)

    df = df.with_columns(
        (2 * math.pi * minute_of_day / 1440).sin().alias("sin_tod"),
        (2 * math.pi * minute_of_day / 1440).cos().alias("cos_tod"),
        (2 * math.pi * day_of_week / 7).sin().alias("sin_dow"),
        (2 * math.pi * day_of_week / 7).cos().alias("cos_dow"),
    )

    result = df.select(["ts"] + FEATURE_COLS)
    logger.debug("build_features: %d bars → %d feature rows", len(bars), len(result))
    return result


def extract_windows(
    features: pl.DataFrame,
    context_len: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slice a feature DataFrame into (X, y) arrays for training.

    Parameters
    ----------
    features:
        Output of build_features, must have column ``r_1m`` plus FEATURE_COLS,
        sorted ascending. NaN rows (early rolling warm-up) must have been
        stripped before calling this.
    context_len:
        Number of minutes in the context window (L).
    horizon:
        Number of minutes in the forecast horizon (H).

    Returns
    -------
    X : np.ndarray, shape (N, context_len, F)
        Context feature matrices.
    y : np.ndarray, shape (N, horizon)
        Target log-return paths.
    """
    feat_np = features.select(FEATURE_COLS).to_numpy()  # (T, F)
    ret_np = features["r_1m"].to_numpy()                # (T,)

    T = len(feat_np)
    n_samples = T - context_len - horizon + 1
    if n_samples <= 0:
        raise ValueError(
            f"Not enough rows: need {context_len + horizon}, got {T}"
        )

    X = np.stack(
        [feat_np[i : i + context_len] for i in range(n_samples)], axis=0
    )  # (N, L, F)
    y = np.stack(
        [ret_np[i + context_len : i + context_len + horizon] for i in range(n_samples)],
        axis=0,
    )  # (N, H)
    return X, y
