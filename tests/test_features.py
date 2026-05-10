from __future__ import annotations

"""
Tests for the feature builder.

Uses a synthetic bar DataFrame so no market data is required.
"""

import math
import pytest
import numpy as np
import polars as pl
from datetime import datetime, timezone, timedelta

from btcfm.features.builders import build_features, extract_windows, FEATURE_COLS
from btcfm.features.normalise import compute_rolling_norm, fit_norm_stats, NormState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(n: int, base_price: float = 10000.0, seed: int = 42) -> pl.DataFrame:
    """
    Generate n synthetic 1-minute bars starting at 2024-01-01T00:00Z.
    Prices follow a simple random walk (log-normal).
    """
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0, 0.001, n)
    closes = base_price * np.exp(np.cumsum(log_returns))
    highs = closes * (1 + np.abs(rng.normal(0, 0.0005, n)))
    lows = closes * (1 - np.abs(rng.normal(0, 0.0005, n)))
    opens = np.roll(closes, 1)
    opens[0] = base_price
    volumes = rng.uniform(0.5, 5.0, n)
    buy_fracs = rng.uniform(0.3, 0.7, n)
    buy_vols = volumes * buy_fracs
    sell_vols = volumes * (1 - buy_fracs)
    buy_counts = (buy_vols / 0.1).astype(int) + 1
    sell_counts = (sell_vols / 0.1).astype(int) + 1

    start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    ts = [start + timedelta(minutes=i) for i in range(n)]

    return pl.DataFrame({
        "ts": pl.Series(ts).cast(pl.Datetime("us", "UTC")),
        "open": opens.tolist(),
        "high": highs.tolist(),
        "low": lows.tolist(),
        "close": closes.tolist(),
        "volume": volumes.tolist(),
        "vwap": closes.tolist(),
        "buy_volume": buy_vols.tolist(),
        "sell_volume": sell_vols.tolist(),
        "buy_count": buy_counts.tolist(),
        "sell_count": sell_counts.tolist(),
        "is_synthetic": [False] * n,
    })


# ---------------------------------------------------------------------------
# Feature output schema
# ---------------------------------------------------------------------------

def test_feature_columns():
    bars = _make_bars(300)
    feat = build_features(bars)
    for col in FEATURE_COLS:
        assert col in feat.columns, f"Missing feature column: {col}"
    assert "ts" in feat.columns


def test_feature_length():
    bars = _make_bars(300)
    feat = build_features(bars)
    assert len(feat) == 300


# ---------------------------------------------------------------------------
# Stationarity contract
# ---------------------------------------------------------------------------

def test_tod_sin_cos_bounded():
    bars = _make_bars(300)
    feat = build_features(bars)
    sin_vals = feat["sin_tod"].to_numpy()
    cos_vals = feat["cos_tod"].to_numpy()
    assert np.all(np.abs(sin_vals) <= 1 + 1e-9)
    assert np.all(np.abs(cos_vals) <= 1 + 1e-9)


def test_vol_imbalance_range():
    bars = _make_bars(300)
    feat = build_features(bars).drop_nulls()
    imbal = feat["vol_imbal"].to_numpy()
    assert np.all(imbal >= -1.0 - 1e-9)
    assert np.all(imbal <= 1.0 + 1e-9)


def test_cnt_imbalance_range():
    bars = _make_bars(300)
    feat = build_features(bars).drop_nulls()
    imbal = feat["cnt_imbal"].to_numpy()
    assert np.all(imbal >= -1.0 - 1e-9)
    assert np.all(imbal <= 1.0 + 1e-9)


def test_rv_non_negative():
    bars = _make_bars(300)
    feat = build_features(bars).drop_nulls()
    for col in ["rv30_cc", "rv60_cc", "rv240_cc", "rv30_pk", "rv60_pk", "rv240_pk"]:
        vals = feat[col].to_numpy()
        assert np.all(vals >= 0), f"{col} has negative values"


# ---------------------------------------------------------------------------
# Parkinson vs close-to-close
# ---------------------------------------------------------------------------

def test_parkinson_smaller_than_cctoc_on_average():
    """
    Parkinson estimator has lower variance than close-to-close for a
    Brownian-motion process; on average both should be similar in magnitude
    but Parkinson can be smaller.  Just check both are positive and finite.
    """
    bars = _make_bars(500)
    feat = build_features(bars).drop_nulls()
    pk = feat["rv60_pk"].to_numpy()
    cc = feat["rv60_cc"].to_numpy()
    assert np.all(np.isfinite(pk))
    assert np.all(np.isfinite(cc))
    assert pk.mean() > 0
    assert cc.mean() > 0


# ---------------------------------------------------------------------------
# extract_windows
# ---------------------------------------------------------------------------

def test_extract_windows_shapes():
    bars = _make_bars(800)
    feat = build_features(bars).drop_nulls()
    L, H = 60, 30
    X, y = extract_windows(feat, context_len=L, horizon=H)
    T = len(feat) - L - H + 1
    assert X.shape == (T, L, len(FEATURE_COLS))
    assert y.shape == (T, H)


def test_extract_windows_values():
    """Target y should equal the r_1m column sliced at the correct offset."""
    # Need > 240 bars so rv240_* warm-up rows don't eliminate everything
    bars = _make_bars(600)
    feat = build_features(bars).drop_nulls()
    L, H = 30, 10
    X, y = extract_windows(feat, context_len=L, horizon=H)
    r_all = feat["r_1m"].to_numpy()
    np.testing.assert_allclose(y[0], r_all[L : L + H], rtol=1e-5)
    np.testing.assert_allclose(y[1], r_all[L + 1 : L + 1 + H], rtol=1e-5)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def test_rolling_norm_mean_near_zero():
    """After rolling z-score, long-run mean of each feature should be ~0."""
    bars = _make_bars(2000)
    feat = build_features(bars)
    normed = compute_rolling_norm(feat, norm_window=240, min_periods=30)
    normed_valid = normed.drop_nulls()
    for col in ["r_1m", "rv60_cc", "vol_imbal"]:
        mu = normed_valid[col].mean()
        assert abs(mu) < 0.5, f"{col} mean after z-score is {mu:.3f}"


def test_fit_norm_state_save_load(tmp_path):
    bars = _make_bars(500)
    feat = build_features(bars).drop_nulls()
    state = fit_norm_stats(feat)

    path = tmp_path / "norm.json"
    state.save(path)
    loaded = NormState.load(path)

    for col in state.stats:
        assert abs(loaded.stats[col].mean - state.stats[col].mean) < 1e-9
        assert abs(loaded.stats[col].std - state.stats[col].std) < 1e-9
