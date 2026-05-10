from __future__ import annotations

"""
Tests for the sliding-window dataset and normalisation leakage detection.
"""

import pytest
import numpy as np
import polars as pl
from datetime import date, datetime, timezone, timedelta

from btcfm.config import load_config
from btcfm.data.coinbase_rest import _candles_to_bar_df
from btcfm.data.dataset import (
    WindowDataset,
    prepare_datasets,
    assert_no_normalisation_leakage,
    _find_gap_mask,
)


CONFIG_PATH = "configs/small.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(n_days: int, base_price: float = 50_000.0, seed: int = 0) -> pl.DataFrame:
    """Generate n_days × 1440 synthetic 1-minute bars."""
    rng = np.random.default_rng(seed)
    frames = []
    start_day = date(2024, 1, 1)
    for d in range(n_days):
        day = start_day + timedelta(days=d)
        day_start = int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())
        candles = []
        price = base_price + rng.normal(0, 100)
        for i in range(1440):
            price = price * np.exp(rng.normal(0, 0.001))
            ts = day_start + i * 60
            candles.append([ts, price * 0.999, price * 1.001, price, price, abs(rng.normal(1, 0.5))])
        frames.append(_candles_to_bar_df(candles, day))
    return pl.concat(frames).sort("ts")


# ---------------------------------------------------------------------------
# Gap mask
# ---------------------------------------------------------------------------

class TestGapMask:
    def test_no_gaps(self):
        is_syn = np.array([False] * 100)
        mask = _find_gap_mask(is_syn, max_gap_bars=5)
        assert not mask.any()

    def test_short_gap_not_masked(self):
        """A run of ≤ max_gap_bars synthetic bars should NOT be masked."""
        is_syn = np.array([False] * 10 + [True] * 3 + [False] * 10)
        mask = _find_gap_mask(is_syn, max_gap_bars=5)
        assert not mask.any()

    def test_long_gap_masked(self):
        """A run of > max_gap_bars synthetic bars should be fully masked."""
        is_syn = np.array([False] * 5 + [True] * 10 + [False] * 5)
        mask = _find_gap_mask(is_syn, max_gap_bars=5)
        assert mask[5:15].all()
        assert not mask[:5].any()
        assert not mask[15:].any()


# ---------------------------------------------------------------------------
# prepare_datasets
# ---------------------------------------------------------------------------

class TestPrepareDatasets:
    @pytest.fixture(scope="class")
    def datasets(self):
        """Build datasets from 35 days of synthetic bars (train=20, val=10, test=5)."""
        config = load_config(CONFIG_PATH)
        bars = _make_bars(n_days=35)
        train_ds, val_ds, test_ds, norm_state = prepare_datasets(
            bars, config, train_days=20, val_days=10,
        )
        return train_ds, val_ds, test_ds, norm_state, config

    def test_datasets_non_empty(self, datasets):
        train_ds, val_ds, test_ds, _, _ = datasets
        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(test_ds) > 0

    def test_window_shapes(self, datasets):
        train_ds, _, _, _, config = datasets
        L = config.data.context_length
        H = config.data.horizon
        F = train_ds.X.shape[2]
        assert train_ds.X.shape[1:] == (L, F)
        assert train_ds.y.shape[1] == H

    def test_train_larger_than_val(self, datasets):
        train_ds, val_ds, _, _, _ = datasets
        assert len(train_ds) > len(val_ds)

    def test_no_nans_in_features(self, datasets):
        train_ds, val_ds, test_ds, _, _ = datasets
        for ds, name in [(train_ds, "train"), (val_ds, "val"), (test_ds, "test")]:
            assert not np.isnan(ds.X).any(), f"NaN in {name} features"
            assert not np.isnan(ds.y).any(), f"NaN in {name} targets"

    def test_sample_batch_shape(self, datasets):
        train_ds, _, _, _, config = datasets
        rng = np.random.default_rng(42)
        X, y = train_ds.sample_batch(8, rng)
        assert X.shape[0] == 8
        assert y.shape[0] == 8


# ---------------------------------------------------------------------------
# Normalisation leakage detection
# ---------------------------------------------------------------------------

class TestNormalisationLeakage:
    def test_no_leakage_passes(self):
        """
        Datasets from different distributions: train N(0,1), val N(2,1).
        After normalising val with train stats, val features should NOT
        have mean≈0 and std≈1, so the leakage check must PASS (not raise).
        """
        rng = np.random.default_rng(0)
        F, L, H = 5, 30, 10

        # Simulate: training features ~ N(0, 1)
        X_train = rng.standard_normal((1000, L, F)).astype(np.float32)
        y_train = rng.standard_normal((1000, H)).astype(np.float32)
        train_ds = WindowDataset(X=X_train, y=y_train, n_skipped=0)

        # Validation features ~ N(2, 0.5) — different distribution from training
        # Normalise val with train stats (mean≈0, std≈1): val mean after norm ≈ 2
        X_val = (rng.standard_normal((300, L, F)) * 0.5 + 2.0).astype(np.float32)
        y_val = rng.standard_normal((300, H)).astype(np.float32)
        val_ds = WindowDataset(X=X_val, y=y_val, n_skipped=0)

        # Should not raise (val mean ≠ 0, val std ≠ 1)
        assert_no_normalisation_leakage(val_ds, norm_state=None, tol=0.05)

    def test_leakage_detected(self):
        """
        If val features are perfectly z-scored (mean=0, std=1 everywhere),
        that's a sign of leakage: the leakage checker must raise.
        """
        rng = np.random.default_rng(1)
        F, L, H = 5, 30, 10

        # Val features already normalised to N(0,1) — suspicious
        X_val = rng.standard_normal((300, L, F)).astype(np.float32)
        # Force exact mean=0, std=1 per feature
        X_flat = X_val.reshape(-1, F)
        X_flat = (X_flat - X_flat.mean(axis=0)) / (X_flat.std(axis=0) + 1e-8)
        X_val = X_flat.reshape(300, L, F).astype(np.float32)

        y_val = rng.standard_normal((300, H)).astype(np.float32)
        val_ds = WindowDataset(X=X_val, y=y_val, n_skipped=0)

        with pytest.raises(AssertionError, match="leakage"):
            assert_no_normalisation_leakage(val_ds, norm_state=None, tol=0.05)
