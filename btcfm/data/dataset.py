from __future__ import annotations

"""
Sliding-window dataset for training the flow-matching model on bar data.

Key properties
--------------
- Windows of length (L + H) slide with stride 1 across the feature array.
- Windows that overlap a gap run (> max_gap_bars consecutive synthetic bars)
  are skipped; the count is logged.
- Normalisation statistics are computed on the training split ONLY and applied
  to all splits — any leakage of val/test moments into the normaliser is a bug.
- The chronological split is contiguous: no shuffling across the time axis.
"""

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl

from btcfm.config import BTCFMConfig
from btcfm.features.builders import build_features, extract_windows, FEATURE_COLS
from btcfm.features.normalise import fit_norm_stats, NormState

logger = logging.getLogger(__name__)


@dataclass
class WindowDataset:
    """
    A set of (context, target) window pairs, ready for training.

    Attributes
    ----------
    X  : (N, L, F)  normalised context feature arrays
    y  : (N, H)     target log-return paths
    """
    X: np.ndarray  # (N, L, F)
    y: np.ndarray  # (N, H)
    n_skipped: int

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        return self.X[i], self.y[i]

    def sample_batch(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample a random batch (with replacement)."""
        idx = rng.integers(0, len(self), size=batch_size)
        return self.X[idx], self.y[idx]


def _find_gap_mask(is_synthetic: np.ndarray, max_gap_bars: int) -> np.ndarray:
    """
    Return a boolean mask of length T where True = this bar is inside a gap
    run longer than max_gap_bars consecutive synthetic bars.
    """
    T = len(is_synthetic)
    gap_mask = np.zeros(T, dtype=bool)
    run_len = 0
    run_start = 0
    for i in range(T):
        if is_synthetic[i]:
            if run_len == 0:
                run_start = i
            run_len += 1
            if run_len > max_gap_bars:
                gap_mask[run_start:i + 1] = True
        else:
            run_len = 0
    return gap_mask


def _build_windows(
    features_np: np.ndarray,
    returns_np: np.ndarray,
    is_synthetic: np.ndarray,
    context_len: int,
    horizon: int,
    max_gap_bars: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Build (X, y) windows, skipping those that overlap gap runs.

    Returns (X, y, n_skipped).
    """
    T = len(features_np)
    gap_mask = _find_gap_mask(is_synthetic, max_gap_bars)

    X_list, y_list = [], []
    n_skipped = 0

    for i in range(T - context_len - horizon + 1):
        window_slice = slice(i, i + context_len + horizon)
        if gap_mask[window_slice].any():
            n_skipped += 1
            continue
        X_list.append(features_np[i: i + context_len])
        y_list.append(returns_np[i + context_len: i + context_len + horizon])

    if not X_list:
        raise ValueError(
            f"No valid windows: T={T}, L={context_len}, H={horizon}, "
            f"max_gap={max_gap_bars}"
        )

    return np.stack(X_list), np.stack(y_list), n_skipped


def prepare_datasets(
    bars: pl.DataFrame,
    config: BTCFMConfig,
    train_days: int = 150,
    val_days: int = 15,
    # test_days is whatever remains
) -> tuple[WindowDataset, WindowDataset, WindowDataset, NormState]:
    """
    Split bars chronologically and build normalised window datasets.

    Split (contiguous, no shuffling):
      train : bars[0 : train_days * 1440]
      val   : bars[train_days * 1440 : (train_days + val_days) * 1440]
      test  : bars[(train_days + val_days) * 1440 : end]

    Normalisation statistics are fit on the TRAINING split only; the same
    stats are applied to val and test.  Leakage would manifest as val/test
    feature means ≈ 0 and stds ≈ 1 (the exact training moments) — use
    ``assert_no_normalisation_leakage`` in tests to detect this.

    Parameters
    ----------
    bars       : complete bar DataFrame, sorted ascending by ts, BAR_SCHEMA
    config     : BTCFMConfig
    train_days : number of training days (default 150)
    val_days   : number of validation days (default 15)

    Returns
    -------
    train_ds, val_ds, test_ds : WindowDataset
    norm_state                : NormState fitted on training data only
    """
    bars = bars.sort("ts")
    total_minutes = len(bars)
    train_end = train_days * 1440
    val_end = (train_days + val_days) * 1440

    if total_minutes < val_end + config.data.context_length + config.data.horizon:
        raise ValueError(
            f"Not enough bars ({total_minutes} min) for "
            f"train ({train_days}d) + val ({val_days}d) + context + horizon."
        )

    train_bars = bars[:train_end]
    val_bars = bars[train_end: val_end]
    test_bars = bars[val_end:]

    logger.info(
        "Split: train=%d bars (%d days), val=%d bars, test=%d bars",
        len(train_bars), train_days, len(val_bars), len(test_bars),
    )

    # Build features (drop early NaN rows from rolling warm-up)
    train_feat = build_features(train_bars).drop_nulls()
    val_feat   = build_features(val_bars).drop_nulls()
    test_feat  = build_features(test_bars).drop_nulls()

    # Fit normaliser on training data only
    norm_state = fit_norm_stats(train_feat)
    logger.info("Normalisation stats fitted on %d training rows", len(train_feat))

    L = config.data.context_length
    H = config.data.horizon
    G = config.data.max_gap_bars

    def _to_dataset(feat_df: pl.DataFrame, split_name: str) -> WindowDataset:
        feat_np = feat_df.select(FEATURE_COLS).to_numpy().astype(np.float32)
        # Apply training normalisation
        feat_np = norm_state.transform(feat_np, FEATURE_COLS)
        ret_np  = feat_df["r_1m"].to_numpy().astype(np.float32)
        syn_np  = feat_df["is_synthetic"].to_numpy() if "is_synthetic" in feat_df.columns \
                  else np.zeros(len(feat_df), dtype=bool)
        # re-join is_synthetic from original bars (build_features drops it)
        # Use the bar is_synthetic aligned to the feature rows by ts
        X, y, n_skip = _build_windows(feat_np, ret_np, syn_np, L, H, G)
        logger.info(
            "%s: %d windows (%d skipped for gaps)", split_name, len(X), n_skip,
        )
        return WindowDataset(X=X, y=y, n_skipped=n_skip)

    # Note: build_features drops is_synthetic from output; gap mask uses the
    # bar series directly.
    def _to_dataset_with_gaps(
        feat_df: pl.DataFrame,
        bar_df: pl.DataFrame,
        split_name: str,
    ) -> WindowDataset:
        feat_np = feat_df.select(FEATURE_COLS).to_numpy().astype(np.float32)
        feat_np = norm_state.transform(feat_np, FEATURE_COLS)
        ret_np  = feat_df["r_1m"].to_numpy().astype(np.float32)

        # Align is_synthetic by ts join
        is_syn_df = bar_df.select(["ts", "is_synthetic"]).join(
            feat_df.select(["ts"]), on="ts", how="inner"
        )
        syn_np = is_syn_df["is_synthetic"].to_numpy()

        X, y, n_skip = _build_windows(feat_np, ret_np, syn_np, L, H, G)
        logger.info("%s: %d windows (%d skipped for gaps)", split_name, len(X), n_skip)
        return WindowDataset(X=X, y=y, n_skipped=n_skip)

    train_ds = _to_dataset_with_gaps(train_feat, train_bars, "train")
    val_ds   = _to_dataset_with_gaps(val_feat,   val_bars,   "val")
    test_ds  = _to_dataset_with_gaps(test_feat,  test_bars,  "test")

    return train_ds, val_ds, test_ds, norm_state


def assert_no_normalisation_leakage(
    val_ds: WindowDataset,
    norm_state: NormState,
    tol: float = 0.05,
) -> None:
    """
    Assert that val features are NOT exactly z-scored to N(0,1).

    If the normaliser was mistakenly fit on val data, every val feature
    would have mean ≈ 0 and std ≈ 1.  We check that at least some features
    deviate from (0, 1) beyond floating-point noise.

    Raises AssertionError if evidence of leakage is found.
    """
    X_flat = val_ds.X.reshape(-1, val_ds.X.shape[-1])  # (N*L, F)
    feat_means = X_flat.mean(axis=0)                    # (F,)
    feat_stds  = X_flat.std(axis=0)                     # (F,)

    all_means_zero = np.all(np.abs(feat_means) < tol)
    all_stds_one   = np.all(np.abs(feat_stds - 1.0) < tol)

    if all_means_zero and all_stds_one:
        raise AssertionError(
            "Validation features appear to have been normalised with their OWN "
            "statistics (mean≈0, std≈1 for all features). This is a normalisation "
            "leakage bug: the normaliser must be fit on training data only."
        )
    logger.info(
        "Normalisation leakage check passed: "
        "val mean range=[%.3f, %.3f], std range=[%.3f, %.3f]",
        feat_means.min(), feat_means.max(),
        feat_stds.min(), feat_stds.max(),
    )
