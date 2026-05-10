from __future__ import annotations

"""
Historical 1-minute bar loader.

Reads from a local Parquet cache of pre-downloaded Coinbase or Binance
1-minute klines. The output schema is identical to BAR_SCHEMA so the live
and historical paths are interchangeable.

Expected cache layout (one file per trading day)::

    {cache_dir}/BTC-USD/YYYY-MM-DD.parquet

Each file must have at minimum the columns in BAR_SCHEMA (or a superset).
Missing `buy_volume`, `sell_volume`, `buy_count`, `sell_count` columns are
filled with 0 / 0 / 0 / 0; `is_synthetic` defaults to False.

All timestamps are stored as UTC-aware Datetime("us", "UTC").
"""

import logging
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from btcfm.data.schema import BAR_SCHEMA, validate_bar_schema

logger = logging.getLogger(__name__)

_OPTIONAL_COLS: dict[str, tuple[type, object]] = {
    "buy_volume": (pl.Float64, 0.0),
    "sell_volume": (pl.Float64, 0.0),
    "buy_count": (pl.Int64, 0),
    "sell_count": (pl.Int64, 0),
    "is_synthetic": (pl.Boolean, False),
    "vwap": (pl.Float64, None),  # fallback: close price
}


def _coerce_bar_df(df: pl.DataFrame) -> pl.DataFrame:
    """Coerce an arbitrary kline DataFrame into BAR_SCHEMA."""
    # Ensure ts is UTC-aware Datetime("us")
    if "ts" in df.columns:
        ts_col = df["ts"]
        if ts_col.dtype == pl.Utf8:
            df = df.with_columns(
                pl.col("ts")
                .str.to_datetime(format=None, use_cache=True, time_unit="us")
                .dt.replace_time_zone("UTC")
                .alias("ts")
            )
        elif ts_col.dtype == pl.Datetime("us", None):
            df = df.with_columns(
                pl.col("ts").dt.replace_time_zone("UTC")
            )
        elif ts_col.dtype == pl.Int64:
            # Assume milliseconds epoch
            df = df.with_columns(
                (pl.col("ts") * 1_000)
                .cast(pl.Datetime("us", "UTC"))
                .alias("ts")
            )
    else:
        raise ValueError("Historical bar file must contain a 'ts' column")

    # Fill optional columns
    for col, (dtype, default) in _OPTIONAL_COLS.items():
        if col not in df.columns:
            if default is None and col == "vwap":
                df = df.with_columns(pl.col("close").alias("vwap"))
            else:
                df = df.with_columns(pl.lit(default).cast(dtype).alias(col))

    return df.select(list(BAR_SCHEMA.keys()))


def load_bars(
    cache_dir: Path,
    symbol: str = "BTC-USD",
    start: date | None = None,
    end: date | None = None,
) -> pl.DataFrame:
    """
    Load 1-minute bars from the local Parquet cache.

    Parameters
    ----------
    cache_dir:
        Root directory of the cache.
    symbol:
        Trading pair subdirectory (default ``BTC-USD``).
    start, end:
        Inclusive date range filter (UTC calendar day). If None, all
        available files are loaded.

    Returns
    -------
    pl.DataFrame
        Sorted ascending by ``ts``, conforming to BAR_SCHEMA.

    Raises
    ------
    FileNotFoundError
        If no Parquet files are found for the requested range.
    """
    symbol_dir = Path(cache_dir) / symbol
    if not symbol_dir.exists():
        raise FileNotFoundError(f"Cache directory not found: {symbol_dir}")

    files = sorted(symbol_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files in {symbol_dir}")

    if start is not None or end is not None:
        files = [
            f for f in files
            if _file_in_range(f.stem, start, end)
        ]
    if not files:
        raise FileNotFoundError(
            f"No files in {symbol_dir} for date range {start}..{end}"
        )

    frames = []
    for f in files:
        try:
            df = pl.read_parquet(f)
            df = _coerce_bar_df(df)
            frames.append(df)
            logger.debug("Loaded %d bars from %s", len(df), f)
        except Exception as exc:
            logger.warning("Skipping %s: %s", f, exc)

    if not frames:
        raise FileNotFoundError("All candidate files failed to load")

    combined = pl.concat(frames).sort("ts").unique("ts", keep="first")
    validate_bar_schema(combined)
    logger.info(
        "load_bars: %d bars loaded [%s → %s]",
        len(combined),
        combined["ts"].min(),
        combined["ts"].max(),
    )
    return combined


def _file_in_range(stem: str, start: date | None, end: date | None) -> bool:
    """Return True if a filename stem (YYYY-MM-DD) falls within [start, end]."""
    try:
        file_date = date.fromisoformat(stem)
    except ValueError:
        return True  # unknown format, include it
    if start is not None and file_date < start:
        return False
    if end is not None and file_date > end:
        return False
    return True
