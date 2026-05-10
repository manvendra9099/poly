from __future__ import annotations

"""
Tick → 1-minute OHLCV bar aggregation, plus gap-filling for pre-aggregated bars.

Invariants (enforced here, tested in tests/test_bars.py):
  - Bars are left-closed, right-open on UTC minute boundaries.
  - Every minute in [first_minute, last_minute] is present.
  - Empty minutes carry forward the previous close as OHLC, volume=0, is_synthetic=True.
  - vwap is computed in the same pass; synthetic bars inherit prev close as vwap.
  - Raises if ticks is empty.
"""

import logging

import polars as pl

from btcfm.data.schema import BAR_SCHEMA, Tick, validate_bar_schema

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared gap-filling logic (used by both ticks_to_bars and coinbase_rest)
# ---------------------------------------------------------------------------

def fill_bar_gaps(df: pl.DataFrame, ts_col: str = "ts") -> pl.DataFrame:
    """
    Fill any missing 1-minute bars in a pre-sorted, possibly gappy bar DataFrame.

    For each gap minute the previous close is carried forward as
    open=high=low=close=vwap, with volume=buy_volume=sell_volume=0,
    buy_count=sell_count=0, and is_synthetic=True.

    Parameters
    ----------
    df:
        Must conform to BAR_SCHEMA (or have ``ts_col`` as a UTC-aware
        Datetime column).  Sorted ascending by ``ts_col``.
    ts_col:
        Name of the timestamp column.

    Returns
    -------
    pl.DataFrame
        Complete regular time series; conforms to BAR_SCHEMA.
    """
    first_ts = df[ts_col].min()
    last_ts = df[ts_col].max()

    all_minutes = pl.DataFrame(
        {
            ts_col: pl.datetime_range(
                first_ts, last_ts,
                interval="1m", eager=True,
                time_unit="us", time_zone="UTC",
            )
        }
    )

    merged = all_minutes.join(df, on=ts_col, how="left")
    merged = merged.with_columns(
        pl.col("close").forward_fill().alias("_prev_close")
    )

    is_syn = pl.col("volume").is_null()
    merged = merged.with_columns(
        pl.when(is_syn).then(pl.col("_prev_close")).otherwise(pl.col("open")).alias("open"),
        pl.when(is_syn).then(pl.col("_prev_close")).otherwise(pl.col("high")).alias("high"),
        pl.when(is_syn).then(pl.col("_prev_close")).otherwise(pl.col("low")).alias("low"),
        pl.when(is_syn).then(pl.col("_prev_close")).otherwise(pl.col("close")).alias("close"),
        pl.when(is_syn).then(pl.lit(0.0)).otherwise(pl.col("volume")).alias("volume"),
        pl.when(is_syn).then(pl.col("_prev_close")).otherwise(pl.col("vwap")).alias("vwap"),
        pl.when(is_syn).then(pl.lit(0.0)).otherwise(pl.col("buy_volume")).alias("buy_volume"),
        pl.when(is_syn).then(pl.lit(0.0)).otherwise(pl.col("sell_volume")).alias("sell_volume"),
        pl.when(is_syn).then(pl.lit(0)).otherwise(pl.col("buy_count")).alias("buy_count"),
        pl.when(is_syn).then(pl.lit(0)).otherwise(pl.col("sell_count")).alias("sell_count"),
        pl.when(is_syn).then(pl.lit(True)).otherwise(pl.lit(False)).alias("is_synthetic"),
    ).drop("_prev_close")

    result = merged.select(list(BAR_SCHEMA.keys()))
    result = result.with_columns(
        pl.col("buy_count").cast(pl.Int64),
        pl.col("sell_count").cast(pl.Int64),
    )
    return result


# ---------------------------------------------------------------------------
# Tick → bar aggregation
# ---------------------------------------------------------------------------

def ticks_to_bars(ticks: list[Tick]) -> pl.DataFrame:
    """
    Aggregate a list of Tick records into 1-minute OHLCV bars.

    The output covers every minute from the first trade to the last trade
    with no gaps (empty minutes are filled synthetically).

    Raises
    ------
    ValueError
        If ticks is empty.
    """
    if not ticks:
        raise ValueError("ticks_to_bars requires at least one tick")

    raw = pl.DataFrame(
        {
            "ts_ns": [t.ts_ns for t in ticks],
            "price": [t.price for t in ticks],
            "size": [t.size for t in ticks],
            "side": [t.side for t in ticks],
        }
    )
    raw = raw.with_columns(
        (pl.col("ts_ns") // 60_000_000_000 * 60_000_000)
        .cast(pl.Datetime("us", "UTC"))
        .alias("bar_ts"),
        pl.col("price").cast(pl.Float64),
        pl.col("size").cast(pl.Float64),
    )

    buy_mask = pl.col("side") == "BUY"

    agg = (
        raw.group_by("bar_ts")
        .agg(
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("size").sum().alias("volume"),
            (pl.col("price") * pl.col("size")).sum().alias("_pv"),
            pl.col("size").filter(buy_mask).sum().alias("buy_volume"),
            pl.col("size").filter(~buy_mask).sum().alias("sell_volume"),
            pl.col("size").filter(buy_mask).len().alias("buy_count"),
            pl.col("size").filter(~buy_mask).len().alias("sell_count"),
        )
        .with_columns(
            (pl.col("_pv") / pl.col("volume")).alias("vwap"),
            pl.lit(False).alias("is_synthetic"),
        )
        .drop("_pv")
        .sort("bar_ts")
        .rename({"bar_ts": "ts"})
    )

    result = fill_bar_gaps(agg)
    validate_bar_schema(result)
    logger.debug(
        "ticks_to_bars: %d ticks → %d bars (%d synthetic)",
        len(ticks), len(result), result["is_synthetic"].sum(),
    )
    return result


def bars_from_dataframe(ticks_df: pl.DataFrame) -> pl.DataFrame:
    """Convenience wrapper: accepts a Polars tick DataFrame."""
    ticks = [
        Tick(
            ts_ns=row["ts_ns"],
            price=row["price"],
            size=row["size"],
            side=row["side"],
            trade_id="",
        )
        for row in ticks_df.iter_rows(named=True)
    ]
    return ticks_to_bars(ticks)
