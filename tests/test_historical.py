from __future__ import annotations

"""
Tests for coinbase_rest.py and the historical bar loader.

No network access: tests use synthetic Parquet files that mimic what
the Coinbase REST API would produce.
"""

import pytest
import polars as pl
import numpy as np
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

from btcfm.data.schema import BAR_SCHEMA
from btcfm.data.coinbase_rest import (
    _candles_to_bar_df,
    _cache_path,
    most_recent_closed_day,
)
from btcfm.data.historical import load_bars


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candle(ts_sec: int, price: float = 50000.0, vol: float = 1.0) -> list:
    """[time, low, high, open, close, volume]"""
    return [ts_sec, price * 0.999, price * 1.001, price, price, vol]


def _day_candles(day: date, n: int = 1440) -> list[list]:
    """Generate n synthetic 1-minute candles for a day, starting at 00:00 UTC."""
    start = int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())
    return [_make_candle(start + i * 60, 50000 + i * 0.01) for i in range(n)]


# ---------------------------------------------------------------------------
# _candles_to_bar_df
# ---------------------------------------------------------------------------

class TestCandlesToBarDf:
    def test_schema(self):
        candles = _day_candles(date(2024, 1, 1), n=1440)
        df = _candles_to_bar_df(candles, date(2024, 1, 1))
        for col, dtype in BAR_SCHEMA.items():
            assert col in df.columns, f"Missing column: {col}"
            assert df[col].dtype == dtype, f"{col}: {df[col].dtype} != {dtype}"

    def test_length_full_day(self):
        candles = _day_candles(date(2024, 1, 1), n=1440)
        df = _candles_to_bar_df(candles, date(2024, 1, 1))
        assert len(df) == 1440

    def test_gap_filled(self):
        """Drop 10 candles → gaps should be synthetic-filled."""
        candles = _day_candles(date(2024, 1, 1), n=1440)
        gapped = candles[:100] + candles[110:]   # remove minutes 100-109
        df = _candles_to_bar_df(gapped, date(2024, 1, 1))
        assert len(df) == 1440  # still complete
        syn_count = df["is_synthetic"].sum()
        assert syn_count >= 10, f"Expected ≥10 synthetic bars, got {syn_count}"

    def test_timestamps_utc(self):
        candles = _day_candles(date(2024, 1, 1), n=10)
        df = _candles_to_bar_df(candles, date(2024, 1, 1))
        assert str(df["ts"].dtype) == "Datetime(time_unit='us', time_zone='UTC')"

    def test_open_high_low_close_order(self):
        candles = _day_candles(date(2024, 1, 1), n=5)
        df = _candles_to_bar_df(candles, date(2024, 1, 1))
        real = df.filter(~pl.col("is_synthetic"))
        assert (real["high"] >= real["open"]).all()
        assert (real["high"] >= real["close"]).all()
        assert (real["low"] <= real["open"]).all()
        assert (real["low"] <= real["close"]).all()

    def test_empty_candles_returns_empty_df(self):
        df = _candles_to_bar_df([], date(2024, 1, 1))
        assert len(df) == 0
        for col in BAR_SCHEMA:
            assert col in df.columns

    def test_deduplication(self):
        """Duplicate candles (same timestamp) should be deduplicated."""
        candles = _day_candles(date(2024, 1, 1), n=5)
        duped = candles + candles   # 10 entries, 5 unique timestamps
        df = _candles_to_bar_df(duped, date(2024, 1, 1))
        real = df.filter(~pl.col("is_synthetic"))
        assert real["ts"].n_unique() == real.height


# ---------------------------------------------------------------------------
# most_recent_closed_day
# ---------------------------------------------------------------------------

def test_most_recent_closed_day():
    d = most_recent_closed_day()
    today = datetime.now(tz=timezone.utc).date()
    assert d == today - timedelta(days=1)


# ---------------------------------------------------------------------------
# load_bars (from historical.py)
# ---------------------------------------------------------------------------

class TestLoadBars:
    def test_load_from_cache(self, tmp_path):
        """Load bars from a pre-built Parquet cache."""
        symbol_dir = tmp_path / "BTC-USD"
        symbol_dir.mkdir()

        # Produce two day-files
        for day in [date(2024, 1, 1), date(2024, 1, 2)]:
            candles = _day_candles(day, n=1440)
            df = _candles_to_bar_df(candles, day)
            df.write_parquet(symbol_dir / f"{day.isoformat()}.parquet")

        bars = load_bars(tmp_path, symbol="BTC-USD")
        # 2 full days = 2 × 1440 = 2880 bars
        assert len(bars) == 2880

    def test_date_filter(self, tmp_path):
        symbol_dir = tmp_path / "BTC-USD"
        symbol_dir.mkdir()
        for d in [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]:
            df = _candles_to_bar_df(_day_candles(d, n=60), d)
            df.write_parquet(symbol_dir / f"{d.isoformat()}.parquet")

        bars = load_bars(tmp_path, symbol="BTC-USD",
                         start=date(2024, 1, 2), end=date(2024, 1, 2))
        assert bars["ts"].min().date() == date(2024, 1, 2)
        assert bars["ts"].max().date() == date(2024, 1, 2)

    def test_sorted_ascending(self, tmp_path):
        symbol_dir = tmp_path / "BTC-USD"
        symbol_dir.mkdir()
        for d in [date(2024, 1, 3), date(2024, 1, 1)]:   # out of order
            df = _candles_to_bar_df(_day_candles(d, n=10), d)
            df.write_parquet(symbol_dir / f"{d.isoformat()}.parquet")

        bars = load_bars(tmp_path, symbol="BTC-USD")
        ts = bars["ts"].to_list()
        assert ts == sorted(ts)

    def test_no_files_raises(self, tmp_path):
        (tmp_path / "BTC-USD").mkdir()
        with pytest.raises(FileNotFoundError):
            load_bars(tmp_path, symbol="BTC-USD")
