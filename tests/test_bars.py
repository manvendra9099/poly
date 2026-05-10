from __future__ import annotations

"""
Tests for the tick → bar aggregator.

All tests use synthetic ticks so no network access or file I/O is needed.
"""

import pytest
import polars as pl
from datetime import datetime, timezone

from btcfm.data.schema import Tick, BAR_SCHEMA
from btcfm.data.bars import ticks_to_bars


def _ts(dt_str: str) -> int:
    """Parse ISO-8601 UTC string to nanoseconds since epoch."""
    dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1_000_000_000)


# Shorthand for a minute boundary
_T0 = "2024-01-01T00:00:30Z"  # in minute 00:00
_T1 = "2024-01-01T00:01:15Z"  # in minute 00:01
_T2 = "2024-01-01T00:02:45Z"  # in minute 00:02
_T3 = "2024-01-01T00:04:00Z"  # in minute 00:04  (gap at 00:03)


def _tick(ts_str, price, size, side, trade_id="1"):
    return Tick(ts_ns=_ts(ts_str), price=price, size=size, side=side, trade_id=trade_id)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_schema_columns():
    ticks = [
        _tick(_T0, 42000.0, 0.1, "BUY", "1"),
        _tick(_T1, 42010.0, 0.2, "SELL", "2"),
    ]
    bars = ticks_to_bars(ticks)
    assert set(bars.columns) == set(BAR_SCHEMA.keys())


def test_schema_dtypes():
    ticks = [
        _tick(_T0, 42000.0, 0.1, "BUY", "1"),
        _tick(_T1, 42010.0, 0.2, "SELL", "2"),
    ]
    bars = ticks_to_bars(ticks)
    for col, dtype in BAR_SCHEMA.items():
        assert bars[col].dtype == dtype, f"{col}: {bars[col].dtype} != {dtype}"


# ---------------------------------------------------------------------------
# OHLCV correctness
# ---------------------------------------------------------------------------

def test_single_tick_bar():
    ticks = [_tick(_T0, 42000.0, 1.5, "BUY", "1")]
    bars = ticks_to_bars(ticks)
    assert len(bars) == 1
    row = bars.row(0, named=True)
    assert row["open"] == 42000.0
    assert row["high"] == 42000.0
    assert row["low"] == 42000.0
    assert row["close"] == 42000.0
    assert row["volume"] == pytest.approx(1.5)
    assert row["vwap"] == pytest.approx(42000.0)
    assert row["buy_volume"] == pytest.approx(1.5)
    assert row["sell_volume"] == pytest.approx(0.0)
    assert row["is_synthetic"] is False


def test_multi_tick_ohlcv():
    """Three ticks in the same minute: open/high/low/close correct."""
    ticks = [
        _tick(_T0, 100.0, 1.0, "BUY", "1"),
        _tick("2024-01-01T00:00:10Z", 105.0, 2.0, "SELL", "2"),
        _tick("2024-01-01T00:00:50Z", 95.0, 1.0, "BUY", "3"),
    ]
    bars = ticks_to_bars(ticks)
    assert len(bars) == 1
    row = bars.row(0, named=True)
    assert row["open"] == pytest.approx(100.0)
    assert row["high"] == pytest.approx(105.0)
    assert row["low"] == pytest.approx(95.0)
    assert row["close"] == pytest.approx(95.0)
    assert row["volume"] == pytest.approx(4.0)
    expected_vwap = (100.0 * 1.0 + 105.0 * 2.0 + 95.0 * 1.0) / 4.0
    assert row["vwap"] == pytest.approx(expected_vwap)


# ---------------------------------------------------------------------------
# Gap filling (synthetic bars)
# ---------------------------------------------------------------------------

def test_gap_filling():
    """Missing minute 00:03 should be filled synthetically."""
    ticks = [
        _tick(_T0, 42000.0, 1.0, "BUY", "1"),   # minute 00:00
        _tick(_T1, 42010.0, 1.0, "SELL", "2"),   # minute 00:01
        _tick(_T2, 42020.0, 1.0, "BUY", "3"),    # minute 00:02
        _tick(_T3, 42030.0, 1.0, "SELL", "4"),   # minute 00:04  (gap at 00:03)
    ]
    bars = ticks_to_bars(ticks)
    # Should have minutes 00:00 → 00:04 inclusive = 5 bars
    assert len(bars) == 5

    # Minute 00:03 should be synthetic
    synthetic_mask = bars["is_synthetic"]
    assert synthetic_mask.sum() == 1

    # Synthetic bar: OHLC = prev close (42020.0), volume = 0
    syn_row = bars.filter(pl.col("is_synthetic")).row(0, named=True)
    assert syn_row["open"] == pytest.approx(42020.0)
    assert syn_row["close"] == pytest.approx(42020.0)
    assert syn_row["volume"] == pytest.approx(0.0)
    assert syn_row["buy_volume"] == pytest.approx(0.0)
    assert syn_row["sell_count"] == 0


def test_no_gaps_when_contiguous():
    ticks = [
        _tick(_T0, 100.0, 1.0, "BUY", "1"),
        _tick(_T1, 101.0, 1.0, "BUY", "2"),
        _tick(_T2, 102.0, 1.0, "SELL", "3"),
    ]
    bars = ticks_to_bars(ticks)
    assert len(bars) == 3
    assert bars["is_synthetic"].sum() == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_ticks_raises():
    with pytest.raises(ValueError, match="at least one tick"):
        ticks_to_bars([])


def test_regular_time_index():
    """All consecutive bars should differ by exactly 60 seconds."""
    ticks = [
        _tick(_T0, 100.0, 1.0, "BUY", "1"),
        _tick(_T3, 102.0, 1.0, "SELL", "2"),
    ]
    bars = ticks_to_bars(ticks)
    ts = bars["ts"].to_list()
    deltas = [(ts[i+1] - ts[i]).total_seconds() for i in range(len(ts) - 1)]
    assert all(abs(d - 60.0) < 1e-6 for d in deltas), f"Irregular gaps: {deltas}"


def test_volume_imbalance_columns():
    """buy_volume + sell_volume == volume for all real bars."""
    ticks = [
        _tick(_T0, 100.0, 2.0, "BUY", "1"),
        _tick("2024-01-01T00:00:30Z", 101.0, 1.0, "SELL", "2"),
        _tick(_T1, 102.0, 0.5, "BUY", "3"),
    ]
    bars = ticks_to_bars(ticks)
    real_bars = bars.filter(~pl.col("is_synthetic"))
    for row in real_bars.iter_rows(named=True):
        assert abs(row["buy_volume"] + row["sell_volume"] - row["volume"]) < 1e-9
