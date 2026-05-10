from __future__ import annotations

"""
Coinbase Exchange REST API — historical 1-minute candle fetcher.

Endpoint
--------
GET https://api.exchange.coinbase.com/products/{product_id}/candles
  ?granularity=60&start=<ISO-8601>&end=<ISO-8601>

Response: JSON array of arrays, each element is
  [time_unix_sec, low, high, open, close, volume]
  returned in **descending** (newest-first) order.
Max candles per request: 300.
No authentication required.

Cache layout
------------
  {cache_dir}/{product_id}/1m/{YYYY-MM-DD}.parquet   (one file per UTC day)

Idempotency
-----------
Fetching a date range twice is safe: existing cache files are read directly;
only missing dates trigger HTTP requests.

# DESIGN: vol_imbal and cnt_imbal features will be 0 for all REST-loaded bars
  because the candle endpoint does not provide trade direction (buy vs sell).
  The normaliser will z-score these to 0 after seeing zero variance, removing
  them from the signal. Real directional flow conditioning requires the live
  WebSocket tick stream from btcfm/data/coinbase_ws.py.

# DESIGN: vwap is approximated as close price for REST candles.
  The REST API does not expose intra-minute VWAP.
"""

import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import polars as pl
import requests

from btcfm.data.bars import fill_bar_gaps
from btcfm.data.schema import BAR_SCHEMA, validate_bar_schema

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.exchange.coinbase.com"
_CANDLE_PATH = "/products/{product}/candles"
_GRANULARITY = 60      # seconds
_MAX_CANDLES = 300     # Coinbase per-request cap
_RATE_LIMIT_DELAY = 0.25   # seconds between requests → 4 req/s (conservative)
_MAX_RETRIES = 5
_RETRY_BACKOFF_BASE = 2.0  # seconds


def _fetch_chunk(
    session: requests.Session,
    product: str,
    chunk_start: datetime,
    chunk_end: datetime,
) -> list[list]:
    """
    Fetch up to _MAX_CANDLES candles for [chunk_start, chunk_end].

    Retries with exponential backoff on 429 / 5xx.

    Returns a list of [time, low, high, open, close, volume].
    """
    url = _BASE_URL + _CANDLE_PATH.format(product=product)
    params = {
        "granularity": _GRANULARITY,
        "start": chunk_start.isoformat(),
        "end": chunk_end.isoformat(),
    }
    for attempt in range(_MAX_RETRIES):
        try:
            resp = session.get(url, params=params, timeout=30)
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = _RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    "HTTP %d for %s – %s; retrying in %.1f s",
                    resp.status_code, chunk_start, chunk_end, wait,
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            wait = _RETRY_BACKOFF_BASE ** attempt
            logger.warning("Request error (%s); retrying in %.1f s", exc, wait)
            time.sleep(wait)
    raise RuntimeError(
        f"Failed to fetch candles for {product} [{chunk_start}, {chunk_end}] "
        f"after {_MAX_RETRIES} attempts"
    )


def _fetch_day(
    session: requests.Session,
    product: str,
    day: date,
) -> list[list]:
    """Fetch all 1440 candles for a UTC calendar day (5 requests)."""
    day_start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    all_candles: list[list] = []

    for offset_min in range(0, 1440, _MAX_CANDLES):
        chunk_start = day_start + timedelta(minutes=offset_min)
        chunk_end = day_start + timedelta(
            minutes=min(offset_min + _MAX_CANDLES - 1, 1439)
        )
        candles = _fetch_chunk(session, product, chunk_start, chunk_end)
        all_candles.extend(candles)
        time.sleep(_RATE_LIMIT_DELAY)
        logger.debug(
            "Fetched %d candles for %s [%s-%s]",
            len(candles), product, chunk_start.strftime("%H:%M"), chunk_end.strftime("%H:%M"),
        )

    return all_candles


def _candles_to_bar_df(candles: list[list], day: date) -> pl.DataFrame:
    """
    Convert raw Coinbase candle list to a gap-filled bar DataFrame for one day.

    Raw format: [time_unix_sec, low, high, open, close, volume]
    """
    if not candles:
        logger.warning("No candles returned for %s", day)
        return pl.DataFrame(schema=BAR_SCHEMA)

    # Sort ascending, deduplicate by time
    sorted_candles = sorted({c[0]: c for c in candles}.values(), key=lambda c: c[0])

    ts_vals = [c[0] for c in sorted_candles]  # Unix seconds

    df = pl.DataFrame(
        {
            "ts": (pl.Series(ts_vals, dtype=pl.Int64) * 1_000_000)
                  .cast(pl.Datetime("us", "UTC")),
            "open": [float(c[3]) for c in sorted_candles],
            "high": [float(c[2]) for c in sorted_candles],
            "low":  [float(c[1]) for c in sorted_candles],
            "close": [float(c[4]) for c in sorted_candles],
            "volume": [float(c[5]) for c in sorted_candles],
            "vwap": [float(c[4]) for c in sorted_candles],  # approximate: close
            "buy_volume":  [0.0] * len(sorted_candles),
            "sell_volume": [0.0] * len(sorted_candles),
            "buy_count":   [0]   * len(sorted_candles),
            "sell_count":  [0]   * len(sorted_candles),
            "is_synthetic": [False] * len(sorted_candles),
        },
        schema=BAR_SCHEMA,
    )

    # Apply gap-filling to produce a complete regular day grid
    df = fill_bar_gaps(df)
    return df


def _cache_path(cache_dir: Path, product: str, day: date) -> Path:
    return cache_dir / product / "1m" / f"{day.isoformat()}.parquet"


def fetch_bars(
    product: str = "BTC-USD",
    start_date: date | None = None,
    end_date: date | None = None,
    cache_dir: Path = Path("data/cache/coinbase"),
    *,
    force_refresh: bool = False,
) -> pl.DataFrame:
    """
    Fetch 1-minute bars from the cache or Coinbase REST API.

    Parameters
    ----------
    product      : trading pair, default "BTC-USD"
    start_date   : first UTC day to fetch (inclusive)
    end_date     : last UTC day to fetch (inclusive); defaults to yesterday
    cache_dir    : root of the local Parquet cache
    force_refresh: if True, re-fetch all dates even if cached

    Returns
    -------
    pl.DataFrame
        Sorted ascending by ``ts``, conforms to BAR_SCHEMA.
        Covers [start_date 00:00 UTC, end_date 23:59 UTC] completely.

    Notes
    -----
    If no dates are cached and the API is unreachable, raises RuntimeError.
    The function is idempotent: running it twice yields the same result.
    """
    if end_date is None:
        end_date = datetime.now(tz=timezone.utc).date() - timedelta(days=1)
    if start_date is None:
        start_date = end_date - timedelta(days=179)  # 180 days including end

    if start_date > end_date:
        raise ValueError(f"start_date {start_date} is after end_date {end_date}")

    session = requests.Session()
    session.headers["User-Agent"] = "btcfm/0.1 (research)"

    dates = _date_range(start_date, end_date)
    frames: list[pl.DataFrame] = []
    fetched = skipped = 0

    for day in dates:
        path = _cache_path(cache_dir, product, day)
        if path.exists() and not force_refresh:
            df = pl.read_parquet(path)
            frames.append(df)
            skipped += 1
        else:
            logger.info("Fetching %s %s from API…", product, day)
            candles = _fetch_day(session, product, day)
            df = _candles_to_bar_df(candles, day)
            if len(df) > 0:
                path.parent.mkdir(parents=True, exist_ok=True)
                df.write_parquet(path)
            frames.append(df)
            fetched += 1

    logger.info(
        "fetch_bars: %d dates loaded (%d from cache, %d fetched)",
        len(dates), skipped, fetched,
    )

    if not frames:
        raise RuntimeError(f"No bars loaded for {product} [{start_date}, {end_date}]")

    combined = (
        pl.concat(frames)
        .sort("ts")
        .unique("ts", keep="first")
    )
    validate_bar_schema(combined)
    return combined


def _date_range(start: date, end: date) -> list[date]:
    days = []
    d = start
    while d <= end:
        days.append(d)
        d += timedelta(days=1)
    return days


def most_recent_closed_day() -> date:
    """Most recent UTC calendar day that is fully closed (i.e., yesterday)."""
    return datetime.now(tz=timezone.utc).date() - timedelta(days=1)
