from __future__ import annotations

"""
Coinbase Advanced Trade WebSocket client — `market_trades` channel, BTC-USD.

No authentication required for public market data.

WebSocket URL:  wss://advanced-trade-ws.coinbase.com
Subscription payload::

    {
        "type": "subscribe",
        "product_ids": ["BTC-USD"],
        "channel": "market_trades"
    }

Message format (trades event)::

    {
        "channel": "market_trades",
        "client_id": "",
        "timestamp": "2024-01-01T00:00:00.000000Z",
        "sequence_num": 1,
        "events": [
            {
                "type": "update",
                "trades": [
                    {
                        "trade_id": "123456",
                        "product_id": "BTC-USD",
                        "price": "42000.00",
                        "size": "0.01",
                        "side": "BUY",
                        "time": "2024-01-01T00:00:00.000000Z"
                    }
                ]
            }
        ]
    }

Persistence: ticks are written to Parquet partitioned by UTC date under
`{output_dir}/year=YYYY/month=MM/day=DD/ticks.parquet`.

Reconnection: exponential backoff, initial 1 s, cap 60 s, jitter ±10 %.
"""

import asyncio
import json
import logging
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Callable

import polars as pl

from btcfm.data.schema import TICK_SCHEMA, Tick

logger = logging.getLogger(__name__)

_WS_URL = "wss://advanced-trade-ws.coinbase.com"

_SUBSCRIBE_MSG = json.dumps(
    {
        "type": "subscribe",
        "product_ids": ["BTC-USD"],
        "channel": "market_trades",
    }
)


def _parse_trade(trade: dict) -> Tick | None:
    """
    Parse one trade dict from the Coinbase payload into a Tick.
    Returns None and logs a warning if the trade cannot be parsed.
    """
    try:
        ts_ns = int(
            datetime.fromisoformat(
                trade["time"].replace("Z", "+00:00")
            ).timestamp()
            * 1_000_000_000
        )
        side = trade["side"].upper()
        if side not in ("BUY", "SELL"):
            logger.warning("Unknown trade side %r, skipping", side)
            return None
        return Tick(
            ts_ns=ts_ns,
            price=float(trade["price"]),
            size=float(trade["size"]),
            side=side,
            trade_id=str(trade["trade_id"]),
        )
    except (KeyError, ValueError) as exc:
        logger.warning("Failed to parse trade: %s — %r", exc, trade)
        return None


def _ticks_to_parquet(ticks: list[Tick], output_dir: Path, date_str: str) -> None:
    partition = output_dir / date_str
    partition.mkdir(parents=True, exist_ok=True)
    path = partition / "ticks.parquet"

    df_new = pl.DataFrame(
        {
            "ts_ns": [t.ts_ns for t in ticks],
            "price": [t.price for t in ticks],
            "size": [t.size for t in ticks],
            "side": [t.side for t in ticks],
            "trade_id": [t.trade_id for t in ticks],
        },
        schema=TICK_SCHEMA,
    )

    if path.exists():
        existing = pl.read_parquet(path)
        df_new = pl.concat([existing, df_new]).unique("trade_id").sort("ts_ns")

    df_new.write_parquet(path)
    logger.debug("Wrote %d ticks → %s", len(df_new), path)


async def _backoff(attempt: int) -> None:
    base = min(60.0, 2 ** attempt)
    jitter = base * random.uniform(-0.1, 0.1)
    delay = base + jitter
    logger.info("Reconnecting in %.1f s (attempt %d)", delay, attempt)
    await asyncio.sleep(delay)


async def stream_ticks(
    on_tick: Callable[[Tick], None],
    *,
    output_dir: Path | None = None,
    flush_every: int = 500,
) -> None:
    """
    Connect to Coinbase WebSocket and call `on_tick` for every incoming trade.

    Parameters
    ----------
    on_tick:
        Callback invoked synchronously for each parsed Tick.
    output_dir:
        If set, ticks are periodically flushed to Parquet under this directory.
    flush_every:
        Number of ticks to buffer before flushing to Parquet (ignored if
        output_dir is None).

    Notes
    -----
    This coroutine runs indefinitely. Cancel it (or raise KeyboardInterrupt)
    to stop. On WebSocket error, reconnects with exponential backoff (cap 60 s).
    """
    try:
        import websockets
    except ImportError as exc:
        raise ImportError("pip install websockets") from exc

    attempt = 0
    buffer: list[Tick] = []

    while True:
        try:
            logger.info("Connecting to %s", _WS_URL)
            async with websockets.connect(_WS_URL) as ws:
                await ws.send(_SUBSCRIBE_MSG)
                attempt = 0  # reset on successful connection
                logger.info("Subscribed to market_trades BTC-USD")

                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("channel") != "market_trades":
                        continue
                    for event in msg.get("events", []):
                        for trade in event.get("trades", []):
                            tick = _parse_trade(trade)
                            if tick is None:
                                continue
                            on_tick(tick)
                            if output_dir is not None:
                                buffer.append(tick)
                                if len(buffer) >= flush_every:
                                    date_str = datetime.fromtimestamp(
                                        buffer[0].ts_ns / 1e9, tz=timezone.utc
                                    ).strftime("%Y-%m-%d")
                                    _ticks_to_parquet(buffer, output_dir, date_str)
                                    buffer.clear()

        except Exception as exc:
            logger.error("WebSocket error: %s", exc, exc_info=True)
            attempt += 1
            await _backoff(attempt)
