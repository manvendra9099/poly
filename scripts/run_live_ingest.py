#!/usr/bin/env python3
"""
Live Coinbase WebSocket ingest: print ticks and optionally persist to Parquet.

Usage::

    python scripts/run_live_ingest.py
    python scripts/run_live_ingest.py --output-dir /data/btc/ticks
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure package root is on path when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from btcfm.data.coinbase_ws import stream_ticks
from btcfm.data.schema import Tick
from btcfm.logging_utils import get_logger

logger = get_logger(__name__)


def on_tick(tick: Tick) -> None:
    logger.info("TICK  trade_id=%-10s  price=%10.2f  size=%.6f  side=%s",
                tick.trade_id, tick.price, tick.size, tick.side)


async def run(output_dir: Path | None) -> None:
    await stream_ticks(on_tick, output_dir=output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live BTC tick ingest")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to persist tick Parquet files")
    args = parser.parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else None
    asyncio.run(run(output_dir))


if __name__ == "__main__":
    main()
