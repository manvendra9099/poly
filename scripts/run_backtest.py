#!/usr/bin/env python3
"""
Backtest: run verification diagnostics over a historical period.

Produces CRPS-per-lead, rank histograms, spread-skill, and pinball loss
for a trained model evaluated on held-out historical bars.

Usage::

    python scripts/run_backtest.py \\
        --config configs/default.yaml \\
        --checkpoint runs/exp01/checkpoint.pkl \\
        --data-dir /data/btc/bars \\
        --output-dir runs/exp01/backtest
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from btcfm.logging_utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="btcfm backtest")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="runs/backtest")
    args = parser.parse_args()

    raise NotImplementedError(
        "Backtest on real data not implemented in session 1. "
        "Complete historical data loader and feature pipeline first."
    )


if __name__ == "__main__":
    main()
