#!/usr/bin/env python3
"""
Training launcher.

Usage::

    # Synthetic smoke-test (session 1):
    python scripts/run_train.py --config configs/small.yaml --synthetic --output-dir runs/smoke

    # Real data (after historical cache is populated):
    python scripts/run_train.py --config configs/default.yaml \\
        --data-dir /data/btc/bars --output-dir runs/exp01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from btcfm.config import load_config
from btcfm.train import train
from btcfm.logging_utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train btcfm")
    parser.add_argument("--config",     default="configs/small.yaml")
    parser.add_argument("--synthetic",  action="store_true", default=False)
    parser.add_argument("--data-dir",   default=None)
    parser.add_argument("--output-dir", default="runs/latest")
    parser.add_argument("--run-id",     default="",
                        help="Unique run identifier. Defaults to config hash.")
    args = parser.parse_args()

    config = load_config(args.config)
    train(
        config,
        synthetic=args.synthetic,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        output_dir=Path(args.output_dir),
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
