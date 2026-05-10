from __future__ import annotations

"""Shared logging helper — single point of logger creation for the package."""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger for ``name`` configured to emit to stderr.

    The root logger is configured on the first call; subsequent calls return
    child loggers without double-configuring the root.
    """
    root = logging.getLogger("btcfm")
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        root.addHandler(handler)
        root.setLevel(logging.INFO)
    return logging.getLogger(name)
