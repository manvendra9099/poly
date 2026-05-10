from __future__ import annotations

import polars as pl
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Tick
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Tick:
    ts_ns: int      # nanoseconds since Unix epoch, UTC
    price: float
    size: float
    side: str       # "BUY" | "SELL"
    trade_id: str

TICK_SCHEMA: dict[str, pl.DataType] = {
    "ts_ns": pl.Int64,
    "price": pl.Float64,
    "size": pl.Float64,
    "side": pl.Utf8,
    "trade_id": pl.Utf8,
}

# ---------------------------------------------------------------------------
# Bar (1-minute OHLCV + derived columns)
# ---------------------------------------------------------------------------

BAR_SCHEMA: dict[str, pl.DataType] = {
    "ts": pl.Datetime("us", "UTC"),   # left edge of the bar (closed)
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "vwap": pl.Float64,
    "buy_volume": pl.Float64,
    "sell_volume": pl.Float64,
    "buy_count": pl.Int64,
    "sell_count": pl.Int64,
    "is_synthetic": pl.Boolean,
}

def empty_bar_dataframe() -> pl.DataFrame:
    """Return a zero-row DataFrame with the canonical bar schema."""
    return pl.DataFrame(schema=BAR_SCHEMA)

def validate_bar_schema(df: pl.DataFrame) -> None:
    """Raise ValueError if df does not conform to BAR_SCHEMA."""
    for col, dtype in BAR_SCHEMA.items():
        if col not in df.columns:
            raise ValueError(f"Bar DataFrame missing column '{col}'")
        if df[col].dtype != dtype:
            raise ValueError(
                f"Column '{col}' has dtype {df[col].dtype}, expected {dtype}"
            )
