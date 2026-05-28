from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yfinance as yf

_YFINANCE_CACHE_READY = False


def _ensure_yfinance_cache_location() -> None:
    global _YFINANCE_CACHE_READY
    if _YFINANCE_CACHE_READY:
        return

    cache_dir = Path("data/yfinance_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))
    _YFINANCE_CACHE_READY = True


def fetch_market_data(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    interval: str = "1d",
    auto_adjust: bool = False,
):
    """
    Fetch raw market data for a given symbol and date range using yfinance.

    Args:
        symbol: Market ticker, e.g. "SPY"
        start_date: YYYY-MM-DD (inclusive)
        end_date: YYYY-MM-DD (exclusive-ish, yfinance behavior depends on interval)
        interval: yfinance interval, e.g. "1d" or "5m"
        auto_adjust: passed through to yfinance

    Returns:
        pandas.DataFrame with OHLCV columns and DatetimeIndex
    """
    try:
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError("Dates must be in YYYY-MM-DD format") from e

    _ensure_yfinance_cache_location()

    df = yf.download(
        tickers=symbol,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned for {symbol} in range {start_date} to {end_date}")

    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df
