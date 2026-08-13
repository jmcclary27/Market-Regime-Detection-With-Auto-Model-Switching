"""Minimal Alpaca daily-bar adapter, isolated from the offline yfinance path."""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


class AlpacaDataError(RuntimeError):
    """Raised when Alpaca cannot provide a valid final daily bar."""


def fetch_daily_bars(
    symbols: list[str],
    *,
    start: date,
    end: date,
    api_key: str | None = None,
    api_secret: str | None = None,
) -> pd.DataFrame:
    api_key = api_key or os.environ.get("ALPACA_API_KEY")
    api_secret = api_secret or os.environ.get("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        raise AlpacaDataError("ALPACA_API_KEY and ALPACA_API_SECRET are required")
    query = urlencode(
        {
            "symbols": ",".join(symbols),
            "timeframe": "1Day",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "adjustment": "all",
            "feed": "iex",
        }
    )
    request = Request(
        f"https://data.alpaca.markets/v2/stocks/bars?{query}",
        headers={"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret},
    )
    try:
        with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed HTTPS provider URL
            payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise AlpacaDataError("Alpaca daily-bar request failed") from exc
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        for bar in payload.get("bars", {}).get(symbol, []):
            rows.append(
                {
                    "timestamp": bar.get("t"),
                    "symbol": symbol,
                    "open": bar.get("o"),
                    "high": bar.get("h"),
                    "low": bar.get("l"),
                    "close": bar.get("c"),
                    "volume": bar.get("v"),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty or set(frame["symbol"].unique()) != set(symbols):
        raise AlpacaDataError("Alpaca did not return daily bars for every requested symbol")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if (
        frame[["open", "high", "low", "close", "volume"]].isna().any().any()
        or (frame["close"] <= 0).any()
    ):
        raise AlpacaDataError("Alpaca returned an invalid daily bar")
    return frame.sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)
