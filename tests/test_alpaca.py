from __future__ import annotations

from datetime import date

import pytest

from src.ingestion.alpaca import AlpacaDataError, fetch_daily_bars


def test_alpaca_requires_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
    with pytest.raises(AlpacaDataError, match="required"):
        fetch_daily_bars(["SPY", "QQQ"], start=date(2026, 8, 1), end=date(2026, 8, 2))
