from __future__ import annotations

import pandas as pd
import pytest

import src.ingestion.fetch_market_data as market_data


def test_fetch_market_data_normalizes_provider_columns_without_network(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(market_data, "_YFINANCE_CACHE_READY", False)
    cache_locations: list[str] = []
    download_calls: list[dict[str, object]] = []

    provider_frame = pd.DataFrame(
        {
            ("Close", "SPY"): [100.0, 101.0],
            ("Volume", "SPY"): [1_000, 1_100],
        }
    )

    def fake_set_cache_location(path: str) -> None:
        cache_locations.append(path)

    def fake_download(**kwargs: object) -> pd.DataFrame:
        download_calls.append(kwargs)
        return provider_frame

    monkeypatch.setattr(market_data.yf, "set_tz_cache_location", fake_set_cache_location)
    monkeypatch.setattr(market_data.yf, "download", fake_download)

    result = market_data.fetch_market_data("SPY", "2020-01-01", "2020-01-10")

    assert list(result.columns) == ["close", "volume"]
    assert result.equals(pd.DataFrame({"close": [100.0, 101.0], "volume": [1_000, 1_100]}))
    assert cache_locations == ["data/yfinance_cache"]
    assert download_calls == [
        {
            "tickers": "SPY",
            "start": "2020-01-01",
            "end": "2020-01-10",
            "interval": "1d",
            "auto_adjust": False,
            "progress": False,
        }
    ]


def test_fetch_market_data_rejects_invalid_dates_before_calling_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        market_data.yf,
        "download",
        lambda **_kwargs: pytest.fail("provider must not be called for invalid dates"),
    )

    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        market_data.fetch_market_data("SPY", "2020/01/01", "2020-01-10")


def test_fetch_market_data_rejects_empty_provider_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(market_data, "_YFINANCE_CACHE_READY", True)
    monkeypatch.setattr(market_data.yf, "download", lambda **_kwargs: pd.DataFrame())

    with pytest.raises(ValueError, match="No data returned"):
        market_data.fetch_market_data("SPY", "2020-01-01", "2020-01-10")
