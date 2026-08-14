from src.ingestion.fetch_market_data import fetch_market_data


def test_fetch_market_data_returns_rows():
    df = fetch_market_data("SPY", "2020-01-01", "2020-01-10")
    assert df is not None
    assert len(df) > 0
    assert "close" in df.columns
