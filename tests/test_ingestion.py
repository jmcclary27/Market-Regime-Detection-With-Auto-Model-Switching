import pytest
from src.ingestion.fetch_market_data import fetch_market_data

def test_fetch_market_data_not_implemented():
    with pytest.raises(NotImplementedError):
        fetch_market_data("SPY", "2020-01-01", "2020-12-31")
