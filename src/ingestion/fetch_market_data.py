def fetch_market_data(symbol: str, start_date: str, end_date: str):
    """
    Fetch raw market data for a given symbol and date range.

    This function will later:
    - Pull data from an external API
    - Cache results locally
    - Be idempotent (safe to re-run)

    Args:
        symbol: Market ticker, e.g. "SPY"
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD

    Returns:
        pandas.DataFrame
    """
    raise NotImplementedError("Market data ingestion not implemented yet.")