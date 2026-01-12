from src.config.load_config import load_config
from src.ingestion.fetch_market_data import fetch_market_data

def main() -> None:
    cfg = load_config()
    symbols = cfg["market"]["symbols"]

    for sym in symbols:
        fetch_market_data(sym, "2020-01-01", "2020-12-31")

def run() -> None:
    main()

if __name__ == "__main__":
    main()